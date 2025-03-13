# app.py
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import os
from ultralytics import YOLO
import shutil
import uuid
from typing import Dict, Set
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import aiofiles
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFW Video Moderation API")

# Configuration
UPLOAD_DIR = "uploaded_videos"
DISCARD_DIR = "discarded_videos"
TEMP_DIR = "temp_processing"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONCURRENT_VIDEOS = 3  # Number of videos to process simultaneously
MAX_QUEUE_SIZE = 100  # Maximum number of requests to queue

# Load model once at startup
MODEL = YOLO("640m.pt")
MODEL.to(DEVICE)

# Create processing queue and semaphore
processing_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VIDEOS)
file_locks = {}
file_locks_lock = threading.Lock()

EXPLICIT_LABELS = {
    "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "MALE_BREAST_EXPOSED"
}

class VideoProcessor:
    def __init__(self, model):
        self.model = model
        
    def get_video_properties(self, video_path: str) -> tuple:
        with file_locks[video_path]:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            fps = fps if fps > 0 else 30
            duration = total_frames / fps
            return duration, total_frames, fps

    def process_video(self, video_path: str) -> Dict:
        with file_locks[video_path]:
            duration, total_frames, fps = self.get_video_properties(video_path)
            threshold_percent = 0.005 if duration > 60 else 0.08
            threshold = max(1, int(total_frames * threshold_percent))
            
            explicit_counts = {label: 0 for label in EXPLICIT_LABELS}
            problematic_labels = set()
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            try:
                frame_count = 0
                frames_batch = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Process every 3rd frame to reduce load
                    if frame_count % 3 == 0:
                        frames_batch.append(frame)
                        
                        # Process in batches of 4 frames
                        if len(frames_batch) == 4:
                            results = self.model(frames_batch, batch=4)
                            frames_batch = []
                            
                            for result in results:
                                current_frame_labels = set()
                                for cls in result.boxes.cls:
                                    label = self.model.names[int(cls)]
                                    if label in EXPLICIT_LABELS:
                                        current_frame_labels.add(label)
                                
                                for label in current_frame_labels:
                                    explicit_counts[label] += 1
                                    if explicit_counts[label] >= threshold:
                                        problematic_labels.add(label)
                    
                    frame_count += 1
                    
                    if len(problematic_labels) == len(EXPLICIT_LABELS):
                        break

                # Process any remaining frames
                if frames_batch:
                    results = self.model(frames_batch, batch=len(frames_batch))
                    # Process results as above...

            finally:
                cap.release()

            return {
                "explicit_counts": explicit_counts,
                "threshold": threshold,
                "problematic_labels": list(problematic_labels),
                "is_safe": not bool(problematic_labels)
            }

processor = VideoProcessor(MODEL)

async def process_queue():
    while True:
        try:
            # Get the next item from the queue
            temp_path, response_future = await processing_queue.get()
            
            try:
                async with processing_semaphore:
                    # Process the video
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, processor.process_video, temp_path
                    )
                    
                    # Set the result
                    response_future.set_result(result)
                    
            except Exception as e:
                response_future.set_exception(e)
                
            finally:
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
                processing_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    # Start the queue processor
    asyncio.create_task(process_queue())

@app.post("/analyze", response_class=JSONResponse)
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Invalid file type. Only video files accepted")

    # Create unique temporary file
    temp_name = f"{uuid.uuid4()}{Path(file.filename).suffix}"
    temp_path = Path(TEMP_DIR) / temp_name

    try:
        # Save uploaded file using aiofiles
        async with aiofiles.open(temp_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        # Create a lock for this file
        with file_locks_lock:
            file_locks[str(temp_path)] = threading.Lock()
        
        # Create a future for the response
        response_future = asyncio.Future()
        
        # Add to processing queue
        try:
            await processing_queue.put((str(temp_path), response_future))
        except asyncio.QueueFull:
            raise HTTPException(503, "Server is currently at capacity. Please try again later.")
        
        # Wait for processing to complete
        try:
            result = await response_future
        except Exception as e:
            raise HTTPException(500, f"Processing error: {str(e)}")
        
        # Move to appropriate directory
        dest_dir = UPLOAD_DIR if result["is_safe"] else DISCARD_DIR
        dest_path = os.path.join(dest_dir, temp_name)
        shutil.move(str(temp_path), dest_path)

        return {
            "filename": temp_name,
            "status": "approved" if result["is_safe"] else "rejected",
            "analysis": {
                "detected_labels": result["problematic_labels"],
                "frame_counts": result["explicit_counts"],
                "threshold": result["threshold"]
            }
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f"Processing error: {str(e)}") from e
    finally:
        # Clean up the file lock
        with file_locks_lock:
            file_locks.pop(str(temp_path), None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)