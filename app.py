import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import os
from ultralytics import YOLO
import shutil
import uuid
from typing import Dict, Set, List
import logging
import asyncio

from concurrent.futures import ThreadPoolExecutor
import tempfile
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFW Video Moderation API")

# Configuration
UPLOAD_DIR = "uploaded_videos"
DISCARD_DIR = "discarded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)

# GPU Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Model Loading
MODEL = YOLO("640m.pt").to(device)
if device == "cuda":
    MODEL.half()  # Use half-precision for faster inference

# Concurrency control
MAX_CONCURRENT_VIDEOS = 4  # Adjust based on GPU memory capacity
semaphore = asyncio.Semaphore(MAX_CONCURRENT_VIDEOS)

EXPLICIT_LABELS = {
    "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "MALE_BREAST_EXPOSED"
}


class VideoProcessor:
    def __init__(self, model):
        self.model = model
        self.batch_size = 8 if device == "cuda" else 1

    def get_video_properties(self, video_path: str) -> tuple:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        fps = fps if fps > 0 else 30
        return total_frames / fps, total_frames, fps

    def process_video(self, video_path: str) -> Dict:
        duration, total_frames, fps = self.get_video_properties(video_path)
        threshold_percent = 0.005 if duration > 60 else 0.08
        threshold = max(1, int(total_frames * threshold_percent))

        cap = cv2.VideoCapture(video_path)
        explicit_counts = {label: 0 for label in EXPLICIT_LABELS}
        problematic_labels = set()
        frame_batch = []
        frame_indices = []

        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # OpenCV reads in BGR, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_batch.append(frame)
                frame_indices.append(frame_idx)
                
                if len(frame_batch) == self.batch_size:
                    self.process_batch(frame_batch, explicit_counts, problematic_labels, threshold)
                    frame_batch = []
                    frame_indices = []
                    if problematic_labels:
                        break

                frame_idx += 1

            # Process remaining frames
            if frame_batch:
                self.process_batch(frame_batch, explicit_counts, problematic_labels, threshold)

        finally:
            cap.release()

        return {
            "explicit_counts": explicit_counts,
            "threshold": threshold,
            "problematic_labels": list(problematic_labels),
            "is_safe": not bool(problematic_labels)
        }

    def process_batch(self, frames, explicit_counts, problematic_labels, threshold):
        # YOLO expects a list of numpy arrays in RGB format
        results = self.model(frames, stream=True)  # stream=True for better memory efficiency
        
        for result in results:
            current_frame_labels = set()
            if result.boxes is not None:  # Check if any detections exist
                for cls in result.boxes.cls:
                    label = self.model.names[int(cls)]
                    if label in EXPLICIT_LABELS:
                        current_frame_labels.add(label)
                
                for label in current_frame_labels:
                    explicit_counts[label] += 1
                    if explicit_counts[label] >= threshold:
                        problematic_labels.add(label)

processor = VideoProcessor(MODEL)
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    async with semaphore:
        if not file.content_type.startswith("video/"):
            raise HTTPException(400, "Invalid file type")

        try:
            # Use temporary file that auto-deletes
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    VideoProcessor(MODEL).process_video,
                    temp_file.name
                )
                
                return {
                    "status": "approved" if result["is_safe"] else "rejected",
                    "analysis": {
                        "detected_labels": result["problematic_labels"],
                        "frame_counts": result["explicit_counts"],
                        "threshold": result["threshold"]
                    }   
                }
                
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise HTTPException(500, f"Processing error: {str(e)}") from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # Use 1 worker with thread pool for CUDA compatibility
    )