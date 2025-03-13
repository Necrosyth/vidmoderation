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
import logging #optional
#optional
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NSFW Video Moderation API")

# Configuration
UPLOAD_DIR = "uploaded_videos"
DISCARD_DIR = "discarded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)

# Load model once at startup

MODEL = YOLO("640m.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
logger.info(f"Device being used: {device}")
MODEL.to(device)  # Move model to the specified device

EXPLICIT_LABELS = {
    "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "MALE_BREAST_EXPOSED"
}

class VideoProcessor:
    def __init__(self, model):
        self.model = model
        
    def get_video_properties(self, video_path: str) -> tuple:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        fps = fps if fps > 0 else 30
        duration = total_frames / fps
        return duration, total_frames, fps

    def process_video(self, video_path: str) -> Dict:
        duration, total_frames, fps = self.get_video_properties(video_path)
        threshold_percent = 0.005 if duration > 60 else 0.08
        threshold = max(1, int(total_frames * threshold_percent))
        
        cap = cv2.VideoCapture(video_path)
        explicit_counts = {label: 0 for label in EXPLICIT_LABELS}
        problematic_labels = set()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame)
                current_frame_labels = set()
                
                for result in results:
                    for cls in result.boxes.cls:
                        label = self.model.names[int(cls)]
                        if label in EXPLICIT_LABELS:
                            current_frame_labels.add(label)
                
                for label in current_frame_labels:
                    explicit_counts[label] += 1
                    if explicit_counts[label] >= threshold:
                        problematic_labels.add(label)
                
                if len(problematic_labels) == len(EXPLICIT_LABELS):
                    break
        finally:
            cap.release()

        return {
            "explicit_counts": explicit_counts,
            "threshold": threshold,
            "problematic_labels": list(problematic_labels),
            "is_safe": not bool(problematic_labels)
        }

processor = VideoProcessor(MODEL)

@app.post("/analyze", response_class=JSONResponse)
async def analyze_video(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Invalid file type. Only video files accepted")

    # Create temp file with unique name
    file_ext = os.path.splitext(file.filename)[1]
    temp_name = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join("/tmp", temp_name)

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video
        result = processor.process_video(temp_path)
        
        # Move to appropriate directory
        dest_dir = UPLOAD_DIR if result["is_safe"] else DISCARD_DIR
        dest_path = os.path.join(dest_dir, temp_name)
        shutil.move(temp_path, dest_path)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)