import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import shutil
import psutil
import threading
import torch
import concurrent.futures
from typing import List, Dict

# Initialize session state for resource monitoring
if 'resource_updates' not in st.session_state:
    st.session_state.resource_updates = 0

# Paths (use os.path.join for Windows compatibility)
UPLOAD_DIR = os.path.join("uploaded_videos")
DISCARD_DIR = os.path.join("discarded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)

# System configurations
MAX_CONCURRENT_VIDEOS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model in session state to prevent reloading
if 'model' not in st.session_state:
    try:
        # Verify model file existence before loading
        model_path = os.path.join(os.getcwd(), "640m.pt")  # Full path to model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '640m.pt' not found at: {model_path}")
            
        st.session_state.model = YOLO(model_path)
        st.session_state.model.to(DEVICE)
        st.session_state.model_initialized = True  # Add initialization flag
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        st.session_state.model_initialized = False
        st.stop()  # Stop execution if model fails to load

# Explicit content definitions
explicit_labels = [
    "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "MALE_BREAST_EXPOSED"
]

@st.cache_data
def get_system_resources() -> Dict:
    """Get current system resource usage."""
    try:
        resources = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "active_threads": threading.active_count(),
            "running_processes": len(psutil.Process().threads())
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            resources.update({
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0)/1024**2:.2f} MB"
            })
        else:
            resources["device"] = "cpu"
            
        return resources
    except Exception as e:
        st.error(f"Error getting system resources: {str(e)}")
        return {"error": str(e)}

def get_video_properties(video_path):
    """Get video duration and frame count."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        fps = 30 if fps <= 0 else fps
        duration = total_frames / fps if fps > 0 else 0
        return duration, total_frames, fps
    except Exception as e:
        st.error(f"Error reading video properties: {str(e)}")
        return 0, 0, 30

def process_single_video(video_path: str) -> Dict:
    """Process a single video file."""
    try:
        duration, total_frames, fps = get_video_properties(video_path)
        threshold_percent = 0.005 if duration > 60 else 0.08
        threshold = max(1, int(total_frames * threshold_percent))
        
        cap = cv2.VideoCapture(video_path)
        explicit_counts = {label: 0 for label in explicit_labels}
        problematic_labels = set()

        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = st.session_state.model(frame, device=DEVICE)
            
            current_frame_labels = set()
            for result in results:
                for cls in result.boxes.cls:
                    label = st.session_state.model.names[int(cls)]
                    if label in explicit_labels:
                        current_frame_labels.add(label)
            
            for label in current_frame_labels:
                explicit_counts[label] += 1
                if explicit_counts[label] >= threshold:
                    problematic_labels.add(label)
            
            frame_count += 1
            
        cap.release()
        return {
            "explicit_counts": explicit_counts,
            "threshold": threshold,
            "problematic_labels": list(problematic_labels),
            "is_safe": len(problematic_labels) == 0
        }
    except Exception as e:
        return {"error": str(e), "is_safe": False}

def process_videos(video_paths: List[str], max_workers: int) -> Dict[str, Dict]:
    """Process multiple videos concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(process_single_video, path): path 
                          for path in video_paths}
        results = {}
        
        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                results[video_path] = future.result()
            except Exception as e:
                results[video_path] = {"error": str(e), "is_safe": False}
    
    return results

def main():
    st.set_page_config(page_title="Video Moderation System", layout="wide")
    st.title("Video Moderation System")
    
    # Critical system check
    if not getattr(st.session_state, 'model_initialized', False):
        st.error("System initialization failed. Unable to start processing.")
        st.stop()

    # ... (rest of your existing main() function remains unchanged)

if __name__ == "__main__":
    main()