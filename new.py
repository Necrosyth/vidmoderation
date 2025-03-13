import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import json
import shutil
import psutil
import threading
import torch
import concurrent.futures
from typing import List, Dict

# Initialize session state for resource monitoring
if 'resource_updates' not in st.session_state:
    st.session_state.resource_updates = 0

# paths
UPLOAD_DIR = "uploaded_videos"
DISCARD_DIR = "discarded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)

# System configurations
MAX_CONCURRENT_VIDEOS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model in session state to prevent reloading
if 'model' not in st.session_state:
    st.session_state.model = YOLO("640m.pt")
    st.session_state.model.to(DEVICE)

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
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Device selection
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    
    selected_device = st.sidebar.selectbox(
        "Processing Device",
        available_devices,
        index=0 if DEVICE == "cpu" else 1
    )
    
    # Update device if changed
    if selected_device != DEVICE:
        try:
            st.session_state.model.to(selected_device)
            st.sidebar.success(f"Model moved to {selected_device}")
        except Exception as e:
            st.sidebar.error(f"Failed to change device: {str(e)}")
    
    # Concurrent processing settings
    max_workers = st.sidebar.slider(
        "Max Concurrent Videos",
        min_value=1,
        max_value=8,
        value=MAX_CONCURRENT_VIDEOS
    )
    
    # System resources
    st.sidebar.subheader("System Resources")
    resources = get_system_resources()
    
    if "error" not in resources:
        st.sidebar.write(f"Device: {resources['device']}")
        st.sidebar.write(f"CPU Usage: {resources['cpu_percent']}%")
        st.sidebar.write(f"Memory Usage: {resources['memory_percent']}%")
        st.sidebar.write(f"Active Threads: {resources['active_threads']}")
        
        if resources['device'] == 'cuda':
            st.sidebar.write(f"GPU: {resources['gpu_name']}")
            st.sidebar.write(f"GPU Memory: {resources['gpu_memory_allocated']}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Approved Videos")
        approved_videos = os.listdir(UPLOAD_DIR)
        if approved_videos:
            for video in approved_videos:
                st.write(f"✅ {video}")
        else:
            st.write("No approved videos")
    
    with col2:
        st.subheader("Rejected Videos")
        rejected_videos = os.listdir(DISCARD_DIR)
        if rejected_videos:
            for video in rejected_videos:
                st.write(f"❌ {video}")
        else:
            st.write("No rejected videos")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload videos for processing",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=True,
        key="video_uploader"
    )
    
    if uploaded_files:
        temp_paths = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save uploaded files
        for i, file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.read())
                temp_paths.append((tmp.name, file.name))
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Saved {i + 1}/{len(uploaded_files)} files")
        
        status_text.text("Processing videos...")
        results = process_videos([path for path, _ in temp_paths], max_workers)
        
        # Process results
        for (tmp_path, original_name) in temp_paths:
            result = results.get(tmp_path, {"error": "Processing failed", "is_safe": False})
            
            if "error" in result:
                st.error(f"Error processing {original_name}: {result['error']}")
                continue
                
            dest_dir = UPLOAD_DIR if result["is_safe"] else DISCARD_DIR
            dest_path = os.path.join(dest_dir, original_name)
            
            try:
                shutil.move(tmp_path, dest_path)
                
                if result["is_safe"]:
                    st.success(f"✅ {original_name} - Approved")
                else:
                    st.error(f"❌ {original_name} - Rejected")
                    st.write("Detected content:")
                    for label in result["problematic_labels"]:
                        count = result["explicit_counts"][label]
                        st.write(f"- {label}: {count} frames")
            except Exception as e:
                st.error(f"Error saving {original_name}: {str(e)}")
        
        status_text.text("Processing complete!")
        progress_bar.empty()

if __name__ == "__main__":
    main()