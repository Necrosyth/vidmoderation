import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import json
import shutil


# paths
UPLOAD_DIR = "uploaded_videos"
DISCARD_DIR = "discarded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DISCARD_DIR, exist_ok=True)

# Load NSFW detection model
model = YOLO("640m.pt")

# Explicit content definitions
explicit_labels = [
    "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "MALE_BREAST_EXPOSED"
]

def get_video_properties(video_path):
    """Get video duration and frame count."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0:
        fps = 30  # Default assumption
    duration = total_frames / fps if fps > 0 else 0
    return duration, total_frames, fps

def process_video(video_path):
    """Process video for explicit content detection."""
    duration, total_frames, fps = get_video_properties(video_path)
    
    # Calculate threshold
    threshold_percent = 0.005 if duration > 60 else 0.08
    threshold = max(1, int(total_frames * threshold_percent))
    
    cap = cv2.VideoCapture(video_path)
    explicit_counts = {label: 0 for label in explicit_labels}
    problematic_labels = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_frame_labels = set()
        
        for result in results:
            for cls in result.boxes.cls:
                label = model.names[int(cls)]
                if label in explicit_labels:
                    current_frame_labels.add(label)
        
        for label in current_frame_labels:
            explicit_counts[label] += 1
            if explicit_counts[label] >= threshold:
                problematic_labels.add(label)
                
        # Early exit if all problematic labels are detected
        if len(problematic_labels) == len(explicit_labels):
            break

    cap.release()
    
    # Prepare results
    result = {
        "explicit_counts": explicit_counts,
        "threshold": threshold,
        "problematic_labels": problematic_labels,
        "is_safe": len(problematic_labels) == 0
    }
    return result

def display_video_names(directory, header):
    """Display video file names in a directory as a list."""
    st.subheader(header)
    videos = sorted(os.listdir(directory))
    
    if videos:
        for vid in videos:
            st.write(f"- {vid}")
    else:
        st.write("No videos available.")

def main():
    st.title("NSFW Video Moderation System")
    
    # Display lists of video names
    col1, col2 = st.columns(2)
    with col1:
        display_video_names(UPLOAD_DIR, "Approved Videos")
    with col2:
        display_video_names(DISCARD_DIR, "Rejected Videos")

    # File uploader
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Process video
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            result = process_video(tmp_path)

        # Move to appropriate directory using shutil.move to handle cross-device moves
        dest_dir = UPLOAD_DIR if result["is_safe"] else DISCARD_DIR
        dest_path = os.path.join(dest_dir, uploaded_file.name)
        shutil.move(tmp_path, dest_path)

        # Display results
        if result["is_safe"]:
            st.success(f"✅ *{uploaded_file.name}* - Approved (No explicit content detected)")
        else:
            st.error(f"❌ *{uploaded_file.name}* - Rejected")
            st.markdown("*Reasons for rejection:*")
            for label in result["problematic_labels"]:
                count = result["explicit_counts"][label]
                st.write(f"- {label} ({count} frames detected, threshold: {result['threshold']})")

if __name__ == "__main__":
    main()
