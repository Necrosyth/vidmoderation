import os
from flask import Flask, request, jsonify
import cv2
import tempfile
from ultralytics import YOLO
import json
import shutil

app = Flask(__name__)

# Paths
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

        if len(problematic_labels) == len(explicit_labels):
            break

    cap.release()
    
    # Prepare results
    result = {
        "explicit_counts": explicit_counts,
        "threshold": threshold,
        "problematic_labels": list(problematic_labels),
        "is_safe": len(problematic_labels) == 0
    }
    return result

@app.route('/moderate_video', methods=['POST'])
def moderate_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Use tempfile for secure file operations
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = process_video(tmp_path)
            # Move file based on result
            dest_dir = UPLOAD_DIR if result["is_safe"] else DISCARD_DIR
            dest_path = os.path.join(dest_dir, file.filename)
            shutil.move(tmp_path, dest_path)

            response = {
                "status": "success",
                "is_safe": result["is_safe"],
                "details": result
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Ensure temporary file is deleted
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False for production