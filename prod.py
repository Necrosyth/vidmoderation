from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import tempfile
import os
import magic
import hashlib
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from ultralytics import YOLO

# --- Flask App Setup ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB file size limit

# Set up logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Rate Limiting
limiter = Limiter(app, key_func=get_remote_address)

# --- API Key Authentication ---
API_KEY = "your-secret-api-key"  # Load securely from environment variables in production

def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-KEY') != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return view_function(*args, **kwargs)
    return decorated_function

# --- Model File Integrity ---
MODEL_PATH = "640m.pt"
EXPECTED_CHECKSUM = "your_precomputed_sha256_hash_here"  # Replace with actual hash

def verify_model_integrity(filepath, expected_checksum):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest() == expected_checksum

if not verify_model_integrity(MODEL_PATH, EXPECTED_CHECKSUM):
    raise Exception("Model file integrity check failed!")

model = YOLO(MODEL_PATH)
explicit_labels = [
    "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "MALE_BREAST_EXPOSED"
]

# --- File Type Validation ---
def is_video_file(file_stream) -> bool:
    file_stream.seek(0)
    file_header = file_stream.read(2048)
    file_stream.seek(0)
    mime = magic.from_buffer(file_header, mime=True)
    return mime.startswith("video/")

# --- Video Processing Function ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps

    threshold_percent = 0.005 if duration > 60 else 0.08
    threshold = max(1, int(total_frames * threshold_percent))
    
    counts = {label: 0 for label in explicit_labels}
    problematic = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_labels = set()
        for result in results:
            for cls in result.boxes.cls:
                label = model.names[int(cls)]
                if label in explicit_labels:
                    current_labels.add(label)
        for label in current_labels:
            counts[label] += 1
            if counts[label] >= threshold and label not in problematic:
                problematic[label] = {"count": counts[label], "threshold": threshold}
        if len(problematic) == len(explicit_labels):
            break

    cap.release()
    return {
        "explicit": len(problematic) > 0,
        "reasons": problematic
    }

# --- Error Handling ---
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception("Unhandled Exception")
    return jsonify({"error": "An internal error occurred."}), 500

# --- Main API Endpoint ---
@app.route('/check', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def video_check():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not is_video_file(file.stream):
        return jsonify({"error": "Uploaded file is not a valid video"}), 400

    filename = secure_filename(file.filename)
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
            file.save(tmp.name)
            result = process_video(tmp.name)
        os.remove(tmp.name)
        return jsonify({
            "explicit": result["explicit"],
            "reasons": result["reasons"]
        })
    except Exception as e:
        app.logger.exception("Error processing video")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # For production, run with Gunicorn or uWSGI instead of Flask's built-in server.
    app.run(host='0.0.0.0', port=5000)
