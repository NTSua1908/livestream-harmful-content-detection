"""
Configuration file for the Harmful Content Detection System
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_VIDEO = "livestream-video"
KAFKA_TOPIC_AUDIO = "livestream-audio"
KAFKA_TOPIC_ALERTS = "livestream-alerts"

# Video Processing
VIDEO_FPS = 25
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 640
VIDEO_SAMPLE_RATE = 0.04  # 1/FPS seconds

# Audio Processing
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 3  # seconds
AUDIO_FORMAT = "wav"

# YOLOv8m Violence Detection Model
# Custom trained model for violence detection
VIOLENCE_MODEL_PATH = str(MODELS_DIR / "yolov8m_violence.pt")
# Enable the YOLOv8m model. If True the consumer will run object detection on frames
USE_VIOLENCE_CLASSIFIER = True
# Confidence threshold for detections (0..1)
# Lower threshold = more sensitive detection (more false positives)
# Higher threshold = less sensitive detection (may miss some violence)
VIOLENCE_CLASSIFIER_THRESHOLD = 0.5  # Confidence threshold for YOLO detections
# How many frames to skip between classifier inferences (1 = every frame)
# Increased from 1 to 5 to reduce resource usage and database bloat
# At 25 FPS: 5 = process every 200ms (~5 frames/second instead of 25)
VIOLENCE_CLASSIFIER_FRAME_SKIP = 5  # Process every 5th frame to save resources
# Batch size for classifier inference
VIOLENCE_CLASSIFIER_BATCH_SIZE = 8

# Harmful object classes that yolov8m_violence model detects
# These are the classes the custom violence detection model was trained on
HARMFUL_CLASSES = [
    "alcohol",
    "blood",
    "cigarette",
    "fight detection - v1 2024-05-10 8-55pm",
    "gun",
    "insulting_gesture",
    "knife",
]

# Alternative: Use ALL detections for demo purposes
# Set to False for real inference; True will mark every detection as harmful (testing only)
USE_ALL_DETECTIONS_AS_HARMFUL = False

# Whisper model configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Toxic words/phrases (Vietnamese and English)
TOXIC_KEYWORDS = [
    # Vietnamese toxic words
    "đồ chó",
    "con chó",
    "đồ ngu",
    "ngu ngốc",
    "khốn nạn",
    "đồ khốn",
    "mất dạy",
    "thằng ngu",
    "con ngu",
    "đồ điên",
    "thằng điên",
    "đồ khùng",
    "vô học",
    "ngu dốt",
    "súc vật",
    "đồ súc sinh",
    "đồ phản bội",
    "đồ phá hoại",
    "đồ lừa đảo",
    "đồ khốn kiếp",
    "chết tiệt",
    "đồ chết",
    "đi chết",
    "bố mày",
    "mẹ mày",
    "cút đi",
    "cút xéo",
    "đồ rác",
    "phế vật",
    "thất bại",
    "thất học",
    "vô dụng",
    # English toxic words
    "fuck",
    "shit",
    "damn",
    "bitch",
    "bastard",
    "asshole",
    "idiot",
    "stupid",
    "moron",
    "dumb",
    "loser",
    "jerk",
    "crap",
    "hell",
    "dickhead",
    "piss",
    "scum",
    "trash",
    "garbage",
    "worthless",
]

# MongoDB Configuration
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "admin123")
MONGO_DB = "livestream_detection"
MONGO_COLLECTION_DETECTIONS = "video_detections"
MONGO_COLLECTION_ALERTS = "alerts"

# Alert Configuration
ALERT_COOLDOWN = 5  # seconds between alerts for same type
ALERT_TYPES = {
    "HIGH": {"level": 3, "color": "red"},
    "MEDIUM": {"level": 2, "color": "orange"},
    "LOW": {"level": 1, "color": "yellow"},
}

# Google Colab Training Configuration
COLAB_API_URL = os.getenv("COLAB_API_URL", "http://localhost:8000")
COLAB_TRAINING_ENDPOINT = "/train"
COLAB_STATUS_ENDPOINT = "/status"
GDRIVE_DATASET_PATH = "/content/drive/MyDrive/harmful_detection_dataset"

# Airflow Configuration
AIRFLOW_RETRAIN_SCHEDULE = "@daily"  # Can be: @daily, @weekly, cron expression
MIN_NEW_SAMPLES_FOR_RETRAIN = 100

# Dashboard Configuration
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_INTERVAL = 1  # seconds

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
