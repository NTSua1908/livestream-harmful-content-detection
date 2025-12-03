"""
Utility functions for the Harmful Content Detection System
"""

import cv2
import numpy as np
import base64
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from pymongo import MongoClient

from config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_USERNAME,
    MONGO_PASSWORD,
    MONGO_DB,
    MONGO_COLLECTION_DETECTIONS,
    MONGO_COLLECTION_ALERTS,
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class MongoDBHandler:
    """Handle MongoDB connections and operations"""

    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        """Connect to MongoDB"""
        try:
            # Nếu không có user/pass, url sẽ gọn hơn. Logic này hỗ trợ cả 2.
            if MONGO_USERNAME and MONGO_PASSWORD:
                connection_string = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
            else:
                connection_string = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"

            self.client = MongoClient(connection_string)
            self.db = self.client[MONGO_DB]
            logger.info(f"Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def save_detection(self, detection_data: Dict[str, Any]):
        """Save detection result to database"""
        try:
            # Thống nhất dùng Unix timestamp nếu chưa có
            if "timestamp" not in detection_data:
                detection_data["timestamp"] = time.time()

            result = self.db[MONGO_COLLECTION_DETECTIONS].insert_one(detection_data)
            logger.debug(f"Saved detection with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save detection: {e}")
            return None

    def save_alert(self, alert_data: Dict[str, Any]):
        """Save alert to database"""
        try:
            # Thống nhất dùng Unix timestamp
            if "timestamp" not in alert_data:
                alert_data["timestamp"] = time.time()

            result = self.db[MONGO_COLLECTION_ALERTS].insert_one(alert_data)
            logger.info(f"Saved alert: {alert_data.get('detection_type', 'UNKNOWN')}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return None

    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """Get recent detections"""
        try:
            detections = (
                self.db[MONGO_COLLECTION_DETECTIONS]
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(detections)
        except Exception as e:
            logger.error(f"Failed to get detections: {e}")
            return []

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        try:
            alerts = (
                self.db[MONGO_COLLECTION_ALERTS]
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(alerts)
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def encode_image_to_base64(frame: np.ndarray) -> str:
    """Encode OpenCV frame to base64 string"""
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return jpg_as_text
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""


def decode_base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV frame"""
    try:
        if not base64_string:
            return None
        jpg_original = base64.b64decode(base64_string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on frame (Skip if bbox is None)"""
    for det in detections:
        bbox = det.get("bbox")
        if bbox is None:
            continue  # Skip drawing if no bounding box (CLIP case)

        x1, y1, x2, y2 = bbox
        label = det["class"]
        conf = det["confidence"]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label_text = f"{label}: {conf:.2f}"
        cv2.putText(
            frame,
            label_text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return frame


def check_toxic_content(text: str, toxic_keywords: List[str]) -> Dict[str, Any]:
    """Check if text contains toxic keywords (Basic substring matching)"""
    text_lower = text.lower()
    matched_keywords = []

    for keyword in toxic_keywords:
        if keyword.lower() in text_lower:
            matched_keywords.append(keyword)

    return {
        "is_toxic": len(matched_keywords) > 0,
        "matched_keywords": matched_keywords,
        "toxic_score": len(matched_keywords),
    }


def calculate_alert_level(detection_type, confidence):
    """Determine alert level based on confidence"""
    if isinstance(confidence, str):
        try:
            confidence = float(confidence.strip("%")) / 100.0
        except:
            confidence = 0.0

    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.60:
        return "MEDIUM"
    elif confidence >= 0.30:
        return "LOW"

    return "LOW"


def save_image_for_training(
    frame: np.ndarray, detection_type: str, save_dir: str = "../data/training_samples"
) -> str:
    """Save detected frame for future model training"""
    try:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{detection_type}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved training sample: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save training image: {e}")
        return ""


class AlertThrottler:
    """Prevent alert spam by throttling similar alerts"""

    def __init__(self, cooldown_seconds: int = 5):
        self.cooldown = cooldown_seconds
        self.last_alerts = {}

    def should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed using Unix timestamp"""
        current_time = time.time()  # Use Unix timestamp

        if alert_type in self.last_alerts:
            time_diff = current_time - self.last_alerts[alert_type]
            if time_diff < self.cooldown:
                return False

        self.last_alerts[alert_type] = current_time
        return True
