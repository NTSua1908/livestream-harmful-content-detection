"""
Video Consumer: Process video frames and detect harmful content using YOLOv8m model
UPDATED VERSION: YOLOv8m violence detection with bounding box drawing
"""

import logging
import argparse
import time
import sys
from pathlib import Path
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
from typing import Dict, List, Any
import cv2
import numpy as np

# Import configurations
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_VIDEO,
    USE_VIOLENCE_CLASSIFIER,
    VIOLENCE_CLASSIFIER_THRESHOLD,
    VIOLENCE_CLASSIFIER_FRAME_SKIP,
    USE_ALL_DETECTIONS_AS_HARMFUL,
    VIOLENCE_MODEL_PATH,
    HARMFUL_CLASSES,
    LOG_LEVEL,
)

# Import utilities
from utils import (
    decode_base64_to_image,
    calculate_alert_level,
    save_image_for_training,
    draw_detections,
    encode_image_to_base64,
    MongoDBHandler,
    AlertThrottler,
)

# --- 1. CẤU HÌNH LOGGING ĐỂ XEM ĐƯỢC TRÊN AIRFLOW/FILE ---
# Tạo logger
logger = logging.getLogger("VideoConsumer")
logger.setLevel(logging.DEBUG)

# Format log
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# A. Handler ghi ra màn hình (Console/Airflow Logs)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# --- Dependency Check ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    logger.warning(
        f"YOLO dependencies missing: {e}. AI detection will be disabled."
    )
    YOLO_AVAILABLE = False


class VideoConsumer:
    """Consumer for processing video frames and detecting harmful content using YOLOv8m"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.kafka_servers = kafka_servers
        self.consumer = None
        self.db_handler = MongoDBHandler()
        self.alert_throttler = AlertThrottler(cooldown_seconds=2)
        self.frame_count = 0

        self.yolo_model = None
        self.device = None

        logger.info("Initializing VideoConsumer with YOLOv8m...")
        self.load_model()

    def load_model(self):
        """Load YOLOv8m model"""
        if not USE_VIOLENCE_CLASSIFIER:
            logger.info("🚫 Violence classifier disabled by config.")
            return

        if not YOLO_AVAILABLE:
            logger.error("❌ YOLO not available. Cannot load model.")
            return

        try:
            logger.info(f"Loading YOLO model from: {VIOLENCE_MODEL_PATH}")
            self.yolo_model = YOLO(VIOLENCE_MODEL_PATH)
            self.device = "cuda" if self.yolo_model.device.type == "cuda" else "cpu"
            logger.info(f"✅ YOLOv8m model loaded successfully on {self.device.upper()}")
            
            # Log model information
            logger.info(f"Model: {self.yolo_model.model}")
            
        except Exception as e:
            logger.error(f"❌ Critical Error loading YOLOv8m: {e}")
            self.yolo_model = None

    def connect_kafka(self):
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_VIDEO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="video-processing-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_poll_records=5,
            )
            logger.info(f"✅ Connected to Kafka: {KAFKA_TOPIC_VIDEO}")
        except KafkaError as e:
            logger.error(f"❌ Kafka Connection Failed: {e}")
            raise

    def detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Detect harmful objects using YOLOv8m model
        Returns list of detections with bounding boxes
        """
        if not USE_VIOLENCE_CLASSIFIER or self.yolo_model is None:
            return []

        if self.frame_count % VIOLENCE_CLASSIFIER_FRAME_SKIP != 0:
            return []

        try:
            # Run inference
            results = self.yolo_model(frame, conf=VIOLENCE_CLASSIFIER_THRESHOLD, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get detection info
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get class name from model
                    class_name = self.yolo_model.model.names[class_id] if hasattr(self.yolo_model.model, 'names') else f"class_{class_id}"
                    
                    # Get bounding box coordinates (normalized to 0-1)
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Denormalize to image dimensions
                    height, width = frame.shape[:2]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    
                    logger.info(f"Detection: {class_name} ({confidence:.2%}) at [{x1},{y1},{x2},{y2}]")
                    
                    detections.append({
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    })
            
            if self.frame_count % 20 == 0 and detections:
                logger.info(f"Frame {self.frame_count}: Found {len(detections)} detections")
            
            return detections

        except Exception as e:
            logger.error(f"Detection Error: {e}")
            return []

    def check_harmful_content(self, detections: List[Dict]) -> Dict[str, Any]:
        # Vì hàm detect_objects đã lọc threshold rồi,
        # nên nếu list detections không rỗng nghĩa là có độc hại.
        return {
            "is_harmful": len(detections) > 0,
            "harmful_detections": detections,
            "total_detections": len(detections),
            "harmful_count": len(detections),
        }

    def process_frame(self, message: Dict[str, Any]):
        try:
            frame_id = message.get("frame_id", -1)
            timestamp = message.get("timestamp", time.time())
            frame_data = message.get("data", "")

            if not frame_data:
                return

            frame = decode_base64_to_image(frame_data)
            if frame is None:
                return

            self.frame_count += 1

            # Detect & Check
            detections = self.detect_objects(frame)
            result = self.check_harmful_content(detections)

            if result["is_harmful"]:
                # Draw bounding boxes on the frame
                frame_with_boxes = draw_detections(frame, detections)
                
                # Encode the annotated frame to base64
                annotated_frame_data = encode_image_to_base64(frame_with_boxes)
                
                # Save to DB with annotated frame
                self.db_handler.save_detection(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "detections": result["harmful_detections"],
                        "is_harmful": True,
                        "data": annotated_frame_data,  # Save annotated frame instead
                        "original_data": frame_data,  # Keep original for reference
                    }
                )
                # Alert
                self.generate_alert(frame_id, result, frame_with_boxes)

            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames...")

        except Exception as e:
            logger.error(f"Frame Processing Error: {e}")

    def generate_alert(self, frame_id: int, harmful_result: Dict, frame):
        try:
            for det in harmful_result["harmful_detections"]:
                det_type = det["class"]
                conf = det["confidence"]
                alert_key = f"video_{det_type}"

                if self.alert_throttler.should_send_alert(alert_key):
                    level = calculate_alert_level(det_type, conf)
                    alert_data = {
                        "source": "video",
                        "frame_id": frame_id,
                        "detection_type": det_type,
                        "confidence": conf,
                        "type": level,
                        "details": f"Detected {det_type} ({conf:.1%})",
                    }
                    self.db_handler.save_alert(alert_data)
                    save_image_for_training(frame, det_type)
                    logger.warning(f"⚠️ ALERT SENT: {det_type} - Frame {frame_id}")

        except Exception as e:
            logger.error(f"Alert Gen Error: {e}")

    def run(self):
        try:
            self.connect_kafka()
            logger.info("👀 Waiting for video stream...")
            for message in self.consumer:
                self.process_frame(message.value)
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.consumer:
            self.consumer.close()
        if self.db_handler:
            self.db_handler.close()
        logger.info("Cleanup done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()
    VideoConsumer(args.kafka).run()


if __name__ == "__main__":
    main()
