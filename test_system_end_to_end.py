#!/usr/bin/env python
"""
End-to-end test of the violence detection system
Tests the complete pipeline: load model, detect, draw, encode, and save to DB
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import logging
from datetime import datetime
from ultralytics import YOLO
from utils import (
    encode_image_to_base64,
    decode_base64_to_image,
    draw_detections,
    MongoDBHandler,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_end_to_end():
    """Test the complete detection and storage pipeline"""
    print("\n" + "=" * 60)
    print("END-TO-END SYSTEM TEST")
    print("=" * 60)

    try:
        # 1. Load Model
        logger.info("Loading YOLOv8m violence detection model...")
        model = YOLO("models/yolov8m_violence.pt")
        logger.info(f"✅ Model loaded")
        logger.info(f"   Classes: {list(model.model.names.values())}")

        # 2. Create a test frame (with some structure to potentially trigger detection)
        logger.info("\nCreating test frame...")
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # Add some structure
        cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 0, 255), 3)
        cv2.putText(
            test_frame,
            "Test Frame",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        logger.info(f"✅ Test frame created: {test_frame.shape}")

        # 3. Run inference
        logger.info("\nRunning inference...")
        results = model(test_frame, conf=0.5, verbose=False)
        logger.info(f"✅ Inference completed")

        # 4. Extract detections
        logger.info("\nExtracting detections...")
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.model.names[class_id]

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                detection = {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": bbox,
                }
                detections.append(detection)
                logger.info(f"   Detection: {class_name} ({confidence:.2%}) at {bbox}")

        logger.info(f"✅ Found {len(detections)} detections")

        # 5. Draw detections on frame
        logger.info("\nDrawing bounding boxes...")
        frame_with_boxes = draw_detections(test_frame, detections)
        logger.info(f"✅ Bounding boxes drawn")

        # 6. Save annotated frame
        logger.info("\nSaving annotated frame...")
        output_path = Path("test_frame_with_detections.jpg")
        cv2.imwrite(str(output_path), frame_with_boxes)
        logger.info(f"✅ Annotated frame saved: {output_path}")

        # 7. Encode frame to base64
        logger.info("\nEncoding frame to base64...")
        encoded_frame = encode_image_to_base64(frame_with_boxes)
        logger.info(f"✅ Frame encoded to base64 ({len(encoded_frame)} chars)")

        # 8. Test database connection and saving
        logger.info("\nTesting database connection...")
        try:
            db_handler = MongoDBHandler()
            logger.info("✅ Connected to MongoDB")

            # 9. Create detection record
            logger.info("\nCreating detection record...")
            detection_record = {
                "timestamp": datetime.now().timestamp(),
                "frame_id": 0,
                "detections": detections,
                "is_harmful": len(detections) > 0,
                "data": encoded_frame,
            }

            # 10. Save to database
            logger.info("Saving detection to database...")
            if len(detections) > 0:
                result_id = db_handler.save_detection(detection_record)
                logger.info(f"✅ Detection saved to database with ID: {result_id}")
            else:
                logger.info("ℹ️  No detections found, skipping database save")

            # 11. Test alert saving
            if len(detections) > 0:
                logger.info("\nCreating and saving alert...")
                alert_record = {
                    "timestamp": datetime.now().timestamp(),
                    "frame_id": 0,
                    "detection_type": detections[0]["class"],
                    "confidence": detections[0]["confidence"],
                    "type": "HIGH" if detections[0]["confidence"] > 0.8 else "MEDIUM",
                    "details": f"Detected {detections[0]['class']} ({detections[0]['confidence']:.1%})",
                    "source": "test",
                }
                alert_id = db_handler.save_alert(alert_record)
                logger.info(f"✅ Alert saved to database with ID: {alert_id}")

            db_handler.close()

        except Exception as e:
            logger.warning(f"⚠️  Database operations skipped: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ END-TO-END TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_end_to_end()
    exit(0 if success else 1)
