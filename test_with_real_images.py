#!/usr/bin/env python
"""
Test with real sample images from training data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
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


def test_with_real_images():
    """Test the system with real sample images"""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL SAMPLE IMAGES")
    print("=" * 60)

    try:
        # Load model
        logger.info("Loading YOLOv8m violence detection model...")
        model = YOLO("models/yolov8m_violence.pt")
        logger.info(f"✅ Model loaded with {len(model.model.names)} classes")

        # Find sample images
        sample_dir = Path("data/training_samples")
        if not sample_dir.exists():
            logger.error(f"Sample directory not found: {sample_dir}")
            return False

        sample_images = list(sample_dir.glob("*.jpg"))[:5]  # Test first 5 images
        logger.info(f"Found {len(sample_images)} sample images")

        db_handler = MongoDBHandler()

        # Process each image
        for idx, img_path in enumerate(sample_images, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing image {idx}/{len(sample_images)}: {img_path.name}")
            logger.info("=" * 60)

            try:
                # Read image
                frame = cv2.imread(str(img_path))
                if frame is None:
                    logger.error(f"Failed to read image: {img_path}")
                    continue

                logger.info(f"Image shape: {frame.shape}")

                # Run inference
                logger.info("Running inference...")
                results = model(frame, conf=0.5, verbose=False)

                # Extract detections
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
                        logger.info(
                            f"✅ Detection: {class_name} ({confidence:.2%}) at {bbox}"
                        )

                if len(detections) == 0:
                    logger.info("ℹ️  No detections found")
                else:
                    # Draw detections
                    logger.info("Drawing bounding boxes...")
                    frame_with_boxes = draw_detections(frame, detections)

                    # Save annotated frame
                    output_path = Path(f"test_result_{idx}_with_detections.jpg")
                    cv2.imwrite(str(output_path), frame_with_boxes)
                    logger.info(f"✅ Annotated frame saved: {output_path}")

                    # Encode frame
                    encoded_frame = encode_image_to_base64(frame_with_boxes)
                    logger.info(f"✅ Frame encoded ({len(encoded_frame)} chars)")

                    # Save to database
                    logger.info("Saving to database...")
                    detection_record = {
                        "timestamp": datetime.now().timestamp(),
                        "frame_id": idx,
                        "file_path": str(img_path),
                        "detections": detections,
                        "is_harmful": len(detections) > 0,
                        "data": encoded_frame,
                    }

                    result_id = db_handler.save_detection(detection_record)
                    logger.info(f"✅ Detection saved with ID: {result_id}")

                    # Save alert
                    for det in detections:
                        alert_record = {
                            "timestamp": datetime.now().timestamp(),
                            "frame_id": idx,
                            "detection_type": det["class"],
                            "confidence": det["confidence"],
                            "type": "HIGH"
                            if det["confidence"] > 0.8
                            else "MEDIUM"
                            if det["confidence"] > 0.6
                            else "LOW",
                            "details": f"Detected {det['class']} ({det['confidence']:.1%})",
                            "source": "test",
                        }
                        alert_id = db_handler.save_alert(alert_record)
                        logger.info(f"✅ Alert saved with ID: {alert_id}")

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                import traceback

                traceback.print_exc()
                continue

        db_handler.close()

        logger.info("\n" + "=" * 60)
        logger.info("✅ REAL IMAGE TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_real_images()
    exit(0 if success else 1)
