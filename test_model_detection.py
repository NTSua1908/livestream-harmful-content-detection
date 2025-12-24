#!/usr/bin/env python
"""
Test script for YOLOv8m violence detection model
Tests model loading, inference, and bounding box drawing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from ultralytics import YOLO
from utils import encode_image_to_base64, decode_base64_to_image, draw_detections


def test_model_loading():
    """Test if model loads correctly"""
    print("=" * 50)
    print("TEST 1: Model Loading")
    print("=" * 50)

    try:
        model_path = Path("models/yolov8m_violence.pt")
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            return False

        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Model path: {model_path}")
        print(f"   Classes: {model.model.names}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_inference():
    """Test inference on a dummy frame"""
    print("\n" + "=" * 50)
    print("TEST 2: Inference on Dummy Frame")
    print("=" * 50)

    try:
        model = YOLO("models/yolov8m_violence.pt")

        # Create a dummy frame (640x640 RGB image with some random noise)
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        print(f"   Frame shape: {dummy_frame.shape}")
        print(f"   Frame dtype: {dummy_frame.dtype}")

        # Run inference
        results = model(dummy_frame, conf=0.5, verbose=False)

        print(f"✅ Inference completed")
        print(f"   Number of results: {len(results)}")

        for i, result in enumerate(results):
            boxes = result.boxes
            print(f"   Result {i}: {len(boxes)} detections")

            for j, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.model.names[class_id]
                print(f"     Detection {j}: {class_name} ({confidence:.2%})")

        return True
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False


def test_bounding_box_drawing():
    """Test drawing bounding boxes on frame"""
    print("\n" + "=" * 50)
    print("TEST 3: Bounding Box Drawing")
    print("=" * 50)

    try:
        model = YOLO("models/yolov8m_violence.pt")

        # Create dummy frame with some structure
        dummy_frame = np.ones((640, 640, 3), dtype=np.uint8) * 200

        # Create mock detections (simulating what model would return)
        mock_detections = [
            {
                "class": "knife",
                "class_id": 6,
                "confidence": 0.85,
                "bbox": [100, 100, 200, 200],
            },
            {
                "class": "gun",
                "class_id": 4,
                "confidence": 0.92,
                "bbox": [300, 150, 400, 250],
            },
        ]

        print(f"   Original frame shape: {dummy_frame.shape}")
        print(f"   Mock detections: {len(mock_detections)}")

        # Draw detections
        frame_with_boxes = draw_detections(dummy_frame, mock_detections)

        print(f"✅ Bounding boxes drawn successfully")
        print(f"   Output frame shape: {frame_with_boxes.shape}")

        # Save test image
        output_path = Path("test_output_with_boxes.jpg")
        cv2.imwrite(str(output_path), frame_with_boxes)
        print(f"   Test image saved: {output_path}")

        return True
    except Exception as e:
        print(f"❌ Error drawing bounding boxes: {e}")
        return False


def test_image_encoding():
    """Test image encoding/decoding"""
    print("\n" + "=" * 50)
    print("TEST 4: Image Encoding/Decoding")
    print("=" * 50)

    try:
        # Create test frame
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print(f"   Original frame shape: {test_frame.shape}")

        # Encode to base64
        encoded = encode_image_to_base64(test_frame)
        print(f"✅ Image encoded to base64")
        print(f"   Encoded string length: {len(encoded)} chars")

        # Decode from base64
        decoded = decode_base64_to_image(encoded)
        print(f"✅ Image decoded from base64")
        print(f"   Decoded frame shape: {decoded.shape}")

        # Verify they match
        if decoded.shape == test_frame.shape:
            print(f"✅ Shapes match!")
        else:
            print(f"❌ Shapes don't match: {test_frame.shape} vs {decoded.shape}")
            return False

        return True
    except Exception as e:
        print(f"❌ Error in encoding/decoding: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🔍 " * 15)
    print("YOLOv8m Violence Detection - System Tests")
    print("🔍 " * 15)

    tests = [
        ("Model Loading", test_model_loading),
        ("Inference", test_inference),
        ("Bounding Box Drawing", test_bounding_box_drawing),
        ("Image Encoding", test_image_encoding),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for r in results.values() if r)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
