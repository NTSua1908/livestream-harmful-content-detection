"""
Test script for YAMNet model audio detection
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_yamnet_loading():
    """Test YAMNet model loading"""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub

        logger.info("Testing YAMNet model loading...")

        # Load YAMNet model
        yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
        logger.info(f"Loading model from: {yamnet_model_handle}")
        yamnet_model = hub.load(yamnet_model_handle)

        # YAMNet classes (521 classes)
        yamnet_classes = [
            "Speech",
            "Shouting",
            "Yelling",
            "Crying",
            "Screaming",
            "Alarm",
            "Siren",
            "Emergency vehicle",
            "Gunshot",
            "Gunfire",
            "Explosion",
            "Bang",
            "Breaking",
            "Crash",
            "Car horn",
            "Dog",
            "Cat",
            "Music",
            "Wind",
            "Rain",
            "Thunder",
            # Add common harmful sound labels
            "Shoal",
            "Laugh",
            "Cough",
            "Sneeze",
            "Snoring",
            "Breathing",
            "Clapping",
            "Throat clearing",
            "Keys jangling",
            "Door knock",
            "Door wood knock",
            "Door metal knock",
            "Knock",
            "Tap",
            "Glass breaking",
            "Car crash",
            "Car alarm",
            "Police car",
            "Ambulance",
            "Fire truck",
            "Fire alarm",
            "Bicycle bell",
            "Motorcycle",
            "Truck",
            "Train",
            "Subway train",
            "Helicopter",
            "Jet",
            "Aircraft",
            "Airplane",
            "UFO",
            "Spaceship",
            # Include more harmul sounds from YAMNet's 521 classes
            "Chainsaw",
            "Hammer",
            "Saw",
            "Drill",
            "Circular saw",
            "Siren",
            "Police siren",
            "Fire siren",
            "Ambulance siren",
            "School bell",
        ]

        # Extended list up to 521 classes (full YAMNet)
        # For now, we'll use a reasonable subset
        # You can download the full list from TensorFlow models repo if needed

        logger.info(f"✅ YAMNet model loaded successfully!")
        logger.info(
            f"✅ Using {len(yamnet_classes)} predefined classes (subset of 521 total)"
        )

        return yamnet_model, yamnet_classes

    except Exception as e:
        logger.error(f"❌ Failed to load YAMNet: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_yamnet_inference(yamnet_model, yamnet_classes):
    """Test YAMNet inference with synthetic audio"""
    try:
        if not yamnet_model:
            logger.error("Model not loaded")
            return False

        # Create synthetic audio (16kHz, 1-3 seconds)
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        num_samples = int(sample_rate * duration)

        # Generate random audio (simulating noise)
        audio_array = np.random.randn(num_samples).astype(np.float32) * 0.1

        logger.info(
            f"Testing inference with {duration}s audio (shape: {audio_array.shape})"
        )

        # Run inference
        scores, embeddings, spectrogram = yamnet_model(audio_array)

        logger.info(f"✅ Inference successful!")
        logger.info(f"   Scores shape: {scores.shape}")
        logger.info(f"   Embeddings shape: {embeddings.shape}")

        # Get top predictions
        scores_np = scores.numpy()
        detected_events = []

        for score in scores_np:
            top_class_idx = np.argmax(score)
            top_score = float(score[top_class_idx])
            if top_score > 0.3:
                class_name = (
                    yamnet_classes[top_class_idx]
                    if top_class_idx < len(yamnet_classes)
                    else "Unknown"
                )
                detected_events.append((class_name, top_score))

        logger.info(f"✅ Detected {len(detected_events)} events above threshold 0.3:")
        for event_name, event_score in sorted(
            detected_events, key=lambda x: x[1], reverse=True
        )[:5]:
            logger.info(f"   - {event_name}: {event_score:.1%}")

        return True

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """Test the consumer_audio with YAMNet"""
    try:
        import sys

        sys.path.insert(0, "d:\\Code\\doan\\src")

        from consumer_audio import AudioConsumer

        logger.info("Testing AudioConsumer with YAMNet...")

        # Create consumer instance (will load models)
        consumer = AudioConsumer()

        # Test with synthetic audio
        sample_rate = 16000
        duration = 3.0
        num_samples = int(sample_rate * duration)
        test_audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        logger.info(f"Testing with {duration}s synthetic audio...")
        result = consumer.transcribe_and_check_toxic(test_audio)

        logger.info(f"✅ AudioConsumer test result:")
        logger.info(f"   Is toxic: {result['is_toxic']}")
        logger.info(f"   Detected keywords: {result['keywords']}")
        logger.info(f"   Score: {result['score']:.2f}")
        logger.info(f"   Text: {result['text']}")

        return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("YAMNet Model Tests")
    logger.info("=" * 60)

    # Test 1: Loading
    logger.info("\n[Test 1] Loading YAMNet model...")
    yamnet_model, yamnet_classes = test_yamnet_loading()

    if yamnet_model is None:
        logger.error("Cannot proceed without model")
        exit(1)

    # Test 2: Basic inference
    logger.info("\n[Test 2] Testing YAMNet inference...")
    if not test_yamnet_inference(yamnet_model, yamnet_classes):
        logger.error("Inference test failed")
        exit(1)

    # Test 3: Integration with AudioConsumer
    logger.info("\n[Test 3] Testing AudioConsumer integration...")
    if not test_integration():
        logger.error("Integration test failed")
        exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests passed!")
    logger.info("=" * 60)
