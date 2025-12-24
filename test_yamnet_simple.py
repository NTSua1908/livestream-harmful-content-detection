"""
Simple end-to-end test for YAMNet-based audio consumer
"""

import sys
import logging
import numpy as np

sys.path.insert(0, "d:\\Code\\doan\\src")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_simple():
    """Simple test - load and run inference"""
    logger.info("=" * 60)
    logger.info("YAMNet Audio Consumer - Simple Test")
    logger.info("=" * 60)

    try:
        from consumer_audio import AudioConsumer

        # Initialize
        logger.info("\n[1] Initializing AudioConsumer...")
        consumer = AudioConsumer()
        logger.info("✅ Initialized successfully")

        # Test YAMNet loading
        if not consumer.yamnet_model:
            logger.error("❌ YAMNet model not loaded!")
            return False
        logger.info("✅ YAMNet model loaded")

        # Create test audio
        logger.info("\n[2] Creating test audio...")
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)

        # Create silence
        silence = np.zeros(num_samples, dtype=np.float32)
        logger.info(f"   - Silence: shape={silence.shape}")

        # Create noise
        noise = np.random.randn(num_samples).astype(np.float32) * 0.1
        logger.info(f"   - Noise: shape={noise.shape}")

        # Create "screaming" (high freq sweep)
        t = np.linspace(0, duration, num_samples)
        freq_start, freq_end = 800, 2000
        frequency = np.linspace(freq_start, freq_end, num_samples)
        phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
        screaming = np.sin(phase).astype(np.float32) * 0.3
        logger.info(f"   - Screaming: shape={screaming.shape}")

        # Test detection
        logger.info("\n[3] Testing YAMNet detection...")

        test_cases = [
            ("Silence", silence),
            ("White Noise", noise),
            ("Screaming-like", screaming),
        ]

        for name, audio in test_cases:
            result = consumer.detect_sound_events(audio)
            logger.info(
                f"\n   {name}:"
                f"\n     - Is Harmful: {result['is_harmful']}"
                f"\n     - Score: {result['score']:.2f}"
                f"\n     - Label: {result['label']}"
            )

        logger.info("\n" + "=" * 60)
        logger.info("✅ Test completed successfully!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple()
    exit(0 if success else 1)
