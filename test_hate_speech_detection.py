"""
Test script for PhoBERT Hate Speech Detection

This script demonstrates how to use the PhoBERT model integrated into AudioConsumer
for detecting hate speech in Vietnamese text transcribed from speech.
"""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the consumer
from consumer_audio import AudioConsumer


def test_hate_speech_detection():
    """Test hate speech detection on sample Vietnamese texts"""

    logger.info("🚀 Initializing AudioConsumer for Hate Speech Detection...")
    try:
        consumer = AudioConsumer()
    except Exception as e:
        logger.error(f"Failed to initialize AudioConsumer: {e}")
        return

    # Sample Vietnamese texts for testing
    test_texts = [
        "Xin chào, đây là một tin nhắn bình thường",  # Normal message
        "Bạn thật tuyệt vời và tốt bụng",  # Positive message
        "Tôi yêu quý bạn rất nhiều",  # Positive message
        "Người khốn kiếp, tôi sẽ hủy hoại bạn",  # Hate speech
        "Đệ tử của ma quỷ, xứng đáng bị giết",  # Hate speech
        "Những kẻ ngu ngốc như vậy không xứng sống",  # Hate speech
    ]

    logger.info("\n" + "=" * 70)
    logger.info("TESTING PHOBERT HATE SPEECH DETECTION")
    logger.info("=" * 70 + "\n")

    for i, text in enumerate(test_texts, 1):
        logger.info(f"Test {i}: {text}")

        result = consumer.check_transcribed_text(text)

        is_hate = result.get("is_hate_speech", False)
        label = result.get("label", "unknown")
        score = result.get("score", 0.0)

        status = "🚨 HATE SPEECH DETECTED" if is_hate else "✅ SAFE"
        logger.info(f"  {status}")
        logger.info(f"  Label: {label}")
        logger.info(f"  Confidence: {score:.4f}")
        logger.info("")

    # Test the integrated method for batch text
    logger.info("\n" + "=" * 70)
    logger.info("TESTING INTEGRATED DETECTION (Multiple Texts)")
    logger.info("=" * 70 + "\n")

    batch_results = []
    for text in test_texts:
        result = consumer.check_transcribed_text(text)
        batch_results.append(
            {
                "text": text,
                "is_hate_speech": result.get("is_hate_speech"),
                "confidence": result.get("score", 0.0),
            }
        )

    # Summary
    hate_count = sum(1 for r in batch_results if r["is_hate_speech"])
    safe_count = len(batch_results) - hate_count

    logger.info(
        f"Summary: {safe_count} safe messages, {hate_count} hate speeches detected\n"
    )

    for r in batch_results:
        status = "🚨" if r["is_hate_speech"] else "✅"
        logger.info(f"{status} [{r['confidence']:.1%}] {r['text'][:50]}...")

    logger.info("\n" + "=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)


def test_direct_model_inference():
    """Direct test of PhoBERT model inference"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info("\n" + "=" * 70)
    logger.info("DIRECT PHOBERT MODEL INFERENCE TEST")
    logger.info("=" * 70 + "\n")

    try:
        model_path = "models/phobert_hate_speech"
        logger.info(f"Loading model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        logger.info("Model loaded successfully!")
        logger.info(f"Model config: {model.config}")
        logger.info(f"ID to label mapping: {model.config.id2label}\n")

        # Test a simple text
        test_text = "Tôi sẽ giết bạn, thằng ngu"
        logger.info(f"Testing text: {test_text}")

        inputs = tokenizer(
            test_text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            score, pred_class = torch.max(probs, dim=-1)

        logger.info(
            f"Predicted class: {pred_class.item()} ({model.config.id2label[pred_class.item()]})"
        )
        logger.info(f"Confidence: {score.item():.4f}")
        logger.info(f"All probabilities: {probs.tolist()}")

    except Exception as e:
        logger.error(f"Error in direct model test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    logger.info("🎤 Starting Hate Speech Detection Tests...\n")

    # First run direct model test to verify model loads
    test_direct_model_inference()

    # Then test through AudioConsumer
    test_hate_speech_detection()
