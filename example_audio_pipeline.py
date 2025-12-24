"""
Example: Integrating PhoBERT Hate Speech Detection with Speech-to-Text

This file demonstrates how to integrate the PhoBERT hate speech detection
with a speech-to-text (STT) system in your audio processing pipeline.
"""

import logging
from typing import Dict, Optional
import numpy as np

# Assuming you have these modules
from src.consumer_audio import AudioConsumer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioProcessingPipeline:
    """
    Complete pipeline for audio processing:
    1. Receive audio chunk
    2. Detect harmful sounds (YAMNet)
    3. Transcribe to text (Speech-to-Text)
    4. Detect hate speech (PhoBERT)
    5. Generate alerts
    6. Save results
    """

    def __init__(self):
        """Initialize the audio consumer with all models"""
        self.consumer = AudioConsumer()
        self.logger = logger

    def process_audio_chunk(
        self,
        audio_data: np.ndarray,
        chunk_id: str,
        timestamp: float,
        transcribed_text: Optional[str] = None,
    ) -> Dict:
        """
        Process an audio chunk with all detection models

        Args:
            audio_data: Audio waveform as numpy array
            chunk_id: Unique identifier for this chunk
            timestamp: Unix timestamp
            transcribed_text: Optional pre-transcribed text

        Returns:
            Dictionary with all detection results
        """

        results = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "sound_detection": None,
            "hate_speech_detection": None,
            "alerts": [],
        }

        # 1. Sound Event Detection (YAMNet)
        self.logger.info(f"🎵 Processing audio chunk {chunk_id}...")
        sound_result = self.consumer.detect_sound_events(audio_data)
        results["sound_detection"] = sound_result

        if sound_result["is_harmful"]:
            self.logger.warning(
                f"  🔊 Harmful sound detected: {sound_result['label']} "
                f"({sound_result['score']:.1%})"
            )
            results["alerts"].append(
                {
                    "type": "harmful_sound",
                    "severity": "HIGH",
                    "message": f"Harmful sound: {sound_result['label']}",
                    "confidence": sound_result["score"],
                }
            )

        # 2. Speech-to-Text (Optional - depends on your implementation)
        if transcribed_text is None:
            # Placeholder: Integrate your STT service here
            # Example with Azure Speech-to-Text:
            # transcribed_text = self.transcribe_audio_azure(audio_data)
            # or with OpenAI Whisper:
            # transcribed_text = self.transcribe_audio_whisper(audio_data)
            self.logger.info("  📝 No transcribed text provided (STT not integrated)")
            transcribed_text = ""

        # 3. Hate Speech Detection (PhoBERT)
        if transcribed_text and len(transcribed_text.strip()) > 0:
            self.logger.info(f"  📄 Transcribed text: '{transcribed_text}'")
            hate_speech_result = self.consumer.check_transcribed_text(transcribed_text)
            results["hate_speech_detection"] = hate_speech_result

            if hate_speech_result["is_hate_speech"]:
                self.logger.error(
                    f"  💬 HATE SPEECH DETECTED: {hate_speech_result['label']} "
                    f"({hate_speech_result['score']:.1%})"
                )
                results["alerts"].append(
                    {
                        "type": "hate_speech",
                        "severity": "CRITICAL",
                        "message": f"Hate speech detected: {transcribed_text}",
                        "confidence": hate_speech_result["score"],
                    }
                )
            else:
                self.logger.info(
                    f"  ✅ Speech is safe ({hate_speech_result['score']:.1%} confidence)"
                )

        # 4. Summary
        if results["alerts"]:
            self.logger.warning(
                f"⚠️  Total alerts for chunk {chunk_id}: {len(results['alerts'])}"
            )
        else:
            self.logger.info(f"✅ No issues detected in chunk {chunk_id}")

        return results

    def transcribe_audio_azure(self, audio_data: np.ndarray) -> str:
        """
        Example: Transcribe using Azure Cognitive Services Speech-to-Text

        Prerequisites:
        - pip install azure-cognitiveservices-speech
        - Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars
        """
        try:
            import azure.cognitiveservices.speech as speechsdk
            import tempfile
            import soundfile as sf

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, 16000)
                tmp_path = tmp.name

            # Initialize speech config
            speech_config = speechsdk.SpeechConfig(
                subscription=os.getenv("AZURE_SPEECH_KEY"),
                region=os.getenv("AZURE_SPEECH_REGION"),
            )
            speech_config.speech_recognition_language = "vi-VN"  # Vietnamese

            # Recognize from file
            audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedFromAudio:
                transcribed_text = result.text
                self.logger.info(f"Azure STT: {transcribed_text}")
                return transcribed_text
            else:
                self.logger.warning(f"Azure STT failed: {result.reason}")
                return ""

        except Exception as e:
            self.logger.error(f"Error in Azure STT: {e}")
            return ""

    def transcribe_audio_whisper(self, audio_data: np.ndarray) -> str:
        """
        Example: Transcribe using OpenAI Whisper

        Prerequisites:
        - pip install openai-whisper
        - Set OPENAI_API_KEY env var (for API version)
        """
        try:
            import whisper
            import tempfile
            import soundfile as sf

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, 16000)
                tmp_path = tmp.name

            # Load model
            model = whisper.load_model("base")  # or "tiny" for speed

            # Transcribe
            result = model.transcribe(tmp_path, language="vi")
            transcribed_text = result["text"]

            self.logger.info(f"Whisper STT: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Error in Whisper STT: {e}")
            return ""

    def transcribe_audio_google(self, audio_data: np.ndarray) -> str:
        """
        Example: Transcribe using Google Cloud Speech-to-Text

        Prerequisites:
        - pip install google-cloud-speech
        - Set GOOGLE_APPLICATION_CREDENTIALS env var
        """
        try:
            from google.cloud import speech_v1
            import tempfile
            import soundfile as sf

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, 16000)
                tmp_path = tmp.name

            # Initialize client
            client = speech_v1.SpeechClient()

            # Load audio
            with open(tmp_path, "rb") as f:
                audio = speech_v1.RecognitionAudio(content=f.read())

            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="vi-VN",  # Vietnamese
            )

            # Recognize
            response = client.recognize(config=config, audio=audio)

            if response.results:
                transcribed_text = response.results[0].alternatives[0].transcript
                self.logger.info(f"Google STT: {transcribed_text}")
                return transcribed_text
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Error in Google STT: {e}")
            return ""


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_1_simple_usage():
    """Example 1: Simple usage with pre-transcribed text"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 1: Simple Usage with Pre-transcribed Text")
    logger.info("=" * 70 + "\n")

    pipeline = AudioProcessingPipeline()

    # Simulate audio data
    import numpy as np

    audio_data = np.random.randn(16000).astype(np.float32)  # 1 second of audio

    # Process with pre-transcribed text
    results = pipeline.process_audio_chunk(
        audio_data=audio_data,
        chunk_id="chunk_001",
        timestamp=1703433600.0,
        transcribed_text="Xin chào, bạn khỏe không?",  # Safe text
    )

    logger.info(f"\nResults: {results}\n")


def example_2_multiple_detections():
    """Example 2: Process multiple chunks with different text"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Multiple Chunks with Various Detections")
    logger.info("=" * 70 + "\n")

    pipeline = AudioProcessingPipeline()

    test_cases = [
        ("chunk_001", "Xin chào, đây là tin nhắn bình thường"),
        ("chunk_002", "Thằng ngu, tôi sẽ giết bạn"),
        ("chunk_003", "Tôi yêu quý bạn rất nhiều"),
        ("chunk_004", "Mày xứng đáng chết, thằng khốn"),
    ]

    for chunk_id, text in test_cases:
        audio_data = np.random.randn(16000).astype(np.float32)

        results = pipeline.process_audio_chunk(
            audio_data=audio_data,
            chunk_id=chunk_id,
            timestamp=1703433600.0,
            transcribed_text=text,
        )

        hate_result = results.get("hate_speech_detection")
        if hate_result:
            status = "🚨 HATE" if hate_result["is_hate_speech"] else "✅ SAFE"
            logger.info(f"{status} [{hate_result['score']:.1%}] {text}\n")


def example_3_with_custom_stt():
    """Example 3: Integration with custom Speech-to-Text"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Custom Speech-to-Text Integration")
    logger.info("=" * 70 + "\n")

    pipeline = AudioProcessingPipeline()

    # Example audio data
    audio_data = np.random.randn(16000).astype(np.float32)

    # Process with audio data only (no pre-transcribed text)
    # In real scenario, you would call one of the STT methods:
    # transcribed_text = pipeline.transcribe_audio_whisper(audio_data)
    # or
    # transcribed_text = pipeline.transcribe_audio_azure(audio_data)

    results = pipeline.process_audio_chunk(
        audio_data=audio_data,
        chunk_id="chunk_with_stt",
        timestamp=1703433600.0,
        transcribed_text=None,  # Will skip text detection
    )

    logger.info(f"Results (without STT): {results}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import os

    logger.info("\n🎤 Audio Processing Pipeline Examples\n")

    # Run examples
    example_1_simple_usage()
    example_2_multiple_detections()
    example_3_with_custom_stt()

    logger.info("✅ All examples completed!\n")
