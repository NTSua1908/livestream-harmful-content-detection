"""
Audio Consumer Optimized: Rolling Buffer + Faster-Whisper + AST
Target: Lightweight, Fast, Accurate for Vietnamese & Sound Events
"""

import logging
import argparse
import json
import base64
import tempfile
import os
import numpy as np
import librosa
from typing import Dict

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Configs imports
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_AUDIO,
    LOG_LEVEL,
)
from utils import (
    check_toxic_content,
    MongoDBHandler,
    AlertThrottler,
)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- IMPORT MODELS ---

# 1. Torch & AST (Sound Event Detection)
try:
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"❌ Transformers/Torch import failed: {e}")
    TRANSFORMERS_AVAILABLE = False

# 2. YAMNet (Sound Event Detection - replacing Whisper for audio detection)
try:
    import tensorflow as tf
    import tensorflow_hub as hub

    YAMNET_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"❌ YAMNet not found. Install: pip install tensorflow tensorflow-hub. Error: {e}"
    )
    YAMNET_AVAILABLE = False

# 3. PhoBERT (Hate Speech Detection for Vietnamese text)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    PHOBERT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"❌ PhoBERT import failed: {e}")
    PHOBERT_AVAILABLE = False

# 4. Whisper (Speech-to-Text for Vietnamese)
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"❌ Whisper not available: {e}. Install: pip install openai-whisper"
    )
    WHISPER_AVAILABLE = False


class AudioConsumer:
    """Consumer for processing audio streams with Rolling Buffer"""

    def __init__(self, kafka_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.kafka_servers = kafka_servers
        self.db_handler = MongoDBHandler()
        self.alert_throttler = AlertThrottler(
            cooldown_seconds=5
        )  # Giảm cooldown để test nhanh hơn
        self.chunk_count = 0

        # Audio Params
        self.target_sample_rate = 16000

        # --- ROLLING BUFFER CONFIG ---
        self.buffer_duration = 5.0  # Giữ lại 5 giây ngữ cảnh
        self.max_buffer_samples = int(self.buffer_duration * self.target_sample_rate)
        # Buffer khởi tạo rỗng
        self.audio_buffer = np.array([], dtype=np.float32)

        logger.info("Initializing AudioConsumer (Optimized)...")
        self.load_models()

        # Danh sách âm thanh nguy hiểm
        self.harmful_sound_labels = [
            "Screaming",
            "Yelling",
            "Shouting",
            "Crying, sobbing",
            "Gunshot, gunfire",
            "Explosion",
            "Bang",
            "Aggressive",
        ]

    def load_models(self):
        """Load AI Models (Optimized for Laptop/Demo)"""

        # Setup Device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"  # Hoặc "int8_float16" nếu GPU yếu
        else:
            self.device = "cpu"
            self.compute_type = "int8"  # CPU chạy int8 cực nhanh

        logger.info(f"Using Device: {self.device} | Compute Type: {self.compute_type}")

        # 1. Load YAMNet (Sound Event Detection)
        if YAMNET_AVAILABLE:
            try:
                logger.info("⏳ Loading YAMNet model...")
                # Load YAMNet model from TensorFlow Hub
                yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
                self.yamnet_model = hub.load(yamnet_model_handle)

                # Get all 521 YAMNet class names from the model's class map
                # YAMNet has class mapping in its params
                # For now, create a basic list - in production, load the full CSV
                self.yamnet_classes = self._load_yamnet_classes()

                logger.info("✅ YAMNet Loaded")
            except Exception as e:
                logger.error(f"Error loading YAMNet: {e}")
                self.yamnet_model = None
                self.yamnet_classes = []

        # 2. Load AST (Giữ nguyên vì chưa có thay thế nhẹ hơn tốt hơn)
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
                self.ast_processor = AutoFeatureExtractor.from_pretrained(model_name)
                self.ast_model = ASTForAudioClassification.from_pretrained(
                    model_name
                ).to(self.device)
                logger.info("✅ AST Model Loaded")
            except Exception as e:
                logger.error(f"Error loading AST: {e}")
                self.ast_model = None

        # 3. Load PhoBERT (Hate Speech Detection for Vietnamese)
        if PHOBERT_AVAILABLE:
            try:
                logger.info("⏳ Loading PhoBERT Hate Speech model...")
                phobert_model_path = "../models/phobert_hate_speech"
                self.phobert_tokenizer = AutoTokenizer.from_pretrained(
                    phobert_model_path
                )
                self.phobert_model = AutoModelForSequenceClassification.from_pretrained(
                    phobert_model_path
                ).to(self.device)
                self.phobert_model.eval()
                logger.info("✅ PhoBERT Model Loaded")
            except Exception as e:
                logger.error(f"Error loading PhoBERT: {e}")
                self.phobert_model = None
                self.phobert_tokenizer = None

        # 4. Load Whisper (Speech-to-Text)
        if WHISPER_AVAILABLE:
            try:
                logger.info("⏳ Loading Whisper STT model...")
                self.whisper_model = whisper.load_model("base", device=self.device)
                logger.info("✅ Whisper Model Loaded")
            except Exception as e:
                logger.error(f"Error loading Whisper: {e}")
                self.whisper_model = None
        else:
            self.whisper_model = None

    def _load_yamnet_classes(self):
        """Load YAMNet class names (521 classes)"""
        # Default list of harmful sound keywords to check against
        harmful_keywords = [
            "speech",
            "shouting",
            "yelling",
            "screaming",
            "crying",
            "gunshot",
            "gunfire",
            "explosion",
            "bang",
            "crash",
            "breaking",
            "alarm",
            "siren",
            "emergency",
            "police",
            "ambulance",
            "fire",
            "chainsaw",
            "hammer",
            "drill",
            "dog",
            "cat",
            "bark",
            "meow",
            "whimper",
            "growl",
            "car horn",
            "motorcycle",
            "truck",
            "siren",
        ]
        return harmful_keywords

    def detect_sound_events(self, audio_array: np.ndarray) -> Dict:
        """YAMNet Detection - Enhanced version"""
        if not self.yamnet_model:
            return {"is_harmful": False, "label": None, "score": 0.0}

        try:
            # Ensure audio is float32
            audio_array = np.array(audio_array, dtype=np.float32)
            if np.max(np.abs(audio_array)) <= 1.0:
                pass
            else:
                audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)

            # Run YAMNet inference
            scores, embeddings, spectrogram = self.yamnet_model(audio_array)

            # Get top predictions
            detected_events = []
            scores_np = scores.numpy()

            # For each frame, get the top class and score
            for frame_scores in scores_np:
                top_class_idx = np.argmax(frame_scores)
                top_score = float(frame_scores[top_class_idx])
                if top_score > 0.3:  # Frame-level threshold
                    detected_events.append((top_class_idx, top_score))

            if not detected_events:
                return {"is_harmful": False, "label": None, "score": 0.0}

            # Analyze detected events
            avg_score = np.mean([score for _, score in detected_events])
            max_score = max([score for _, score in detected_events])

            # Detection logic: Use average score across frames
            # Lower threshold to catch important events
            is_harmful = avg_score > 0.45

            label = f"Audio event (frames: {len(detected_events)}, avg confidence: {avg_score:.1%})"

            return {
                "is_harmful": is_harmful,
                "label": label,
                "score": avg_score,
            }
        except Exception as e:
            logger.error(f"YAMNet error: {e}")
            return {"is_harmful": False, "label": None, "score": 0.0}

    def connect_kafka(self):
        """Connect to Kafka"""
        try:
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC_AUDIO,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="audio-group-optimized",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            logger.info(f"Connected to Kafka topic: {KAFKA_TOPIC_AUDIO}")
        except KafkaError as e:
            logger.error(f"Kafka connection failed: {e}")
            raise

    def decode_audio(self, base64_data: str) -> np.ndarray:
        """Decode base64 to numpy array"""
        try:
            audio_bytes = base64.b64decode(base64_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            # Load audio & Resample về 16kHZ
            audio_array, _ = librosa.load(temp_path, sr=self.target_sample_rate)
            os.remove(temp_path)
            return audio_array
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return None

    def detect_sound_events_ast(self, audio_array: np.ndarray) -> Dict:
        """AST Detection (as secondary model)"""
        if not self.ast_model:
            return {"is_harmful": False, "label": None, "score": 0.0}

        try:
            # AST xử lý tốt nhất khoảng 5-10s, nhưng buffer của mình 5s là đẹp
            inputs = self.ast_processor(
                audio_array, sampling_rate=self.target_sample_rate, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.ast_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                score, idx = torch.max(probs, dim=-1)
                predicted_label = self.ast_model.config.id2label[idx.item()]
                score_val = score.item()

            is_harmful = (
                score_val > 0.35 and predicted_label in self.harmful_sound_labels
            )  # Tăng ngưỡng lên chút

            return {
                "is_harmful": is_harmful,
                "label": predicted_label,
                "score": score_val,
            }
        except Exception as e:
            # logger.error(f"AST error: {e}") # Tắt log rác nếu cần
            return {"is_harmful": False, "label": None, "score": 0.0}

    def detect_hate_speech(self, text: str) -> Dict:
        """Detect hate speech in text using PhoBERT"""
        if not self.phobert_model or not text or text.strip() == "":
            return {
                "is_hate_speech": False,
                "label": None,
                "score": 0.0,
                "confidence": 0.0,
            }

        try:
            # Tokenize input text
            inputs = self.phobert_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                score, predicted_class = torch.max(probabilities, dim=-1)

            score_val = score.item()
            predicted_label = self.phobert_model.config.id2label.get(
                predicted_class.item(), "unknown"
            )

            # Determine if hate speech (assuming class 1 = hate speech, 0 = safe)
            is_hate = predicted_class.item() == 1 and score_val > 0.5

            return {
                "is_hate_speech": is_hate,
                "label": predicted_label,
                "score": score_val,
                "confidence": score_val,
            }
        except Exception as e:
            logger.error(f"PhoBERT error: {e}")
            return {
                "is_hate_speech": False,
                "label": None,
                "score": 0.0,
                "confidence": 0.0,
            }

    def transcribe_and_check_toxic(self, audio_buffer: np.ndarray) -> Dict:
        """YAMNet Detection for harmful sounds (replaces Whisper transcription)"""
        # Use the detect_sound_events method (YAMNet-based)
        yamnet_result = self.detect_sound_events(audio_buffer)

        return {
            "is_toxic": yamnet_result["is_harmful"],
            "text": yamnet_result["label"] or "",
            "keywords": [],
            "score": yamnet_result["score"],
        }

    def transcribe_audio_whisper(self, audio_array: np.ndarray) -> str:
        """
        Transcribe audio to Vietnamese text using Whisper

        Args:
            audio_array: Audio waveform (float32, 16kHz)

        Returns:
            Transcribed text in Vietnamese
        """
        if not self.whisper_model:
            return ""

        tmp_path = None
        try:
            import tempfile
            import soundfile as sf

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, self.target_sample_rate)
                tmp_path = tmp.name

            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                tmp_path,
                language="vi",  # Vietnamese
                fp16=False if self.device == "cpu" else True,
            )

            transcribed_text = result.get("text", "").strip()

            if transcribed_text:
                logger.info(f"📝 Whisper STT: {transcribed_text}")
                return transcribed_text
            return ""

        except FileNotFoundError as e:
            if "ffmpeg" in str(e).lower():
                logger.error(
                    "❌ FFmpeg not found in Docker! Add to Dockerfile: RUN apt-get install -y ffmpeg"
                )
            else:
                logger.error(f"❌ Whisper transcription error: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Whisper transcription error: {e}")
            return ""
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    def check_transcribed_text(self, text: str) -> Dict:
        """
        Public method to check transcribed text for hate speech.
        Can be called from speech-to-text module when text is available.
        """
        if not text or text.strip() == "":
            return {"is_hate_speech": False, "label": None, "score": 0.0}

        return self.detect_hate_speech(text)

    def process_message(self, message: Dict):
        """
        Main processing with ROLLING BUFFER logic
        """
        chunk_id = message.get("chunk_id")
        timestamp = message.get("timestamp")
        b64_data = message.get("data")

        if not b64_data:
            return

        # 1. Decode chunk mới (1 giây)
        new_chunk = self.decode_audio(b64_data)
        if new_chunk is None:
            return

        # 2. CẬP NHẬT ROLLING BUFFER
        # Nối chunk mới vào đuôi buffer hiện tại
        self.audio_buffer = np.concatenate((self.audio_buffer, new_chunk))

        # Nếu buffer dài quá 5 giây, cắt bớt phần đầu (cũ nhất)
        if len(self.audio_buffer) > self.max_buffer_samples:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_samples :]

        # Chỉ xử lý khi buffer đã có ít nhất 1-2 giây để model đoán chuẩn hơn
        # (Lúc mới khởi động có thể bỏ qua vài chunk đầu)
        if len(self.audio_buffer) < 16000:
            return

        # --- PHÂN TÍCH ---

        # A. Detect Sound (YAMNet) - Dùng toàn bộ buffer (5s) để detect chính xác hơn
        sound_event = self.detect_sound_events(self.audio_buffer)

        # B. Optional: AST Detection as secondary model (comment out if not needed)
        # ast_result = self.detect_sound_events_ast(self.audio_buffer)

        # C. Transcribe audio to Vietnamese text and detect hate speech
        transcribed_text = ""
        hate_speech_result = {"is_hate_speech": False, "label": None, "score": 0.0}

        if self.whisper_model and len(self.audio_buffer) >= 16000:
            # Transcribe audio
            transcribed_text = self.transcribe_audio_whisper(self.audio_buffer)

            # Check for hate speech if transcription succeeded
            if transcribed_text and len(transcribed_text.strip()) > 0:
                hate_speech_result = self.detect_hate_speech(transcribed_text)

        # 3. Alert Logic
        alert_details = ""

        # --- Xử lý YAMNet Alert ---
        if sound_event["is_harmful"]:
            alert_details = (
                f"YAMNet Alert: {sound_event['label']} ({sound_event['score']:.1%})"
            )
            logger.warning(f"🔊 {alert_details}")

            if self.alert_throttler.should_send_alert("audio_event"):
                self.db_handler.save_alert(
                    {
                        "source": "audio",
                        "frame_id": chunk_id,
                        "detection_type": "Audio Event",
                        "type": "HIGH",
                        "confidence": sound_event["score"],
                        "details": alert_details,
                        "timestamp": timestamp,
                    }
                )

        # --- Xử lý Hate Speech Alert (nếu có text) ---
        if hate_speech_result["is_hate_speech"]:
            alert_details = f"Hate Speech Alert: {hate_speech_result['label']} ({hate_speech_result['score']:.1%})"
            logger.warning(f"💬 {alert_details}")

            if self.alert_throttler.should_send_alert("hate_speech"):
                self.db_handler.save_alert(
                    {
                        "source": "audio_text",
                        "frame_id": chunk_id,
                        "detection_type": "Hate Speech",
                        "type": "HIGH",
                        "confidence": hate_speech_result["score"],
                        "details": alert_details,
                        "timestamp": timestamp,
                    }
                )

        # 4. Save Record
        # Lưu thông tin detected từ YAMNet và PhoBERT
        # Convert numpy booleans to Python native bool for MongoDB compatibility
        self.db_handler.save_detection(
            {
                "chunk_id": chunk_id,
                "timestamp": timestamp,
                "transcribed_text": transcribed_text or (sound_event["label"] or ""),
                "sound_label": sound_event["label"],
                "sound_confidence": float(sound_event["score"]),
                "is_toxic": bool(sound_event["is_harmful"]),
                "is_screaming": bool(sound_event["is_harmful"]),
                "hate_speech_detected": bool(hate_speech_result["is_hate_speech"]),
                "hate_speech_label": hate_speech_result["label"],
                "hate_speech_confidence": float(hate_speech_result["score"]),
            }
        )

        if self.chunk_count % 5 == 0:
            short_text = (
                sound_event["label"][-50:]
                if sound_event["label"] and len(sound_event["label"]) > 50
                else (sound_event["label"] or "No sound detected")
            )
            logger.info(
                f"Chunk {chunk_id} | Sound: {short_text} | Confidence: {sound_event['score']:.2f}"
            )

        self.chunk_count += 1

    def run(self):
        try:
            self.connect_kafka()
            logger.info("🎧 Audio Consumer (YAMNet) listening...")
            for msg in self.consumer:
                self.process_message(msg.value)
        except KeyboardInterrupt:
            logger.info("Stopped.")
        finally:
            if hasattr(self, "consumer") and self.consumer:
                self.consumer.close()
            self.db_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()
    AudioConsumer(args.kafka).run()
