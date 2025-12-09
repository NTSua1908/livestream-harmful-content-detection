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

# 2. Faster Whisper (Optimized Speech to Text)
try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"❌ Faster-Whisper not found. Install: pip install faster-whisper. Error: {e}"
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

        # 1. Load Faster-Whisper (Thay cho Whisper gốc)
        if WHISPER_AVAILABLE:
            try:
                # Model 'small' là cân bằng nhất cho tiếng Việt trên máy cá nhân
                # 'tiny' quá tệ, 'base' tạm được, 'small' khá tốt.
                logger.info("⏳ Loading Faster-Whisper 'small' model...")
                self.whisper_model = WhisperModel(
                    "small", device=self.device, compute_type=self.compute_type
                )
                logger.info("✅ Faster-Whisper Loaded")
            except Exception as e:
                logger.error(f"Error loading Faster-Whisper: {e}")
                self.whisper_model = None

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

    def detect_sound_events(self, audio_array: np.ndarray) -> Dict:
        """AST Detection"""
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

    def transcribe_and_check_toxic(self, audio_buffer: np.ndarray) -> Dict:
        """Faster-Whisper Transcription + Keyword Check"""
        if not self.whisper_model:
            return {"is_toxic": False, "text": "", "keywords": []}

        try:
            # Faster-whisper cực nhanh
            # beam_size=1 để nhanh nhất có thể (greedy search)
            segments, _ = self.whisper_model.transcribe(
                audio_buffer,
                language="vi",
                beam_size=1,
                vad_filter=True,  # Tự động lọc khoảng lặng, giúp chính xác hơn
            )

            # Gộp text từ các segments
            text = " ".join([s.text for s in segments]).strip()

            if not text:
                return {"is_toxic": False, "text": "", "keywords": []}

            # Check toxic
            from config import TOXIC_KEYWORDS

            toxic_result = check_toxic_content(text, TOXIC_KEYWORDS)

            return {
                "is_toxic": toxic_result["is_toxic"],
                "text": text,
                "keywords": toxic_result.get("matched_keywords", []),
                "score": toxic_result.get("toxic_score", 0),
            }

        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return {"is_toxic": False, "text": "", "keywords": []}

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

        # A. Detect Sound (AST) - Dùng toàn bộ buffer (5s) để detect chính xác hơn
        sound_event = self.detect_sound_events(self.audio_buffer)

        # B. Transcribe (Whisper) - Dùng toàn bộ buffer (5s) để lấy ngữ cảnh
        speech_result = self.transcribe_and_check_toxic(self.audio_buffer)

        # 3. Alert Logic
        alert_details = ""

        # --- Xử lý AST Alert ---
        if sound_event["is_harmful"]:
            alert_details = (
                f"Detected: {sound_event['label']} ({sound_event['score']:.1%})"
            )
            logger.warning(f"🔊 {alert_details}")

            if self.alert_throttler.should_send_alert("audio_scream"):
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

        # --- Xử lý Toxic Alert ---
        if speech_result["is_toxic"]:
            alert_details = (
                f"Toxic: {speech_result['keywords']} | '{speech_result['text']}'"
            )
            logger.warning(f"🤬 {alert_details}")

            if self.alert_throttler.should_send_alert("audio_toxic"):
                self.db_handler.save_alert(
                    {
                        "source": "audio",
                        "frame_id": chunk_id,
                        "detection_type": "Toxic Speech",
                        "type": "MEDIUM",
                        "confidence": 1.0,
                        "details": alert_details,
                        "timestamp": timestamp,
                    }
                )

        # 4. Save Record
        # Lưu text đầy đủ để hiển thị lên dashboard
        self.db_handler.save_detection(
            {
                "chunk_id": chunk_id,
                "timestamp": timestamp,
                "transcribed_text": speech_result["text"],  # Text này sẽ dài (5s)
                "sound_label": sound_event["label"],
                "sound_confidence": sound_event["score"],
                "is_toxic": speech_result["is_toxic"],
                "is_screaming": sound_event["is_harmful"],
            }
        )

        if self.chunk_count % 5 == 0:
            short_text = (
                speech_result["text"][-50:]
                if len(speech_result["text"]) > 50
                else speech_result["text"]
            )
            logger.info(
                f"Chunk {chunk_id} | Sound: {sound_event['label']} ({sound_event['score']:.2f}) | Text: ...{short_text}"
            )

        self.chunk_count += 1

    def run(self):
        try:
            self.connect_kafka()
            logger.info("🎧 Audio Consumer (Optimized) listening...")
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
