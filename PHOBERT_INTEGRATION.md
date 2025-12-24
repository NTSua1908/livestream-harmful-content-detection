# PhoBERT Hate Speech Detection Integration

## Overview

The `consumer_audio.py` module has been enhanced with **PhoBERT** (BERT model trained on Vietnamese text) to detect hate speech and harmful content in transcribed text from speech-to-text systems.

## Features

### 1. **PhoBERT Model Loading**

- Loads from `models/phobert_hate_speech` directory
- Supports GPU (CUDA) and CPU inference
- Automatic device selection and optimization

### 2. **Hate Speech Detection Method**

```python
def detect_hate_speech(self, text: str) -> Dict:
    """
    Detect hate speech in text using PhoBERT

    Args:
        text: Vietnamese text to analyze

    Returns:
        {
            "is_hate_speech": bool,
            "label": str,  # Label from model (e.g., "hate", "safe")
            "score": float,  # Confidence score (0-1)
            "confidence": float  # Same as score
        }
    """
```

### 3. **Public Interface Method**

```python
def check_transcribed_text(self, text: str) -> Dict:
    """
    Public method to check transcribed text.
    Can be called from speech-to-text modules.

    Args:
        text: Transcribed Vietnamese text

    Returns:
        Hate speech detection result
    """
```

### 4. **Integrated Processing**

The `process_message()` method now:

- Detects harmful sounds using YAMNet
- Detects hate speech in transcribed text using PhoBERT
- Saves both detection results to MongoDB
- Generates alerts for detected hate speech

## Usage Examples

### Example 1: Check Single Text

```python
from src.consumer_audio import AudioConsumer

consumer = AudioConsumer()

# Check if text contains hate speech
result = consumer.check_transcribed_text("Thằng ngu, tôi sẽ hủy hoại bạn")
if result["is_hate_speech"]:
    print(f"⚠️ Hate speech detected: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Example 2: Batch Processing

```python
texts = [
    "Xin chào, bạn khỏe không?",
    "Thằng khốn kiếp đáng bị chết",
    "Tôi yêu bạn rất nhiều"
]

for text in texts:
    result = consumer.check_transcribed_text(text)
    status = "🚨 HATE" if result["is_hate_speech"] else "✅ SAFE"
    print(f"{status}: {text}")
```

### Example 3: Integration with Speech-to-Text

```python
# When you receive transcribed text from speech-to-text module:
def handle_transcribed_audio(audio_chunk, transcript_text):
    # Your existing audio processing
    sound_result = consumer.detect_sound_events(audio_chunk)

    # NEW: Check transcribed text for hate speech
    speech_result = consumer.check_transcribed_text(transcript_text)

    # Process both results
    if speech_result["is_hate_speech"]:
        # Alert or log hate speech detection
        logger.warning(f"Hate speech detected in transcript: {transcript_text}")
```

## Data Structure

### Detection Result in MongoDB

```python
{
    "chunk_id": "chunk_12345",
    "timestamp": 1234567890,
    "transcribed_text": "Thằng ngu",
    "sound_label": "Yelling",
    "sound_confidence": 0.87,
    "is_toxic": True,
    "is_screaming": True,
    "hate_speech_detected": True,          # NEW
    "hate_speech_label": "hate",           # NEW
    "hate_speech_confidence": 0.92         # NEW
}
```

### Alert Triggered by Hate Speech

```python
{
    "source": "audio_text",
    "frame_id": "chunk_12345",
    "detection_type": "Hate Speech",
    "type": "HIGH",
    "confidence": 0.92,
    "details": "Hate Speech Alert: hate (92.0%)",
    "timestamp": 1234567890
}
```

## Model Information

### PhoBERT Model Location

```
models/phobert_hate_speech/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── vocab.txt
└── bpe.codes
```

### Model Specifications

- **Type**: Sequence Classification (Binary or Multi-class)
- **Input**: Vietnamese text (up to 256 tokens)
- **Output**: Classification label + confidence score
- **Framework**: Hugging Face Transformers

## Configuration

### Alert Thresholds

Hate speech is flagged when:

- Predicted class index = 1 (hate class)
- Confidence score > 0.5 (50%)

To adjust these thresholds, modify in `detect_hate_speech()`:

```python
# Change threshold
is_hate = predicted_class.item() == 1 and score_val > 0.6  # 60% threshold
```

### Device Selection

The consumer automatically selects:

- **GPU (CUDA)**: If available, uses `float16` precision for speed
- **CPU**: Uses `int8` quantization for efficiency

## Testing

### Run Hate Speech Detection Test

```bash
# From d:\Code\doan directory
python test_hate_speech_detection.py
```

This will:

1. Load the PhoBERT model
2. Test on various Vietnamese texts
3. Display detection results
4. Show summary statistics

### Sample Test Output

```
Test 1: Xin chào, đây là một tin nhắn bình thường
  ✅ SAFE
  Label: safe
  Confidence: 0.9523

Test 2: Tôi sẽ giết bạn, thằng ngu
  🚨 HATE SPEECH DETECTED
  Label: hate
  Confidence: 0.9847
```

## Requirements

### Python Packages

- `transformers` - For loading PhoBERT model
- `torch` - For inference
- `numpy` - For data processing
- `kafka-python` - For Kafka integration
- `pymongo` - For database storage

### Installation

```bash
pip install transformers torch numpy kafka-python pymongo
```

## Integration Points

1. **Kafka Message Flow**

   - Audio chunks → Rolling buffer → YAMNet detection
   - (Future) Transcribed text → PhoBERT detection

2. **Database Storage**

   - Hate speech results saved to MongoDB
   - Alert triggered if threshold exceeded

3. **Dashboard**
   - Results displayed in Streamlit dashboard
   - Shows toxic speech count and content

## Performance Notes

### Inference Speed

- **GPU**: ~50-100ms per text (depending on length)
- **CPU**: ~200-500ms per text

### Memory Usage

- Model size: ~370MB (safetensors)
- Typical memory: 1-2GB during inference

## Troubleshooting

### Model Loading Error

```
❌ Error loading PhoBERT
```

**Solution**: Ensure model files exist in `models/phobert_hate_speech/`

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Switch to CPU by setting `self.device = "cpu"` in `load_models()`

### Empty Results

If `check_transcribed_text()` returns all zeros:

- Check if text is empty
- Verify model loaded successfully
- Check logs for PhoBERT errors

## Future Enhancements

1. **Multilingual Support**: Extend to other languages (English, Chinese, etc.)
2. **Fine-tuning**: Train on domain-specific hate speech
3. **Confidence Threshold Optimization**: Adjust based on deployment needs
4. **Batch Processing**: Process multiple texts simultaneously
5. **Explainability**: Show which words triggered hate speech detection

## References

- **PhoBERT**: [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- **Transformers**: [Hugging Face](https://huggingface.co/transformers/)
- **Model Weights**: Community-trained models available on Hugging Face Model Hub
