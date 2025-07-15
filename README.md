# Audio Scam Detection System

A comprehensive FastAPI-based system for real-time detection of scam phone calls using AI voice analysis, emotion detection, speaker diarization, and LLM-powered analysis.

## System Architecture

The system processes 10-second audio chunks from phone calls through multiple AI modules:

1. **Audio Preprocessing** - Converts audio to standard format (16kHz mono WAV)
2. **Parallel Processing** - Three modules run simultaneously:
   - **Whisper Transcription** - Converts speech to text
   - **AI Voice Detection** - Identifies artificially generated voices
   - **Background Noise Analysis** - Detects suspicious call center environments
3. **Speaker Diarization** - Separates speakers and identifies who speaks when
4. **Emotion Analysis** - Analyzes emotions in **diarized speaker audio only**
5. **LLM Analysis** - Local LLM analyzes all data + **previous LLM outputs** for context
6. **Context Caching** - Stores LLM outputs and feeds them back for next chunk analysis

## Workflow

```
Node.js Server → FastAPI Backend → Audio Processing → 3 Parallel Modules → Speaker Diarization → Emotion (Diarized Audio) → LLM (+ Cached Context) → Scam Report
     ↓                                                      ↓                                                           ↓
Audio Chunks (10s)                                    Cache/Context                                              LLM Output Cache
```

## Features

- **Real-time Processing** - Handles streaming 10-second audio chunks
- **Multi-modal Analysis** - Voice, emotion, background, and content analysis
- **Context Awareness** - LLM outputs are cached and fed back for context continuity
- **Scam Pattern Detection** - Identifies common scam tactics and red flags
- **Speaker Profiling** - Analyzes individual speaker behavior using diarized audio
- **Escalation Detection** - Monitors if scam tactics intensify over time
- **Emotion Analysis on Separated Audio** - Emotion detection only on speaker-diarized audio for accuracy

## Key Processing Flow

1. **Parallel Processing Phase**: Transcription, AI voice detection, and background noise analysis run simultaneously
2. **Speaker Diarization**: Runs separately to identify and separate speakers
3. **Emotion Detection**: Processes **only the diarized/separated speaker audio** for accurate emotion analysis
4. **LLM Analysis**: Analyzes current chunk data + **feeds in cached previous LLM outputs** for context-aware decision making
5. **Caching**: Stores LLM analysis results and feeds them back into the next chunk analysis

## Installation

1. **Clone and Setup**
   ```bash
   cd /home/antonyshane/linux/INTER_NIT_2
   pip install -r requirements.txt
   ```

2. **Install Ollama** (for LLM analysis)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull hermes3:8b
   ```

3. **Set Environment Variables**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

## Usage

### Start the API Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Analyze Audio Chunk
```http
POST /analyze-audio
Content-Type: multipart/form-data

file: audio_file.wav
call_id: unique_call_identifier
chunk_number: sequence_number
```

**Response:**
```json
{
  "status": "success",
  "call_id": "call_123",
  "chunk_number": 0,
  "results": {
    "transcription": {
      "transcript": "Hello, this is important...",
      "scam_keywords": ["urgent", "money"],
      "risk_score": 0.7
    },
    "ai_voice_detection": {
      "is_ai_voice": false,
      "confidence": 0.8923
    },
    "background_noise": {
      "suspicious_sounds": ["typing", "office"],
      "is_suspicious": true
    },
    "speaker_analysis": {
      "diarization": {
        "num_speakers": 2,
        "segments": [...]
      },
      "emotions": {
        "SPEAKER_00": {
          "top_emotion": "fearful",
          "confidence": 0.85
        }
      }
    },
    "scam_analysis": {
      "is_scam": true,
      "confidence": 85,
      "red_flags": ["urgency_pressure", "money_request"],
      "recommended_action": "Hang up immediately"
    }
  }
}
```

#### Get Call Summary
```http
GET /call-summary/{call_id}
```

#### Health Check
```http
GET /health
```

### Test the System

Run the test client:
```bash
python test_client.py
```

This simulates how a Node.js server would send audio chunks to the API.

## Configuration

Edit `config.py` to customize:

- Model sizes and thresholds
- Processing parameters
- Directory paths
- API settings

## Models Used

- **Whisper** (OpenAI) - Speech transcription
- **as1605/Deepfake-audio-detection-V2** - AI voice detection
- **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition** - Emotion detection
- **pyannote/speaker-diarization-3.1** - Speaker separation
- **hermes3:8b** (via Ollama) - Scam pattern analysis

## Scam Detection Capabilities

The system can identify various scam types:

- **Tech Support Scams** - Fake computer virus warnings
- **IRS/Government Scams** - Fake tax or legal threats
- **Romance Scams** - Emotional manipulation for money
- **Prize/Lottery Scams** - Fake winnings requiring payment
- **Phishing** - Attempts to steal personal information
- **Investment Scams** - Fraudulent financial opportunities

### Detection Methods

- **Keyword Analysis** - Scam-related terms and phrases
- **Emotion Patterns** - Victim stress and manipulation tactics
- **Voice Analysis** - AI-generated voice detection
- **Environmental Audio** - Fake call center sounds
- **Behavioral Patterns** - Escalation and pressure tactics
- **Context Tracking** - Cross-chunk pattern recognition

## Integration with Node.js

For Node.js integration, use `multipart/form-data` to send audio chunks:

```javascript
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('audio_chunk.wav'));
form.append('call_id', 'unique_call_id');
form.append('chunk_number', chunkIndex);

const response = await fetch('http://localhost:8000/analyze-audio', {
  method: 'POST',
  body: form
});
```

## Performance

- **Processing Time** - ~5-15 seconds per 10-second chunk (depending on hardware)
- **Memory Usage** - ~2-4GB RAM (with GPU acceleration)
- **Parallel Processing** - Up to 4 modules process simultaneously
- **Caching** - Maintains context for up to 7 days

## Hardware Requirements

**Minimum:**
- 8GB RAM
- 4 CPU cores
- 10GB storage

**Recommended:**
- 16GB+ RAM
- NVIDIA GPU with CUDA support
- 8+ CPU cores
- SSD storage

## Troubleshooting

**Common Issues:**

1. **Import errors** - Install missing packages: `pip install -r requirements.txt`
2. **CUDA errors** - Install PyTorch with CUDA support
3. **Ollama not found** - Install Ollama and pull the model
4. **HuggingFace token** - Set `HF_TOKEN` environment variable
5. **Audio format errors** - Ensure audio is in supported format (WAV, OGG, MP3)

## Security Considerations

- Set proper CORS origins in production
- Use environment variables for tokens
- Implement rate limiting
- Secure audio file storage
- Regular cache cleanup

## License

This project is for educational and research purposes.
