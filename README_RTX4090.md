# RTX 4090 Optimized Audio Scam Detection - RunPod Deployment

## 🚀 **Optimized for RTX 4090 on RunPod PyTorch Template**

This system is specifically optimized for RTX 4090 GPUs running on RunPod's PyTorch template with maximum performance optimizations.

### 🎯 **RTX 4090 Optimizations**

- **Mixed Precision (FP16)**: 2x faster inference with 24GB VRAM
- **Model Compilation**: PyTorch 2.0+ compilation for RTX 4090
- **Memory Management**: Optimized GPU memory allocation (90% utilization)
- **Batch Processing**: Efficient batch processing for multiple requests
- **CUDA Optimizations**: Advanced CUDA memory pooling and allocation

### 📦 **Quick Deployment**

#### **1. Choose RunPod Template**
- Template: **PyTorch 2.1.0 (Python 3.10, CUDA 11.8)**
- GPU: **RTX 4090** (24GB VRAM)
- Storage: **50GB+ Persistent Volume**
- Environment Variables: `HF_TOKEN=your_huggingface_token`

#### **2. Deploy in One Command**
```bash
# Clone and deploy
cd /workspace
git clone <your-repo-url> audio-scam-detection
cd audio-scam-detection
./deploy_rtx4090.sh
```

#### **3. Alternative Manual Setup**
```bash
# Manual setup
chmod +x setup_rtx4090.sh
./setup_rtx4090.sh
pip install -r requirements_rtx4090.txt
python runpod_main.py
```

### 🔧 **RTX 4090 Configuration**

#### **GPU Settings**
- **Memory Fraction**: 90% (21.6GB out of 24GB)
- **Mixed Precision**: Enabled (FP16)
- **Batch Size**: 8 (optimized for RTX 4090)
- **Model Compilation**: PyTorch 2.0+ compile mode

#### **Model Optimizations**
- **Whisper**: `large-v3` (RTX 4090 can handle large models)
- **Emotion Detection**: FP16 + compiled model
- **Speaker Diarization**: GPU-accelerated pyannote.audio
- **LLM**: Hermes3 8B with full GPU layers (33 layers)

### 📊 **Performance Benchmarks**

#### **Expected Performance on RTX 4090**
- **Audio Processing**: ~2-3 seconds per 10-second chunk
- **Whisper Transcription**: ~1 second for 10-second audio
- **Emotion Detection**: ~0.5 seconds with batch processing
- **Speaker Diarization**: ~1-2 seconds
- **LLM Analysis**: ~1-2 seconds

#### **Memory Usage**
- **Models in Memory**: ~8-12GB VRAM
- **Processing Overhead**: ~2-4GB VRAM
- **Available for Batch**: ~8-12GB VRAM remaining

### 🌐 **API Endpoints**

#### **Main Endpoints**
- `POST /analyze-audio` - Main audio analysis endpoint
- `GET /health` - Basic health check
- `GET /rtx4090-status` - RTX 4090 specific status
- `GET /performance-test` - GPU performance benchmark
- `GET /docs` - Interactive API documentation

#### **Example Usage**
```bash
# Your RunPod URL
BASE_URL="https://your-pod-id-8000.proxy.runpod.net"

# Test server
curl $BASE_URL/health

# Check RTX 4090 status
curl $BASE_URL/rtx4090-status

# Analyze audio
curl -X POST $BASE_URL/analyze-audio \
     -F "file=@audio.wav" \
     -F "call_id=call_123" \
     -F "chunk_number=1"
```

### 🧪 **Testing**

#### **Run Tests**
```bash
# Test RTX 4090 optimizations
./test_rtx4090.sh

# Manual tests
curl https://your-pod-id-8000.proxy.runpod.net/rtx4090-status
curl https://your-pod-id-8000.proxy.runpod.net/performance-test
```

### 📈 **Monitoring**

#### **GPU Monitoring**
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Memory usage
gpustat -i 1

# Server logs
tail -f /workspace/logs/server.log
```

#### **Performance Metrics**
The server provides detailed performance metrics:
- GPU memory usage
- Processing times per module
- Batch processing efficiency
- Model compilation status

### 🔧 **Troubleshooting**

#### **Common Issues**

1. **Out of Memory**
   ```bash
   # Reduce batch size in runpod_config.py
   MAX_BATCH_SIZE = 4  # Reduce from 8
   ```

2. **Model Loading Errors**
   ```bash
   # Check HuggingFace token
   export HF_TOKEN=your_token
   ```

3. **CUDA Errors**
   ```bash
   # Reset GPU memory
   python -c "import torch; torch.cuda.empty_cache()"
   ```

#### **Optimization Tips**

1. **For Maximum Speed**: Enable all optimizations
2. **For Memory Efficiency**: Reduce batch size and model sizes
3. **For Stability**: Disable model compilation if issues occur

### 📝 **File Structure**

```
/workspace/
├── runpod_config.py         # RTX 4090 configuration
├── runpod_main.py          # Optimized server entry point
├── setup_rtx4090.sh        # Complete setup script
├── deploy_rtx4090.sh       # One-command deployment
├── test_rtx4090.sh         # Testing script
├── requirements_rtx4090.txt # Optimized requirements
├── modules/
│   ├── emotion_detector.py (optimized)
│   └── transcription_optimized.py
├── logs/                   # Server logs
├── models/                 # Cached models
└── uploads/               # Audio uploads
```

### 🎯 **Production Deployment**

#### **Environment Variables**
```bash
HF_TOKEN=your_huggingface_token
RUNPOD_POD_ID=your_pod_id
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### **Scaling**
- **Single RTX 4090**: Handles 3-5 concurrent requests
- **Multiple GPUs**: Scale horizontally with multiple pods
- **Load Balancing**: Use RunPod's load balancing features

### 🚀 **Next Steps**

1. **Deploy**: Use the deployment scripts
2. **Test**: Run the test suite
3. **Monitor**: Check performance metrics
4. **Scale**: Add more pods as needed
5. **Optimize**: Fine-tune based on your specific use case

**Your RTX 4090 optimized audio scam detection system is ready for production! 🎉**
