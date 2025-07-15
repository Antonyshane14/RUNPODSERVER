#!/bin/bash
# RunPod RTX 4090 Optimized Setup Script

echo "🚀 Setting up Audio Scam Detection for RTX 4090 on RunPod..."

# RTX 4090 environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,roundup_power2_divisions:16"
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# Update system packages
apt-get update

# Install system dependencies for audio processing
echo "📦 Installing system dependencies..."
apt-get install -y \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    git \
    curl \
    wget

# Install Python packages optimized for RTX 4090
echo "🐍 Installing optimized Python packages..."

# Core ML frameworks (RTX 4090 optimized)
pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 11.8 (already installed in template, upgrade if needed)
pip install --upgrade torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Audio processing libraries
pip install \
    librosa>=0.10.1 \
    soundfile>=0.12.1 \
    pyannote.audio>=3.1.1 \
    speechbrain>=0.5.15

# Transformers and ML utilities
pip install \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    datasets>=2.14.0 \
    optimum>=1.14.0

# Whisper for transcription
pip install openai-whisper>=20231117

# Web framework
pip install \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6

# Utilities
pip install \
    requests>=2.31.0 \
    python-dotenv>=1.0.0 \
    psutil>=5.9.5 \
    gpustat>=1.1.1

# Install Ollama for LLM analysis
echo "🧠 Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Configure Ollama for RTX 4090
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=33
export OLLAMA_FLASH_ATTENTION=1

# Start Ollama service
echo "🚀 Starting Ollama service..."
nohup ollama serve > /workspace/ollama.log 2>&1 &

# Wait for Ollama to start
sleep 15

# Pull optimized models
echo "📥 Pulling Hermes3 8B model..."
ollama pull hermes3:8b

# Create workspace structure
mkdir -p /workspace/{uploads,cache,models,logs,temp}

# Set permissions
chmod -R 755 /workspace

# Verify RTX 4090 setup
echo "🔍 Verifying RTX 4090 setup..."

python3 -c "
import torch
print(f'🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else 'No CUDA')
print(f'🔧 PyTorch Version: {torch.__version__}')
print(f'⚡ CUDA Available: {torch.cuda.is_available()}')
print(f'🚀 Compiled with CUDA: {torch.version.cuda}')
"

# Test audio libraries
python3 -c "
try:
    import librosa
    import soundfile
    import torchaudio
    import transformers
    print('✅ All audio libraries imported successfully')
except Exception as e:
    print(f'❌ Error importing libraries: {e}')
"

# Check Ollama status
echo "🧠 Checking Ollama status..."
ollama list

echo "🎉 RTX 4090 setup complete!"
echo "📊 GPU Memory Info:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo "🚀 Ready to start server with: python runpod_main.py"
