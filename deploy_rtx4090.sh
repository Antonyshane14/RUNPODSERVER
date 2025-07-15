#!/bin/bash
# Quick deployment script for RunPod RTX 4090

echo "🚀 Quick Deploy to RunPod RTX 4090..."

# Set executable permissions
chmod +x setup_rtx4090.sh

# Run the full setup
echo "📦 Running RTX 4090 setup..."
./setup_rtx4090.sh

# Install project requirements
echo "🐍 Installing project requirements..."
pip install -r requirements_rtx4090.txt

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import torchaudio
import transformers
import fastapi
import librosa
print('✅ All core libraries installed successfully')
print(f'🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')
"

# Start the server
echo "🚀 Starting RTX 4090 optimized server..."
python runpod_main.py
