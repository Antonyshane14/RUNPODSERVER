#!/bin/bash
# RunPod Environment Setup Helper

echo "🔧 RunPod Audio Scam Detection Setup Helper"
echo "============================================"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "❌ HF_TOKEN environment variable is not set"
    echo ""
    echo "📋 To get your Hugging Face token:"
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'Read' permissions"
    echo "3. Copy the token"
    echo ""
    echo "🔧 To set the token in RunPod:"
    echo "export HF_TOKEN='your_token_here'"
    echo ""
    echo "Or add it to your RunPod pod environment variables"
    echo ""
    exit 1
else
    echo "✅ HF_TOKEN is set: ${HF_TOKEN:0:10}..."
fi

# Check Python environment
echo ""
echo "🐍 Checking Python environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('❌ PyTorch not installed')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not installed')

try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except ImportError:
    print('❌ FastAPI not installed')
"

echo ""
echo "🚀 Environment check complete!"
echo "💡 If any packages are missing, run: pip install -r requirements_rtx4090.txt"
echo ""
echo "▶️  To start the server: python runpod_main.py"
