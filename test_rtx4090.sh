#!/bin/bash
# Test RTX 4090 optimized deployment

echo "🧪 Testing RTX 4090 Audio Scam Detection Server..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get RunPod URL (replace with your actual URL)
POD_ID=${RUNPOD_POD_ID:-"your-pod-id"}
BASE_URL="https://${POD_ID}-8000.proxy.runpod.net"

echo "🔗 Testing server at: $BASE_URL"

# Test 1: Basic health check
echo -e "\n${YELLOW}Test 1: Basic Health Check${NC}"
curl -s "$BASE_URL/health" | jq '.' || echo -e "${RED}❌ Health check failed${NC}"

# Test 2: RTX 4090 status
echo -e "\n${YELLOW}Test 2: RTX 4090 Status${NC}"
curl -s "$BASE_URL/rtx4090-status" | jq '.' || echo -e "${RED}❌ RTX 4090 status failed${NC}"

# Test 3: Performance test
echo -e "\n${YELLOW}Test 3: Performance Test${NC}"
curl -s "$BASE_URL/performance-test" | jq '.' || echo -e "${RED}❌ Performance test failed${NC}"

# Test 4: GPU memory info
echo -e "\n${YELLOW}Test 4: Local GPU Memory Check${NC}"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

# Test 5: Create sample audio for testing
echo -e "\n${YELLOW}Test 5: Creating Sample Audio${NC}"
python3 -c "
import numpy as np
import soundfile as sf

# Create 10-second sample audio (16kHz)
duration = 10
sample_rate = 16000
samples = duration * sample_rate

# Generate sample audio with some speech-like patterns
t = np.linspace(0, duration, samples)
audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(samples)

# Save as WAV
sf.write('/workspace/test_audio.wav', audio, sample_rate)
print('✅ Sample audio created: /workspace/test_audio.wav')
"

# Test 6: Test audio upload (if sample audio exists)
if [ -f "/workspace/test_audio.wav" ]; then
    echo -e "\n${YELLOW}Test 6: Audio Upload Test${NC}"
    curl -X POST "$BASE_URL/analyze-audio" \
         -F "file=@/workspace/test_audio.wav" \
         -F "call_id=test_call_123" \
         -F "chunk_number=1" \
         --max-time 120 \
         -s | jq '.' || echo -e "${RED}❌ Audio upload test failed${NC}"
else
    echo -e "\n${RED}❌ Sample audio not found, skipping upload test${NC}"
fi

# Test 7: Ollama status
echo -e "\n${YELLOW}Test 7: Ollama Status${NC}"
ollama list | grep hermes3 && echo -e "${GREEN}✅ Hermes3 model available${NC}" || echo -e "${RED}❌ Hermes3 model not found${NC}"

# Test 8: Python dependencies
echo -e "\n${YELLOW}Test 8: Python Dependencies${NC}"
python3 -c "
try:
    import torch
    import torchaudio
    import transformers
    import fastapi
    import librosa
    print('✅ All dependencies imported successfully')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
        print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    else:
        print('❌ CUDA not available')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Summary
echo -e "\n${GREEN}🎉 RTX 4090 Testing Complete!${NC}"
echo -e "\n${YELLOW}Quick Commands:${NC}"
echo "🔗 Server URL: $BASE_URL"
echo "🏥 Health Check: curl $BASE_URL/health"
echo "🎮 RTX 4090 Status: curl $BASE_URL/rtx4090-status"
echo "⚡ Performance Test: curl $BASE_URL/performance-test"

echo -e "\n${YELLOW}For live testing, visit:${NC}"
echo "$BASE_URL/docs (FastAPI Swagger UI)"
