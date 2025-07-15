#!/bin/bash

# Audio Scam Detection System Startup Script

echo "=== Audio Scam Detection System ==="
echo "Starting up FastAPI backend..."

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install Ollama for LLM analysis:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    echo "   ollama pull hermes3:8b"
    echo ""
fi

# Check if required model is available
if command -v ollama &> /dev/null; then
    if ! ollama list | grep -q "hermes3:8b"; then
        echo "Downloading required LLM model (this may take a while)..."
        ollama pull hermes3:8b
    fi
fi

# Set environment variables if not already set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN environment variable must be set"
    echo "💡 Get your token from: https://huggingface.co/settings/tokens"
    echo "🔧 Set it with: export HF_TOKEN='your_token_here'"
    exit 1
fi

# Create necessary directories
mkdir -p temp_audio cache speaker_segments

echo ""
echo "🚀 Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI server
python main.py
