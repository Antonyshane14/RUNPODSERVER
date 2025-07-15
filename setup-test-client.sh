#!/bin/bash

echo "🚀 Setting up Node.js Test Client for Audio Scam Detection API"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Please run this script from the main project directory (where main.py is located)"
    exit 1
fi

# Create test client directory if it doesn't exist
mkdir -p test-client
cd test-client

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Copy the audio file to a more accessible location
echo "🎵 Setting up test audio file..."
AUDIO_SOURCE="/home/antonyshane/Downloads/WhatsApp Ptt 2025-07-15 at 10.26.55 PM.ogg"
AUDIO_DEST="./test-audio.ogg"

if [ -f "$AUDIO_SOURCE" ]; then
    cp "$AUDIO_SOURCE" "$AUDIO_DEST"
    echo "✅ Audio file copied to: $AUDIO_DEST"
else
    echo "⚠️  Original audio file not found, you may need to update the path in test-api.js"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔧 To run the test client:"
echo "   cd test-client"
echo "   npm start"
echo ""
echo "🌐 The test client will be available at: http://localhost:3000"
echo "📋 You can also run tests directly with: npm test"
echo ""
echo "💡 Make sure your Python FastAPI server is running on port 8000!"
