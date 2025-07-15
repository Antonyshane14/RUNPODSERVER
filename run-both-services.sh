#!/bin/bash

echo "🚀 Starting Audio Scam Detection System + Test Client"
echo "====================================================="

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $PYTHON_PID $NODE_PID 2>/dev/null
    wait
    echo "✅ Cleanup complete"
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Check if Python server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ Python server already running on port 8000"
else
    echo "🐍 Starting Python FastAPI server..."
    python runpod_main.py &
    PYTHON_PID=$!
    echo "   Python server PID: $PYTHON_PID"
    
    # Wait a moment for Python server to start
    sleep 3
fi

# Change to test client directory
cd test-client

# Check if Node.js dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start Node.js test client
echo "🌐 Starting Node.js test client..."
node client.js &
NODE_PID=$!
echo "   Node.js client PID: $NODE_PID"

echo ""
echo "🎉 Both services are now running!"
echo ""
echo "📊 Available services:"
echo "   🐍 Python API: http://localhost:8000"
echo "   🌐 Test Client: http://localhost:3000"
echo ""
echo "💡 Open http://localhost:3000 in your browser to test the API!"
echo ""
echo "Press Ctrl+C to stop both services..."

# Wait for processes
wait
