const express = require('express');
const AudioScamTestClient = require('./test-api');

const app = express();
const PORT = 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Create test client instance
const testClient = new AudioScamTestClient();

// Routes
app.get('/', (req, res) => {
    res.send(`
        <html>
            <head>
                <title>Audio Scam Detection Test Client</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .button { 
                        padding: 10px 20px; 
                        margin: 10px; 
                        background: #007bff; 
                        color: white; 
                        border: none; 
                        border-radius: 5px; 
                        cursor: pointer; 
                    }
                    .button:hover { background: #0056b3; }
                    .result { 
                        margin: 20px 0; 
                        padding: 15px; 
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                        background: #f8f9fa; 
                    }
                </style>
            </head>
            <body>
                <h1>🎵 Audio Scam Detection Test Client</h1>
                <p>Test your Python FastAPI server running on the same machine!</p>
                
                <div>
                    <button class="button" onclick="testHealth()">🔍 Test Health</button>
                    <button class="button" onclick="testAudio()">🎵 Test Audio Analysis</button>
                    <button class="button" onclick="testSummary()">📊 Test Call Summary</button>
                    <button class="button" onclick="runAllTests()">🚀 Run All Tests</button>
                </div>
                
                <div id="results" class="result">
                    <strong>Results will appear here...</strong>
                </div>

                <script>
                    function showResult(result) {
                        document.getElementById('results').innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                    }

                    async function testHealth() {
                        try {
                            const response = await fetch('/api/test-health');
                            const result = await response.json();
                            showResult(result);
                        } catch (error) {
                            showResult({ error: error.message });
                        }
                    }

                    async function testAudio() {
                        try {
                            document.getElementById('results').innerHTML = '<strong>🎵 Testing audio analysis... This may take a minute...</strong>';
                            const response = await fetch('/api/test-audio');
                            const result = await response.json();
                            showResult(result);
                        } catch (error) {
                            showResult({ error: error.message });
                        }
                    }

                    async function testSummary() {
                        try {
                            const response = await fetch('/api/test-summary');
                            const result = await response.json();
                            showResult(result);
                        } catch (error) {
                            showResult({ error: error.message });
                        }
                    }

                    async function runAllTests() {
                        try {
                            document.getElementById('results').innerHTML = '<strong>🚀 Running all tests... Please wait...</strong>';
                            const response = await fetch('/api/run-all-tests');
                            const result = await response.json();
                            showResult(result);
                        } catch (error) {
                            showResult({ error: error.message });
                        }
                    }
                </script>
            </body>
        </html>
    `);
});

// API endpoints for testing
app.get('/api/test-health', async (req, res) => {
    try {
        const result = await testClient.testHealthEndpoint();
        res.json({ success: result, message: result ? 'Health check passed' : 'Health check failed' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.get('/api/test-audio', async (req, res) => {
    try {
        const result = await testClient.testAudioAnalysis();
        res.json({ success: result, message: result ? 'Audio analysis completed' : 'Audio analysis failed' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.get('/api/test-summary', async (req, res) => {
    try {
        const result = await testClient.testCallSummary();
        res.json({ success: result, message: result ? 'Call summary retrieved' : 'Call summary failed' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.get('/api/run-all-tests', async (req, res) => {
    try {
        const result = await testClient.runAllTests();
        res.json({ success: result, message: result ? 'All tests passed' : 'Some tests failed' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🌐 Test Client Server running at:`);
    console.log(`   Local: http://localhost:${PORT}`);
    console.log(`   Network: http://0.0.0.0:${PORT}`);
    console.log(`\n🎯 Ready to test Audio Scam Detection API!`);
});

module.exports = app;
