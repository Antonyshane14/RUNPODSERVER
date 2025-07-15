const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

// Configuration
const API_BASE_URL = 'http://localhost:8000';  // Python FastAPI server on same machine
const AUDIO_FILE_PATH = '/home/antonyshane/Downloads/WhatsApp Ptt 2025-07-15 at 10.26.55 PM.ogg';

class AudioScamTestClient {
    constructor() {
        this.apiUrl = API_BASE_URL;
    }

    async testHealthEndpoint() {
        console.log('🔍 Testing Health Endpoint...');
        try {
            const response = await axios.get(`${this.apiUrl}/health`);
            console.log('✅ Health Check Success:', response.data);
            return true;
        } catch (error) {
            console.error('❌ Health Check Failed:', error.message);
            return false;
        }
    }

    async testAudioAnalysis() {
        console.log('🎵 Testing Audio Analysis...');
        
        try {
            // Check if audio file exists
            if (!fs.existsSync(AUDIO_FILE_PATH)) {
                console.error(`❌ Audio file not found: ${AUDIO_FILE_PATH}`);
                return false;
            }

            // Create form data
            const formData = new FormData();
            formData.append('file', fs.createReadStream(AUDIO_FILE_PATH));
            formData.append('call_id', 'test_call_123');
            formData.append('chunk_number', '1');

            console.log(`📤 Uploading: ${path.basename(AUDIO_FILE_PATH)}`);
            console.log(`📊 File size: ${fs.statSync(AUDIO_FILE_PATH).size} bytes`);

            // Send request
            const response = await axios.post(`${this.apiUrl}/analyze-audio`, formData, {
                headers: {
                    ...formData.getHeaders(),
                    'Content-Type': 'multipart/form-data'
                },
                timeout: 60000 // 60 second timeout
            });

            console.log('✅ Audio Analysis Success!');
            console.log('📋 Response Status:', response.status);
            console.log('🎯 Results Summary:');
            
            const results = response.data.results;
            
            // Display results
            if (results.transcription) {
                console.log(`   🗣️  Transcription: "${results.transcription.text || 'N/A'}"`);
            }
            
            if (results.ai_voice_detection) {
                console.log(`   🤖 AI Voice: ${results.ai_voice_detection.is_ai_voice ? 'DETECTED' : 'NOT DETECTED'} (${(results.ai_voice_detection.confidence * 100).toFixed(1)}%)`);
            }
            
            if (results.background_noise) {
                console.log(`   🔊 Background Noise: ${results.background_noise.noise_level?.toFixed(2) || 'N/A'}`);
            }
            
            if (results.speaker_analysis?.diarization) {
                const speakers = results.speaker_analysis.diarization.num_speakers || 0;
                console.log(`   👥 Speakers Detected: ${speakers}`);
            }
            
            if (results.speaker_analysis?.emotions) {
                console.log(`   😊 Emotions Analyzed: ${Object.keys(results.speaker_analysis.emotions).length} speakers`);
            }
            
            if (results.scam_analysis) {
                console.log(`   ⚠️  Scam Analysis: ${results.scam_analysis.summary || 'Completed'}`);
            }

            console.log(`⏱️  Processing Time: ${response.data.processing_time?.toFixed(2) || 'N/A'} seconds`);
            
            return true;

        } catch (error) {
            console.error('❌ Audio Analysis Failed:');
            if (error.response) {
                console.error(`   Status: ${error.response.status}`);
                console.error(`   Error: ${error.response.data?.detail || error.response.statusText}`);
            } else {
                console.error(`   Error: ${error.message}`);
            }
            return false;
        }
    }

    async testCallSummary(callId = 'test_call_123') {
        console.log(`📊 Testing Call Summary for: ${callId}`);
        try {
            const response = await axios.get(`${this.apiUrl}/call-summary/${callId}`);
            console.log('✅ Call Summary Success:', response.data);
            return true;
        } catch (error) {
            console.error('❌ Call Summary Failed:', error.response?.data?.detail || error.message);
            return false;
        }
    }

    async runAllTests() {
        console.log('🚀 Starting Audio Scam Detection API Tests');
        console.log('=' * 50);
        
        const tests = [
            { name: 'Health Check', func: () => this.testHealthEndpoint() },
            { name: 'Audio Analysis', func: () => this.testAudioAnalysis() },
            { name: 'Call Summary', func: () => this.testCallSummary() }
        ];

        let passed = 0;
        let total = tests.length;

        for (const test of tests) {
            console.log(`\n🧪 Running: ${test.name}`);
            try {
                const result = await test.func();
                if (result) {
                    console.log(`✅ ${test.name}: PASSED`);
                    passed++;
                } else {
                    console.log(`❌ ${test.name}: FAILED`);
                }
            } catch (error) {
                console.log(`💥 ${test.name}: CRASHED - ${error.message}`);
            }
        }

        console.log('\n' + '=' * 50);
        console.log(`📈 TEST SUMMARY: ${passed}/${total} tests passed`);
        
        if (passed === total) {
            console.log('🎉 ALL TESTS PASSED! Your API is working perfectly!');
        } else {
            console.log('⚠️  Some tests failed. Check the logs above for details.');
        }

        return passed === total;
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    const client = new AudioScamTestClient();
    client.runAllTests()
        .then(success => {
            process.exit(success ? 0 : 1);
        })
        .catch(error => {
            console.error('💥 Test suite crashed:', error);
            process.exit(1);
        });
}

module.exports = AudioScamTestClient;
