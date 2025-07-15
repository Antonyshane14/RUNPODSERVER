from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import concurrent.futures
import logging

# Import configuration
import config

# Import our processing modules
from modules.audio_processor import AudioProcessor
from modules.ai_voice_detector import AIVoiceDetector
from modules.background_noise_detector import BackgroundNoiseDetector
from modules.speaker_diarization import SpeakerDiarizer
from modules.emotion_detector import EmotionDetector
from modules.transcription import WhisperTranscriber
from modules.llm_analyzer import LLMAnalyzer
from modules.cache_manager import CacheManager

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)

# CORS middleware for Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=config.ALLOWED_METHODS,
    allow_headers=config.ALLOWED_HEADERS,
)

# Initialize components
audio_processor = AudioProcessor()
ai_voice_detector = AIVoiceDetector(config.AI_VOICE_MODEL)
background_noise_detector = BackgroundNoiseDetector(config.BACKGROUND_NOISE_THRESHOLD)
speaker_diarizer = SpeakerDiarizer(config.HF_TOKEN)
emotion_detector = EmotionDetector(config.EMOTION_MODEL)
transcriber = WhisperTranscriber(config.WHISPER_MODEL_SIZE)
llm_analyzer = LLMAnalyzer(config.OLLAMA_MODEL)
cache_manager = CacheManager(str(config.CACHE_DIR))

# Create directories
config.TEMP_AUDIO_DIR.mkdir(exist_ok=True)
config.CACHE_DIR.mkdir(exist_ok=True)
config.SPEAKER_SEGMENTS_DIR.mkdir(exist_ok=True)

@app.post("/analyze-audio")
async def analyze_audio_chunk(
    file: UploadFile = File(...),
    call_id: str = None,
    chunk_number: int = 0
):
    """
    Process a 10-second audio chunk from the Node.js server.
    
    Workflow:
    1. Save audio chunk locally
    2. Process in parallel: Whisper, AI voice detection, background noise, speaker diarization
    3. Run emotion detection on speaker-separated audio
    4. Combine results and send to LLM for analysis
    5. Update cache for continuous call analysis
    """
    
    try:
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        audio_filename = config.TEMP_AUDIO_DIR / f"{call_id}_{chunk_number}_{timestamp}.wav"
        
        # Save uploaded file
        logger.info(f"Saving audio chunk: {audio_filename}")
        with open(audio_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Convert to standard format (16kHz mono WAV)
        processed_audio_path = audio_processor.convert_to_standard_format(audio_filename)
        
        # Load previous context from cache
        previous_context = cache_manager.load_context(call_id) if call_id else {}
        
        # STEP 1: Process first 3 modules in parallel (NOT emotion detection)
        logger.info("Starting parallel processing (transcription, AI voice, background noise)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit parallel tasks (excluding emotion detection)
            future_transcription = executor.submit(transcriber.transcribe, processed_audio_path)
            future_ai_voice = executor.submit(ai_voice_detector.detect, processed_audio_path)
            future_background_noise = executor.submit(background_noise_detector.detect, processed_audio_path)
            
            # Get results
            transcription_result = future_transcription.result()
            ai_voice_result = future_ai_voice.result()
            background_noise_result = future_background_noise.result()
        
        # STEP 2: Run speaker diarization separately
        logger.info("Running speaker diarization...")
        diarization_result = speaker_diarizer.diarize(processed_audio_path)
        
        # STEP 3: Process emotion detection ONLY on diarized speaker audio
        logger.info("Processing emotion detection on speaker-separated audio...")
        emotion_results = {}
        speaker_files = {}
        
        if diarization_result['segments']:
            # Split audio by speakers first
            speaker_files = speaker_diarizer.split_audio_by_speakers(
                processed_audio_path, 
                diarization_result['segments']
            )
            
            # NOW detect emotions ONLY on the separated speaker audio
            for speaker_id, speaker_file in speaker_files.items():
                logger.info(f"Analyzing emotions for {speaker_id} using separated audio")
                emotion_results[speaker_id] = emotion_detector.detect(speaker_file)
        else:
            logger.warning("No speaker segments found - skipping emotion detection")
        
        # Combine all results
        analysis_data = {
            "call_id": call_id,
            "chunk_number": chunk_number,
            "timestamp": timestamp,
            "transcription": transcription_result,
            "ai_voice_detection": ai_voice_result,
            "background_noise": background_noise_result,
            "speaker_diarization": diarization_result,
            "emotion_detection": emotion_results,
            "previous_context": previous_context
        }
        
        logger.info("Sending to LLM for analysis...")
        
        # Send to LLM for analysis
        llm_analysis = llm_analyzer.analyze(analysis_data)
        
        # Update cache with new data
        cache_manager.update_context(call_id, analysis_data, llm_analysis)
        
        # Cleanup temporary files
        if config.CLEANUP_TEMP_FILES:
            cleanup_temp_files([str(audio_filename), processed_audio_path] + list(speaker_files.values()) if 'speaker_files' in locals() else [])
        
        # Return comprehensive results
        response = {
            "status": "success",
            "call_id": call_id,
            "chunk_number": chunk_number,
            "results": {
                "transcription": transcription_result,
                "ai_voice_detection": ai_voice_result,
                "background_noise": background_noise_result,
                "speaker_analysis": {
                    "diarization": diarization_result,
                    "emotions": emotion_results
                },
                "scam_analysis": llm_analysis
            },
            "processing_time": time.time() - timestamp/1000
        }
        
        logger.info(f"Analysis complete for chunk {chunk_number}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/call-summary/{call_id}")
async def get_call_summary(call_id: str):
    """Get a summary of the entire call analysis"""
    try:
        summary = cache_manager.get_call_summary(call_id)
        return {"status": "success", "call_id": call_id, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Call not found: {str(e)}")

@app.delete("/call-data/{call_id}")
async def clear_call_data(call_id: str):
    """Clear cache data for a specific call"""
    try:
        cache_manager.clear_call_data(call_id)
        return {"status": "success", "message": f"Data cleared for call {call_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Audio Scam Detection API"}

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
