import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoProcessor, WhisperForConditionalGeneration
import logging
from typing import Dict, Any
import warnings
import time

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class OptimizedWhisperTranscriber:
    """RTX 4090 optimized Whisper transcriber"""
    
    def __init__(self, model_size="large-v3"):
        self.model_size = model_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.mixed_precision = torch.cuda.is_available()
        
        # RTX 4090 optimizations
        self.batch_size = 2  # Batch processing for efficiency
        self.compile_model = True
        
        self.load_model()
        self.warmup_model()
    
    def load_model(self):
        """Load optimized Whisper model for RTX 4090"""
        try:
            logger.info(f"🔧 Loading Whisper {self.model_size} for RTX 4090...")
            
            model_name = f"openai/whisper-{self.model_size}"
            
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir="/workspace/models"
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir="/workspace/models",
                torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
            
            # RTX 4090 optimizations
            if torch.cuda.is_available():
                self.model.half()  # Use FP16
                
                # Compile model (PyTorch 2.0+)
                if self.compile_model and hasattr(torch, 'compile'):
                    logger.info("🚀 Compiling Whisper for RTX 4090...")
                    self.model = torch.compile(self.model, mode="max-autotune")
            
            self.model.eval()
            logger.info("✅ Whisper model loaded and optimized")
            
        except Exception as e:
            logger.error(f"❌ Error loading Whisper: {e}")
            raise
    
    def warmup_model(self):
        """Warmup model for consistent performance"""
        try:
            logger.info("🔥 Warming up Whisper model...")
            
            # Create dummy audio (3 seconds)
            dummy_audio = torch.randn(1, 48000, dtype=torch.float16 if self.mixed_precision else torch.float32)
            
            with torch.no_grad():
                for _ in range(2):
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            inputs = self.processor(
                                dummy_audio.squeeze(),
                                sampling_rate=16000,
                                return_tensors="pt"
                            )
                            _ = self.model.generate(**inputs.to(self.device), max_length=50)
                    else:
                        inputs = self.processor(
                            dummy_audio.squeeze(),
                            sampling_rate=16000,
                            return_tensors="pt"
                        )
                        _ = self.model.generate(**inputs.to(self.device), max_length=50)
            
            torch.cuda.empty_cache()
            logger.info("✅ Whisper warmup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Whisper warmup failed: {e}")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        RTX 4090 optimized transcription with scam keyword detection
        """
        try:
            start_time = time.time()
            
            # Load and preprocess audio
            waveform, sr = torchaudio.load(audio_path)
            
            if self.mixed_precision:
                waveform = waveform.half()
            
            # Resample to 16kHz for Whisper
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Process with Whisper
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    result = self._transcribe_optimized(waveform.squeeze())
            else:
                result = self._transcribe_optimized(waveform.squeeze())
            
            # Add scam keyword detection
            result["scam_keywords"] = self._detect_scam_keywords(result["transcript"])
            result["processing_time"] = time.time() - start_time
            
            # Memory cleanup
            del waveform
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return {
                "transcript": "",
                "language": "unknown",
                "confidence": 0.0,
                "scam_keywords": {"keywords": [], "risk_score": 0},
                "error": str(e)
            }
    
    def _transcribe_optimized(self, audio) -> Dict[str, Any]:
        """Optimized transcription processing"""
        
        # Process audio
        inputs = self.processor(
            audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=448,
                        num_beams=5,
                        temperature=0.0,
                        task="transcribe",
                        language="en"
                    )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=448,
                    num_beams=5,
                    temperature=0.0,
                    task="transcribe",
                    language="en"
                )
        
        # Decode transcript
        transcript = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return {
            "transcript": transcript.strip(),
            "language": "en",  # Could be detected
            "confidence": 0.95  # Placeholder - Whisper doesn't return confidence
        }
    
    def _detect_scam_keywords(self, transcript: str) -> Dict[str, Any]:
        """Detect scam-related keywords in transcript"""
        
        scam_keywords = {
            "urgency": ["urgent", "immediately", "right now", "expires today", "limited time"],
            "authority": ["IRS", "government", "police", "FBI", "court", "legal action"],
            "financial": ["account", "credit card", "bank", "money", "payment", "refund"],
            "personal_info": ["social security", "SSN", "password", "PIN", "verify", "confirm"],
            "threats": ["arrest", "lawsuit", "penalty", "fine", "suspended", "frozen"],
            "tech_support": ["computer", "virus", "malware", "Microsoft", "Windows", "Apple"],
            "prizes": ["won", "winner", "lottery", "prize", "congratulations", "selected"],
            "charity": ["donation", "charity", "help", "disaster", "fundraising"]
        }
        
        found_keywords = []
        category_scores = {}
        
        transcript_lower = transcript.lower()
        
        for category, keywords in scam_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword in transcript_lower:
                    found_keywords.append(keyword)
                    category_score += 1
            
            if category_score > 0:
                category_scores[category] = category_score
        
        # Calculate risk score (0-100)
        total_keywords = len(found_keywords)
        risk_score = min(total_keywords * 15, 100)  # Each keyword adds 15 points, max 100
        
        return {
            "keywords": found_keywords,
            "categories": category_scores,
            "risk_score": risk_score,
            "total_count": total_keywords
        }

# For backward compatibility
WhisperTranscriber = OptimizedWhisperTranscriber
