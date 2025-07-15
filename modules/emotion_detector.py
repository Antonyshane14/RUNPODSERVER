import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForAudioClassification, AutoConfig
import logging
from typing import Dict, Any
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class OptimizedEmotionDetector:
    """RTX 4090 optimized emotion detector for RunPod deployment"""
    
    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.mixed_precision = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # RTX 4090 optimizations
        self.batch_size = 4  # Process multiple audio files in batch
        self.compile_model = True  # PyTorch 2.0+ compilation
        
        self.load_model()
        self.warmup_model()
    
    def load_model(self):
        """Load and optimize the emotion detection model for RTX 4090"""
        try:
            # Check for HF_TOKEN
            import os
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.error("❌ HF_TOKEN environment variable is required for Hugging Face model access")
                logger.error("💡 Get your token from: https://huggingface.co/settings/tokens")
                logger.error("🔧 Set it with: export HF_TOKEN='your_token_here'")
                raise ValueError("HF_TOKEN environment variable must be set")
            
            logger.info(f"🔧 Loading emotion model: {self.model_name}")
            
            # Load with optimized settings and explicit tokenizer handling
            logger.info("📥 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir="/workspace/models",
                use_auth_token=hf_token,
                trust_remote_code=True,
                force_download=False,
                resume_download=True
            )
            
            logger.info("📥 Loading model...")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                cache_dir="/workspace/models",
                torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                use_auth_token=hf_token,
                trust_remote_code=True,
                force_download=False,
                resume_download=True
            ).to(self.device)
            
            # RTX 4090 optimizations
            if torch.cuda.is_available():
                self.model.half()  # Use FP16 for RTX 4090
                
                # Compile model for RTX 4090 (PyTorch 2.0+)
                if self.compile_model and hasattr(torch, 'compile'):
                    logger.info("🚀 Compiling model for RTX 4090...")
                    self.model = torch.compile(self.model, mode="max-autotune")
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info("✅ Emotion detection model loaded and optimized")
            
        except Exception as e:
            logger.error(f"❌ Error loading emotion model: {e}")
            
            # Try alternative approach for problematic models
            if "expected str, bytes or os.PathLike object, not NoneType" in str(e):
                logger.warning("🔄 Trying alternative loading approach for tokenizer issue...")
                try:
                    # Force download all files first
                    config = AutoConfig.from_pretrained(
                        self.model_name,
                        use_auth_token=hf_token,
                        cache_dir="/workspace/models"
                    )
                    
                    # Try loading processor again
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        cache_dir="/workspace/models",
                        use_auth_token=hf_token,
                        local_files_only=False,
                        trust_remote_code=True
                    )
                    
                    self.model = AutoModelForAudioClassification.from_pretrained(
                        self.model_name,
                        config=config,
                        cache_dir="/workspace/models",
                        torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        use_auth_token=hf_token,
                        local_files_only=False,
                        trust_remote_code=True
                    ).to(self.device)
                    
                    logger.info("✅ Alternative loading successful")
                    
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback loading also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise
    
    def warmup_model(self):
        """Warmup model for consistent performance"""
        try:
            logger.info("🔥 Warming up emotion detection model...")
            
            # Create dummy input
            dummy_input = torch.randn(1, 16000, dtype=torch.float16 if self.mixed_precision else torch.float32).to(self.device)
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(3):
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            inputs = self.processor(
                                dummy_input.cpu().numpy(),
                                sampling_rate=16000,
                                return_tensors="pt"
                            ).to(self.device)
                            _ = self.model(**inputs)
                    else:
                        inputs = self.processor(
                            dummy_input.cpu().numpy(),
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).to(self.device)
                        _ = self.model(**inputs)
            
            # Clear cache after warmup
            torch.cuda.empty_cache()
            logger.info("✅ Model warmup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")

    def detect(self, audio_path: str) -> Dict[str, Any]:
        """
        RTX 4090 optimized emotion detection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion detection results
        """
        try:
            # Load and preprocess audio with optimizations
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to half precision for RTX 4090
            if self.mixed_precision:
                waveform = waveform.half()
            
            # Resample if needed (optimized)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(self.device)
                waveform = resampler(waveform.to(self.device))
            else:
                waveform = waveform.to(self.device)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Process with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    result = self._process_audio_optimized(waveform)
            else:
                result = self._process_audio_optimized(waveform)
            
            # Memory cleanup
            del waveform
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in emotion detection: {e}")
            return self._get_error_result(str(e))
    
    def _process_audio_optimized(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """Optimized audio processing for RTX 4090"""
        
        # Process audio
        inputs = self.processor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference with optimizations
        with torch.no_grad():
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(**inputs).logits
            else:
                logits = self.model(**inputs).logits
            
            # Use optimized softmax
            probs = F.softmax(logits, dim=1)[0]
        
        # Convert to CPU for processing
        probs_cpu = probs.cpu().float()
        
        # Get emotion probabilities
        emotion_probs = {}
        for i, prob in enumerate(probs_cpu):
            emotion_label = self.model.config.id2label[i]
            emotion_probs[emotion_label] = float(prob)
        
        # Get top emotion
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        top_confidence = emotion_probs[top_emotion]
        
        # Optimized categorization
        stress_emotions = ['angry', 'fearful', 'sad', 'surprised']
        calm_emotions = ['calm', 'neutral', 'happy']
        
        stress_level = sum(emotion_probs.get(emotion, 0) for emotion in stress_emotions)
        calm_level = sum(emotion_probs.get(emotion, 0) for emotion in calm_emotions)
        
        result = {
            "top_emotion": top_emotion,
            "confidence": round(top_confidence, 4),
            "all_emotions": {k: round(v, 4) for k, v in emotion_probs.items()},
            "stress_level": round(stress_level, 4),
            "calm_level": round(calm_level, 4),
            "emotional_state": self._categorize_emotional_state(top_emotion, top_confidence),
            "scam_indicators": self._get_scam_emotional_indicators(emotion_probs),
            "processing_info": {
                "device": str(self.device),
                "mixed_precision": self.mixed_precision,
                "model_compiled": self.compile_model
            }
        }
        
        logger.info(f"🎯 Emotion: {top_emotion} ({top_confidence:.3f})")
        return result
    
    def _categorize_emotional_state(self, emotion: str, confidence: float) -> str:
        """Categorize emotional state for analysis"""
        high_stress = ['angry', 'fearful']
        moderate_stress = ['sad', 'surprised', 'disgust']
        calm_states = ['calm', 'neutral', 'happy']
        
        if confidence < 0.4:
            return "uncertain"
        elif emotion in high_stress:
            return "high_stress"
        elif emotion in moderate_stress:
            return "moderate_stress"
        elif emotion in calm_states:
            return "calm"
        else:
            return "neutral"
    
    def _get_scam_emotional_indicators(self, emotion_probs: Dict[str, float]) -> Dict[str, Any]:
        """Analyze emotions for scam indicators"""
        indicators = {
            "victim_stress": False,
            "emotional_manipulation": False,
            "fear_tactics": False,
            "urgency_pressure": False
        }
        
        # High fear or stress might indicate victim under pressure
        if emotion_probs.get('fearful', 0) > 0.6 or emotion_probs.get('angry', 0) > 0.5:
            indicators["victim_stress"] = True
        
        # Rapid emotional changes or extreme emotions
        emotion_values = list(emotion_probs.values())
        if max(emotion_values) > 0.8 or (max(emotion_values) - min(emotion_values)) > 0.7:
            indicators["emotional_manipulation"] = True
        
        # Specific fear-based emotions
        if emotion_probs.get('fearful', 0) > 0.5:
            indicators["fear_tactics"] = True
        
        # Combination of stress emotions might indicate urgency tactics
        stress_combo = emotion_probs.get('angry', 0) + emotion_probs.get('fearful', 0) + emotion_probs.get('surprised', 0)
        if stress_combo > 1.0:
            indicators["urgency_pressure"] = True
        
        return indicators
    
    def _get_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Return error result with proper structure"""
        return {
            "top_emotion": "unknown",
            "confidence": 0.0,
            "all_emotions": {},
            "stress_level": 0.0,
            "calm_level": 0.0,
            "emotional_state": "error",
            "scam_indicators": {
                "victim_stress": False,
                "emotional_manipulation": False,
                "fear_tactics": False,
                "urgency_pressure": False
            },
            "error": error_msg
        }
    
    def cleanup_memory(self):
        """Cleanup GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        logger.info("🧹 GPU memory cleaned")

# For backward compatibility
EmotionDetector = OptimizedEmotionDetector
