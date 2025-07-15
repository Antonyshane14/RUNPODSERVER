import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AIVoiceDetector:
    """Detects AI-generated/deepfake voices"""
    
    def __init__(self, model_name="as1605/Deepfake-audio-detection-V2"):
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.load_model()
    
    def load_model(self):
        """Load the AI voice detection model"""
        try:
            logger.info(f"Loading AI voice detection model: {self.model_name}")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name, 
                weights_only=True
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            logger.info("AI voice detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading AI voice model: {e}")
            raise
    
    def load_audio(self, path: str, target_sr: int = 16000) -> torch.Tensor:
        """Load and resample audio"""
        try:
            wav, sr = torchaudio.load(path)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            return wav.squeeze().numpy()
        except Exception as e:
            logger.error(f"Error loading audio {path}: {e}")
            raise
    
    def detect(self, audio_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect if voice is AI-generated
        
        Args:
            audio_path: Path to audio file
            threshold: Threshold for fake classification
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Load audio
            audio = self.load_audio(audio_path)
            
            # Extract features
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)[0]
            
            # Assumes index=1 is 'fake'
            fake_prob = probs[1].item()
            is_fake = fake_prob > threshold
            
            result = {
                "is_ai_voice": is_fake,
                "confidence": round(fake_prob, 4),
                "classification": "FAKE" if is_fake else "REAL",
                "threshold_used": threshold,
                "raw_probabilities": {
                    "real": round(probs[0].item(), 4),
                    "fake": round(probs[1].item(), 4)
                }
            }
            
            logger.info(f"AI voice detection complete: {result['classification']} ({result['confidence']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in AI voice detection: {e}")
            return {
                "is_ai_voice": False,
                "confidence": 0.0,
                "classification": "ERROR",
                "error": str(e)
            }
