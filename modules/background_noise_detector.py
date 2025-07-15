import librosa
import numpy as np
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class BackgroundNoiseDetector:
    """Detects suspicious background noises that might indicate fake call centers"""
    
    # Suspicious office/call center sounds
    SUSPICIOUS_TAGS = {
        'Computer keyboard': 0.4,
        'Typing': 0.5,
        'Printer': 0.3,
        'Chatter': 0.6,
        'Telephone': 0.4,
        'Office': 0.7,
        'White noise': 0.5,
        'Air conditioning': 0.4,
        'Click': 0.3,
        'Writing': 0.3,
        'Background music': 0.6,
        'Crowd': 0.5,
        'Traffic': 0.4
    }
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.model = None
        self.class_labels = []
        self.load_model()
    
    def load_model(self):
        """Load audio tagging model"""
        try:
            # Try to import and load PANNs model
            try:
                from panns_inference import AudioTagging
                self.model = AudioTagging(checkpoint_path=None, device='cpu')
                logger.info("PANNs audio tagging model loaded successfully")
                
                # Load class labels if available
                try:
                    with open('audio_class_labels.txt', 'r') as f:
                        self.class_labels = [line.strip() for line in f.readlines()]
                except FileNotFoundError:
                    logger.warning("audio_class_labels.txt not found, using fallback detection")
                    self.class_labels = list(self.SUSPICIOUS_TAGS.keys())
                    
            except ImportError:
                logger.warning("PANNs not available, using simple spectral analysis")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading background noise model: {e}")
            self.model = None
    
    def detect(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect suspicious background noises
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detection results
        """
        try:
            if self.model is not None:
                return self._detect_with_panns(audio_path)
            else:
                return self._detect_with_spectral_analysis(audio_path)
                
        except Exception as e:
            logger.error(f"Error in background noise detection: {e}")
            return {
                "suspicious_sounds": [],
                "suspicion_score": 0.0,
                "is_suspicious": False,
                "error": str(e)
            }
    
    def _detect_with_panns(self, audio_path: str) -> Dict[str, Any]:
        """Detect using PANNs audio tagging model"""
        try:
            # Load and resample audio
            audio, sr = librosa.load(audio_path, sr=32000)
            audio = audio[None, :]  # Add batch dimension
            
            # Run inference
            clipwise_output, _ = self.model.inference(audio)
            clipwise_output = clipwise_output[0]  # Remove batch dim
            
            # Analyze results
            detected_sounds = []
            suspicion_score = 0.0
            
            for idx, score in enumerate(clipwise_output):
                if idx < len(self.class_labels):
                    class_name = self.class_labels[idx]
                    if class_name in self.SUSPICIOUS_TAGS:
                        threshold = self.SUSPICIOUS_TAGS[class_name]
                        if score > threshold:
                            detected_sounds.append({
                                "sound": class_name,
                                "confidence": round(float(score), 4),
                                "threshold": threshold
                            })
                        suspicion_score += score * self.SUSPICIOUS_TAGS[class_name]
            
            # Normalize suspicion score
            suspicion_score = min(suspicion_score, 1.0)
            is_suspicious = suspicion_score > self.threshold
            
            result = {
                "suspicious_sounds": detected_sounds,
                "suspicion_score": round(suspicion_score, 4),
                "is_suspicious": is_suspicious,
                "detection_method": "PANNs",
                "threshold_used": self.threshold
            }
            
            logger.info(f"Background noise detection complete: {len(detected_sounds)} suspicious sounds found")
            return result
            
        except Exception as e:
            logger.error(f"Error in PANNs detection: {e}")
            return self._detect_with_spectral_analysis(audio_path)
    
    def _detect_with_spectral_analysis(self, audio_path: str) -> Dict[str, Any]:
        """Fallback detection using spectral analysis"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Extract features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Simple heuristics for suspicious sounds
            detected_sounds = []
            suspicion_score = 0.0
            
            # High frequency content (typing, clicking)
            if np.mean(spectral_centroids) > 3000:
                detected_sounds.append({
                    "sound": "High frequency activity (typing/clicking)",
                    "confidence": 0.7,
                    "method": "spectral_analysis"
                })
                suspicion_score += 0.3
            
            # Regular patterns (air conditioning, machinery)
            if np.std(zero_crossing_rate) < 0.01:
                detected_sounds.append({
                    "sound": "Regular background hum",
                    "confidence": 0.6,
                    "method": "spectral_analysis"
                })
                suspicion_score += 0.2
            
            # Multiple voice activity (chatter)
            energy_variance = np.var(librosa.feature.rms(y=y)[0])
            if energy_variance > 0.01:
                detected_sounds.append({
                    "sound": "Multiple voice activity",
                    "confidence": 0.5,
                    "method": "spectral_analysis"
                })
                suspicion_score += 0.25
            
            is_suspicious = suspicion_score > 0.4
            
            result = {
                "suspicious_sounds": detected_sounds,
                "suspicion_score": round(suspicion_score, 4),
                "is_suspicious": is_suspicious,
                "detection_method": "spectral_analysis",
                "note": "Fallback detection method - install PANNs for better accuracy"
            }
            
            logger.info(f"Spectral analysis complete: {len(detected_sounds)} potential suspicious sounds")
            return result
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {e}")
            return {
                "suspicious_sounds": [],
                "suspicion_score": 0.0,
                "is_suspicious": False,
                "error": str(e)
            }
