import torch
import torchaudio
import torch.nn.functional as F
import logging
from typing import Dict, Any
import warnings
import librosa
import numpy as np

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class SimpleEmotionDetector:
    """Simple emotion detector that works without problematic HuggingFace models"""
    
    def __init__(self, model_name: str = "simple", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔧 Initializing Simple Emotion Detector on {self.device}")
        
        # Simple rule-based emotion detection for now
        self.emotion_labels = [
            "neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"
        ]
        
        logger.info("✅ Simple Emotion Detector initialized successfully")
    
    def detect_emotion(self, audio_path: str) -> Dict[str, Any]:
        """
        Simple emotion detection based on audio features
        This is a placeholder implementation that analyzes basic audio properties
        """
        try:
            logger.info(f"🎵 Analyzing emotions in: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract basic features
            features = self._extract_features(audio, sr)
            
            # Simple rule-based emotion classification
            emotion_scores = self._classify_emotion(features)
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            result = {
                "dominant_emotion": dominant_emotion[0],
                "confidence": dominant_emotion[1],
                "all_emotions": emotion_scores,
                "features": features
            }
            
            logger.info(f"✅ Emotion detected: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in emotion detection: {e}")
            # Return neutral emotion as fallback
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.5,
                "all_emotions": {"neutral": 0.5},
                "features": {},
                "error": str(e)
            }
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract basic audio features for emotion analysis"""
        try:
            # Basic features
            features = {}
            
            # Energy/amplitude features
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            
            # MFCC features (first few coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
            for i in range(5):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            
            return features
            
        except Exception as e:
            logger.warning(f"⚠️ Feature extraction error: {e}")
            return {"error": str(e)}
    
    def _classify_emotion(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Simple rule-based emotion classification
        This is a basic implementation - in production, you'd use a trained model
        """
        try:
            emotions = {emotion: 0.1 for emotion in self.emotion_labels}  # Base probability
            
            if not features or "error" in features:
                emotions["neutral"] = 0.7
                return emotions
            
            # Simple rules based on audio features
            energy = features.get('rms_energy', 0.1)
            tempo = features.get('tempo', 120)
            spectral_centroid = features.get('spectral_centroid', 1000)
            zcr = features.get('zero_crossing_rate', 0.1)
            
            # High energy + fast tempo = happy/excited
            if energy > 0.15 and tempo > 140:
                emotions["happy"] += 0.4
                emotions["surprise"] += 0.2
            
            # Low energy + slow tempo = sad
            elif energy < 0.08 and tempo < 100:
                emotions["sad"] += 0.4
                emotions["neutral"] += 0.2
            
            # High energy + high spectral centroid = angry
            elif energy > 0.12 and spectral_centroid > 2000:
                emotions["angry"] += 0.4
                emotions["fear"] += 0.1
            
            # High zero crossing rate = fear/anxiety
            elif zcr > 0.15:
                emotions["fear"] += 0.3
                emotions["angry"] += 0.2
            
            # Default to neutral
            else:
                emotions["neutral"] += 0.5
            
            # Normalize probabilities
            total = sum(emotions.values())
            emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            logger.warning(f"⚠️ Classification error: {e}")
            return {"neutral": 1.0}

# Keep the original class name for compatibility
class EmotionDetector(SimpleEmotionDetector):
    """Alias for compatibility with existing code"""
    pass
