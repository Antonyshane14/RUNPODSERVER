import librosa
import soundfile as sf
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio format conversion and preprocessing"""
    
    def __init__(self, target_sr=16000, target_channels=1):
        self.target_sr = target_sr
        self.target_channels = target_channels
        
    def convert_to_standard_format(self, input_path: str) -> str:
        """
        Convert audio to standard format (16kHz mono WAV)
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to converted audio file
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
            
            # Generate output path
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_processed.wav")
            
            # Save as WAV
            sf.write(output_path, y, self.target_sr)
            
            logger.info(f"Audio converted: {input_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio {input_path}: {e}")
            raise
    
    def validate_audio(self, audio_path: str) -> bool:
        """
        Validate that audio file is readable and has reasonable duration
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            
            # Check if duration is reasonable (between 1 and 30 seconds for chunks)
            if 1.0 <= duration <= 30.0:
                return True
            else:
                logger.warning(f"Audio duration {duration}s is outside expected range")
                return False
                
        except Exception as e:
            logger.error(f"Error validating audio {audio_path}: {e}")
            return False
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get audio file information
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            
            return {
                "sample_rate": sr,
                "duration": round(duration, 2),
                "samples": len(y),
                "channels": 1 if len(y.shape) == 1 else y.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info {audio_path}: {e}")
            return {}
