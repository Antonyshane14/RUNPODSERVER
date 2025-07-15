import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import warnings
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class SpeakerDiarizer:
    """Handles speaker diarization and audio splitting"""
    
    def __init__(self, hf_token: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN must be provided or set as environment variable")
        self.load_pipeline()
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
    def load_pipeline(self):
        """Load the speaker diarization pipeline"""
        try:
            logger.info("Loading speaker diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            ).to(self.device)
            logger.info("Speaker diarization pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading diarization pipeline: {e}")
            raise
    
    def diarize(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with diarization results
        """
        try:
            logger.info(f"Running speaker diarization on {audio_path}")
            
            # Run diarization
            diarization = self.pipeline(audio_path)
            
            # Extract segments
            segments = []
            speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "duration": round(turn.duration, 2)
                }
                segments.append(segment)
                speakers.add(speaker)
            
            result = {
                "segments": segments,
                "num_speakers": len(speakers),
                "speakers": sorted(list(speakers)),
                "total_speech_time": round(sum(seg["duration"] for seg in segments), 2)
            }
            
            logger.info(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            return {
                "segments": [],
                "num_speakers": 0,
                "speakers": [],
                "total_speech_time": 0.0,
                "error": str(e)
            }
    
    def split_audio_by_speakers(self, audio_path: str, segments: List[Dict]) -> Dict[str, str]:
        """
        Split audio into separate files for each speaker
        
        Args:
            audio_path: Path to original audio file
            segments: List of speaker segments from diarization
            
        Returns:
            Dictionary mapping speaker IDs to their audio file paths
        """
        try:
            if not segments:
                return {}
                
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Create output directory
            output_dir = Path("speaker_segments")
            output_dir.mkdir(exist_ok=True)
            
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker = segment["speaker"]
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)
            
            # Create audio files for each speaker
            speaker_files = {}
            
            for speaker, speaker_segs in speaker_segments.items():
                # Combine all segments for this speaker
                combined_audio = AudioSegment.empty()
                
                for seg in speaker_segs:
                    start_ms = int(seg["start"] * 1000)
                    end_ms = int(seg["end"] * 1000)
                    segment_audio = audio[start_ms:end_ms]
                    combined_audio += segment_audio
                
                # Save speaker audio
                speaker_filename = output_dir / f"speaker_{speaker}.wav"
                combined_audio.export(str(speaker_filename), format="wav")
                speaker_files[speaker] = str(speaker_filename)
                
                logger.info(f"Created audio file for {speaker}: {speaker_filename}")
            
            return speaker_files
            
        except Exception as e:
            logger.error(f"Error splitting audio by speakers: {e}")
            return {}
    
    def get_speaker_statistics(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Get statistics about speaker activity
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dictionary with speaker statistics
        """
        try:
            if not segments:
                return {}
            
            # Calculate per-speaker statistics
            speaker_stats = {}
            
            for segment in segments:
                speaker = segment["speaker"]
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "total_time": 0.0,
                        "segment_count": 0,
                        "avg_segment_length": 0.0,
                        "longest_segment": 0.0,
                        "shortest_segment": float('inf')
                    }
                
                duration = segment["duration"]
                stats = speaker_stats[speaker]
                stats["total_time"] += duration
                stats["segment_count"] += 1
                stats["longest_segment"] = max(stats["longest_segment"], duration)
                stats["shortest_segment"] = min(stats["shortest_segment"], duration)
            
            # Calculate averages
            for speaker, stats in speaker_stats.items():
                stats["avg_segment_length"] = round(stats["total_time"] / stats["segment_count"], 2)
                stats["total_time"] = round(stats["total_time"], 2)
                stats["longest_segment"] = round(stats["longest_segment"], 2)
                stats["shortest_segment"] = round(stats["shortest_segment"], 2)
            
            return speaker_stats
            
        except Exception as e:
            logger.error(f"Error calculating speaker statistics: {e}")
            return {}
