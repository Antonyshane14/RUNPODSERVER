import whisper
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Handles audio transcription using OpenAI Whisper"""
    
    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: str, task="translate") -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_path: Path to audio file
            task: "translate" or "transcribe"
            
        Returns:
            Dictionary with transcription results
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Run transcription
            result = self.model.transcribe(audio_path, task=task)
            
            # Extract text and metadata
            transcript = result["text"].strip()
            language = result.get("language", "unknown")
            
            # Analyze transcript for scam keywords
            scam_keywords = self._detect_scam_keywords(transcript)
            
            # Calculate confidence metrics
            segments = result.get("segments", [])
            avg_confidence = self._calculate_avg_confidence(segments)
            
            transcription_result = {
                "transcript": transcript,
                "language": language,
                "task": task,
                "word_count": len(transcript.split()) if transcript else 0,
                "character_count": len(transcript),
                "segments": segments,
                "average_confidence": avg_confidence,
                "scam_keywords": scam_keywords,
                "quality_metrics": self._assess_transcript_quality(transcript, segments)
            }
            
            logger.info(f"Transcription complete: {len(transcript)} characters, {len(segments)} segments")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {
                "transcript": "",
                "language": "unknown",
                "task": task,
                "word_count": 0,
                "character_count": 0,
                "segments": [],
                "average_confidence": 0.0,
                "scam_keywords": [],
                "error": str(e)
            }
    
    def _detect_scam_keywords(self, transcript: str) -> Dict[str, Any]:
        """Detect scam-related keywords in transcript"""
        if not transcript:
            return {"keywords": [], "categories": [], "risk_score": 0.0}
        
        transcript_lower = transcript.lower()
        
        # Define scam keyword categories
        scam_categories = {
            "urgency": ["urgent", "immediately", "right now", "quickly", "hurry", "deadline", "expires"],
            "money": ["money", "payment", "bank", "account", "credit card", "wire transfer", "bitcoin", "cash", "refund"],
            "authority": ["police", "irs", "government", "arrest", "warrant", "legal action", "court", "officer"],
            "tech_support": ["computer", "virus", "malware", "microsoft", "windows", "apple", "tech support", "remote access"],
            "prizes": ["won", "winner", "prize", "lottery", "sweepstakes", "congratulations", "claim"],
            "personal_info": ["social security", "ssn", "password", "pin", "date of birth", "mother's maiden name"],
            "threats": ["arrest", "lawsuit", "legal trouble", "suspended", "frozen", "closed", "terminated"],
            "verification": ["verify", "confirm", "validate", "authenticate", "security check"]
        }
        
        detected_keywords = []
        detected_categories = []
        risk_score = 0.0
        
        for category, keywords in scam_categories.items():
            category_matches = []
            for keyword in keywords:
                if keyword in transcript_lower:
                    category_matches.append(keyword)
                    risk_score += 0.1
            
            if category_matches:
                detected_categories.append(category)
                detected_keywords.extend(category_matches)
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        return {
            "keywords": list(set(detected_keywords)),
            "categories": detected_categories,
            "risk_score": round(risk_score, 3)
        }
    
    def _calculate_avg_confidence(self, segments) -> float:
        """Calculate average confidence from segments"""
        if not segments:
            return 0.0
        
        confidences = []
        for segment in segments:
            # Whisper doesn't always provide confidence, use alternative metrics
            words = segment.get("words", [])
            if words:
                word_confidences = [word.get("probability", 0.5) for word in words if "probability" in word]
                if word_confidences:
                    confidences.extend(word_confidences)
            else:
                # Fallback: use segment timing as quality indicator
                duration = segment.get("end", 0) - segment.get("start", 0)
                text_length = len(segment.get("text", ""))
                if duration > 0 and text_length > 0:
                    # Rough confidence based on speech rate
                    words_per_second = (text_length / 5) / duration  # Rough words estimate
                    confidence = min(max(words_per_second / 3, 0.3), 0.9)  # Normalize
                    confidences.append(confidence)
        
        return round(sum(confidences) / len(confidences), 3) if confidences else 0.5
    
    def _assess_transcript_quality(self, transcript: str, segments) -> Dict[str, Any]:
        """Assess the quality of the transcription"""
        if not transcript:
            return {"quality": "poor", "issues": ["empty_transcript"]}
        
        issues = []
        quality_score = 1.0
        
        # Check for common transcription issues
        if len(transcript) < 10:
            issues.append("very_short_transcript")
            quality_score -= 0.3
        
        # Check for repeated words (often indicates poor audio quality)
        words = transcript.lower().split()
        if len(words) > 5:
            repeated_words = len(words) - len(set(words))
            if repeated_words / len(words) > 0.3:
                issues.append("high_repetition")
                quality_score -= 0.2
        
        # Check for incomplete sentences
        sentence_endings = transcript.count('.') + transcript.count('!') + transcript.count('?')
        if len(transcript) > 50 and sentence_endings == 0:
            issues.append("no_sentence_structure")
            quality_score -= 0.1
        
        # Check segment consistency
        if segments:
            avg_segment_length = sum(len(s.get("text", "")) for s in segments) / len(segments)
            if avg_segment_length < 5:
                issues.append("fragmented_segments")
                quality_score -= 0.1
        
        # Determine overall quality
        if quality_score >= 0.8:
            quality = "good"
        elif quality_score >= 0.6:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "score": round(max(quality_score, 0.0), 3),
            "issues": issues
        }
