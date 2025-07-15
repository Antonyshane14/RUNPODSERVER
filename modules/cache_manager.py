import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages cache files for ongoing call analysis and context tracking"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Cache manager initialized with directory: {cache_dir}")
    
    def load_context(self, call_id: str) -> Dict[str, Any]:
        """
        Load previous context for a call, including previous LLM outputs
        
        Args:
            call_id: Unique identifier for the call
            
        Returns:
            Dictionary with previous context data including LLM analysis history
        """
        if not call_id:
            return {}
        
        try:
            cache_file = self.cache_dir / f"{call_id}.json"
            
            if not cache_file.exists():
                logger.info(f"No previous context found for call {call_id}")
                return {}
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                context = json.load(f)
            
            # Extract previous LLM outputs for feeding back into next analysis
            previous_llm_analyses = []
            for chunk in context.get('chunks', []):
                scam_analysis = chunk.get('scam_analysis', {})
                if scam_analysis:
                    previous_llm_analyses.append({
                        "chunk_number": chunk.get('chunk_number', 0),
                        "is_scam": scam_analysis.get('is_scam', False),
                        "confidence": scam_analysis.get('confidence', 0),
                        "red_flags": scam_analysis.get('red_flags', []),
                        "scam_type": scam_analysis.get('scam_type', 'none'),
                        "analysis": scam_analysis.get('analysis', ''),
                        "escalation_level": scam_analysis.get('escalation_level', 'low')
                    })
            
            # Add LLM history to context for next analysis
            context['previous_llm_analyses'] = previous_llm_analyses
            context['overall_scam_trend'] = self._calculate_scam_trend(previous_llm_analyses)
            
            logger.info(f"Loaded context for call {call_id}: {len(context.get('chunks', []))} previous chunks, {len(previous_llm_analyses)} LLM analyses")
            return context
            
        except Exception as e:
            logger.error(f"Error loading context for call {call_id}: {e}")
            return {}
    
    def update_context(self, call_id: str, chunk_data: Dict[str, Any], llm_analysis: Dict[str, Any]) -> None:
        """
        Update context cache with new chunk data and analysis
        
        Args:
            call_id: Unique identifier for the call
            chunk_data: Data from the current chunk processing
            llm_analysis: Analysis results from the LLM
        """
        if not call_id:
            return
        
        try:
            # Load existing context
            context = self.load_context(call_id)
            
            # Initialize context structure if new call
            if not context:
                context = {
                    "call_id": call_id,
                    "created_at": time.time(),
                    "last_updated": time.time(),
                    "chunks": [],
                    "overall_analysis": {
                        "scam_likelihood": 0.0,
                        "persistent_patterns": [],
                        "escalation_trend": "stable",
                        "risk_level": "low"
                    },
                    "speaker_profiles": {},
                    "timeline": []
                }
            
            # Update timestamp
            context["last_updated"] = time.time()
            
            # Add current chunk
            chunk_summary = {
                "chunk_number": chunk_data.get("chunk_number", 0),
                "timestamp": chunk_data.get("timestamp", time.time()),
                "transcript_snippet": chunk_data.get("transcription", {}).get("transcript", "")[:200],
                "ai_voice_detected": chunk_data.get("ai_voice_detection", {}).get("is_ai_voice", False),
                "suspicious_background": chunk_data.get("background_noise", {}).get("is_suspicious", False),
                "num_speakers": chunk_data.get("speaker_diarization", {}).get("num_speakers", 0),
                "dominant_emotions": self._extract_dominant_emotions(chunk_data.get("emotion_detection", {})),
                "scam_analysis": {
                    "is_scam": llm_analysis.get("is_scam", False),
                    "confidence": llm_analysis.get("confidence", 0),
                    "red_flags": llm_analysis.get("red_flags", []),
                    "scam_type": llm_analysis.get("scam_type", "none")
                }
            }
            
            context["chunks"].append(chunk_summary)
            
            # Update overall analysis
            self._update_overall_analysis(context, llm_analysis)
            
            # Update speaker profiles
            self._update_speaker_profiles(context, chunk_data)
            
            # Update timeline
            self._update_timeline(context, chunk_summary)
            
            # Save updated context
            cache_file = self.cache_dir / f"{call_id}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated context for call {call_id}")
            
        except Exception as e:
            logger.error(f"Error updating context for call {call_id}: {e}")
    
    def get_call_summary(self, call_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the call analysis
        
        Args:
            call_id: Unique identifier for the call
            
        Returns:
            Dictionary with call summary
        """
        try:
            context = self.load_context(call_id)
            
            if not context:
                return {"error": "Call not found"}
            
            chunks = context.get("chunks", [])
            overall = context.get("overall_analysis", {})
            
            # Calculate summary statistics
            scam_chunks = sum(1 for chunk in chunks if chunk.get("scam_analysis", {}).get("is_scam", False))
            ai_voice_chunks = sum(1 for chunk in chunks if chunk.get("ai_voice_detected", False))
            suspicious_bg_chunks = sum(1 for chunk in chunks if chunk.get("suspicious_background", False))
            
            avg_confidence = sum(chunk.get("scam_analysis", {}).get("confidence", 0) for chunk in chunks) / len(chunks) if chunks else 0
            
            # Extract all red flags
            all_red_flags = []
            for chunk in chunks:
                all_red_flags.extend(chunk.get("scam_analysis", {}).get("red_flags", []))
            
            unique_red_flags = list(set(all_red_flags))
            
            # Get most recent analysis
            latest_chunk = chunks[-1] if chunks else {}
            
            summary = {
                "call_id": call_id,
                "total_chunks": len(chunks),
                "duration_analyzed": f"{len(chunks) * 10} seconds",  # Assuming 10-second chunks
                "overall_assessment": {
                    "is_likely_scam": overall.get("scam_likelihood", 0) > 0.6,
                    "scam_likelihood": round(overall.get("scam_likelihood", 0), 2),
                    "risk_level": overall.get("risk_level", "low"),
                    "escalation_trend": overall.get("escalation_trend", "stable")
                },
                "statistics": {
                    "scam_chunks": scam_chunks,
                    "scam_percentage": round((scam_chunks / len(chunks)) * 100, 1) if chunks else 0,
                    "ai_voice_detected_chunks": ai_voice_chunks,
                    "suspicious_background_chunks": suspicious_bg_chunks,
                    "average_confidence": round(avg_confidence, 1)
                },
                "red_flags": unique_red_flags,
                "speaker_analysis": context.get("speaker_profiles", {}),
                "timeline": context.get("timeline", []),
                "latest_analysis": latest_chunk.get("scam_analysis", {}),
                "created_at": context.get("created_at"),
                "last_updated": context.get("last_updated")
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting call summary for {call_id}: {e}")
            return {"error": str(e)}
    
    def clear_call_data(self, call_id: str) -> None:
        """
        Clear cache data for a specific call
        
        Args:
            call_id: Unique identifier for the call
        """
        try:
            cache_file = self.cache_dir / f"{call_id}.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache data for call {call_id}")
            else:
                logger.warning(f"No cache data found for call {call_id}")
                
        except Exception as e:
            logger.error(f"Error clearing cache data for call {call_id}: {e}")
    
    def _extract_dominant_emotions(self, emotion_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract dominant emotions for each speaker"""
        dominant_emotions = {}
        
        for speaker, emotions in emotion_data.items():
            top_emotion = emotions.get("top_emotion", "unknown")
            confidence = emotions.get("confidence", 0)
            dominant_emotions[speaker] = f"{top_emotion} ({confidence:.2f})"
        
        return dominant_emotions
    
    def _update_overall_analysis(self, context: Dict[str, Any], llm_analysis: Dict[str, Any]) -> None:
        """Update overall analysis based on new chunk"""
        overall = context["overall_analysis"]
        chunks = context["chunks"]
        
        # Update scam likelihood (weighted average favoring recent chunks)
        current_confidence = llm_analysis.get("confidence", 0) / 100.0
        current_is_scam = llm_analysis.get("is_scam", False)
        
        if current_is_scam:
            # If current chunk is flagged as scam, increase likelihood
            overall["scam_likelihood"] = min(overall["scam_likelihood"] + current_confidence * 0.3, 1.0)
        else:
            # Slight decrease if not scam, but don't go too low
            overall["scam_likelihood"] = max(overall["scam_likelihood"] - 0.1, 0.0)
        
        # Update persistent patterns
        red_flags = llm_analysis.get("red_flags", [])
        for flag in red_flags:
            if flag not in overall["persistent_patterns"]:
                # Count occurrences
                flag_count = sum(1 for chunk in chunks if flag in chunk.get("scam_analysis", {}).get("red_flags", []))
                if flag_count >= 2:  # Flag appears in multiple chunks
                    overall["persistent_patterns"].append(flag)
        
        # Update risk level
        if overall["scam_likelihood"] > 0.8:
            overall["risk_level"] = "high"
        elif overall["scam_likelihood"] > 0.5:
            overall["risk_level"] = "medium"
        else:
            overall["risk_level"] = "low"
        
        # Update escalation trend
        if len(chunks) >= 3:
            recent_confidences = [chunk.get("scam_analysis", {}).get("confidence", 0) for chunk in chunks[-3:]]
            if recent_confidences[2] > recent_confidences[1] > recent_confidences[0]:
                overall["escalation_trend"] = "escalating"
            elif recent_confidences[2] < recent_confidences[1] < recent_confidences[0]:
                overall["escalation_trend"] = "de-escalating"
            else:
                overall["escalation_trend"] = "stable"
    
    def _update_speaker_profiles(self, context: Dict[str, Any], chunk_data: Dict[str, Any]) -> None:
        """Update speaker behavior profiles"""
        speaker_profiles = context["speaker_profiles"]
        emotion_data = chunk_data.get("emotion_detection", {})
        
        for speaker, emotions in emotion_data.items():
            if speaker not in speaker_profiles:
                speaker_profiles[speaker] = {
                    "appearances": 0,
                    "dominant_emotions": [],
                    "stress_levels": [],
                    "emotional_stability": "stable"
                }
            
            profile = speaker_profiles[speaker]
            profile["appearances"] += 1
            profile["dominant_emotions"].append(emotions.get("top_emotion", "unknown"))
            profile["stress_levels"].append(emotions.get("stress_level", 0))
            
            # Keep only recent data (last 10 appearances)
            if len(profile["dominant_emotions"]) > 10:
                profile["dominant_emotions"] = profile["dominant_emotions"][-10:]
                profile["stress_levels"] = profile["stress_levels"][-10:]
            
            # Update emotional stability
            if len(profile["stress_levels"]) >= 3:
                recent_stress = profile["stress_levels"][-3:]
                stress_variance = sum((x - sum(recent_stress)/3)**2 for x in recent_stress) / 3
                if stress_variance > 0.2:
                    profile["emotional_stability"] = "unstable"
                elif stress_variance > 0.1:
                    profile["emotional_stability"] = "variable"
                else:
                    profile["emotional_stability"] = "stable"
    
    def _update_timeline(self, context: Dict[str, Any], chunk_summary: Dict[str, Any]) -> None:
        """Update call timeline with significant events"""
        timeline = context["timeline"]
        
        # Add significant events to timeline
        chunk_num = chunk_summary.get("chunk_number", 0)
        
        # High-confidence scam detection
        if chunk_summary.get("scam_analysis", {}).get("confidence", 0) > 80:
            timeline.append({
                "chunk": chunk_num,
                "event": "high_confidence_scam_detection",
                "details": f"Scam confidence: {chunk_summary['scam_analysis']['confidence']}%"
            })
        
        # AI voice detection
        if chunk_summary.get("ai_voice_detected"):
            timeline.append({
                "chunk": chunk_num,
                "event": "ai_voice_detected",
                "details": "Artificial voice generation detected"
            })
        
        # Speaker changes
        current_speakers = chunk_summary.get("num_speakers", 0)
        if len(context["chunks"]) > 1:
            prev_speakers = context["chunks"][-2].get("num_speakers", 0)
            if current_speakers != prev_speakers:
                timeline.append({
                    "chunk": chunk_num,
                    "event": "speaker_change",
                    "details": f"Speakers changed from {prev_speakers} to {current_speakers}"
                })
        
        # Keep timeline manageable (last 20 events)
        if len(timeline) > 20:
            context["timeline"] = timeline[-20:]
    
    def _calculate_scam_trend(self, previous_llm_analyses: List[Dict]) -> Dict[str, Any]:
        """Calculate overall scam trend from previous LLM analyses"""
        if not previous_llm_analyses:
            return {
                "trend": "unknown",
                "average_confidence": 0.0,
                "consistent_red_flags": [],
                "escalation_pattern": "stable"
            }
        
        # Calculate trend metrics
        confidences = [analysis.get('confidence', 0) for analysis in previous_llm_analyses]
        scam_flags = [analysis.get('is_scam', False) for analysis in previous_llm_analyses]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        scam_percentage = (sum(scam_flags) / len(scam_flags)) * 100 if scam_flags else 0
        
        # Find consistent red flags
        all_red_flags = []
        for analysis in previous_llm_analyses:
            all_red_flags.extend(analysis.get('red_flags', []))
        
        flag_counts = {}
        for flag in all_red_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Red flags that appear in multiple chunks
        consistent_red_flags = [flag for flag, count in flag_counts.items() if count >= 2]
        
        # Determine trend
        if scam_percentage >= 70:
            trend = "highly_suspicious"
        elif scam_percentage >= 40:
            trend = "moderately_suspicious"
        elif scam_percentage >= 15:
            trend = "potentially_suspicious"
        else:
            trend = "likely_clean"
        
        # Check escalation pattern
        if len(confidences) >= 3:
            recent_trend = confidences[-3:]
            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                escalation_pattern = "escalating"
            elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                escalation_pattern = "de-escalating"
            else:
                escalation_pattern = "variable"
        else:
            escalation_pattern = "insufficient_data"
        
        return {
            "trend": trend,
            "average_confidence": round(avg_confidence, 1),
            "scam_percentage": round(scam_percentage, 1),
            "consistent_red_flags": consistent_red_flags,
            "escalation_pattern": escalation_pattern,
            "total_chunks_analyzed": len(previous_llm_analyses)
        }
