import subprocess
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Analyzes audio processing results using local LLM via Ollama"""
    
    def __init__(self, model_name="hermes3:8b"):
        self.model_name = model_name
        self.verify_ollama()
    
    def verify_ollama(self):
        """Verify that Ollama is available and the model exists"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("Ollama not available or not running")
            elif self.model_name not in result.stdout:
                logger.warning(f"Model {self.model_name} not found in Ollama")
            else:
                logger.info(f"Ollama and model {self.model_name} verified")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not verify Ollama: {e}")
    
    def analyze(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze all processing results to detect scam patterns
        
        Args:
            analysis_data: Combined results from all audio processing modules
            
        Returns:
            Dictionary with LLM analysis results
        """
        try:
            # Build comprehensive prompt
            prompt = self._build_analysis_prompt(analysis_data)
            
            # Send to LLM
            logger.info("Sending data to LLM for scam analysis...")
            llm_response = self._query_ollama(prompt)
            
            # Parse response
            analysis_result = self._parse_llm_response(llm_response)
            
            # Add metadata
            analysis_result["analysis_metadata"] = {
                "model_used": self.model_name,
                "chunk_number": analysis_data.get("chunk_number", 0),
                "has_previous_context": bool(analysis_data.get("previous_context")),
                "processing_modules": list(analysis_data.keys())
            }
            
            logger.info(f"LLM analysis complete: {analysis_result.get('is_scam', 'unknown')} (confidence: {analysis_result.get('confidence', 0)})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "is_scam": False,
                "confidence": 0,
                "red_flags": [],
                "targets": ["none"],
                "analysis": "Error occurred during analysis",
                "error": str(e)
            }
    
    def _build_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Build comprehensive analysis prompt for the LLM with previous context"""
        
        transcription = data.get("transcription", {})
        ai_voice = data.get("ai_voice_detection", {})
        background_noise = data.get("background_noise", {})
        diarization = data.get("speaker_diarization", {})
        emotions = data.get("emotion_detection", {})
        previous_context = data.get("previous_context", {})
        
        prompt = f"""
Analyze this phone call audio chunk for scam indicators. This is chunk #{data.get('chunk_number', 0)} from call ID {data.get('call_id', 'unknown')}.

=== CURRENT CHUNK ANALYSIS ===
Text: "{transcription.get('transcript', 'No transcript available')[:2000]}"
Language: {transcription.get('language', 'unknown')}
Scam Keywords Found: {transcription.get('scam_keywords', {}).get('keywords', [])}
Keyword Risk Score: {transcription.get('scam_keywords', {}).get('risk_score', 0)}

=== VOICE AUTHENTICITY ===
AI Voice Detected: {ai_voice.get('is_ai_voice', False)}
AI Confidence: {ai_voice.get('confidence', 0)}
Classification: {ai_voice.get('classification', 'unknown')}

=== BACKGROUND ANALYSIS ===
Suspicious Sounds: {[sound['sound'] for sound in background_noise.get('suspicious_sounds', [])]}
Suspicion Score: {background_noise.get('suspicion_score', 0)}
Is Suspicious Environment: {background_noise.get('is_suspicious', False)}

=== SPEAKER ANALYSIS ===
Number of Speakers: {diarization.get('num_speakers', 0)}
Total Speech Time: {diarization.get('total_speech_time', 0)} seconds

=== EMOTION ANALYSIS (FROM DIARIZED AUDIO) ===
"""
        
        # Add emotion analysis for each speaker (from diarized audio only)
        if emotions:
            for speaker, emotion_data in emotions.items():
                prompt += f"""
{speaker} (analyzed from separated audio):
- Top Emotion: {emotion_data.get('top_emotion', 'unknown')} (confidence: {emotion_data.get('confidence', 0)})
- Stress Level: {emotion_data.get('stress_level', 0)}
- Emotional State: {emotion_data.get('emotional_state', 'unknown')}
- Scam Indicators: {emotion_data.get('scam_indicators', {})}
"""
        else:
            prompt += "\nNo emotion data available (no speakers detected or diarization failed)"
        
        # Add PREVIOUS LLM ANALYSES for context continuity
        previous_llm_analyses = previous_context.get('previous_llm_analyses', [])
        if previous_llm_analyses:
            prompt += f"""
=== PREVIOUS LLM ANALYSIS HISTORY ===
This call has {len(previous_llm_analyses)} previous chunks analyzed:

"""
            for i, prev_analysis in enumerate(previous_llm_analyses[-3:]):  # Show last 3 analyses
                prompt += f"""
Chunk {prev_analysis.get('chunk_number', i)}:
- Scam Assessment: {prev_analysis.get('is_scam', False)} (confidence: {prev_analysis.get('confidence', 0)}%)
- Scam Type: {prev_analysis.get('scam_type', 'none')}
- Red Flags: {prev_analysis.get('red_flags', [])}
- Escalation: {prev_analysis.get('escalation_level', 'unknown')}
- Analysis: "{prev_analysis.get('analysis', '')[:200]}..."

"""
            
            # Add overall trend information
            overall_trend = previous_context.get('overall_scam_trend', {})
            prompt += f"""
OVERALL CALL TREND:
- Trend: {overall_trend.get('trend', 'unknown')}
- Average Confidence: {overall_trend.get('average_confidence', 0)}%
- Scam Percentage: {overall_trend.get('scam_percentage', 0)}%
- Consistent Red Flags: {overall_trend.get('consistent_red_flags', [])}
- Escalation Pattern: {overall_trend.get('escalation_pattern', 'unknown')}
"""
        else:
            prompt += "\n=== FIRST CHUNK ===\nThis is the first chunk of this call - no previous context available."
        
        prompt += """
=== ANALYSIS INSTRUCTIONS ===
Based on ALL the above information (current chunk + previous LLM analyses), analyze for scam patterns:

IMPORTANT: Consider how this chunk fits with previous analyses. Look for:
- Escalating patterns from previous chunks
- Consistency with previous red flags
- Changes in tactics or approach
- Building narrative or pressure tactics

1. COMMON SCAM TYPES:
   - Phishing/Identity theft
   - Tech support scams
   - IRS/Government impersonation
   - Romance scams
   - Prize/Lottery scams
   - Business email compromise
   - Investment/Crypto scams

2. RED FLAGS TO CONSIDER:
   - Urgency and pressure tactics
   - Requests for personal/financial information
   - Emotional manipulation
   - AI-generated voice
   - Suspicious background environment
   - Inconsistent speaker emotions
   - Scam-related keywords
   - PATTERN CHANGES from previous chunks

3. CONTEXT ANALYSIS:
   - How does this chunk compare to previous analyses?
   - Are scam tactics escalating or changing?
   - Is there narrative consistency?
   - What's the overall call trajectory?

Respond ONLY in this JSON format:
{
    "is_scam": boolean,
    "confidence": 0-100,
    "red_flags": ["list", "of", "specific", "red", "flags", "found"],
    "targets": ["money", "personal_info", "credentials", "access", "none"],
    "scam_type": "most_likely_scam_type_or_none",
    "analysis": "detailed explanation considering previous context and current chunk",
    "escalation_level": "low|medium|high",
    "immediate_risk": boolean,
    "recommended_action": "advice for call recipient",
    "context_consistency": "how_this_chunk_relates_to_previous_analyses",
    "pattern_changes": "any_notable_changes_from_previous_chunks"
}
"""
        
        return prompt
    
    def _query_ollama(self, prompt: str) -> str:
        """Send query to Ollama and get response"""
        try:
            cmd = ["ollama", "run", self.model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"Ollama command failed: {result.stderr}")
                return ""
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            logger.error("Ollama query timed out")
            return ""
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            analysis = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["is_scam", "confidence", "red_flags", "targets", "analysis"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = self._get_default_value(field)
            
            # Ensure confidence is in valid range
            analysis["confidence"] = max(0, min(100, analysis.get("confidence", 0)))
            
            # Add context-related fields if missing
            if "context_consistency" not in analysis:
                analysis["context_consistency"] = "No context analysis available"
            if "pattern_changes" not in analysis:
                analysis["pattern_changes"] = "No pattern changes detected"
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Return fallback analysis
            return {
                "is_scam": False,
                "confidence": 0,
                "red_flags": ["llm_parsing_error"],
                "targets": ["none"],
                "scam_type": "unknown",
                "analysis": "Could not parse LLM response properly",
                "escalation_level": "low",
                "immediate_risk": False,
                "recommended_action": "Manual review needed",
                "parsing_error": str(e)
            }
    
    def _get_default_value(self, field: str):
        """Get default value for missing fields"""
        defaults = {
            "is_scam": False,
            "confidence": 0,
            "red_flags": [],
            "targets": ["none"],
            "scam_type": "unknown",
            "analysis": "No analysis available",
            "escalation_level": "low",
            "immediate_risk": False,
            "recommended_action": "Continue monitoring",
            "context_consistency": "No context analysis available",
            "pattern_changes": "No pattern analysis available"
        }
        return defaults.get(field, "")
