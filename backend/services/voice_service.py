import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        """Initialize voice service with fallback to browser-based solutions"""
        
        # ElevenLabs Configuration (optional)
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = None
        
        # Try to initialize ElevenLabs if available
        if self.elevenlabs_api_key:
            try:
                from elevenlabs import ElevenLabs
                self.client = ElevenLabs(api_key=self.elevenlabs_api_key)
                logger.info("✅ ElevenLabs TTS + STT service initialized")
            except ImportError:
                logger.info("ElevenLabs package not installed - using browser-based voice")
            except Exception as e:
                logger.warning(f"ElevenLabs initialization failed: {e} - using browser-based voice")
        else:
            logger.info("🎤 Using browser-based Web Speech API (no API key required)")
        
        # Indian Language Support (for reference - Web Speech API supports many of these)
        self.indian_language_support = {
            "hi": {
                "name": "Hindi",
                "elevenlabs_code": "hin",
                "web_speech_code": "hi-IN",
                "tts_supported": True,  # Browser TTS
                "stt_supported": True,  # Browser STT
                "accuracy": "good"
            },
            "bn": {
                "name": "Bengali", 
                "elevenlabs_code": "ben",
                "web_speech_code": "bn-BD",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "te": {
                "name": "Telugu",
                "elevenlabs_code": "tel",
                "web_speech_code": "te-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "ta": {
                "name": "Tamil",
                "elevenlabs_code": "tam",
                "web_speech_code": "ta-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "mr": {
                "name": "Marathi",
                "elevenlabs_code": "mar",
                "web_speech_code": "mr-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "gu": {
                "name": "Gujarati",
                "elevenlabs_code": "guj",
                "web_speech_code": "gu-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "kn": {
                "name": "Kannada",
                "elevenlabs_code": "kan",
                "web_speech_code": "kn-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "ml": {
                "name": "Malayalam",
                "elevenlabs_code": "mal",
                "web_speech_code": "ml-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "pa": {
                "name": "Punjabi",
                "elevenlabs_code": "pan",
                "web_speech_code": "pa-IN",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "or": {
                "name": "Odia",
                "elevenlabs_code": "ori",
                "web_speech_code": "or-IN",
                "tts_supported": False,  # Limited browser support
                "stt_supported": False,
                "accuracy": "limited"
            },
            "as": {
                "name": "Assamese",
                "elevenlabs_code": "asm",
                "web_speech_code": "as-IN",
                "tts_supported": False,  # Limited browser support
                "stt_supported": False,
                "accuracy": "limited"
            },
            "ur": {
                "name": "Urdu",
                "elevenlabs_code": "urd",
                "web_speech_code": "ur-PK",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "good"
            },
            "en": {
                "name": "English",
                "elevenlabs_code": "eng",
                "web_speech_code": "en-US",
                "tts_supported": True,
                "stt_supported": True,
                "accuracy": "excellent"
            }
        }
        
        # Default settings
        self.default_voice_id = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    def is_tts_available(self) -> bool:
        """TTS is always available via browser Web Speech API"""
        return True
    
    def is_stt_available(self) -> bool:
        """STT is always available via browser Web Speech API"""
        return True
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get supported languages for both STT and TTS"""
        return self.indian_language_support
    
    def is_language_supported(self, language_code: str, service: str = "both") -> bool:
        """Check if a language is supported for STT, TTS, or both"""
        if language_code not in self.indian_language_support:
            return False
        
        lang_info = self.indian_language_support[language_code]
        
        if service == "stt":
            return lang_info.get("stt_supported", False)
        elif service == "tts":
            return lang_info.get("tts_supported", False)
        else:  # both
            return lang_info.get("stt_supported", False) and lang_info.get("tts_supported", False)
    
    async def speech_to_text(
        self, 
        audio_data: bytes, 
        audio_format: str = "webm", 
        language: str = "en"
    ) -> Optional[str]:
        """
        Speech to text - handled by browser Web Speech API
        This method exists for API compatibility but should not be used
        when using browser-based STT
        """
        if self.client and self.elevenlabs_api_key:
            # If ElevenLabs is available, use it
            try:
                import tempfile
                import asyncio
                
                temp_audio_path = None
                try:
                    # Create temporary file for audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as temp_file:
                        temp_file.write(audio_data)
                        temp_audio_path = temp_file.name
                    
                    logger.info(f"Transcribing audio using ElevenLabs: {len(audio_data)} bytes")
                    
                    # ElevenLabs STT API call
                    with open(temp_audio_path, "rb") as audio_file:
                        response = await asyncio.to_thread(
                            self.client.speech_to_text.transcribe,
                            audio_file,
                            model_id="scribe-v1"
                        )
                    
                    if hasattr(response, 'text'):
                        return response.text.strip()
                    elif isinstance(response, dict) and 'text' in response:
                        return response['text'].strip()
                    else:
                        logger.error("Unexpected response format from ElevenLabs STT")
                        return None
                        
                except Exception as e:
                    logger.error(f"ElevenLabs STT error: {e}")
                    return None
                
                finally:
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                        except Exception as e:
                            logger.warning(f"Could not remove temp audio file: {e}")
            except Exception as e:
                logger.error(f"STT processing error: {e}")
                return None
        else:
            # Return message indicating browser-based STT should be used
            logger.info("Using browser-based speech recognition")
            return "Browser-based speech recognition is being used. This message should not appear in production."
    
    async def text_to_speech(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> Optional[str]:
        """
        Text to speech - handled by browser Web Speech API
        This method exists for API compatibility but should not be used
        when using browser-based TTS
        """
        if self.client and self.elevenlabs_api_key:
            # If ElevenLabs is available, use it
            try:
                import base64
                
                if not text or len(text.strip()) == 0:
                    return None
                
                # Limit text length
                max_length = int(os.getenv("TTS_MAX_CHARACTERS", "5000"))
                if len(text) > max_length:
                    text = text[:max_length-3] + "..."
                
                voice_id = voice_id or self.default_voice_id
                settings = voice_settings or {
                    "stability": 0.75,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
                
                logger.info(f"Generating speech using ElevenLabs: {len(text)} characters")
                
                # Use multilingual model for Indian languages
                model = "eleven_multilingual_v2" if language != "en" else "eleven_monolingual_v1"
                
                # Generate speech
                audio_generator = self.client.generate(
                    text=text,
                    voice=voice_id,
                    voice_settings=settings,
                    model=model
                )
                
                # Convert to bytes
                audio_bytes = b"".join(audio_generator)
                
                # Encode to base64
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                logger.info(f"Generated {len(audio_bytes)} bytes of audio")
                return audio_base64
                
            except Exception as e:
                logger.error(f"ElevenLabs TTS error: {e}")
                return None
        else:
            # Return message indicating browser-based TTS should be used
            logger.info("Using browser-based text-to-speech")
            return "Browser-based text-to-speech is being used. This message should not appear in production."
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """Get available TTS voices"""
        if self.client and self.elevenlabs_api_key:
            try:
                voices = self.client.voices.get_all()
                
                voice_list = []
                for voice in voices.voices:
                    voice_info = {
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "description": getattr(voice, 'description', ''),
                        "category": getattr(voice, 'category', ''),
                        "labels": getattr(voice, 'labels', {}),
                        "preview_url": getattr(voice, 'preview_url', ''),
                        "supports_multilingual": True
                    }
                    voice_list.append(voice_info)
                
                return {
                    "voices": voice_list, 
                    "default_voice_id": self.default_voice_id,
                    "supported_languages": self.indian_language_support,
                    "service": "ElevenLabs"
                }
            except Exception as e:
                logger.error(f"Error retrieving ElevenLabs voices: {e}")
                return self._get_browser_voice_info()
        else:
            return self._get_browser_voice_info()
    
    def _get_browser_voice_info(self) -> Dict[str, Any]:
        """Return browser-based voice information"""
        return {
            "voices": [
                {
                    "voice_id": "browser_default",
                    "name": "Browser Default Voice",
                    "description": "Uses browser's built-in text-to-speech",
                    "category": "browser",
                    "labels": {"language": "multiple"},
                    "preview_url": "",
                    "supports_multilingual": True
                }
            ],
            "default_voice_id": "browser_default",
            "supported_languages": self.indian_language_support,
            "service": "Browser Web Speech API",
            "note": "Actual voices depend on user's browser and operating system"
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get voice service information"""
        service_type = "ElevenLabs + Browser Fallback" if (self.client and self.elevenlabs_api_key) else "Browser Web Speech API"
        
        return {
            "service_type": service_type,
            "elevenlabs_available": bool(self.client and self.elevenlabs_api_key),
            "browser_fallback": True,
            "tts": {
                "service": service_type,
                "available": True,  # Always available via browser
                "default_voice_id": self.default_voice_id if (self.client and self.elevenlabs_api_key) else "browser_default",
                "max_text_length": int(os.getenv("TTS_MAX_CHARACTERS", "5000")),
                "supported_formats": ["mp3"] if (self.client and self.elevenlabs_api_key) else ["browser_audio"],
                "note": "Uses browser Web Speech API when ElevenLabs unavailable",
                "supported_languages": [
                    {
                        "code": code,
                        "name": info["name"],
                        "supported": info["tts_supported"],
                        "web_speech_code": info.get("web_speech_code", code)
                    }
                    for code, info in self.indian_language_support.items()
                ]
            },
            "stt": {
                "service": service_type,
                "available": True,  # Always available via browser
                "supported_formats": ["webm", "mp3", "wav", "m4a", "ogg", "aac", "flac", "mp4"] if (self.client and self.elevenlabs_api_key) else ["browser_audio"],
                "max_file_size_mb": int(os.getenv("STT_MAX_FILE_SIZE_MB", "3000")) if (self.client and self.elevenlabs_api_key) else "unlimited",
                "note": "Uses browser Web Speech API when ElevenLabs unavailable",
                "supported_languages": [
                    {
                        "code": code,
                        "name": info["name"],
                        "supported": info["stt_supported"],
                        "web_speech_code": info.get("web_speech_code", code),
                        "accuracy": info.get("accuracy", "good")
                    }
                    for code, info in self.indian_language_support.items()
                ]
            },
            "api_keys": {
                "elevenlabs_configured": bool(self.elevenlabs_api_key),
                "elevenlabs_working": bool(self.client and self.elevenlabs_api_key)
            },
            "recommendations": [
                "Browser-based voice features work without any API keys",
                "For production use, consider adding ElevenLabs API key for better quality",
                "Web Speech API requires HTTPS in production",
                "Voice features require user interaction to work properly"
            ]
        }