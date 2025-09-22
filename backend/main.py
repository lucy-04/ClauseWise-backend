import os
from dotenv import load_dotenv

# Load environment variables from .env (from project root)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()  # fallback to root if needed
from fastapi import FastAPI, HTTPException, File, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging


# Load environment variables
load_dotenv()

# Import your services
from services.llm_service import EnhancedLLMService
from services.document_processor import DocumentProcessor  
from services.vector_store import VectorStoreManager
from services.voice_service import VoiceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ClauseWise API",
    description="Legal Document AI Assistant with Voice Support",
    version="1.0.0"
)

# Configure CORS - UPDATED FOR VERCEL FRONTEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:8501", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "https://clausewise-green.vercel.app",  # Your production frontend
        "https://clausewise.vercel.app",  # Alternative domain
        "*"  # For development - remove in production if security is critical
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Language detection function
def detect_language_from_text(text: str) -> str:
    """
    Detect language from text by checking for script patterns
    """
    # Hindi/Devanagari script range (also used for Marathi)
    if any('\u0900' <= char <= '\u097F' for char in text):
        return "hi"
    
    # Bengali script range
    if any('\u0980' <= char <= '\u09FF' for char in text):
        return "bn"
    
    # Telugu script range
    if any('\u0C00' <= char <= '\u0C7F' for char in text):
        return "te"
    
    # Tamil script range
    if any('\u0B80' <= char <= '\u0BFF' for char in text):
        return "ta"
    
    # Gujarati script range
    if any('\u0A80' <= char <= '\u0AFF' for char in text):
        return "gu"
    
    # Kannada script range
    if any('\u0C80' <= char <= '\u0CFF' for char in text):
        return "kn"
    
    # Malayalam script range
    if any('\u0D00' <= char <= '\u0D7F' for char in text):
        return "ml"
    
    # Punjabi/Gurmukhi script range
    if any('\u0A00' <= char <= '\u0A7F' for char in text):
        return "pa"
    
    # Odia script range
    if any('\u0B00' <= char <= '\u0B7F' for char in text):
        return "or"
    
    # Urdu/Arabic script range
    if any('\u0600' <= char <= '\u06FF' for char in text) or any('\uFB50' <= char <= '\uFDFF' for char in text):
        return "ur"
    
    # Assamese (uses Bengali script with some additions)
    if any('\u0980' <= char <= '\u09FF' for char in text):
        # Check for Assamese-specific characters
        if any(char in '\u09F0\u09F1' for char in text):
            return "as"
        return "bn"  # Default to Bengali if no Assamese-specific chars
    
    # Default to English
    return "en"

# UPDATED BACKEND SECRET MIDDLEWARE FOR FREE TIER
@app.middleware("http")
async def verify_secret(request: Request, call_next):
    # Allow health checks and OPTIONS requests
    if request.url.path == "/health" or request.method == "OPTIONS":
        return await call_next(request)
    
    # Get backend secret from environment
    backend_secret = os.getenv("BACKEND_SECRET")
    
    # If backend secret is configured, validate it
    if backend_secret and backend_secret != "optional_for_free_tier":
        secret = request.headers.get("x-backend-secret")
        if secret != backend_secret:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Invalid or missing backend secret"}
            )
    
    return await call_next(request)

# Initialize services
try:
    logger.info("Initializing services...")
    
    # Initialize LLM service
    llm_service = EnhancedLLMService()
    logger.info("✅ LLM Service initialized")
    
    # Initialize document processor
    document_processor = DocumentProcessor()
    logger.info("✅ Document Processor initialized")
    
    # Initialize vector store manager
    vector_store_manager = VectorStoreManager()
    logger.info("✅ Vector Store Manager initialized")
    
    # Initialize voice service
    voice_service = VoiceService()
    logger.info("✅ Voice Service initialized")
    
    logger.info("🚀 All services initialized successfully!")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize services: {e}")
    raise

# Pydantic models - UPDATED WITH LANGUAGE SUPPORT
class SimplificationRequest(BaseModel):
    document_id: str
    language: str = "en"  # ADDED LANGUAGE SUPPORT

class RiskAssessmentRequest(BaseModel):
    document_id: str
    language: str = "en"  # ADDED LANGUAGE SUPPORT

class ChatRequest(BaseModel):
    document_id: str
    question: str
    language: str = "en"
    use_document_language: bool = False  # Changed default to False for better auto-detection

class TranslationRequest(BaseModel):
    text: str
    target_language: str = "hi"

class VoiceTranscriptionRequest(BaseModel):
    audio_format: str = "webm"
    language: str = "en"

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    voice_settings: Optional[Dict[str, Any]] = None
    language: str = "en"

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        legal_kb_info = llm_service.get_legal_kb_info()
        return {
            "status": "healthy",
            "message": "ClauseWise API is running",
            "services": {
                "document_processor": "healthy",
                "vector_store": "healthy", 
                "llm_service": "healthy",
                "legal_kb": legal_kb_info.get("status", "unknown"),
                "voice_service": "healthy" if voice_service.is_tts_available() else "limited"
            },
            "legal_kb_documents": legal_kb_info.get("count", 0),
            "voice_available": {
                "tts": voice_service.is_tts_available(),
                "stt": voice_service.is_stt_available()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Document upload endpoint
@app.post("/upload-document", tags=["Document"])
async def upload_document(
    file: UploadFile = File(...),
    language: str = "en"
):
    """Upload and process a PDF document with language preference"""
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Validate file size (10MB limit)
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > 10:
        raise HTTPException(status_code=400, detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 10MB.")
    
    logger.info(f"Processing uploaded file: {file.filename} ({file_size_mb:.1f}MB) in {language}")
    
    try:
        # Process the document
        doc_id, chunks = document_processor.process_uploaded_file(file_content, file.filename)

        if not doc_id or not chunks:
            logger.error(f"Failed to process {file.filename} - no content extracted")
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. The file might be corrupted, password-protected, or contain only images.")
        
        logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks created")
        
        # Create vector store
        vector_store_manager.create_vector_store(doc_id, chunks)
        
        # Store document language preference in metadata
        doc_info = document_processor.get_document_info(chunks)
        doc_info["language"] = language
        
        # Store language in vector store metadata if possible
        try:
            collection = vector_store_manager.get_vector_store(doc_id)
            if collection:
                # Update collection metadata with language info
                metadata = collection.metadata or {}
                metadata["language"] = language
                collection.modify(metadata=metadata)
        except Exception as e:
            logger.warning(f"Could not store language metadata: {e}")
        
        return {
            "message": "Document processed successfully",
            "document_id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "file_size_mb": round(file_size_mb, 2),
            "document_info": doc_info,
            "language": language,
            "voice_support": {
                "stt_supported": voice_service.is_language_supported(language, "stt"),
                "tts_supported": voice_service.is_language_supported(language, "tts")
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# Explicit OPTIONS handler for upload endpoint
@app.options("/upload-document")
async def upload_document_options():
    """Handle OPTIONS request for upload endpoint"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Get documents endpoint
@app.get("/documents", tags=["Document"])
async def get_documents():
    """Get list of uploaded documents"""
    try:
        document_ids = vector_store_manager.list_documents()
        documents = []
        
        for doc_id in document_ids:
            collection_info = vector_store_manager.get_collection_info(doc_id)
            if collection_info:
                documents.append({
                    "document_id": doc_id,
                    "filename": f"Document_{doc_id[:8]}",  # Placeholder filename
                    "upload_date": None,  # Could be added to metadata
                    "chunks": collection_info.get("count", 0)
                })
        
        return {
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

# Delete document endpoint
@app.delete("/documents/{document_id}", tags=["Document"])
async def delete_document(document_id: str):
    """Delete a document and its vector store"""
    try:
        success = vector_store_manager.delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# Get document language info
@app.get("/documents/{document_id}/language", tags=["Document"])
async def get_document_language(document_id: str):
    """Get language information for a specific document"""
    try:
        collection_info = vector_store_manager.get_collection_info(document_id)
        if not collection_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        language = collection_info.get("metadata", {}).get("language", "en")
        
        return {
            "document_id": document_id,
            "language": language,
            "language_name": voice_service.get_supported_languages().get(language, {}).get("name", "Unknown"),
            "voice_support": {
                "stt_supported": voice_service.is_language_supported(language, "stt"),
                "tts_supported": voice_service.is_language_supported(language, "tts")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# UPDATED Simplify document endpoint with language auto-detection
@app.post("/simplify", tags=["Analysis"])
async def simplify_document_endpoint(request: SimplificationRequest):
    """Simplify a legal document into plain language with language support"""
    try:
        # Auto-detect language if not specified
        if request.language == "en":
            # Try to get from document metadata
            collection_info = vector_store_manager.get_collection_info(request.document_id)
            stored_language = collection_info.get("metadata", {}).get("language", "en")
            if stored_language != "en":
                request.language = stored_language
        
        logger.info(f"Simplifying document {request.document_id} in {request.language}")
        
        # Get document chunks
        chunks = vector_store_manager.get_all_chunks(request.document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found or has no content")
        
        # Simplify using LLM with language parameter
        simplified_text = await llm_service.simplify_document(chunks, request.language)
        
        return {
            "simplified_text": simplified_text,
            "document_id": request.document_id,
            "language": request.language,
            "chunks_processed": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simplifying document {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")

# UPDATED Risk assessment endpoint with language support
@app.post("/assess-risk", tags=["Analysis"])
async def assess_risk_endpoint(request: RiskAssessmentRequest):
    """Assess legal risks in a document with language support"""
    try:
        # Auto-detect language if not specified
        if request.language == "en":
            # Try to get from document metadata
            collection_info = vector_store_manager.get_collection_info(request.document_id)
            stored_language = collection_info.get("metadata", {}).get("language", "en")
            if stored_language != "en":
                request.language = stored_language
        
        logger.info(f"Assessing risks for document {request.document_id} in {request.language}")
        
        # Get document chunks
        chunks = vector_store_manager.get_all_chunks(request.document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found or has no content")
        
        # Assess risks using LLM with language support
        risk_assessment = await llm_service.assess_risks(chunks, request.language)
        
        return {
            "risk_assessment": risk_assessment,
            "document_id": request.document_id,
            "language": request.language,
            "chunks_processed": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assessing risks for document {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# UPDATED Chat endpoint with auto language detection
@app.post("/chat", tags=["Interaction"])
async def chat_with_document_endpoint(request: ChatRequest):
    """Chat with a document using AI with language-aware voice support"""
    try:
        logger.info(f"Chat request for document {request.document_id}: {request.question[:50]}...")
        logger.info(f"Request language parameter: {request.language}")
        
        # Get vector store
        vector_store = vector_store_manager.get_vector_store(request.document_id)
        if not vector_store:
            raise HTTPException(status_code=404, detail="Document vector store not found.")

        # AUTO-DETECT LANGUAGE FROM QUESTION
        detected_language = detect_language_from_text(request.question)
        logger.info(f"Detected language from question text: {detected_language}")
        
        # Priority: detected language > request language > document language
        document_language = detected_language
        
        # If no language detected from text, use request language
        if detected_language == "en" and request.language != "en":
            document_language = request.language
            logger.info(f"Using request language: {document_language}")
        
        # If still English and use_document_language is True, try document metadata
        if document_language == "en" and request.use_document_language:
            try:
                collection_info = vector_store_manager.get_collection_info(request.document_id)
                stored_language = collection_info.get("metadata", {}).get("language")
                if stored_language and stored_language != "en":
                    document_language = stored_language
                    logger.info(f"Using stored document language: {document_language}")
            except Exception as e:
                logger.warning(f"Could not retrieve document language: {e}")

        logger.info(f"Final language for response: {document_language}")

        # Get answer from LLM
        answer, sources = await llm_service.chat_with_document(
            document_vector_store=vector_store,
            question=request.question,
            language=document_language
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "question": request.question,
            "language": document_language,
            "document_id": request.document_id,
            "detected_language": detected_language,
            "voice_support": {
                "can_speak_response": voice_service.is_language_supported(document_language, "tts"),
                "can_transcribe_input": voice_service.is_language_supported(document_language, "stt")
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in chat for document {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Translation endpoint
@app.post("/translate", tags=["Language"])
async def translate_text_endpoint(request: TranslationRequest):
    """Translate text to target language"""
    try:
        logger.info(f"Translating {len(request.text)} characters to {request.target_language}")
        
        translated_text = await llm_service.translate_text(request.text, request.target_language)
        
        return {
            "translated_text": translated_text,
            "target_language": request.target_language,
            "original_length": len(request.text),
            "translated_length": len(translated_text)
        }
        
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Voice transcription endpoint
@app.post("/voice/transcribe", tags=["Voice"])
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = "en"
):
    """Convert speech to text using ElevenLabs STT with Indian language support"""
    try:
        if not voice_service.is_stt_available():
            raise HTTPException(
                status_code=503,
                detail="Speech-to-text service unavailable. Please check ElevenLabs API key."
            )
        
        # Check language support
        if not voice_service.is_language_supported(language, "stt"):
            raise HTTPException(
                status_code=400,
                detail=f"Speech-to-text not supported for language: {language}"
            )
        
        # Validate file size (3GB limit for ElevenLabs)
        file_content = await audio_file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > 3000:  # 3GB limit
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large ({file_size_mb:.1f}MB). Maximum size is 3GB."
            )
        
        logger.info(f"Transcribing audio in {language}: {audio_file.filename} ({file_size_mb:.1f}MB)")
        
        # Determine audio format from filename
        audio_format = "webm"
        if audio_file.filename:
            ext = audio_file.filename.split('.')[-1].lower()
            if ext in ['mp3', 'wav', 'm4a', 'ogg', 'aac', 'flac', 'mp4']:
                audio_format = ext
        
        # Transcribe audio
        transcribed_text = await voice_service.speech_to_text(file_content, audio_format, language)
        
        if not transcribed_text:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        
        return {
            "transcribed_text": transcribed_text,
            "language": language,
            "audio_format": audio_format,
            "file_size_mb": round(file_size_mb, 2),
            "text_length": len(transcribed_text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# Text-to-speech endpoint
@app.post("/voice/speak", tags=["Voice"])
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using ElevenLabs with Indian language support"""
    try:
        if not voice_service.is_tts_available():
            raise HTTPException(
                status_code=503,
                detail="Text-to-speech service unavailable. Please check ElevenLabs API key."
            )
        
        # Check language support
        if not voice_service.is_language_supported(request.language, "tts"):
            raise HTTPException(
                status_code=400,
                detail=f"Text-to-speech not supported for language: {request.language}"
            )
        
        logger.info(f"TTS request in {request.language}: {len(request.text)} characters")
        
        # Validate text
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Text too long. Maximum 5000 characters allowed."
            )
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate speech
        audio_base64 = await voice_service.text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            voice_settings=request.voice_settings,
            language=request.language
        )
        
        if not audio_base64:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        return {
            "audio_base64": audio_base64,
            "voice_id": request.voice_id or voice_service.default_voice_id,
            "language": request.language,
            "text_length": len(request.text),
            "format": "mp3"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TTS: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

# Get voice service info
@app.get("/voice/info", tags=["Voice"])
async def get_voice_service_info():
    """Get voice service information with Indian language support"""
    try:
        return voice_service.get_service_info()
    except Exception as e:
        logger.error(f"Error getting voice service info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get available voices
@app.get("/voice/voices", tags=["Voice"])
async def get_available_voices():
    """Get available TTS voices"""
    try:
        return await voice_service.get_available_voices()
    except Exception as e:
        logger.error(f"Error getting available voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get supported Indian languages
@app.get("/voice/languages", tags=["Voice"])
async def get_voice_supported_languages():
    """Get supported Indian languages for voice services"""
    try:
        languages = voice_service.get_supported_languages()
        return {
            "supported_languages": languages,
            "tts_languages": [
                code for code, info in languages.items() 
                if info.get("tts_supported", False)
            ],
            "stt_languages": [
                code for code, info in languages.items() 
                if info.get("stt_supported", False)
            ]
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get supported languages
@app.get("/supported-languages", tags=["Language"])
async def get_supported_languages():
    """Get supported languages for the application"""
    try:
        return {
            "en": "English",
            "hi": "Hindi", 
            "bn": "Bengali",
            "te": "Telugu",
            "mr": "Marathi",
            "ta": "Tamil",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "or": "Odia",
            "as": "Assamese",
            "ur": "Urdu"
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get legal knowledge base info
@app.get("/legal-kb-info", tags=["System"])
async def get_legal_kb_info():
    """Get legal knowledge base information"""
    try:
        return {
            "legal_kb_info": llm_service.get_legal_kb_info(),
            "document_count": llm_service.get_legal_kb_info().get("count", 0),
            "datasets_path": llm_service.legal_datasets_path if hasattr(llm_service, 'legal_datasets_path') else None
        }
    except Exception as e:
        logger.error(f"Error getting legal KB info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ClauseWise Legal Document AI Assistant API",
        "version": "1.0.0",
        "features": [
            "PDF document processing",
            "Legal document simplification", 
            "Risk assessment",
            "Multi-language chat with auto-detection",
            "Voice input/output with ElevenLabs",
            "13+ Indian languages support"
        ],
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable, default to 8080
    port = int(os.getenv("PORT", 1000))
    
    print(f"🚀 Starting ClauseWise API on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )