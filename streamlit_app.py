import streamlit as st
import requests
import os
import time
from typing import Dict, Any, List
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="ClauseWise - Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
.stContainer > div {
    padding-top: 1rem;
}
.status-good {
    color: #00ff00;
}
.status-bad {
    color: #ff0000;
}
.chat-message {
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
}
.voice-controls {
    display: flex;
    gap: 10px;
    align-items: center;
    padding: 10px;
    background-color: #f0f2f6;
    border-radius: 10px;
    margin: 10px 0;
}
.mic-button {
    background-color: #ff4b4b;
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    cursor: pointer;
    font-size: 20px;
}
.mic-button:hover {
    background-color: #ff6b6b;
}
.mic-button.recording {
    background-color: #ff0000;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "document_id": None,
        "document_name": "",
        "simplified_text": None,
        "risk_assessment": None,
        "translated_text": None,
        "chat_history": [],
        "sources": [],
        "processing_complete": False,
        "last_backend_check": 0,
        "backend_status": False,
        "voice_input": "",
        "last_speech_result": ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- API Communication Functions ---
@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_backend_health() -> bool:
    """Check if backend is running with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    except Exception as e:
        st.error(f"Backend connection error: {e}")
        return False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_supported_languages() -> Dict[str, str]:
    """Get supported languages with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/supported-languages", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Could not fetch languages: {e}")
    # Fallback languages
    return {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali",
        "te": "Telugu",
        "mr": "Marathi",
        "ta": "Tamil",
        "gu": "Gujarati"
    }

def upload_document(file) -> tuple[bool, Dict[str, Any]]:
    """Upload document to backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(
            f"{API_BASE_URL}/upload-document",
            files=files,
            timeout=120  # Increased timeout for large files
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            return False, {"error": error_detail}
    except requests.exceptions.Timeout:
        return False, {"error": "Upload timeout - file may be too large"}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Could not connect to backend server"}
    except Exception as e:
        return False, {"error": f"Upload failed: {str(e)}"}

def simplify_document(doc_id: str) -> tuple[bool, Dict[str, Any]]:
    """Simplify document"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/simplify",
            json={"document_id": doc_id},
            timeout=120
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            return False, {"error": error_detail}
    except requests.exceptions.Timeout:
        return False, {"error": "Simplification timeout - document may be too long"}
    except Exception as e:
        return False, {"error": f"Simplification failed: {str(e)}"}

def assess_risk(doc_id: str) -> tuple[bool, Dict[str, Any]]:
    """Assess document risks"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/assess-risk",
            json={"document_id": doc_id},
            timeout=120
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            return False, {"error": error_detail}
    except requests.exceptions.Timeout:
        return False, {"error": "Risk assessment timeout"}
    except Exception as e:
        return False, {"error": f"Risk assessment failed: {str(e)}"}

def chat_with_document(doc_id: str, question: str, lang: str) -> tuple[bool, Dict[str, Any]]:
    """Chat with document"""
    try:
        payload = {"document_id": doc_id, "question": question, "language": lang}
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            return False, {"error": error_detail}
    except requests.exceptions.Timeout:
        return False, {"error": "Chat timeout - question may be too complex"}
    except Exception as e:
        return False, {"error": f"Chat failed: {str(e)}"}

def translate_text(text: str, lang: str) -> tuple[bool, Dict[str, Any]]:
    """Translate text"""
    try:
        payload = {"text": text, "target_language": lang}
        response = requests.post(
            f"{API_BASE_URL}/translate",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", f"HTTP {response.status_code}")
            return False, {"error": error_detail}
    except Exception as e:
        return False, {"error": f"Translation failed: {str(e)}"}

# --- Voice Helper Functions ---
def create_tts_button(text: str, button_text: str = "🔊 Speak", button_id: str = "ttsBtn"):
    """Create a text-to-speech button for any text"""
    # Clean the text for JavaScript
    clean_text = text.replace('`', '').replace('"', '\\"').replace('\n', ' ').replace('\r', '')
    tts_html = f"""
    <div style="margin: 10px 0;">
        <button onclick="{button_id}Speak()" style="
            background: #00c851; color: white; border: none; padding: 8px 16px;
            border-radius: 4px; cursor: pointer; margin-right: 10px;">
            {button_text}
        </button>
        <button onclick="stopAllSpeech()" style="
            background: #666; color: white; border: none; padding: 8px 16px;
            border-radius: 4px; cursor: pointer;">
            ⏹️ Stop
        </button>
    </div>

    <script>
    function {button_id}Speak() {{
        const text = `{clean_text}`;
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 1;
            // Try to get a good English voice
            const voices = window.speechSynthesis.getVoices();
            const englishVoice = voices.find(voice => voice.lang.startsWith('en'));
            if (englishVoice) {{
                utterance.voice = englishVoice;
            }}
            window.speechSynthesis.speak(utterance);
        }} else {{
            alert('Text-to-speech not supported in this browser');
        }}
    }}
    function stopAllSpeech() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
        }}
    }}
    </script>
    """
    return tts_html

def create_voice_input():
    """Create voice input component"""
    voice_html = """
    <div style="padding: 20px; background: #f0f2f6; border-radius: 10px; margin: 10px 0;">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <button id="micBtn" onclick="toggleRecording()" style="
                background: #ff4b4b; color: white; border: none; border-radius: 50%;
                width: 60px; height: 60px; font-size: 24px; cursor: pointer;
                transition: all 0.3s ease;">
                🎤
            </button>
            <div id="status">Click to start recording</div>
        </div>
        <div id="transcript" style="
            background: white; padding: 15px; border-radius: 5px;
            min-height: 50px; border: 2px solid #ddd;"></div>
        <div style="margin-top: 10px;">
            <button onclick="clearTranscript()" style="
                background: #666; color: white; border: none; padding: 8px 16px;
                border-radius: 4px; margin-right: 10px; cursor: pointer;">Clear</button>
            <button onclick="useTranscript()" style="
                background: #00c851; color: white; border: none; padding: 8px 16px;
                border-radius: 4px; cursor: pointer;">Use This Text</button>
        </div>
    </div>

    <script>
        let recognition;
        let isRecording = false;
        let finalTranscript = '';

        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'en-US';

                recognition.onresult = function(event) {
                    let interim = '';
                    let final = '';
                    for (let i = 0; i < event.results.length; i++) {
                        if (event.results[i].isFinal) {
                            final += event.results[i][0].transcript;
                        } else {
                            interim += event.results[i][0].transcript;
                        }
                    }
                    document.getElementById('transcript').innerHTML =
                        '<strong>Final:</strong> ' + final + '<br><em>Interim:</em> ' + interim;
                    if (final) {
                        finalTranscript = final;
                    }
                };

                recognition.onstart = function() {
                    isRecording = true;
                    document.getElementById('micBtn').style.background = '#ff0000';
                    document.getElementById('micBtn').style.animation = 'pulse 1s infinite';
                    document.getElementById('status').textContent = 'Recording... Click to stop';
                };

                recognition.onend = function() {
                    isRecording = false;
                    document.getElementById('micBtn').style.background = '#ff4b4b';
                    document.getElementById('micBtn').style.animation = 'none';
                    document.getElementById('status').textContent = 'Click to start recording';
                };

                recognition.onerror = function(event) {
                    document.getElementById('transcript').innerHTML =
                        '<span style="color: red;">Error: ' + event.error + '</span>';
                    isRecording = false;
                    document.getElementById('micBtn').style.background = '#ff4b4b';
                    document.getElementById('micBtn').style.animation = 'none';
                    document.getElementById('status').textContent = 'Error occurred. Click to try again.';
                };

                return true;
            } else {
                document.getElementById('transcript').innerHTML =
                    '<span style="color: red;">Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.</span>';
                document.getElementById('micBtn').disabled = true;
                return false;
            }
        }

        function toggleRecording() {
            if (!recognition && !initSpeechRecognition()) return;
            if (isRecording) {
                recognition.stop();
            } else {
                finalTranscript = '';
                document.getElementById('transcript').innerHTML = 'Listening...';
                recognition.start();
            }
        }

        function clearTranscript() {
            finalTranscript = '';
            document.getElementById('transcript').innerHTML = '';
        }

        function useTranscript() {
            if (finalTranscript.trim()) {
                // Copy to clipboard as fallback
                navigator.clipboard.writeText(finalTranscript.trim()).then(() => {
                    alert('Voice text copied to clipboard! Paste it in the chat input below and press Enter.');
                }).catch(() => {
                    // If clipboard fails, just show the text
                    alert('Your voice input: "' + finalTranscript.trim() + '"\\n\\nPlease copy this text and paste it in the chat below.');
                });
            } else {
                alert('No speech recorded. Please try again.');
            }
        }

        // Auto-initialize
        setTimeout(initSpeechRecognition, 1000);

        // Add CSS for pulse animation
        const style = document.createElement('style');
        style.textContent = `
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        `;
        document.head.appendChild(style);
    </script>
    """
    return voice_html

# --- UI Components ---
def render_sidebar(language_options: Dict[str, str]):
    """Render sidebar with settings and status"""
    with st.sidebar:
        st.title("⚖️ ClauseWise")
        st.markdown("**Legal Document AI Assistant**")
        # Backend status check
        current_time = time.time()
        if current_time - st.session_state.last_backend_check > 30:  # Check every 30 seconds
            st.session_state.backend_status = check_backend_health()
            st.session_state.last_backend_check = current_time
        if st.session_state.backend_status:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Offline")
            st.warning("Start backend: `python run_backend.py`")
            if st.button("🔄 Retry Connection"):
                st.session_state.last_backend_check = 0  # Force recheck
                st.rerun()

        st.markdown("---")
        # Language selection
        st.header("🌐 Settings")
        selected_language = st.selectbox(
            "Select Language:",
            options=list(language_options.keys()),
            format_func=lambda x: language_options.get(x, x),
            index=0,
            help="Select your preferred language for analysis and chat"
        )
        # Voice settings
        st.subheader("🎤 Voice Features")
        st.info("🎤 **Speech-to-Text**: Click microphone in chat\n🔊 **Text-to-Speech**: Click speak buttons on responses")
        # Rate limiting info
        st.info("💡 **Rate Limits**: Using Gemini Free Tier. Please wait 5-10 seconds between requests.")
        st.markdown("---")
        # Document status
        st.header("📄 Document Status")
        if st.session_state.document_id:
            st.success(f"**Active:** {st.session_state.document_name}")
            if st.button("🗑️ Clear Document"):
                # Reset relevant state
                for key in ["document_id", "document_name", "simplified_text",
                           "risk_assessment", "translated_text", "chat_history"]:
                    st.session_state[key] = None if key in ["document_id", "simplified_text", "risk_assessment", "translated_text"] else [] if key == "chat_history" else ""
                st.rerun()
        else:
            st.warning("No document loaded")
        st.markdown("---")
        # Help section
        st.subheader("📚 Help")
        with st.expander("How to Use"):
            st.markdown("""
            1. **Upload PDF**: Upload a legal document
            2. **Analyze**: Click 'Simplify' or 'Assess Risks'
            3. **Chat**: Ask questions about the document
            4. **Voice**: Use microphone for voice input
            5. **Listen**: Click speak button to hear responses
            """)
        with st.expander("Voice Features"):
            st.markdown("""
            - **🎤 Voice Input**: Click microphone to speak your question
            - **🔊 Text-to-Speech**: Click speak button to hear responses
            - **Browser Support**: Works in Chrome, Edge, Safari
            - **Language**: Currently English only
            """)
        with st.expander("Troubleshooting"):
            st.markdown("""
            - **Upload fails**: Check file size (<10MB) and format (PDF only)
            - **Backend offline**: Run `python run_backend.py`
            - **Voice not working**: Check microphone permissions
            - **No audio**: Check browser compatibility and speakers
            """)
    return selected_language

def render_document_upload():
    """Render document upload section"""
    st.header("📄 Step 1: Upload Legal Document")
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a legal document (contracts, agreements, etc.) - Max 10MB",
            disabled=not st.session_state.backend_status
        )
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 **File**: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            if file_size_mb > 10:
                st.error("❌ File too large! Maximum size is 10MB.")
                return
            col1, col2 = st.columns([1, 3])
            with col1:
                process_button = st.button(
                    "🚀 Process Document",
                    type="primary",
                    disabled=not st.session_state.backend_status
                )
            if process_button:
                with st.spinner("📖 Processing document... This may take 30-60 seconds."):
                    # Add progress bar
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    success, result = upload_document(uploaded_file)
                    progress_bar.progress(75)
                    if success:
                        progress_bar.progress(100)
                        # Reset relevant state for new document
                        st.session_state.document_id = result["document_id"]
                        st.session_state.document_name = result["filename"]
                        st.session_state.simplified_text = None
                        st.session_state.risk_assessment = None
                        st.session_state.translated_text = None
                        st.session_state.chat_history = []
                        st.session_state.processing_complete = True
                        st.success(f"✅ Document '{result['filename']}' processed successfully!")
                        st.info(f"📊 Created {result.get('num_chunks', 0)} text chunks for analysis")
                        time.sleep(1)  # Brief pause before rerun
                        st.rerun()
                    else:
                        progress_bar.progress(0)
                        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

def render_analysis_section(selected_language: str, lang_name: str):
    """Render document analysis section"""
    st.header("🔍 Step 2: Analyze Document")
    if not st.session_state.document_id:
        st.info("👆 Please upload and process a document first.")
        return
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📖 Simplify Document", help="Get a plain-English summary"):
                with st.spinner("🤖 Generating simplified summary..."):
                    success, result = simplify_document(st.session_state.document_id)
                    if success:
                        st.session_state.simplified_text = result["simplified_text"]
                        st.success("✅ Document simplified!")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")

        with col2:
            if st.button("⚠️ Assess Risks", help="Identify potential legal risks"):
                with st.spinner("🔍 Performing risk assessment..."):
                    success, result = assess_risk(st.session_state.document_id)
                    if success:
                        st.session_state.risk_assessment = result["risk_assessment"]
                        st.success("✅ Risk assessment complete!")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
        # Translation option
        if st.session_state.simplified_text and selected_language != "en":
            if st.button(f"🌐 Translate to {lang_name}"):
                with st.spinner(f"🔄 Translating to {lang_name}..."):
                    success, result = translate_text(st.session_state.simplified_text, selected_language)
                    if success:
                        st.session_state.translated_text = result["translated_text"]
                        st.success("✅ Translation complete!")
                    else:
                        st.error(f"❌ Translation Error: {result.get('error')}")

def render_results():
    """Render analysis results with TTS"""
    if not (st.session_state.simplified_text or st.session_state.risk_assessment):
        return
    st.header("📋 Analysis Results")
    with st.container(border=True):
        if st.session_state.simplified_text:
            with st.expander("📖 Simplified Summary", expanded=True):
                st.markdown(st.session_state.simplified_text)
                # Add TTS button
                tts_html = create_tts_button(
                    st.session_state.simplified_text,
                    "🔊 Listen to Summary",
                    "summary"
                )
                components.html(tts_html, height=60)
        if st.session_state.risk_assessment:
            with st.expander("⚠️ Risk Assessment", expanded=True):
                st.markdown(st.session_state.risk_assessment)
                # Add TTS button
                tts_html = create_tts_button(
                    st.session_state.risk_assessment,
                    "🔊 Listen to Risk Assessment",
                    "risk"
                )
                components.html(tts_html, height=60)

        if st.session_state.translated_text:
            with st.expander(f"🌐 Translation", expanded=True):
                st.markdown(st.session_state.translated_text)

def render_chat_section(selected_language: str):
    """Render chat interface with voice input"""
    st.header("💬 Step 3: Chat with Document")
    if not st.session_state.document_id:
        st.info("👆 Please upload and process a document first.")
        return
    with st.container(border=True):
        # Voice input section
        st.subheader("🎤 Voice Input")
        voice_html = create_voice_input()
        components.html(voice_html, height=200)
        # Sample questions
        st.subheader("💡 Try these questions:")
        sample_questions = [
            "What are the main obligations in this document?",
            "What are the payment terms?",
            "Are there any penalties mentioned?",
            "What happens if the contract is breached?",
            "What are the termination conditions?"
        ]
        cols = st.columns(len(sample_questions))
        for i, question in enumerate(sample_questions):
            with cols[i % len(cols)]:
                if st.button(f"❓ {question[:20]}...", key=f"sample_{i}", help=question):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    # Process the question
                    with st.spinner("🤖 Thinking..."):
                        success, result = chat_with_document(st.session_state.document_id, question, selected_language)
                        if success:
                            response_text = result["answer"]
                            sources = result.get("sources", [])
                            if sources:
                                response_text += f"\n\n*📚 Sources: {', '.join(sources)}*"
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        else:
                            error_msg = f"❌ Error: {result.get('error', 'Could not process question.')}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    st.rerun()
        st.markdown("---")
        # Chat history display with TTS
        if st.session_state.chat_history:
            st.subheader("💬 Conversation")
            for i, message in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Add TTS button for assistant responses
                    if message["role"] == "assistant":
                        tts_html = create_tts_button(
                            message["content"],
                            "🔊 Speak",
                            f"chat{i}"
                        )
                        components.html(tts_html, height=50)
        # Chat input
        if prompt := st.chat_input("Ask a question about the document... (or use voice input above)", key="main_chat"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("🤖 Thinking..."):
                success, result = chat_with_document(st.session_state.document_id, prompt, selected_language)
                if success:
                    response_text = result["answer"]
                    sources = result.get("sources", [])
                    if sources:
                        response_text += f"\n\n*📚 Sources: {', '.join(sources)}*"
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                else:
                    error_msg = f"❌ Error: {result.get('error', 'Could not process question.')}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.rerun()

# --- Main Application ---
def main():
    """Main application function"""
    initialize_session_state()
    # Get language options
    language_options = get_supported_languages()
    selected_lang_code = render_sidebar(language_options)
    selected_lang_name = language_options.get(selected_lang_code, "Unknown")
    # Main content area
    st.title("⚖️ ClauseWise - Legal AI Assistant")
    st.markdown("**Simplify legal documents, assess risks, and get answers in plain language.**")
    st.info("🎤 **New Voice Features**: Click microphone to speak questions, click speak buttons to hear responses!")
    if not st.session_state.backend_status:
        st.error("🚫 Backend server is not running. Please start it to use the application.")
        st.code("python run_backend.py", language="bash")
        st.stop()
    # Render main sections
    render_document_upload()
    # Only show analysis and chat if document is uploaded
    if st.session_state.document_id:
        st.markdown("---")
        render_analysis_section(selected_lang_code, selected_lang_name)
        if st.session_state.simplified_text or st.session_state.risk_assessment:
            st.markdown("---")
            render_results()
        st.markdown("---")
        render_chat_section(selected_lang_code)

if __name__ == "__main__":
    main()