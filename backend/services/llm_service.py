import os
import json
import pandas as pd
import time
import asyncio
import glob
import re
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

load_dotenv()

class EnhancedLLMService:
    def __init__(self):
        # Initialize Gemini with correct model name
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Try different model names until one works
        model_names = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-pro",
            "gemini-1.0-pro"
        ]
        
        self.gemini_llm = None
        for model_name in model_names:
            try:
                print(f"🔄 Trying Gemini model: {model_name}")
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.3,
                    max_tokens=2048,
                    top_k=40,
                    top_p=0.95
                )
                print(f"✅ Successfully initialized Gemini with: {model_name}")
                break
            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                continue
        
        if not self.gemini_llm:
            raise ValueError("Could not initialize any Gemini model. Check your API key.")
        
        # Rate limiting
        self.rate_limit = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "15"))
        self.last_request_time = 0
        self.min_request_interval = 60 / self.rate_limit
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Legal knowledge base (simplified)
        self.legal_kb_collection = None
        self.chroma_client = None
        self.legal_datasets_path = os.getenv("LEGAL_DATASETS_PATH", "./backend/legal_datasets")
        self.enable_legal_kb = os.getenv("ENABLE_LEGAL_KB", "true").lower() == "true"
        
        if self.enable_legal_kb:
            self._initialize_legal_knowledge_base()
        
        print("✅ Enhanced LLM Service initialized successfully")
    
    def _clean_markdown_output(self, text: str) -> str:
        """Clean and format markdown output for better display"""
        # Remove excessive asterisks
        text = re.sub(r'\*{3,}', '**', text)
        
        # Convert markdown bold to simple text with emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        # Clean up bullet points - ensure proper spacing
        text = re.sub(r'^(\s*)-\s+', r'• ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*)\*\s+', r'• ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*)(\d+)\.\s+', r'\1\2. ', text, flags=re.MULTILINE)
        
        # Clean up headers - remove markdown headers but keep the emphasis
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\n\1\n', text, flags=re.MULTILINE)
        
        # Clean up emojis if present (keep them as they add visual appeal)
        # But ensure they're properly spaced
        text = re.sub(r'([🔴🟡🟢⚠️💡✅❌📄📝🔍⚡️⏰📊])\s*', r'\1 ', text)
        
        # Ensure proper paragraph spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up any remaining markdown artifacts
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove inline code formatting
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)  # Remove blockquotes
        
        return text.strip()
    
    async def _rate_limit_check(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            print(f"⏰ Rate limiting: waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _initialize_legal_knowledge_base(self):
        """Initialize ChromaDB-based legal knowledge base"""
        try:
            chroma_path = os.path.join(os.getenv("CHROMA_DB_PATH", "./chroma_db"), "legal_kb")
            os.makedirs(chroma_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.legal_kb_collection = self.chroma_client.get_or_create_collection(
                name="legal_knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Create some sample legal knowledge if empty
            if self.legal_kb_collection.count() == 0:
                print("📚 Creating sample legal knowledge base...")
                sample_legal_data = [
                    {
                        "text": "A contract is a legally binding agreement between two or more parties. For a contract to be valid, it must have offer, acceptance, consideration, and legal capacity.",
                        "type": "definition",
                        "topic": "contract_law"
                    },
                    {
                        "text": "The Indian Contract Act, 1872 governs contracts in India. Section 10 states that all agreements are contracts if made by free consent of parties competent to contract.",
                        "type": "law",
                        "topic": "indian_contract_act"
                    },
                    {
                        "text": "A breach of contract occurs when one party fails to fulfill obligations. Remedies include damages, specific performance, or contract rescission.",
                        "type": "legal_concept",
                        "topic": "breach_of_contract"
                    }
                ]
                
                texts = [item["text"] for item in sample_legal_data]
                metadatas = [{"source": "sample_data", **{k: v for k, v in item.items() if k != "text"}} for item in sample_legal_data]
                ids = [f"sample_{i}" for i in range(len(sample_legal_data))]
                embeddings = self.embeddings.embed_documents(texts)
                
                self.legal_kb_collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Created sample legal knowledge base with {len(sample_legal_data)} documents")
            else:
                print(f"✅ Loaded existing legal knowledge base with {self.legal_kb_collection.count()} documents")

        except Exception as e:
            print(f"❌ Error initializing legal knowledge base: {e}")
            self.legal_kb_collection = None
    
    async def simplify_document(self, documents: List[Document], language: str = "en") -> str:
        """Simplify legal document using Gemini with language support"""
        try:
            await self._rate_limit_check()
            
            combined_text = ""
            for doc in documents:
                if len(combined_text) + len(doc.page_content) > 4000: 
                    break
                combined_text += doc.page_content + "\n\n"
            
            # Language-specific instructions
            lang_instruction = ""
            if language != "en":
                language_names = {
                    "hi": "Hindi (हिंदी)",
                    "bn": "Bengali (বাংলা)",
                    "te": "Telugu (తెలుగు)",
                    "mr": "Marathi (मराठी)",
                    "ta": "Tamil (தமிழ்)",
                    "gu": "Gujarati (ગુજરાતી)",
                    "kn": "Kannada (ಕನ್ನಡ)",
                    "ml": "Malayalam (മലയാളം)",
                    "or": "Odia (ଓଡ଼ିଆ)",
                    "pa": "Punjabi (ਪੰਜਾਬੀ)",
                    "ur": "Urdu (اردو)",
                    "as": "Assamese (অসমীয়া)"
                }
                lang_name = language_names.get(language, "Hindi")
                lang_instruction = f"\n\nIMPORTANT: Provide your ENTIRE explanation in {lang_name} language only. Do not use English."
            
            prompt = f"""You are a legal expert helping ordinary people understand legal documents. 

DOCUMENT TO SIMPLIFY:
{combined_text}

Please provide a clear, simple explanation in PLAIN TEXT format (no markdown, no special characters):

MAIN PURPOSE:
What is this document for? (in 2-3 sentences)

KEY POINTS:
List the most important things to know (3-5 points)

YOUR RIGHTS:
What rights do you have under this document?

YOUR OBLIGATIONS:
What must you do or pay?

IMPORTANT DATES:
Any deadlines or time limits mentioned?

WHAT TO WATCH OUT FOR:
Any concerning clauses or conditions?

Use simple language that a person with basic education can understand. Explain legal terms in plain language.
DO NOT use markdown formatting, asterisks, or special characters. Use simple numbered lists and clear paragraphs.
{lang_instruction}"""
            
            result = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            cleaned_output = self._clean_markdown_output(result.content)
            return cleaned_output
            
        except Exception as e:
            raise Exception(f"Error simplifying document: {str(e)}")
    
    async def assess_risks(self, documents: List[Document], language: str = "en") -> str:
        """Assess legal risks in the document with language support"""
        try:
            await self._rate_limit_check()
            
            combined_text = ""
            for doc in documents:
                if len(combined_text) + len(doc.page_content) > 3500: 
                    break
                combined_text += doc.page_content + "\n\n"
            
            # Language-specific instructions
            lang_instruction = ""
            if language != "en":
                language_names = {
                    "hi": "Hindi (हिंदी)",
                    "bn": "Bengali (বাংলা)",
                    "te": "Telugu (తెలుగు)",
                    "mr": "Marathi (मराठी)",
                    "ta": "Tamil (தமிழ்)",
                    "gu": "Gujarati (ગુજરાતી)",
                    "kn": "Kannada (ಕನ್ನಡ)",
                    "ml": "Malayalam (മലയാളം)",
                    "or": "Odia (ଓଡ଼ିଆ)",
                    "pa": "Punjabi (ਪੰਜਾਬੀ)",
                    "ur": "Urdu (اردو)",
                    "as": "Assamese (অসমীয়া)"
                }
                lang_name = language_names.get(language, "Hindi")
                lang_instruction = f"\n\nCRITICAL: Provide your ENTIRE risk assessment in {lang_name} language only. Do not use English."
            
            prompt = f"""As a legal risk analyst, evaluate the following document for potential risks:

DOCUMENT:
{combined_text}

Provide a comprehensive risk assessment in PLAIN TEXT format:

HIGH RISK AREAS:
List any high-risk clauses or terms
Include financial liabilities or penalties

MEDIUM RISK AREAS:
List potentially problematic terms
Include unclear obligations

OVERALL RISK LEVEL: (Low/Medium/High)

IMMEDIATE ACTION REQUIRED:
List any urgent deadlines
List critical decisions needed

RECOMMENDATIONS:
Provide suggestions to reduce risks
Indicate when to consult a lawyer

Be specific and practical in your assessment. Use plain language without markdown formatting.
{lang_instruction}"""
            
            result = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            cleaned_output = self._clean_markdown_output(result.content)
            return cleaned_output
            
        except Exception as e:
            raise Exception(f"Error assessing risks: {str(e)}")
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target Indian language"""
        try:
            await self._rate_limit_check()
            
            language_names = {
                "hi": "Hindi (हिंदी)", 
                "bn": "Bengali (বাংলা)", 
                "te": "Telugu (తెలుగు)", 
                "mr": "Marathi (मराठी)", 
                "ta": "Tamil (தமிழ்)", 
                "gu": "Gujarati (ગુજરાતી)", 
                "kn": "Kannada (ಕನ್ನಡ)", 
                "ml": "Malayalam (മലയാളം)", 
                "or": "Odia (ଓଡ଼ିଆ)", 
                "pa": "Punjabi (ਪੰਜਾਬੀ)", 
                "ur": "Urdu (اردو)", 
                "as": "Assamese (অসমীয়া)"
            }
            
            target_lang_name = language_names.get(target_language, "Hindi")
            text_to_translate = text[:3000]
            
            prompt = f"""Translate the following legal document explanation to {target_lang_name}. 
Maintain clarity and simplicity. Do not use markdown formatting.

{text_to_translate}

Translation in {target_lang_name}:"""
            
            result = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            return result.content
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation unavailable. Original text: {text[:300]}..."
    
    async def chat_with_document(self, document_vector_store, question: str, language: str = "en") -> Tuple[str, List[str]]:
        """Enhanced chat using document and legal knowledge base with proper language support"""
        try:
            await self._rate_limit_check()
            
            # Get document context
            doc_context = ""
            if hasattr(document_vector_store, 'query'):
                try:
                    query_embedding = self.embeddings.embed_query(question)
                    doc_results = document_vector_store.query(query_embeddings=[query_embedding], n_results=3)
                    if doc_results['documents'] and doc_results['documents'][0]:
                        doc_context = "\n".join([doc[:400] for doc in doc_results['documents'][0]])
                except Exception as e:
                    print(f"Document query error: {e}")
            
            # Get legal context
            legal_context = ""
            if self.legal_kb_collection:
                try:
                    legal_query_embedding = self.embeddings.embed_query(question)
                    legal_results = self.legal_kb_collection.query(query_embeddings=[legal_query_embedding], n_results=2)
                    if legal_results['documents'] and legal_results['documents'][0]:
                        legal_context = "\n".join([doc[:300] for doc in legal_results['documents'][0]])
                except Exception as e:
                    print(f"Legal KB query error: {e}")
            
            # Language instruction - CRITICAL FIX
            lang_instruction = ""
            language_names = {
                "hi": "Hindi (हिंदी)",
                "bn": "Bengali (বাংলা)",
                "te": "Telugu (తెలుగు)",
                "mr": "Marathi (मराठी)",
                "ta": "Tamil (தமிழ்)",
                "gu": "Gujarati (ગુજરાતી)",
                "kn": "Kannada (ಕನ್ನಡ)",
                "ml": "Malayalam (മലയാളം)",
                "or": "Odia (ଓଡ଼ିଆ)",
                "pa": "Punjabi (ਪੰਜਾਬੀ)",
                "ur": "Urdu (اردو)",
                "as": "Assamese (অসমীয়া)"
            }
            
            if language != "en":
                lang_name = language_names.get(language, "Hindi")
                # CRITICAL: More emphatic language instruction
                lang_instruction = f"""
IMPORTANT: You MUST respond ENTIRELY in {lang_name} language.
Your ENTIRE answer should be in {lang_name}, NOT in English.
कृपया {lang_name} में उत्तर दें। Do not use English in your response.
"""
            
            # Modified prompt with stronger language enforcement
            prompt = f"""You are a helpful legal assistant. Answer the user's question based on the provided context.

DOCUMENT CONTEXT:
{doc_context}

LEGAL KNOWLEDGE:
{legal_context}

USER QUESTION: {question}

{lang_instruction}

CRITICAL INSTRUCTIONS:
1. Provide a clear, helpful answer based on the context
2. Use simple paragraphs and numbered lists if needed
3. Do not use asterisks, markdown, or special formatting
{f"4. YOUR ENTIRE RESPONSE MUST BE IN {language_names.get(language, language).upper()} LANGUAGE ONLY!" if language != "en" else ""}

{"Reply in " + language_names.get(language, language) + " language:" if language != "en" else "Answer:"}"""
            
            response = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            cleaned_output = self._clean_markdown_output(response.content)
            
            sources = []
            if doc_context: sources.append("Your uploaded document")
            if legal_context: sources.append("Legal knowledge base")
            
            return cleaned_output, sources
            
        except Exception as e:
            print(f"Chat error: {e}")
            # Return error message in requested language
            error_messages = {
                "hi": "क्षमा करें, मैं इस समय आपके प्रश्न को संसाधित नहीं कर सका।",
                "bn": "দুঃখিত, আমি এই মুহূর্তে আপনার প্রশ্ন প্রক্রিয়া করতে পারছি না।",
                "te": "క్షమించండి, నేను ప్రస్తుతం మీ ప్రశ్నను ప్రాసెస్ చేయలేకపోతున్నాను.",
                "mr": "माफ करा, मी सध्या तुमचा प्रश्न प्रक्रिया करू शकत नाही.",
                "ta": "மன்னிக்கவும், உங்கள் கேள்வியை இப்போது செயலாக்க முடியவில்லை.",
                "gu": "માફ કરશો, હું હાલમાં તમારા પ્રશ્નની પ્રક્રિયા કરી શકતો નથી.",
                "en": "I apologize, but I couldn't process your question at the moment."
            }
            return error_messages.get(language, error_messages["en"]), []
    
    def get_legal_kb_info(self) -> Dict[str, Any]:
        """Get information about the legal knowledge base"""
        if not self.legal_kb_collection:
            return {"status": "not_initialized", "count": 0}
        
        try:
            return {
                "status": "active",
                "count": self.legal_kb_collection.count(),
                "datasets_path": self.legal_datasets_path
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}