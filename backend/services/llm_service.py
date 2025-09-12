import os
import json
import pandas as pd
import time
import asyncio
import glob
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
    
    async def simplify_document(self, documents: List[Document]) -> str:
        """Simplify legal document using Gemini"""
        try:
            await self._rate_limit_check()
            
            combined_text = ""
            for doc in documents:
                if len(combined_text) + len(doc.page_content) > 4000: 
                    break
                combined_text += doc.page_content + "\n\n"
            
            prompt = f"""You are a legal expert helping ordinary people understand legal documents. 

DOCUMENT TO SIMPLIFY:
{combined_text}

Please provide a clear, simple explanation that includes:
1. **MAIN PURPOSE**: What is this document for? (in 2-3 sentences)
2. **KEY POINTS**: What are the most important things to know? (3-5 bullet points)
3. **YOUR RIGHTS**: What rights do you have under this document?
4. **YOUR OBLIGATIONS**: What must you do or pay?
5. **IMPORTANT DATES**: Any deadlines or time limits mentioned?
6. **WHAT TO WATCH OUT FOR**: Any concerning clauses or conditions?

Use simple English that a person with basic education can understand. Explain legal terms in plain language."""
            
            result = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            return result.content
            
        except Exception as e:
            raise Exception(f"Error simplifying document: {str(e)}")
    
    async def assess_risks(self, documents: List[Document]) -> str:
        """Assess legal risks in the document"""
        try:
            await self._rate_limit_check()
            
            combined_text = ""
            for doc in documents:
                if len(combined_text) + len(doc.page_content) > 3500: 
                    break
                combined_text += doc.page_content + "\n\n"
            
            prompt = f"""As a legal risk analyst, evaluate the following document for potential risks:

DOCUMENT:
{combined_text}

Provide a comprehensive risk assessment:
**🔴 HIGH RISK AREAS:**
- List any high-risk clauses or terms
- Financial liabilities or penalties

**🟡 MEDIUM RISK AREAS:**
- Potentially problematic terms
- Unclear obligations

**🟢 OVERALL RISK LEVEL:** [Low/Medium/High]

**⚠️ IMMEDIATE ACTION REQUIRED:**
- Any urgent deadlines
- Critical decisions needed

**💡 RECOMMENDATIONS:**
- Suggestions to reduce risks
- When to consult a lawyer

Be specific and practical in your assessment."""
            
            result = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            return result.content
            
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
            
            prompt = f"""Translate the following legal document explanation to {target_lang_name}. Maintain the structure and formatting.

{text_to_translate}

Translation in {target_lang_name}:"""
            
            result = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            return result.content
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation unavailable. Original text: {text[:300]}..."
    
    async def chat_with_document(self, document_vector_store, question: str, language: str = "en") -> Tuple[str, List[str]]:
        """Enhanced chat using document and legal knowledge base"""
        try:
            await self._rate_limit_check()
            
            doc_context = ""
            if hasattr(document_vector_store, 'query'):
                try:
                    query_embedding = self.embeddings.embed_query(question)
                    doc_results = document_vector_store.query(query_embeddings=[query_embedding], n_results=3)
                    if doc_results['documents'] and doc_results['documents'][0]:
                        doc_context = "\n".join([doc[:400] for doc in doc_results['documents'][0]])
                except Exception as e:
                    print(f"Document query error: {e}")
            
            legal_context = ""
            if self.legal_kb_collection:
                try:
                    legal_query_embedding = self.embeddings.embed_query(question)
                    legal_results = self.legal_kb_collection.query(query_embeddings=[legal_query_embedding], n_results=2)
                    if legal_results['documents'] and legal_results['documents'][0]:
                        legal_context = "\n".join([doc[:300] for doc in legal_results['documents'][0]])
                except Exception as e:
                    print(f"Legal KB query error: {e}")
            
            lang_instruction = ""
            if language != "en":
                language_names = {
                    "hi": "Hindi", "bn": "Bengali", "te": "Telugu", 
                    "mr": "Marathi", "ta": "Tamil", "gu": "Gujarati", 
                    "kn": "Kannada", "ml": "Malayalam", "or": "Odia", 
                    "pa": "Punjabi", "ur": "Urdu", "as": "Assamese"
                }
                lang_name = language_names.get(language, "Hindi")
                lang_instruction = f"Please answer in {lang_name}."
            
            prompt = f"""You are a helpful legal assistant for legal documents. Answer the user's question based on the provided context.

DOCUMENT CONTEXT:
{doc_context}

LEGAL KNOWLEDGE CONTEXT:
{legal_context}

USER QUESTION: {question}

{lang_instruction}

Provide a helpful, accurate answer based on the context. If you cannot find relevant information, say so clearly. Try answering in the selected language"""
            
            response = await asyncio.to_thread(self.gemini_llm.invoke, prompt)
            
            sources = []
            if doc_context: sources.append("Your uploaded document")
            if legal_context: sources.append("Legal knowledge base")
            
            return response.content, sources
            
        except Exception as e:
            print(f"Chat error: {e}")
            return ("I apologize, but I couldn't process your question at the moment."), []
    
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