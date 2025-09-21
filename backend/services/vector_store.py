import os
import uuid
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # ChromaDB configuration
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        os.makedirs(self.chroma_db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collection for document chunks
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "legal_documents")
        self.document_collections: Dict[str, chromadb.Collection] = {}
        
        # FREE TIER OPTIMIZED SETTINGS
        self.max_documents = int(os.getenv("MAX_DOCUMENTS_STORED", "5"))  # Reduced from 10
        self.document_ttl_hours = int(os.getenv("DOCUMENT_TTL_HOURS", "12"))  # Reduced from 24
        self.auto_cleanup_enabled = os.getenv("AUTO_CLEANUP_ENABLED", "true").lower() == "true"
        
        print(f"✅ ChromaDB initialized at: {self.chroma_db_path}")
        print(f"📦 Free Tier Mode: Max documents: {self.max_documents}, TTL: {self.document_ttl_hours} hours")
        
        # Perform initial cleanup on startup
        if self.auto_cleanup_enabled:
            self._cleanup_old_documents()
    
    def _cleanup_old_documents(self):
        """Clean up old documents based on TTL and max document limit"""
        try:
            collections = self.chroma_client.list_collections()
            collection_info = []
            
            for collection in collections:
                if collection.name.startswith("doc_"):
                    # Get collection metadata
                    metadata = collection.metadata or {}
                    created_at = metadata.get("created_at")
                    
                    if created_at:
                        # Parse creation time
                        created_time = datetime.fromisoformat(created_at)
                        age_hours = (datetime.now() - created_time).total_seconds() / 3600
                        
                        collection_info.append({
                            "name": collection.name,
                            "created_at": created_time,
                            "age_hours": age_hours,
                            "count": collection.count()
                        })
            
            # Sort by creation time (oldest first)
            collection_info.sort(key=lambda x: x["created_at"])
            
            # Remove documents older than TTL
            for info in collection_info:
                if info["age_hours"] > self.document_ttl_hours:
                    print(f"🗑️ Removing expired document: {info['name']} (age: {info['age_hours']:.1f} hours)")
                    self.chroma_client.delete_collection(name=info["name"])
                    collection_info.remove(info)
            
            # Remove oldest documents if we exceed max limit
            while len(collection_info) > self.max_documents:
                oldest = collection_info.pop(0)
                print(f"🗑️ Removing oldest document to maintain limit: {oldest['name']}")
                self.chroma_client.delete_collection(name=oldest["name"])
            
            print(f"✅ Cleanup complete. Current documents: {len(collection_info)}/{self.max_documents}")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
    
    def _get_collection_count(self) -> int:
        """Get current number of document collections"""
        collections = self.chroma_client.list_collections()
        return sum(1 for c in collections if c.name.startswith("doc_"))
    
    def create_vector_store(self, document_id: str, documents: List[Document]) -> None:
        """Create and store a vector store for the document using ChromaDB"""
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            # Perform cleanup before adding new document
            if self.auto_cleanup_enabled:
                current_count = self._get_collection_count()
                
                # If we're at the limit, remove the oldest document
                if current_count >= self.max_documents:
                    print(f"⚠️ Document limit reached ({current_count}/{self.max_documents}). Cleaning up...")
                    self._cleanup_old_documents()
            
            # Create a unique collection for this document
            collection_name = f"doc_{document_id}"
            
            # Try to delete existing collection if it exists
            try:
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if collection_name in existing_collections:
                    print(f"🗑️ Deleting existing collection: {collection_name}")
                    self.chroma_client.delete_collection(name=collection_name)
            except Exception as e:
                print(f"Note: Could not delete existing collection {collection_name}: {e}")
            
            # Create new collection with creation timestamp
            try:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "created_at": datetime.now().isoformat(),
                        "document_id": document_id
                    }
                )
                print(f"✅ Created new collection: {collection_name}")
            except Exception as e:
                print(f"Failed to create collection {collection_name}: {e}")
                # Try to get existing collection if creation failed
                try:
                    collection = self.chroma_client.get_collection(name=collection_name)
                    print(f"✅ Retrieved existing collection: {collection_name}")
                except Exception as e2:
                    raise Exception(f"Could not create or retrieve collection {collection_name}: {e2}")
            
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                texts.append(doc.page_content)
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["chunk_id"] = i
                metadata["document_id"] = document_id
                metadatas.append(metadata)
                ids.append(f"{document_id}_{i}")
            
            # Generate embeddings
            print(f"🔄 Generating embeddings for {len(texts)} documents...")
            embeddings = self.embeddings.embed_documents(texts)
            print(f"✅ Generated {len(embeddings)} embeddings")
            
            # Add to ChromaDB with batch processing for large documents
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"✅ Added batch {i//batch_size + 1} to collection")
            
            # Store collection reference
            self.document_collections[document_id] = collection
            
            # Verify collection was created successfully
            count = collection.count()
            print(f"✅ Created vector store for document {document_id} with {count} chunks")
            
            # Show current storage status
            current_count = self._get_collection_count()
            print(f"📊 Storage status: {current_count}/{self.max_documents} documents")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def get_vector_store(self, document_id: str) -> Optional[chromadb.Collection]:
        """Get vector store by document ID"""
        try:
            if document_id in self.document_collections:
                return self.document_collections[document_id]
            
            # Try to load from ChromaDB
            collection_name = f"doc_{document_id}"
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                self.document_collections[document_id] = collection
                
                # Update last accessed time
                if self.auto_cleanup_enabled:
                    try:
                        metadata = collection.metadata or {}
                        metadata["last_accessed"] = datetime.now().isoformat()
                        collection.modify(metadata=metadata)
                    except Exception as e:
                        print(f"Could not update last accessed time: {e}")
                
                return collection
            except ValueError:
                print(f"Collection not found for document {document_id}")
                return None
                
        except Exception as e:
            print(f"Error getting vector store: {e}")
            return None
    
    def get_all_chunks(self, document_id: str) -> List[Document]:
        """Get all document chunks for a given document ID"""
        collection = self.get_vector_store(document_id)
        if not collection:
            return []
        
        try:
            # Get all documents from the collection
            results = collection.get()
            
            documents = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    documents.append(Document(
                        page_content=doc,
                        metadata=metadata or {}
                    ))
            
            return documents
            
        except Exception as e:
            print(f"Error getting all chunks: {e}")
            return []
    
    def search_similar(self, document_id: str, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents using ChromaDB"""
        collection = self.get_vector_store(document_id)
        if not collection:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count())
            )
            
            # Convert results to LangChain Document format
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0]
                )):
                    documents.append(Document(
                        page_content=doc,
                        metadata=metadata or {}
                    ))
            
            return documents
            
        except Exception as e:
            print(f"Error searching similar documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document's vector store"""
        try:
            collection_name = f"doc_{document_id}"
            self.chroma_client.delete_collection(name=collection_name)
            
            if document_id in self.document_collections:
                del self.document_collections[document_id]
            
            print(f"✅ Deleted vector store for document {document_id}")
            
            # Show current storage status
            current_count = self._get_collection_count()
            print(f"📊 Storage status: {current_count}/{self.max_documents} documents")
            
            return True
            
        except Exception as e:
            print(f"Error deleting vector store: {e}")
            return False
    
    def list_documents(self) -> List[str]:
        """List all document IDs with vector stores"""
        try:
            collections = self.chroma_client.list_collections()
            document_ids = []
            
            for collection in collections:
                if collection.name.startswith("doc_"):
                    document_id = collection.name[4:]  # Remove "doc_" prefix
                    document_ids.append(document_id)
            
            return document_ids
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def get_collection_info(self, document_id: str) -> Dict:
        """Get information about a document's collection"""
        collection = self.get_vector_store(document_id)
        if not collection:
            return {}
        
        try:
            metadata = collection.metadata or {}
            created_at = metadata.get("created_at")
            age_hours = 0
            
            if created_at:
                created_time = datetime.fromisoformat(created_at)
                age_hours = (datetime.now() - created_time).total_seconds() / 3600
            
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": metadata,
                "age_hours": age_hours,
                "ttl_remaining_hours": max(0, self.document_ttl_hours - age_hours)
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def cleanup_all(self):
        """Manually trigger cleanup of all expired documents"""
        if self.auto_cleanup_enabled:
            print("🧹 Manual cleanup triggered...")
            self._cleanup_old_documents()
        else:
            print("⚠️ Auto-cleanup is disabled")