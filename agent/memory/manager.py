import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class MemoryManager:
    def __init__(self, persist_directory: str = "agent_memory_db"):
        self.persist_directory = persist_directory
        
        # Initialize Embedding Function (using local SentenceTransformer)
        # This ensures consistent and offline-capable embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="agent_knowledge",
            embedding_function=self.embedding_fn
        )
        self.tool_collection = self.chroma_client.get_or_create_collection(
            name="agent_tools",
            embedding_function=self.embedding_fn
        ) # New collection for tools
        
        # Simple Key-Value Store (mocking Redis)
        self.kv_store: Dict[str, Any] = {}


    def index_tool(self, tool_name: str, tool_description: str):
        """Index a tool for retrieval"""
        description = tool_description or ""
        upsert = getattr(self.tool_collection, "upsert", None)
        if callable(upsert):
            upsert(
                documents=[description],
                metadatas=[{"name": tool_name}],
                ids=[tool_name],
            )
            return

        try:
            self.tool_collection.delete(ids=[tool_name])
        except Exception:
            pass
        self.tool_collection.add(
            documents=[description],
            metadatas=[{"name": tool_name}],
            ids=[tool_name],
        )

    def retrieve_tools(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve relevant tool names"""
        results = self.tool_collection.query(
            query_texts=[query],
            n_results=limit
        )
        if results['metadatas']:
            return [m['name'] for m in results['metadatas'][0]]
        return []

    def store_context(self, key: str, value: Any):
        """Store short-term context (Redis-like)"""
        self.kv_store[key] = value

    def retrieve_context(self, key: str) -> Optional[Any]:
        """Retrieve short-term context"""
        return self.kv_store.get(key)

    def add_memory(self, content: str, metadata: Dict[str, Any] = None):
        """Add long-term memory to Vector DB"""
        metadata = metadata or {}
        doc_id = str(len(self.collection.get()['ids']) + 1)
        
        # In a real app, we would use embeddings model to embed the content
        # Chroma handles default embedding if none provided, but here we just pass content
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def retrieve_relevant(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve relevant memories from Vector DB"""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        if results['documents']:
            return results['documents'][0]
        return []
