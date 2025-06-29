# src/rag/vector_store.py
import logging
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from config.config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for RAG retrieval."""
    
    def __init__(self):
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.embeddings = []
    
    def build_index(self, embeddings: List[List[float]], chunks: List[Dict[str, any]]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks with metadata
        """
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        # Store chunks and embeddings
        self.chunks = chunks
        self.embeddings = embeddings
        
        logger.info(f"Built FAISS index with {len(chunks)} vectors")
    
    def save_index(self):
        """Save FAISS index and associated data."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.config.VECTOR_STORE_PATH))
        
        # Save chunks
        chunks_path = self.config.PROCESSED_DIR / 'vector_store_chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved vector store to {self.config.VECTOR_STORE_PATH}")
    
    def load_index(self):
        """Load FAISS index from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.config.VECTOR_STORE_PATH))
            
            # Load chunks
            chunks_path = self.config.PROCESSED_DIR / 'vector_store_chunks.pkl'
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info("Loaded vector store from disk")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, any]]:
        """
        Search for relevant chunks based on query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        if self.index is None:
            logger.error("Vector store index not initialized")
            return []
        
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(1 / (1 + dist))  # Convert distance to similarity score
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def get_context_for_query(self, query: str) -> str:
        """
        Get formatted context for a query.
        
        Args:
            query: Search query
            
        Returns:
            Formatted context string
        """
        results = self.search(query)
        
        if not results:
            return "No relevant compliance rules found."
        
        # Format results as context
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Rule {i+1} from {result['source']}]:\n{result['text']}")
        
        return "\n\n".join(context_parts)