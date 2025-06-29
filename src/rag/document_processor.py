# src/rag/document_processor.py
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
import pickle
from config.config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process compliance documents for RAG pipeline."""
    
    def __init__(self):
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
    
    def load_documents(self) -> List[Dict[str, str]]:
        """Load all compliance documents from the directory."""
        documents = []
        
        # Process all text files in compliance directory
        for file_path in self.config.COMPLIANCE_DOCS_DIR.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                documents.append({
                    'source': file_path.name,
                    'content': content
                })
                logger.info(f"Loaded document: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Chunk documents into semantic segments.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for doc in documents:
            # Split by paragraphs or sections
            sections = self._split_into_sections(doc['content'])
            
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:  # Minimum chunk size
                    chunks.append({
                        'text': section.strip(),
                        'source': doc['source'],
                        'chunk_id': f"{doc['source']}_chunk_{i}",
                        'metadata': {
                            'source_file': doc['source'],
                            'chunk_index': i
                        }
                    })
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into semantic sections."""
        # Split by multiple newlines or section headers
        sections = re.split(r'\n\n+|(?=\d+\.\s+[A-Z])|(?=Section \d+)', text)
        
        # Further split very long sections
        final_sections = []
        for section in sections:
            if len(section) > 1000:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 800:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            final_sections.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    final_sections.append(current_chunk.strip())
            else:
                final_sections.append(section)
        
        return final_sections
    
    def generate_embeddings(self, chunks: List[Dict[str, any]]) -> Tuple[List[List[float]], List[Dict[str, any]]]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            Tuple of (embeddings, chunks)
        """
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        return embeddings.tolist(), chunks
    
    def save_processed_data(self, embeddings: List[List[float]], chunks: List[Dict[str, any]]):
        """Save processed embeddings and chunks."""
        # Save embeddings
        embeddings_path = self.config.PROCESSED_DIR / 'embeddings.pkl'
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save chunks
        chunks_path = self.config.PROCESSED_DIR / 'chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        logger.info(f"Saved processed data to {self.config.PROCESSED_DIR}")

