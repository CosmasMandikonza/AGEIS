# config/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    # IBM watsonx.ai settings
    WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
    WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
    WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
    
    # Model settings
    SPEECH_MODEL = os.getenv('SPEECH_MODEL', 'ibm/granite-3.3-speech-8b')
    LOCAL_MODEL = os.getenv('LOCAL_MODEL', 'granite-4.0-tiny')
    GUARDIAN_MODEL = os.getenv('GUARDIAN_MODEL', 'granite-guardian-8b')
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Audio settings
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '16000'))
    CHUNK_DURATION = int(os.getenv('CHUNK_DURATION', '2'))
    CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
    
    # Ollama settings
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    COMPLIANCE_DOCS_DIR = DATA_DIR / 'compliance_docs'
    PROCESSED_DIR = DATA_DIR / 'processed'
    VECTOR_STORE_PATH = PROCESSED_DIR / 'compliance_vectors.faiss'
    
    # RAG settings
    CHUNK_SIZE_TOKENS = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 3
    
    # Create directories if they don't exist
    COMPLIANCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

