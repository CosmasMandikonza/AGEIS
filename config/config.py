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

---
# src/cloud/watsonx_client.py
import asyncio
import logging
from typing import Optional, AsyncGenerator
import aiohttp
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import Model
from config.config import Config

logger = logging.getLogger(__name__)

class WatsonxClient:
    """Client for IBM watsonx.ai services."""
    
    def __init__(self):
        self.config = Config()
        self.credentials = None
        self.speech_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Watson credentials and models."""
        try:
            # Set up credentials
            self.credentials = Credentials(
                api_key=self.config.WATSONX_API_KEY,
                url=self.config.WATSONX_URL
            )
            
            # Initialize speech model
            self.speech_model = Model(
                model_id=self.config.SPEECH_MODEL,
                credentials=self.credentials,
                project_id=self.config.WATSONX_PROJECT_ID
            )
            
            logger.info("Watson client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Watson client: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data using Granite Speech model.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
        """
        try:
            # Note: This is a simplified version. The actual implementation
            # will depend on the specific Watson Speech API endpoint
            
            headers = {
                'Authorization': f'Bearer {self.config.WATSONX_API_KEY}',
                'Content-Type': 'audio/wav',
                'Accept': 'application/json'
            }
            
            params = {
                'model': self.config.SPEECH_MODEL,
                'project_id': self.config.WATSONX_PROJECT_ID
            }
            
            # Make async request to Watson Speech API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.WATSONX_URL}/v1/speech-to-text/recognize",
                    headers=headers,
                    params=params,
                    data=audio_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Extract transcript from response
                        transcript = result.get('results', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
                        return transcript
                    else:
                        error_msg = await response.text()
                        logger.error(f"Transcription failed: {error_msg}")
                        return ""
                        
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
    
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for text using Watson embeddings API.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use the embeddings endpoint
            embeddings = []
            
            for text in texts:
                params = {
                    "input": text,
                    "model_id": "ibm/all-minilm-l12-v2",  # IBM's embedding model
                    "project_id": self.config.WATSONX_PROJECT_ID
                }
                
                # Note: Actual implementation would use the proper Watson API
                # This is a placeholder for the structure
                response = await self._make_embedding_request(params)
                embeddings.append(response)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    async def _make_embedding_request(self, params: dict) -> list[float]:
        """Make embedding request to Watson API."""
        # Placeholder - actual implementation would make HTTP request
        return [0.0] * 384  # Return dummy embedding
    
    def test_connection(self) -> bool:
        """Test connection to Watson services."""
        try:
            # Simple test to verify credentials
            test_params = {
                "prompt": "Test",
                "max_new_tokens": 1
            }
            
            # Try a simple generation to test connection
            response = self.speech_model.generate(test_params)
            logger.info("Watson connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Watson connection test failed: {e}")
            return False

---
# src/audio/audio_handler.py
import asyncio
import logging
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import io
import wave
from typing import Callable, Optional
from config.config import Config

logger = logging.getLogger(__name__)

class AudioHandler:
    """Handles real-time audio capture and streaming."""
    
    def __init__(self, on_audio_chunk: Callable[[bytes], None]):
        self.config = Config()
        self.on_audio_chunk = on_audio_chunk
        self.is_recording = False
        self.audio_queue = asyncio.Queue()
        
    def start_recording(self):
        """Start recording audio from microphone."""
        self.is_recording = True
        logger.info("Starting audio recording...")
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream."""
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            if self.is_recording:
                # Convert to bytes and add to queue
                audio_data = indata.copy()
                asyncio.create_task(self.audio_queue.put(audio_data))
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.config.SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=self.config.CHUNK_SIZE,
            dtype='float32'
        )
        self.stream.start()
        
        # Start processing loop
        asyncio.create_task(self._process_audio_chunks())
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        logger.info("Audio recording stopped")
    
    async def _process_audio_chunks(self):
        """Process audio chunks from the queue."""
        buffer = []
        buffer_duration = 0
        
        while self.is_recording:
            try:
                # Get audio data from queue
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=0.1
                )
                
                buffer.append(audio_data)
                buffer_duration += len(audio_data) / self.config.SAMPLE_RATE
                
                # Send chunk when we have enough audio
                if buffer_duration >= self.config.CHUNK_DURATION:
                    # Combine buffer into single array
                    combined_audio = np.concatenate(buffer)
                    
                    # Convert to WAV format
                    wav_data = self._numpy_to_wav(combined_audio)
                    
                    # Send to callback
                    await self.on_audio_chunk(wav_data)
                    
                    # Reset buffer
                    buffer = []
                    buffer_duration = 0
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
    
    def _numpy_to_wav(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy array to WAV format bytes."""
        # Normalize float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()