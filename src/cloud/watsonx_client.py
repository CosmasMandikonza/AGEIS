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
