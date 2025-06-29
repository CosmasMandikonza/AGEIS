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