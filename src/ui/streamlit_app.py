# src/ui/streamlit_app.py
import streamlit as st
import asyncio
import logging
from datetime import datetime
import json
from typing import Optional

# Import our components
from src.audio.audio_handler import AudioHandler
from src.cloud.watsonx_client import WatsonxClient
from src.agents.worker_agent import WorkerAgent
from src.agents.guardian_agent import GuardianAgent
from src.rag.vector_store import VectorStore
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Aegis - Real-time Compliance Guardian",
    page_icon="🛡️",
    layout="wide"
)

class AegisApp:
    """Main Aegis application class."""
    
    def __init__(self):
        self.config = Config()
        self.watson_client = None
        self.worker_agent = None
        self.guardian_agent = None
        self.audio_handler = None
        self.vector_store = None
        
        # Initialize session state
        if 'transcript' not in st.session_state:
            st.session_state.transcript = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def initialize_components(self):
        """Initialize all components."""
        with st.spinner("Initializing Aegis components..."):
            try:
                # Initialize vector store
                self.vector_store = VectorStore()
                if not self.vector_store.load_index():
                    st.error("Failed to load vector store. Please run data preparation first.")
                    return False
                
                # Initialize Watson client
                self.watson_client = WatsonxClient()
                
                # Initialize agents
                self.worker_agent = WorkerAgent(self.vector_store)
                self.guardian_agent = GuardianAgent()
                
                # Initialize audio handler
                self.audio_handler = AudioHandler(self.on_audio_chunk)
                
                st.success("✅ All components initialized successfully!")
                return True
                
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
                logger.error(f"Failed to initialize components: {e}")
                return False
    
    async def on_audio_chunk(self, audio_data: bytes):
        """Handle incoming audio chunks."""
        try:
            # Transcribe audio
            transcript = await self.watson_client.transcribe_audio(audio_data)
            
            if transcript:
                # Add to transcript
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.transcript.append({
                    'time': timestamp,
                    'text': transcript
                })
                
                # Update conversation history
                st.session_state.conversation_history.append(transcript)
                
                # Analyze for compliance
                analysis = await self.worker_agent.analyze_utterance(
                    transcript,
                    context=st.session_state.conversation_history[:-1]
                )
                
                # If risk detected, get guardian review
                if analysis['risk_detected']:
                    analysis = await self.guardian_agent.review_alert(
                        analysis,
                        transcript
                    )
                    
                    # Add alert
                    st.session_state.alerts.append({
                        'time': timestamp,
                        'original': transcript,
                        'analysis': analysis
                    })
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def render_ui(self):
        """Render the Streamlit UI."""
        # Header
        st.title("🛡️ Aegis - Real-time Compliance Guardian")
        st.markdown("*Protecting conversations with AI-powered compliance monitoring*")
        
        # Control panel
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            if st.button("🎤 Start Recording" if not st.session_state.is_recording else "⏹️ Stop Recording"):
                if not st.session_state.is_recording:
                    self.start_recording()
                else:
                    self.stop_recording()
        
        with col2:
            if st.button("🗑️ Clear Session"):
                self.clear_session()
        
        with col3:
            status = "🔴 Recording..." if st.session_state.is_recording else "⚪ Ready"
            st.markdown(f"**Status:** {status}")
        
        st.divider()
        
        # Main content area
        col_transcript, col_alerts = st.columns([1, 1])
        
        # Transcript column
        with col_transcript:
            st.subheader("📝 Live Transcript")
            transcript_container = st.container()
            
            with transcript_container:
                if st.session_state.transcript:
                    for entry in st.session_state.transcript[-10:]:  # Show last 10
                        st.text(f"[{entry['time']}] {entry['text']}")
                else:
                    st.info("Transcript will appear here when recording starts...")
        
        # Alerts column
        with col_alerts:
            st.subheader("⚠️ Compliance Alerts")
            alerts_container = st.container()
            
            with alerts_container:
                if st.session_state.alerts:
                    for alert in st.session_state.alerts[-5:]:  # Show last 5
                        with st.expander(f"🚨 Alert at {alert['time']}", expanded=True):
                            st.error(f"**Original:** {alert['original']}")
                            st.warning(f"**Risk:** {alert['analysis']['explanation']}")
                            st.success(f"**Suggestion:** {alert['analysis']['suggestion']}")
                            
                            if alert['analysis'].get('guardian_reviewed'):
                                quality = alert['analysis'].get('quality_score', 'N/A')
                                st.caption(f"✓ Guardian reviewed (Quality: {quality}/10)")
                else:
                    st.info("Compliance alerts will appear here...")
        
        # Statistics sidebar
        with st.sidebar:
            st.header("📊 Session Statistics")
            st.metric("Total Utterances", len(st.session_state.transcript))
            st.metric("Alerts Generated", len(st.session_state.alerts))
            
            if st.session_state.alerts and st.session_state.transcript:
                compliance_rate = (1 - len(st.session_state.alerts) / len(st.session_state.transcript)) * 100
                st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
            
            st.divider()
            
            st.header("🔧 Configuration")
            st.text(f"Speech Model: {self.config.SPEECH_MODEL}")
            st.text(f"Analysis Model: {self.config.LOCAL_MODEL}")
            st.text(f"Guardian Model: {self.config.GUARDIAN_MODEL}")
    
    def start_recording(self):
        """Start audio recording."""
        st.session_state.is_recording = True
        self.audio_handler.start_recording()
        st.rerun()
    
    def stop_recording(self):
        """Stop audio recording."""
        st.session_state.is_recording = False
        self.audio_handler.stop_recording()
        st.rerun()
    
    def clear_session(self):
        """Clear session data."""
        st.session_state.transcript = []
        st.session_state.alerts = []
        st.session_state.conversation_history = []
        st.rerun()

# Main application entry point
def main():
    """Main application entry point."""
    app = AegisApp()
    
    # Initialize components on first run
    if 'initialized' not in st.session_state:
        if app.initialize_components():
            st.session_state.initialized = True
        else:
            st.stop()
    
    # Set up the app components from session state if needed
    if st.session_state.initialized and app.watson_client is None:
        app.initialize_components()
    
    # Render UI
    app.render_ui()
    
    # Auto-refresh for live updates
    if st.session_state.is_recording:
        st.empty()
        import time
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    # Run with: streamlit run src/ui/streamlit_app.py
    main()
