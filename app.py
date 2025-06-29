# app.py (Main entry point)
#!/usr/bin/env python3
"""
Aegis - Real-time AI Compliance Guardian
Main application launcher
"""

import sys
import subprocess
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Launch the Aegis application."""
    logger.info("Starting Aegis application...")
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main()