"""Test connections to all required services."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from config.config import Config
from src.cloud.watsonx_client import WatsonxClient
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_watson_connection():
    """Test Watson connection."""
    logger.info("Testing Watson connection...")
    try:
        client = WatsonxClient()
        success = client.test_connection()
        if success:
            logger.info("✅ Watson connection successful")
        else:
            logger.error("❌ Watson connection failed")
        return success
    except Exception as e:
        logger.error(f"❌ Watson connection error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection."""
    logger.info("Testing Ollama connection...")
    try:
        config = Config()
        client = ollama.Client(host=config.OLLAMA_HOST)
        
        # List models
        models = client.list()
        logger.info(f"Available models: {[m['name'] for m in models['models']]}")
        
        # Test generation
        response = client.generate(
            model=config.LOCAL_MODEL,
            prompt="Hello",
            options={"num_predict": 5}
        )
        
        logger.info("✅ Ollama connection successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama connection error: {e}")
        return False

def test_file_system():
    """Test file system setup."""
    logger.info("Testing file system...")
    try:
        config = Config()
        
        # Check directories
        dirs_ok = True
        for dir_path in [config.DATA_DIR, config.COMPLIANCE_DOCS_DIR, config.PROCESSED_DIR]:
            if dir_path.exists():
                logger.info(f"✅ Directory exists: {dir_path}")
            else:
                logger.error(f"❌ Directory missing: {dir_path}")
                dirs_ok = False
        
        return dirs_ok
        
    except Exception as e:
        logger.error(f"❌ File system error: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("=== Aegis Connection Tests ===\n")
    
    results = {
        "Watson": await test_watson_connection(),
        "Ollama": test_ollama_connection(),
        "File System": test_file_system()
    }
    
    logger.info("\n=== Test Results ===")
    for service, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{service}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! Ready to run Aegis.")
    else:
        logger.error("\n⚠️ Some tests failed. Please fix issues before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)