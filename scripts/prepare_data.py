"""Prepare compliance documents and build vector store."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStore
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample compliance rules (you'll replace with real documents)
SAMPLE_COMPLIANCE_DOCS = {
    "finra_guarantees.txt": """
FINRA Rule 2210 - Communications with the Public

Section 1: Prohibited Statements
Financial advisors and registered representatives are strictly prohibited from:
- Making any false, exaggerated, unwarranted, promissory or misleading statement or claim
- Publishing, circulating or distributing any public communication that contains an untrue statement of material fact
- Making predictions or projections of investment results without clear disclaimers
- Using terms such as "guaranteed", "assured", or "certain" when referring to investment returns

Section 2: Required Disclosures
All communications must include appropriate risk disclosures, including:
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal
- Diversification does not ensure profit or protect against loss
""",
    
    "sec_suitability.txt": """
SEC Regulation Best Interest - Suitability Requirements

Section 1: Know Your Customer
Investment professionals must:
- Obtain and document essential facts about each client
- Understand the client's investment profile, including risk tolerance
- Consider the client's age, financial situation, and investment objectives
- Review and update client information regularly

Section 2: Reasonable Basis Suitability
Before recommending any investment, advisors must:
- Have a reasonable basis to believe the recommendation is suitable
- Understand the potential risks and rewards
- Consider less risky alternatives
- Document the rationale for recommendations
""",
    
    "compliance_keywords.txt": """
High-Risk Compliance Keywords and Phrases

The following terms often indicate potential compliance violations:
- "Guaranteed returns" - Never promise specific investment outcomes
- "No risk" or "risk-free" - All investments carry some level of risk
- "Sure thing" - Implies certainty where none exists
- "Can't lose" - Misleading statement about investment safety
- "Hot tip" or "inside information" - May suggest insider trading
- "Limited time offer" - Creates artificial urgency
- "Exclusive opportunity" - May violate fair dealing requirements
"""
}

def create_sample_documents():
    """Create sample compliance documents."""
    config = Config()
    
    logger.info("Creating sample compliance documents...")
    for filename, content in SAMPLE_COMPLIANCE_DOCS.items():
        file_path = config.COMPLIANCE_DOCS_DIR / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created: {file_path}")

def main():
    """Main data preparation pipeline."""
    logger.info("=== Aegis Data Preparation ===\n")
    
    # Create sample documents (replace with real document loading in production)
    create_sample_documents()
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Load documents
    logger.info("\nLoading compliance documents...")
    documents = processor.load_documents()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    logger.info("\nChunking documents...")
    chunks = processor.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    logger.info("\nGenerating embeddings...")
    embeddings, chunks = processor.generate_embeddings(chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Build vector store
    logger.info("\nBuilding vector store...")
    vector_store = VectorStore()
    vector_store.build_index(embeddings, chunks)
    
    # Save vector store
    logger.info("\nSaving vector store...")
    vector_store.save_index()
    processor.save_processed_data(embeddings, chunks)
    
    # Test retrieval
    logger.info("\n=== Testing Retrieval ===")
    test_queries = [
        "guaranteed returns",
        "investment recommendations",
        "risk disclosure"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = vector_store.search(query, top_k=2)
        for i, result in enumerate(results):
            logger.info(f"  Result {i+1} (score: {result['score']:.3f}): {result['text'][:100]}...")
    
    logger.info("\n✅ Data preparation complete!")

if __name__ == "__main__":
    main()