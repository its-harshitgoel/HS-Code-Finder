"""
Tool: build_index.py

Purpose: Standalone script to build and test the FAISS vector index from the HS dataset.
         Useful for pre-building the index and verifying search quality.

Inputs:  data/hs_codes.csv
Outputs: Prints search results for test queries to verify index quality.
Logging: Logs index building progress and test results.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.embedding import EmbeddingService
from backend.services.hs_knowledge import HSKnowledgeBase
from backend.services.vector_search import VectorSearchService
from backend.utils.logger import get_logger

logger = get_logger("build_index")

CSV_PATH = PROJECT_ROOT / "data" / "hs_codes.csv"

# Test queries to verify search quality
TEST_QUERIES = [
    "cotton t-shirt",
    "frozen shrimp",
    "wooden dining table",
    "laptop computer",
    "olive oil",
    "rubber car tires",
]


def main():
    """Build the FAISS index and run test queries."""
    # 1. Load dataset
    logger.info("Loading HS dataset...")
    kb = HSKnowledgeBase()
    kb.load(CSV_PATH)

    # 2. Load embedding model
    logger.info("Loading embedding model...")
    embed = EmbeddingService()
    embed.load_model()

    # 3. Build index (subheadings + headings)
    logger.info("Building FAISS index...")
    entries = kb.get_subheadings() + kb.get_headings()
    vs = VectorSearchService()
    vs.build_index(entries, embed)

    # 4. Run test queries
    logger.info("=" * 60)
    logger.info("Running test queries...")
    logger.info("=" * 60)

    for query in TEST_QUERIES:
        vector = embed.encode(query)
        results = vs.search(vector, top_k=5)

        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.hs_code}] {r.description[:80]}... ({r.similarity_score:.2%})")

    logger.info("Index build and test complete.")


if __name__ == "__main__":
    main()
