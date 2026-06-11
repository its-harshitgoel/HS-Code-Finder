"""Builds the FAISS index and runs test queries to verify search quality."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.services.embedding import EmbeddingService
from backend.services.hs_knowledge import HSKnowledgeBase
from backend.services.vector_search import VectorSearchService
from backend.utils.logger import get_logger

logger = get_logger("build_index")

CSV_PATH = PROJECT_ROOT / "data" / "hs_codes.csv"

TEST_QUERIES = [
    "cotton t-shirt",
    "frozen shrimp",
    "wooden dining table",
    "laptop computer",
    "olive oil",
    "rubber car tires",
    "metal water bottle",
    "silicone phone case",
]


def main():
    kb = HSKnowledgeBase()
    kb.load(CSV_PATH)

    embed = EmbeddingService()
    embed.load_model()

    # Index only 6-digit subheadings with hierarchy enrichment — must match main.py exactly
    entries = kb.get_subheadings()
    hierarchy: dict[str, list[str]] = {}
    for entry in entries:
        path = kb.get_hierarchy_path(entry.hs_code)
        descs = list(dict.fromkeys(e.description for e in path))
        if len(descs) > 1:
            hierarchy[entry.hs_code] = descs

    vs = VectorSearchService()
    vs.build_index(entries, embed, hierarchy=hierarchy)

    logger.info("Running test queries...")
    for query in TEST_QUERIES:
        results = vs.search(embed.encode(query), top_k=5)
        print(f"\nQuery: '{query}'")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.hs_code}] {r.description[:80]} ({r.similarity_score:.2%})")


if __name__ == "__main__":
    main()
