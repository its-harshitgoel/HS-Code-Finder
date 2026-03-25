"""
Vector Search Service (FAISS).

Purpose: Builds and queries a FAISS index over HS description embeddings for
         fast semantic similarity search.
Inputs:  Query embedding vector (384-dim).
Outputs: Ranked list of Candidate objects with similarity scores.
Index:   IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).
"""

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from backend.models.schemas import Candidate, HSEntry
from backend.services.embedding import EmbeddingService
from backend.utils.logger import get_logger

logger = get_logger("vector_search")


class VectorSearchService:
    """Manages the FAISS index for semantic search over HS descriptions."""

    def __init__(self) -> None:
        self._index = None
        self._entries: list[HSEntry] = []
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def index_size(self) -> int:
        return len(self._entries)

    def build_index(
        self,
        entries: list[HSEntry],
        embedding_service: EmbeddingService,
    ) -> None:
        """Build the FAISS index from HS entries.

        Encodes all HS descriptions into embeddings and stores them in
        a flat inner-product index (cosine similarity on normalized vectors).

        Args:
            entries: List of HSEntry objects to index.
            embedding_service: Service to generate embeddings.
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        logger.info("Building FAISS index for %d entries...", len(entries))

        self._entries = entries
        descriptions = [entry.description for entry in entries]

        # Batch encode all descriptions
        vectors = embedding_service.encode_batch(descriptions)

        # Build flat inner-product index (cosine sim on normalized vectors)
        dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(vectors)

        self._is_built = True
        logger.info(
            "FAISS index built: %d vectors, %d dimensions",
            self._index.ntotal,
            dimension,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[Candidate]:
        """Search the index for the most similar HS entries.

        Args:
            query_vector: Normalized embedding vector of shape (384,).
            top_k: Number of top results to return.

        Returns:
            List of Candidate objects sorted by similarity (highest first).
        """
        if not self._is_built or self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Reshape for FAISS: (1, dim)
        query = query_vector.reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self._index.search(query, top_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue

            entry = self._entries[idx]
            candidates.append(
                Candidate(
                    hs_code=entry.hs_code,
                    description=entry.description,
                    section=entry.section,
                    level=entry.level,
                    parent=entry.parent,
                    similarity_score=round(float(max(0.0, min(1.0, score))), 4),
                )
            )

        logger.info(
            "Search returned %d candidates (top score: %.4f)",
            len(candidates),
            candidates[0].similarity_score if candidates else 0.0,
        )

        return candidates
