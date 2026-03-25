"""
Embedding Generation Service.

Purpose: Converts text into dense vector representations using sentence-transformers.
Inputs:  Raw text strings (product descriptions, HS descriptions).
Outputs: Numpy arrays of shape (n, 384) — normalized embeddings.
Model:   all-MiniLM-L6-v2 (384 dimensions, fast, runs locally).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.utils.logger import get_logger
from backend.utils.text_processing import prepare_for_embedding

logger = get_logger("embedding")

# Model name — matches gemini.md architectural invariant
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Generates and manages text embeddings using sentence-transformers."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._dimension: int = 384

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Load the sentence-transformer model into memory.

        First run downloads ~80MB model files. Subsequent runs use cache.
        """
        logger.info("Loading embedding model: %s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dimension=%d)", self._dimension)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a normalized embedding vector.

        Args:
            text: Raw text to encode.

        Returns:
            Normalized numpy array of shape (384,).
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        processed = prepare_for_embedding(text)
        vector = self._model.encode(processed, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into normalized embedding vectors.

        Args:
            texts: List of raw text strings.

        Returns:
            Numpy array of shape (n, 384) with normalized vectors.
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        processed = [prepare_for_embedding(t) for t in texts]
        vectors = self._model.encode(
            processed,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
        )
        return np.array(vectors, dtype=np.float32)
