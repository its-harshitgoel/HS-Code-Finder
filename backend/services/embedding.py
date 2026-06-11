import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger
from utils.text_processing import prepare_for_embedding

logger = get_logger("embedding")

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def load_model(self) -> None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded")

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load_model() first.")
        vector = self._model.encode(prepare_for_embedding(text), normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load_model() first.")
        vectors = self._model.encode(
            [prepare_for_embedding(t) for t in texts],
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
        )
        return np.array(vectors, dtype=np.float32)
