import pickle
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from models.schemas import Candidate, HSEntry
from services.embedding import EmbeddingService
from utils.logger import get_logger

logger = get_logger("vector_search")

_INDEX_FILE = "hs_index.faiss"
_ENTRIES_FILE = "hs_entries.pkl"


class VectorSearchService:
    def __init__(self) -> None:
        self._index = None
        self._entries: list[HSEntry] = []
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    def load_from_disk(self, index_dir: Path) -> bool:
        """Load the pre-built index from disk. Returns True on success."""
        if faiss is None:
            return False

        index_path = index_dir / _INDEX_FILE
        entries_path = index_dir / _ENTRIES_FILE

        if not (index_path.exists() and entries_path.exists()):
            return False

        self._index = faiss.read_index(str(index_path))
        with open(entries_path, "rb") as f:
            self._entries = pickle.load(f)
        self._is_built = True
        logger.info("Loaded FAISS index from disk: %d vectors", self._index.ntotal)
        return True

    def save_to_disk(self, index_dir: Path) -> None:
        """Save the built index to disk permanently."""
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / _INDEX_FILE))
        with open(index_dir / _ENTRIES_FILE, "wb") as f:
            pickle.dump(self._entries, f)
        logger.info("FAISS index saved to disk")

    def build_index(
        self,
        entries: list[HSEntry],
        embedding_service: EmbeddingService,
        hierarchy: dict[str, list[str]] | None = None,
    ) -> None:
        """Build FAISS index.

        hierarchy: optional mapping of hs_code → [desc0, desc1, ...] from root to leaf.
        When provided, each entry is indexed as its full classification path, giving
        richer context without any hardcoded synonyms.
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        logger.info("Building FAISS index for %d entries...", len(entries))

        self._entries = entries

        def _index_text(entry: HSEntry) -> str:
            if hierarchy and entry.hs_code in hierarchy:
                return " | ".join(hierarchy[entry.hs_code])
            return entry.description

        enriched = [_index_text(e) for e in entries]
        vectors = embedding_service.encode_batch(enriched)

        dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(vectors)
        self._is_built = True

        logger.info("FAISS index built: %d vectors, %d dimensions", self._index.ntotal, dimension)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[Candidate]:
        if not self._is_built or self._index is None:
            raise RuntimeError("Call build_index() first.")

        scores, indices = self._index.search(query_vector.reshape(1, -1).astype(np.float32), top_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            candidates.append(Candidate(
                hs_code=entry.hs_code,
                description=entry.description,
                section=entry.section,
                level=entry.level,
                parent=entry.parent,
                similarity_score=round(float(max(0.0, min(1.0, score))), 4),
            ))

        logger.info("Search returned %d candidates (top: %.4f)", len(candidates), candidates[0].similarity_score if candidates else 0.0)
        return candidates
