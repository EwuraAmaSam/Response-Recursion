from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any

import numpy as np

# Optional: embedder depends on sentence-transformers
try:
    from .embedder import TextEmbedder  # type: ignore
except Exception:  # pragma: no cover
    TextEmbedder = Any  # type: ignore


@dataclass(frozen=True)
class AnchorBankConfig:
    """Refusal anchor configuration.

    - refusal_anchor_embeddings_path: path to .npy file storing (n, d) embeddings
    - refusal_anchor_texts_path: optional path to a text file with benign refusal examples
      (one refusal per line) to build embeddings if the .npy is missing.
    """

    refusal_anchor_embeddings_path: Optional[Path] = None
    refusal_anchor_texts_path: Optional[Path] = None
    allow_build_from_texts: bool = False


class RefusalAnchorBank:
    def __init__(self, centroid: np.ndarray) -> None:
        self.centroid = centroid.astype(np.float32)

    @staticmethod
    def _centroid(embs: np.ndarray) -> np.ndarray:
        if embs.ndim != 2 or embs.shape[0] == 0:
            raise ValueError("refusal_anchor_embeddings must be a non-empty 2D array")
        return embs.mean(axis=0).astype(np.float32)

    @classmethod
    def load_or_build(
        cls,
        cfg: AnchorBankConfig,
        *,
        embedder: Optional[TextEmbedder] = None,
    ) -> Optional["RefusalAnchorBank"]:
        path = cfg.refusal_anchor_embeddings_path
        if path is not None:
            path = Path(path).expanduser().resolve()
            if path.exists():
                embs = np.load(path)
                return cls(cls._centroid(embs))

        if not cfg.allow_build_from_texts:
            return None

        texts_path = cfg.refusal_anchor_texts_path
        if texts_path is None:
            return None

        texts_path = Path(texts_path).expanduser().resolve()
        if not texts_path.exists():
            return None
        if embedder is None:
            raise ValueError("embedder must be provided to build anchor bank from texts")

        lines: List[str] = [ln.strip() for ln in texts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return None
        embs = embedder.embed_texts(lines)
        if path is not None:
            # Best-effort persist for reproducibility.
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, embs.astype(np.float32))
            except Exception:
                pass
        return cls(cls._centroid(embs))

    def distance(self, response_embedding: np.ndarray) -> float:
        from .utils import cosine_similarity

        return float(1.0 - cosine_similarity(response_embedding.astype(np.float32), self.centroid))
