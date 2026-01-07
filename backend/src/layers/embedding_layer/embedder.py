from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    """
    Configuration for the embedding backend.

    This should be wired to core.config, and can be logged
    into external storage for traceability.
    """
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: Optional[str] = None      # "cuda", "cpu" or None for auto-detect
    batch_size: int = 32
    normalize_embeddings: bool = True


class TextEmbedder:
    """
    Wrapper around a SentenceTransformer encoder.

    Responsibilities:
    - Take embedding_units[].clean_text and encode to vectors
    - Support batching and GPU/CPU selection
    - Produce deterministic embeddings for a fixed model + config
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

        device = config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = SentenceTransformer(config.model_name, device=device)
        self._device = device

    @classmethod
    def from_defaults(cls) -> "TextEmbedder":
        return cls(EmbeddingConfig())

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode a list/iterable of texts into a 2D numpy array (n, d).
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience wrapper for a single text. Returns a 1D numpy array (d,).
        """
        return self.embed_texts([text])[0]

