from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset

# Adjust this import if the backend package root is named differently.
# This assumes project root has "backend" as a package, and we run with PYTHONPATH=.
from backend.src.layers.embedding_layer.embedder import EmbeddingConfig, TextEmbedder


def load_dolly_responses(max_samples: int | None = None) -> List[str]:
    """
    Load the databricks/databricks-dolly-15k dataset and extract safe responses.

    For now we treat all responses as safe reference geometry. You can add
    additional filtering later (length, keywords, categories, etc.).
    """
    print("🔹 Loading Dolly dataset from Hugging Face: databricks/databricks-dolly-15k")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    responses: List[str] = []
    for ex in dataset:
        resp = (ex.get("response") or "").strip()
        if not resp:
            continue
        responses.append(resp)

    if max_samples is not None and len(responses) > max_samples:
        responses = responses[:max_samples]

    print(f"✅ Loaded {len(responses)} non-empty responses from Dolly.")
    return responses


def build_embedder() -> TextEmbedder:
    """
    Create a TextEmbedder with the same config as the embedding layer.
    """
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device=None,          # auto: "cuda" if available, else "cpu"
        batch_size=32,
        normalize_embeddings=True,
    )
    print(f"🔹 Initializing TextEmbedder with model: {config.model_name}")
    return TextEmbedder(config)


def embed_texts_in_chunks(
    embedder: TextEmbedder,
    texts: List[str],
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Embed a list of texts in chunks to avoid RAM spikes.

    Returns:
        embeddings: np.ndarray of shape (N, d)
    """
    all_embeddings: list[np.ndarray] = []
    total = len(texts)
    print(f"🔹 Embedding {total} texts in chunks of {chunk_size}...")

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        batch = texts[start:end]
        print(f"  - Embedding batch {start}:{end}...")
        emb_batch = embedder.embed_texts(batch)  # shape: (batch_size, d)
        all_embeddings.append(emb_batch)

    embeddings = np.vstack(all_embeddings)
    print(f"✅ Finished embedding. Final shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, out_path: Path) -> None:
    """
    Save embeddings as a .npy file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"✅ Saved embeddings to: {out_path.resolve()}")


def main() -> None:
    # 1) Determine project root: scripts/ → backend/ → project root
    project_root = Path(__file__).resolve().parents[2]
    reference_dir = project_root / "data" / "reference"
    out_path = reference_dir / "safe_ref_embeddings_dolly15k.npy"

    print(f"📂 Project root: {project_root}")
    print(f"📂 Reference dir: {reference_dir}")
    print(f"📄 Output file: {out_path}")

    # 2) Load Dolly responses
    responses = load_dolly_responses(max_samples=None)  # use all available

    # 3) Build embedder
    embedder = build_embedder()

    # 4) Embed in chunks
    embeddings = embed_texts_in_chunks(embedder, responses, chunk_size=512)

    # 5) Save embeddings
    save_embeddings(embeddings, out_path)

    print("🎉 Done. You can now load these embeddings as safe_ref_embeddings in the embedding layer.")


if __name__ == "__main__":
    main()

