from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def main() -> None:
    # 1) Resolve project root and input/output paths
    project_root = Path(__file__).resolve().parents[2]
    reference_dir = project_root / "data" / "reference"
    clusters_dir = project_root / "data" / "clusters"

    ref_path = reference_dir / "safe_ref_embeddings_dolly15k.npy"
    centers_path = clusters_dir / "cluster_centers_dolly15k_k32.npy"
    safe_indices_path = clusters_dir / "safe_cluster_indices_dolly15k_k32.json"

    print(f"📂 Project root: {project_root}")
    print(f"📂 Reference dir: {reference_dir}")
    print(f"📂 Clusters dir: {clusters_dir}")
    print(f"📄 Input embeddings: {ref_path}")
    print(f"📄 Output centers: {centers_path}")
    print(f"📄 Output safe-cluster indices: {safe_indices_path}")

    clusters_dir.mkdir(parents=True, exist_ok=True)

    # 2) Load reference embeddings
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference embeddings not found at {ref_path}. "
            f"Run build_safe_ref_embeddings_dolly.py first."
        )

    print("🔹 Loading reference embeddings...")
    embeddings = np.load(ref_path)
    print(f"✅ Loaded embeddings with shape: {embeddings.shape}")

    # 3) Choose number of clusters K
    # You can adjust this based on dataset size & dimensionality.
    K = 32
    print(f"🔹 Running KMeans with K={K} clusters...")

    kmeans = KMeans(
        n_clusters=K,
        random_state=0,
        n_init="auto",
    )
    kmeans.fit(embeddings)

    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"✅ KMeans finished. Cluster centers shape: {centers.shape}")

    # 4) Save cluster centers
    np.save(centers_path, centers)
    print(f"✅ Saved cluster centers to: {centers_path.resolve()}")

    # 5) Define safe cluster indices.
    # For now, since the entire reference corpus is constructed from safe Dolly responses,
    # we treat *all* clusters as safe. You can refine this later by manual inspection
    # or additional labeling.
    safe_indices = list(range(K))

    with safe_indices_path.open("w", encoding="utf-8") as f:
        json.dump(safe_indices, f, indent=2)

    print(f"✅ Saved safe cluster indices (all clusters) to: {safe_indices_path.resolve()}")

    print("🎉 Done. You can now load cluster_centers and safe_cluster_indices in your embedding layer config.")


if __name__ == "__main__":
    main()

