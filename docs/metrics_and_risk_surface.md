## Safe Reference Embeddings (Dolly15k)

- We construct a safe reference embedding corpus from `databricks/databricks-dolly-15k`.

- Script: `backend/scripts/build_safe_ref_embeddings_dolly.py` downloads the dataset,
  extracts the `response` field, embeds all responses with
  `sentence-transformers/all-mpnet-base-v2`, and stores them in:

  - `data/reference/safe_ref_embeddings_dolly15k.npy` (shape: (N, d))

- Script: `backend/scripts/build_clusters_from_safe_ref_embeddings.py` runs KMeans with K=32
  over this embedding space and stores:

  - `data/clusters/cluster_centers_dolly15k_k32.npy`
  - `data/clusters/safe_cluster_indices_dolly15k_k32.json` (currently marking all clusters as safe)

These artifacts are used by the embedding layer (Memb, Mprob, Mcls) as the stable
reference geometry for novelty, semantic distance, and safe-density metrics.

