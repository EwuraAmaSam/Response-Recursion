"""Embedding + metrics layer.

This package exposes the embedding layer entrypoint and metric models.

Some optional dependencies (e.g. `sentence-transformers`, `pyarrow`) are heavy
and may not be installed in minimal unit test environments. Imports that rely
on those packages are wrapped so that *metric-logic* unit tests can run without
needing the full ML stack.
"""

from __future__ import annotations

from typing import Any

# Optional: embedder (depends on sentence-transformers)
try:
    from .embedder import EmbeddingConfig, TextEmbedder  # noqa: F401
except Exception:  # pragma: no cover
    EmbeddingConfig = Any  # type: ignore
    TextEmbedder = Any  # type: ignore

# Optional: parquet IO (depends on pyarrow/pandas)
try:
    from .io import EmbeddingStore, MetricStore  # noqa: F401
except Exception:  # pragma: no cover
    EmbeddingStore = Any  # type: ignore
    MetricStore = Any  # type: ignore

from .anchor_bank import AnchorBankConfig, RefusalAnchorBank  # noqa: F401
from .constants import RefusalType, ALL_BOUNDARY_FLAGS  # noqa: F401
from .metrics_models import MCore, MDef, MDiag, TurnMetrics, RunAggregates  # noqa: F401

# Optional: orchestrator entrypoint (imports IO which may depend on pyarrow)
try:
    from .aggregator import (  # noqa: F401
        EmbeddingMetricSpaceConfig,
        OpenWeightComponentsConfig,
        EmbeddingLayerConfig,
        process_breakdown_message,
    )
except Exception:  # pragma: no cover
    EmbeddingMetricSpaceConfig = Any  # type: ignore
    OpenWeightComponentsConfig = Any  # type: ignore
    EmbeddingLayerConfig = Any  # type: ignore
    process_breakdown_message = Any  # type: ignore
