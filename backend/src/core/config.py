from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from core.private_storage import PrivateArtifactConfig
from layers.embedding_layer import EmbeddingConfig
from layers.embedding_layer.anchor_bank import AnchorBankConfig
from layers.embedding_layer.aggregator import (
    EmbeddingLayerConfig,
    EmbeddingMetricSpaceConfig,
    OpenWeightComponentsConfig,
)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v not in (None, "") else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


@dataclass(frozen=True)
class ArtifactPaths:
    seed_library_path: Path
    parquet_base_dir: Path

    # Anchor bank
    refusal_anchor_embeddings_path: Optional[Path]
    refusal_anchor_texts_path: Optional[Path]


@dataclass(frozen=True)
class AppConfig:
    version: str
    artifacts: ArtifactPaths
    embedding: EmbeddingConfig

    privacy: PrivateArtifactConfig
    components: OpenWeightComponentsConfig

    # Kept for compatibility (controller uses this to decide mock vs local)
    llm_mode: str  # "mock" | "local"


def load_app_config(project_root: Optional[Path] = None) -> AppConfig:
    if project_root is None:
        # backend/src/core/config.py -> ... -> repo root
        project_root = Path(__file__).resolve().parents[4]

    data_dir = project_root / "data"
    default_parquet = project_root / "external_storage"

    artifacts = ArtifactPaths(
        seed_library_path=_as_path(
            _env("RR_SEED_LIBRARY_PATH", str(data_dir / "seeds" / "seed_library.json"))
            or str(data_dir / "seeds" / "seed_library.json")
        ),
        parquet_base_dir=_as_path(_env("RR_PARQUET_BASE_DIR", str(default_parquet)) or str(default_parquet)),
        refusal_anchor_embeddings_path=(
            _as_path(_env("RR_REFUSAL_ANCHOR_EMB_PATH", str(data_dir / "anchors" / "refusal_anchor_embeddings.npy")))
            if _env("RR_REFUSAL_ANCHOR_EMB_PATH")
            else None
        ),
        refusal_anchor_texts_path=(
            _as_path(_env("RR_REFUSAL_ANCHOR_TEXT_PATH", str(data_dir / "anchors" / "refusal_anchor_texts.txt")))
            if _env("RR_REFUSAL_ANCHOR_TEXT_PATH")
            else None
        ),
    )

    embed_model = _env("RR_EMBED_MODEL", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2"
    embedding_cfg = EmbeddingConfig(model_name=embed_model, normalize_embeddings=True)

    privacy = PrivateArtifactConfig(
        store_raw_text=_env_bool("STORE_RAW_TEXT", False),
        private_artifacts_dir=_as_path(
            _env("PRIVATE_ARTIFACTS_DIR", str(project_root / "private_artifacts")) or str(project_root / "private_artifacts")
        ),
    )

    components = OpenWeightComponentsConfig(
        inference_backend=_env("RR_INFERENCE_BACKEND", "transformers") or "transformers",
        judge_model=_env("RR_JUDGE_MODEL"),
        guard_model=_env("RR_GUARD_MODEL"),
        nli_model=_env("RR_NLI_MODEL"),
        target_model=_env("RR_TARGET_MODEL"),
        vllm_tensor_parallel_size=int(_env("RR_VLLM_TP", "1") or "1"),
        vllm_dtype=_env("RR_VLLM_DTYPE"),
        transformers_torch_dtype=_env("RR_TORCH_DTYPE"),
    )

    return AppConfig(
        version=_env("RR_VERSION", "0.3") or "0.3",
        artifacts=artifacts,
        embedding=embedding_cfg,
        privacy=privacy,
        components=components,
        llm_mode=_env("RR_LLM_MODE", "mock") or "mock",
    )


def build_embedding_layer_config(app_cfg: AppConfig) -> EmbeddingLayerConfig:
    anchor_cfg = AnchorBankConfig(
        refusal_anchor_embeddings_path=app_cfg.artifacts.refusal_anchor_embeddings_path,
        refusal_anchor_texts_path=app_cfg.artifacts.refusal_anchor_texts_path,
        allow_build_from_texts=_env_bool("RR_BUILD_ANCHOR_FROM_TEXT", False),
    )
    metric_space = EmbeddingMetricSpaceConfig(anchor_bank=anchor_cfg)
    return EmbeddingLayerConfig(
        embedding_config=app_cfg.embedding,
        metric_space=metric_space,
        parquet_base_dir=app_cfg.artifacts.parquet_base_dir,
        privacy=app_cfg.privacy,
        components=app_cfg.components,
    )
