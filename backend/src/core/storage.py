from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import datetime as dt

import numpy as np
import pandas as pd

from layers.embedding_layer.io import MetricStore, EmbeddingStore

@dataclass(frozen=True)
class StorageRefs:
    embeddings_parquet: str
    metrics_parquet: str
    risk_surface_dir: str
    prompts_parquet: str

class RiskSurfaceStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.dir = self.base_dir / "risk_surface"
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, run_id: str, iteration: int, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> Dict[str, str]:
        snap_dir = self.dir / f"run_{run_id}" / f"iter_{iteration:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        np.save(snap_dir / "X.npy", X.astype(np.float32))
        np.save(snap_dir / "y.npy", y.astype(np.float32))
        with open(snap_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {"risk_surface_path": str(snap_dir)}

    def load_latest(self, run_id: str) -> Optional[Dict[str, Any]]:
        run_dir = self.dir / f"run_{run_id}"
        if not run_dir.exists():
            return None
        iters = sorted([p for p in run_dir.iterdir() if p.is_dir()])
        if not iters:
            return None
        snap_dir = iters[-1]
        X = np.load(snap_dir / "X.npy")
        y = np.load(snap_dir / "y.npy")
        meta = json.loads((snap_dir / "meta.json").read_text(encoding="utf-8"))
        return {"X": X, "y": y, "meta": meta, "risk_surface_path": str(snap_dir)}

class PromptCandidateStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.dir = self.base_dir / "prompt_candidates"
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_candidates(self, run_id: str, iteration: int, candidates: List[Dict[str, Any]]) -> str:
        path = self.dir / f"run_{run_id}_iter_{iteration:04d}.parquet"
        df = pd.DataFrame(candidates)
        df["created_at"] = dt.datetime.utcnow().isoformat() + "Z"
        df.to_parquet(path, index=False)
        return str(path)