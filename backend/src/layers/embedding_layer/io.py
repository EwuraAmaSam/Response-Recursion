from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class EmbeddingRecord:
    """
    Single embedding row for Parquet.

    This is how embeddings produced from the breakdown layer's
    embedding_units are persisted, keyed by run/iteration/response/unit.
    """
    run_id: str
    iteration: int
    response_id: str
    unit_id: str         # typically cs_id from embedding_units
    text_sha256: str     # hash of the embedded text (no raw text in public parquet)
    embedding: np.ndarray
    embedding_model: str
    created_at: dt.datetime

    def to_row_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base["embedding"] = self.embedding.tolist()
        base["created_at"] = self.created_at.isoformat()
        return base


@dataclass
class MetricRecord:
    """
    Single per-response metric bundle to persist.

    This captures the jailbreak-focused misuse metric set:
      - Mcore
      - Mdef
      - Mdiag
    for a particular response in a given run & iteration.
    """
    run_id: str
    iteration: int
    turn_index: int
    prompt_id: str
    response_id: str
    prompt_sha256: str
    response_sha256: str
    Mcore: Dict[str, Any]
    Mdef: Dict[str, Any]
    Mdiag: Dict[str, Any]
    judge_confidence: float | None = None
    judge_flags: List[str] | None = None
    guard_p_unsafe: float | None = None
    guard_categories: List[str] | None = None
    created_at: dt.datetime

    def to_row_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base["created_at"] = self.created_at.isoformat()
        return base


class EmbeddingStore:
    """
    Handles persistence of embeddings to Parquet.

    Directory layout (relative to configured base_dir):
      base_dir/embeddings/run_{run_id}_iter_{iteration}.parquet
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.embeddings_dir = self.base_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, run_id: str, iteration: int) -> Path:
        return self.embeddings_dir / f"run_{run_id}_iter_{iteration}.parquet"

    def save_embeddings(self, records: Iterable[EmbeddingRecord]) -> Path:
        records_list = list(records)
        if not records_list:
            raise ValueError("No embedding records provided to save.")

        rows = [r.to_row_dict() for r in records_list]
        df = pd.DataFrame(rows)

        run_id = records_list[0].run_id
        iteration = records_list[0].iteration
        path = self._parquet_path(run_id, iteration)

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)

        return path

    def load_embeddings(self, run_id: str, iteration: int) -> pd.DataFrame:
        path = self._parquet_path(run_id, iteration)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        table = pq.read_table(path)
        return table.to_pandas()


class MetricStore:
    """
    Handles persistence of metrics to Parquet.

    Directory layout:
      base_dir/metrics/run_{run_id}_iter_{iteration}.parquet
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.metrics_dir = self.base_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, run_id: str, iteration: int) -> Path:
        return self.metrics_dir / f"run_{run_id}_iter_{iteration}.parquet"

    def save_metrics(self, records: Iterable[MetricRecord]) -> Path:
        records_list = list(records)
        if not records_list:
            raise ValueError("No metric records provided to save.")

        rows = [r.to_row_dict() for r in records_list]
        df = pd.DataFrame(rows)

        run_id = records_list[0].run_id
        iteration = records_list[0].iteration
        path = self._parquet_path(run_id, iteration)

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)

        return path

    def load_metrics(self, run_id: str, iteration: int) -> pd.DataFrame:
        path = self._parquet_path(run_id, iteration)
        if not path.exists():
            raise FileNotFoundError(f"Metric file not found: {path}")
        table = pq.read_table(path)
        return table.to_pandas()

    def list_iterations(self, run_id: str) -> List[int]:
        iters: List[int] = []
        for p in self.metrics_dir.glob(f"run_{run_id}_iter_*.parquet"):
            try:
                s = p.stem
                # run_{run_id}_iter_{iteration}
                it = int(s.rsplit("_", 1)[-1])
                iters.append(it)
            except Exception:
                continue
        return sorted(set(iters))

    def load_all_metrics(self, run_id: str) -> pd.DataFrame:
        parts: List[pd.DataFrame] = []
        for it in self.list_iterations(run_id):
            try:
                parts.append(self.load_metrics(run_id, it))
            except FileNotFoundError:
                continue
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def load_metrics_upto(self, run_id: str, iteration: int) -> pd.DataFrame:
        """Load all metrics parquet files for run_id with iter <= iteration."""
        dfs: List[pd.DataFrame] = []
        for p in sorted(self.metrics_dir.glob(f"run_{run_id}_iter_*.parquet")):
            try:
                it = int(p.stem.split("_iter_")[-1])
            except Exception:
                continue
            if it <= int(iteration):
                dfs.append(pq.read_table(p).to_pandas())
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

