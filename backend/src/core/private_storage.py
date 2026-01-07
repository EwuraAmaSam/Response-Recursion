from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import datetime as dt


@dataclass(frozen=True)
class PrivateArtifactConfig:
    store_raw_text: bool = False
    private_artifacts_dir: Path = Path("private_artifacts")


class PrivateArtifactWriter:
    """Writes raw prompt/response artifacts to a private directory.

    Public parquet artifacts must not contain raw text. This writer exists so
    the user can keep private data local to the evaluation environment.
    """

    def __init__(self, cfg: PrivateArtifactConfig) -> None:
        self.cfg = cfg
        self.base_dir = Path(cfg.private_artifacts_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_turn(
        self,
        *,
        run_id: str,
        iteration: int,
        turn_index: int,
        prompt_id: str,
        response_id: str,
        prompt_text: str,
        response_text: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.cfg.store_raw_text:
            return None

        # Path layout: run_id=<run_id>/iteration=<iteration>/turn=<turn_index>.json
        turn_dir = self.base_dir / f"run_id={run_id}" / f"iteration={iteration}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        path = turn_dir / f"turn={turn_index}.json"

        rec: Dict[str, Any] = {
            "run_id": run_id,
            "iteration": int(iteration),
            "turn_index": int(turn_index),
            "prompt": prompt_text,
            "response": response_text,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return str(path)
