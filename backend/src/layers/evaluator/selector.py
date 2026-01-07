from __future__ import annotations

from typing import Dict, Any, List
import datetime as dt

def select_prompt(candidates_envelope: Dict[str, Any]) -> Dict[str, Any]:
    run_id = candidates_envelope["run_id"]
    iteration = int(candidates_envelope["iteration"])
    payload = candidates_envelope["payload"]
    cands: List[Dict[str, Any]] = payload.get("prompt_candidates", []) or []
    if not cands:
        raise ValueError("No prompt candidates to select from.")

    def score(c: Dict[str, Any]) -> float:
        v = c.get("score_hint")
        try:
            return float(v)
        except Exception:
            return 0.0

    best = max(cands, key=score)

    out_payload = {
        "type": "selected_prompt_to_orchestration",
        "selected_prompt": best,
        "selection_strategy": "argmax_score_hint",
    }

    return {
        "version": candidates_envelope.get("version", "0.2"),
        "run_id": run_id,
        "iteration": iteration,
        "from_layer": "evaluator",
        "to_layer": "orchestration",
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "payload": out_payload,
    }