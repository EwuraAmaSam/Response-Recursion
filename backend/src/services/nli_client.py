from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from layers.embedding_layer.utils import clamp


@dataclass(frozen=True)
class NLIResult:
    contradiction_prob: float


class NLIContradictionScorer:
    """Transformer NLI scorer that returns P(contradiction)."""

    def __init__(self, model_name_or_path: str, *, device: Optional[int] = None) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError("transformers is required for NLI scoring") from e

        # Use text-classification pipeline for MNLI-style models.
        self._pipe = pipeline(
            "text-classification",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device=device if device is not None else -1,
            return_all_scores=True,
        )

    def score(self, *, premise: str, hypothesis: str) -> NLIResult:
        if not premise.strip() or not hypothesis.strip():
            return NLIResult(contradiction_prob=0.0)
        out = self._pipe({"text": premise, "text_pair": hypothesis})
        scores = out[0] if isinstance(out, list) and out else []
        # Normalize labels
        prob = 0.0
        for item in scores:
            label = str(item.get("label", "")).upper()
            if "CONTR" in label:
                prob = float(item.get("score", 0.0))
                break
        return NLIResult(contradiction_prob=clamp(prob))


class MockNLIContradictionScorer(NLIContradictionScorer):
    def __init__(self) -> None:
        self._pipe = None  # type: ignore

    def score(self, *, premise: str, hypothesis: str) -> NLIResult:
        # If both non-empty and share negation markers, pretend contradiction.
        p = (premise or "").lower()
        h = (hypothesis or "").lower()
        if ("cannot" in p or "can't" in p or "won't" in p) and ("do" in h or "steps" in h):
            return NLIResult(contradiction_prob=0.7)
        return NLIResult(contradiction_prob=0.1)