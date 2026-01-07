from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Optional, List

import re
from datetime import datetime

from .segmenter import SentenceSegmenter
from .merger import SemanticMerger, CoreStatement
from .type_classifier import StatementTyper
from .extractors import BreakdownExtractor


def _basic_preprocess(text: str) -> str:
    """
    Minimal preprocessing:
    - normalize whitespace
    - keep quotes/examples
    - strip obvious boilerplate prefixes
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Drop leading boilerplate like "As an AI language model, ..."
    text = re.sub(
        r"^As an AI language model,?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


class BreakdownPipeline:
    """
    Main Iterative Breakdown Layer pipeline.

    Contract (input):
        {
          "t": int,
          "run_id": str,
          "prompt_id": str,
          "response_id": str,
          "prompt_text": str,
          "response_text": str,
          # optional:
          "prior_core_statement_ids": [str, ...]
        }

    Contract (output): unified record compatible with both the embedding
    system and the prompt generator, as per the spec.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.segmenter = SentenceSegmenter(model_name=spacy_model)
        self.merger = SemanticMerger()
        self.typer = StatementTyper()
        self.extractor = BreakdownExtractor()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # ----- unpack & preprocess -----
        response_text: str = payload.get("response_text", "") or ""
        prompt_text: str = payload.get("prompt_text", "") or ""
        t: int = int(payload.get("t", 0))
        run_id: str = payload.get("run_id", "run-unknown")
        prompt_id: str = payload.get("prompt_id", "prompt-unknown")
        response_id: str = payload.get("response_id", "response-unknown")
        prior_ids: Optional[List[str]] = payload.get("prior_core_statement_ids")

        preprocessed = _basic_preprocess(response_text)

        # ----- sentence & clause segmentation -----
        sentences = self.segmenter.segment_sentences(preprocessed)
        if not sentences:
            return self._empty_output(
                run_id=run_id,
                t=t,
                prompt_id=prompt_id,
                response_id=response_id,
            )

        clauses = [
            {
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "sentence_index": c.sentence_index,
                "clause_index": c.clause_index,
            }
            for c in self.segmenter.decompose_clauses(sentences)
        ]

        # ----- semantic merge -> core statements -----
        core_statements: List[CoreStatement] = self.merger.process_clauses(clauses)
        if not core_statements:
            return self._empty_output(
                run_id=run_id,
                t=t,
                prompt_id=prompt_id,
                response_id=response_id,
            )

        core_dicts: List[Dict[str, Any]] = [
            {
                "id": cs.id,
                "text": cs.text,
                "sentence_index": cs.sentence_index,
                "clause_index": cs.clause_index,
                "start_char": cs.start_char,
                "end_char": cs.end_char,
            }
            for cs in core_statements
        ]

        # ----- statement typing -----
        typed = self.typer.classify(core_dicts)

        typed_dicts: List[Dict[str, Any]] = [
            {
                "id": ts.id,
                "text": ts.text,
                "type": ts.type,
                "confidence": ts.confidence,
                "sentence_index": ts.sentence_index,
                "clause_index": ts.clause_index,
                "start_char": ts.start_char,
                "end_char": ts.end_char,
            }
            for ts in typed
        ]

        # ----- extract all downstream structures -----
        embedding_units = [
            asdict(u) for u in self.extractor.build_embedding_units(core_dicts)
        ]
        risk_targets = [
            asdict(rt) for rt in self.extractor.build_risk_targets(typed_dicts)
        ]
        action_frames = [
            asdict(af)
            for af in self.extractor.build_action_frames(
                typed_dicts,
                [rt for rt in self.extractor.build_risk_targets(typed_dicts)]
                or [],
            )
        ]
        refusal_evidence = [
            asdict(rev) for rev in self.extractor.build_refusal_evidence(typed_dicts)
        ]
        attack_signal = asdict(self.extractor.build_attack_signal(typed_dicts))

        delta = self.extractor.compute_delta(
            current_ids=[cs.id for cs in core_statements],
            prior_ids=prior_ids,
        )

        # ----- summary & quality flags -----
        summary = self._build_summary(core_statements, risk_targets, refusal_evidence)
        quality_flags = self._build_quality_flags(response_text, preprocessed)

        # ----- assemble unified output -----
        output: Dict[str, Any] = {
            "statements": typed_dicts,
            "embedding_units": embedding_units,
            "risk_targets": risk_targets,
            "action_frames": action_frames,
            "refusal_evidence": refusal_evidence,
            "attack_signal": attack_signal,
            "summary": summary,
            "delta": delta,
            "quality_flags": quality_flags,
            "provenance": {
                "run_id": run_id,
                "iteration": t,
                "prompt_id": prompt_id,
                "response_id": response_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
        }
        return output

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_output(
        run_id: str,
        t: int,
        prompt_id: str,
        response_id: str,
    ) -> Dict[str, Any]:
        return {
            "statements": [],
            "embedding_units": [],
            "risk_targets": [],
            "action_frames": [],
            "refusal_evidence": [],
            "attack_signal": {"level": "none", "patterns": [], "support_cs": []},
            "summary": "No core statements extracted.",
            "delta": {
                "new": [],
                "kept": [],
                "removed": [],
                "label": "empty",
            },
            "quality_flags": {"empty_response": True},
            "provenance": {
                "run_id": run_id,
                "iteration": t,
                "prompt_id": prompt_id,
                "response_id": response_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
        }

    @staticmethod
    def _build_summary(
        core_statements: List[CoreStatement],
        risk_targets: List[Dict[str, Any]],
        refusal_evidence: List[Dict[str, Any]],
    ) -> str:
        n = len(core_statements)
        if n == 0:
            return "No core statements extracted."

        has_unsafe = any(rt["risk_tag"] == "unsafe" for rt in risk_targets)
        has_refusal = len(refusal_evidence) > 0

        parts = [f"{n} core statements extracted."]
        if has_unsafe:
            parts.append("At least one unsafe or high-risk concept identified.")
        if has_refusal:
            parts.append("Refusal or meta-safety language detected.")
        return " ".join(parts)

    @staticmethod
    def _build_quality_flags(original: str, preprocessed: str) -> Dict[str, Any]:
        flags: Dict[str, Any] = {}
        if len(original.strip()) == 0:
            flags["empty_original"] = True
        if len(preprocessed) < 20:
            flags["very_short"] = True
        if len(original) - len(preprocessed) > 200:
            flags["heavy_preprocessing"] = True
        return flags


def BREAKDOWN(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience functional API, mirroring the original spec:

        output = BREAKDOWN({
            "t": 0,
            "run_id": "...",
            "prompt_id": "...",
            "response_id": "...",
            "prompt_text": "...",
            "response_text": "...",
        })
    """
    pipeline = BreakdownPipeline()
    return pipeline.run(payload)

