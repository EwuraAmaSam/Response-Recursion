from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import re

try:
    import spacy  # type: ignore
except ImportError:
    spacy = None


_SPACY_CACHE: Dict[str, Any] = {}


def _get_nlp(model_name: str = "en_core_web_sm"):
    """Lazy-load and cache a spaCy model, falling back gracefully if unavailable."""
    if spacy is None:
        return None

    if model_name in _SPACY_CACHE:
        return _SPACY_CACHE[model_name]

    try:
        nlp = spacy.load(model_name)
    except OSError:
        # If the model is not installed, fall back to the small English pipeline
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Final fallback: blank pipeline, we'll just do regex splitting
            nlp = spacy.blank("en")

    _SPACY_CACHE[model_name] = nlp
    return nlp


@dataclass
class Sentence:
    text: str
    start_char: int
    end_char: int
    index: int


@dataclass
class Clause:
    text: str
    start_char: int
    end_char: int
    sentence_index: int
    clause_index: int


class SentenceSegmenter:
    """Segment raw text into sentences and clauses."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self.nlp = _get_nlp(model_name)

    def segment_sentences(self, text: str) -> List[Sentence]:
        """Segment into sentences; keeps char offsets."""
        text = text.strip()
        if not text:
            return []

        if self.nlp is None or not getattr(self.nlp, "pipe_names", None):
            # Very simple fallback segmentation
            parts = re.split(r"(?<=[.!?])\s+", text)
            offset = 0
            sentences: List[Sentence] = []
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                start = text.find(part, offset)
                end = start + len(part)
                sentences.append(Sentence(text=part, start_char=start, end_char=end, index=i))
                offset = end
            return sentences

        doc = self.nlp(text)
        sentences: List[Sentence] = []
        for i, sent in enumerate(doc.sents):
            sentences.append(
                Sentence(
                    text=sent.text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    index=i,
                )
            )
        return sentences

    def decompose_clauses(self, sentences: List[Sentence]) -> List[Clause]:
        """
        Naive clause decomposition: split on coordinating conjunctions and 'if/when/because/...'
        while keeping offsets. This is intentionally simple but structurally correct.
        """
        clauses: List[Clause] = []
        clause_idx = 0

        splitter = re.compile(
            r"\b(and|but|or|so|because|if|when|while|whereas|although|however)\b",
            flags=re.IGNORECASE,
        )

        for sent in sentences:
            text = sent.text
            base = sent.start_char

            # Use regex to split but keep delimiters as part of next clause to avoid losing semantics
            parts = splitter.split(text)
            current = ""
            local_offset = 0

            def flush_current(cur: str, start_local: int, end_local: int):
                nonlocal clause_idx
                cur = cur.strip()
                if not cur:
                    return
                clauses.append(
                    Clause(
                        text=cur,
                        start_char=base + start_local,
                        end_char=base + end_local,
                        sentence_index=sent.index,
                        clause_index=clause_idx,
                    )
                )
                clause_idx += 1

            i = 0
            while i < len(parts):
                part = parts[i]
                if i + 1 < len(parts) and splitter.match(parts[i + 1] or ""):
                    # part (clause) + conj
                    conj = parts[i + 1]
                    segment = (part + " " + conj).strip()
                    start_local = text.find(part, local_offset)
                    end_local = start_local + len(segment)
                    flush_current(segment, start_local, end_local)
                    local_offset = end_local
                    i += 2
                else:
                    segment = part.strip()
                    start_local = text.find(segment, local_offset)
                    end_local = start_local + len(segment)
                    flush_current(segment, start_local, end_local)
                    local_offset = end_local
                    i += 1

        return clauses

