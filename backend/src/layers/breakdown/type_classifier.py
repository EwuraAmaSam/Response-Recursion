from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Literal

import re

StatementType = Literal[
    "Instruction",
    "Conditional",
    "Warning/Policy",
    "Definition",
    "Assertion/Fact",
    "Meta-safety",
    "Refusal",
    "Other",
]


@dataclass
class TypedStatement:
    id: str
    text: str
    type: StatementType
    confidence: float
    sentence_index: int
    clause_index: int
    start_char: int
    end_char: int


_TYPE_PATTERNS = {
    "Refusal": [
        r"\b(i\s+cannot\b|\bi\s+can't\b|\bnot able to\b)",
        r"\b(as an ai language model)\b",
    ],
    "Meta-safety": [
        r"\bi do not encourage\b",
        r"\bfor safety reasons\b",
        r"\bthis may be dangerous\b",
    ],
    "Warning/Policy": [
        r"\b(warning|caution|danger|risk|hazard)\b",
        r"\b(policy|regulation|compliance|violation)\b",
        r"\b(prohibited|forbidden|illegal|unlawful)\b",
    ],
    "Instruction": [
        r"\b(should|must|need to|ought to|required to)\b",
        r"^(you|one|they|we)\s+(should|must|need|ought)\b",
    ],
    "Conditional": [
        r"\b(if|when|unless|provided that|assuming|given that)\b",
        r"\b(then|else|otherwise)\b",
    ],
    "Definition": [
        r"\b(is|are|means|refers to|defined as|denotes)\b",
        r"\b(consists of|comprises|constitutes)\b",
        r"^(a|an|the)\s+\w+\s+(is|are)\b",
    ],
    "Assertion/Fact": [
        r"\b(is|are|was|were|will be|can be|cannot be)\b",
    ],
}


class StatementTyper:
    """Rule-first statement type classifier."""

    def __init__(self) -> None:
        pass

    def classify(self, statements: List[Dict[str, Any]]) -> List[TypedStatement]:
        typed: List[TypedStatement] = []

        for st in statements:
            text = st["text"]
            st_type: StatementType = "Other"
            confidence = 0.3  # default low confidence

            lowered = text.lower()

            for candidate_type, patterns in _TYPE_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, lowered, flags=re.IGNORECASE):
                        st_type = candidate_type  # type: ignore[assignment]
                        confidence = 0.9
                        break
                if confidence >= 0.9:
                    break

            typed.append(
                TypedStatement(
                    id=st["id"],
                    text=text,
                    type=st_type,
                    confidence=confidence,
                    sentence_index=st["sentence_index"],
                    clause_index=st["clause_index"],
                    start_char=st["start_char"],
                    end_char=st["end_char"],
                )
            )

        return typed

