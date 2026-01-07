from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import re


@dataclass
class CoreStatement:
    id: str
    text: str
    sentence_index: int
    clause_index: int
    start_char: int
    end_char: int


class SemanticMerger:
    """
    Merge/split clauses into minimal semantic units (Core Statements).

    Current implementation:
    - drops obviously empty/boilerplate fragments
    - merges extremely short trailing fragments into previous clause
    You can later extend this with embedding similarity based grouping.
    """

    def __init__(self, min_len: int = 5) -> None:
        self.min_len = min_len

    def process_clauses(self, clauses: List[Dict[str, Any]]) -> List[CoreStatement]:
        core_statements: List[CoreStatement] = []
        buffer_text: Optional[str] = None
        buffer_meta: Optional[Dict[str, Any]] = None
        idx = 0

        for cl in clauses:
            text = cl["text"].strip()
            if not text:
                continue

            # Remove trivial boilerplate fragments
            if re.match(r"^(as an ai language model|sorry,? i cannot)", text, re.IGNORECASE):
                continue

            if buffer_text is None:
                buffer_text = text
                buffer_meta = cl
                continue

            if len(text) < self.min_len:
                # Merge very short continuation into previous statement
                buffer_text = buffer_text + " " + text
            else:
                # Flush buffer as a statement
                core_statements.append(
                    CoreStatement(
                        id=f"CS_{idx}",
                        text=buffer_text,
                        sentence_index=buffer_meta["sentence_index"],
                        clause_index=buffer_meta["clause_index"],
                        start_char=buffer_meta["start_char"],
                        end_char=buffer_meta["end_char"],
                    )
                )
                idx += 1
                buffer_text = text
                buffer_meta = cl

        # Flush remaining buffer
        if buffer_text is not None and buffer_meta is not None:
            core_statements.append(
                CoreStatement(
                    id=f"CS_{idx}",
                    text=buffer_text,
                    sentence_index=buffer_meta["sentence_index"],
                    clause_index=buffer_meta["clause_index"],
                    start_char=buffer_meta["start_char"],
                    end_char=buffer_meta["end_char"],
                )
            )

        return core_statements

