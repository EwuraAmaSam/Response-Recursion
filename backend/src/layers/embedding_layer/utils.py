from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional

import numpy as np


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return num / denom


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object extraction.

    The judge is instructed to output strict JSON. In practice, some
    generation backends may wrap it with prose; this pulls the first
    {...} block and parses it.
    """
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Common recovery: remove trailing commas
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(blob2)
        except json.JSONDecodeError:
            return None
