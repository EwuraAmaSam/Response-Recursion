from __future__ import annotations

from enum import Enum
from typing import Final, Set


class RefusalType(str, Enum):
    HARD_REFUSAL = "HARD_REFUSAL"
    REFUSAL_WITH_SAFE_ALTERNATIVE = "REFUSAL_WITH_SAFE_ALTERNATIVE"
    PARTIAL_REFUSAL = "PARTIAL_REFUSAL"
    CONDITIONAL_COMPLIANCE = "CONDITIONAL_COMPLIANCE"
    EVASIVE_DEFLECT = "EVASIVE_DEFLECT"
    COMPLIANT = "COMPLIANT"


CATEGORY_VIOLATION_PREFIX: Final[str] = "CATEGORY_VIOLATION_"

BOUNDARY_FLAG_VOCAB_BASE: Final[Set[str]] = {
    "INSTRUCTION_HIERARCHY_CONFLICT",
    "REFUSAL_WEAKENING_MARKERS",
    "PARTIAL_COMPLIANCE_AFTER_REFUSAL",
    "CONDITIONAL_COMPLIANCE_MARKERS",
    "EVASIVE_NONANSWER",
    "SAFE_ALTERNATIVE_GIVEN",
    "SYSTEM_PROMPT_PROBE_PATTERN",
    "FORMAT_OBFUSCATION_SIGNAL",
    "MULTI_TURN_ESCALATION_PATTERN",
    # Optional / best-effort:
    "LANGUAGE_SWITCH_OR_TRANSLATION",
    "ENUMERATED_ACTION_PLAN_DETECTED",
}

# Alias used across the codebase
ALL_BOUNDARY_FLAGS: Final[Set[str]] = set(BOUNDARY_FLAG_VOCAB_BASE)


def is_valid_boundary_flag(flag: str) -> bool:
    """Validate against the fixed vocabulary.

    CATEGORY_VIOLATION_<X> is allowed for any non-empty <X>.
    """
    if flag in BOUNDARY_FLAG_VOCAB_BASE:
        return True
    return flag.startswith(CATEGORY_VIOLATION_PREFIX) and len(flag) > len(CATEGORY_VIOLATION_PREFIX)
