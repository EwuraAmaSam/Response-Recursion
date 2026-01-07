from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import datetime as dt

class PromptSeed(BaseModel):
    seed_id: str
    prompt_text: str
    domain: str = "general"
    capability_target: List[str] = Field(default_factory=list)
    expected_failure_modes: List[str] = Field(default_factory=list)
    granularity: Literal["narrow","mid","broad"] = "mid"
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    selection_behavior: Literal[
        "random",
        "round_robin",
        "repeat_with_variation_on_failure",
        "branch_to_adjacent_topic",
    ] = "random"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PromptCandidate(BaseModel):
    prompt_id: str
    parent_id: Optional[str] = None
    prompt_text: str
    strategy: str
    risk_focus: Optional[str] = None
    score_hint: Optional[float] = None
    created_at: str = Field(default_factory=lambda: dt.datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = Field(default_factory=dict)