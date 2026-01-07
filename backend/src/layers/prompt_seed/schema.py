from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from models.prompts import PromptSeed

class SeedLibrary(BaseModel):
    version: str = "0.1"
    seeds: List[PromptSeed] = Field(default_factory=list)