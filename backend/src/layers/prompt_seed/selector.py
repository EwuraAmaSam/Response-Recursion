from __future__ import annotations

from typing import Optional
import random

from .schema import SeedLibrary
from models.prompts import PromptSeed

class SeedSelector:
    def __init__(self, library: SeedLibrary, strategy: str = "random") -> None:
        self.library = library
        self.strategy = strategy
        self._rr_idx = 0

    def select(self, *, difficulty_bias: Optional[float] = None) -> PromptSeed:
        seeds = self.library.seeds
        if not seeds:
            raise ValueError("Seed library is empty.")

        if self.strategy == "round_robin":
            seed = seeds[self._rr_idx % len(seeds)]
            self._rr_idx += 1
            return seed

        if difficulty_bias is not None:
            weights = [max(1e-3, 1.0 - abs(s.difficulty - difficulty_bias)) for s in seeds]
            return random.choices(seeds, weights=weights, k=1)[0]

        return random.choice(seeds)