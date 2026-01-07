"""
Iterative Breakdown Layer (IBL) – public interface.

Exports a single convenience function `BREAKDOWN(...)` that the orchestration
layer can call, plus the main pipeline class.
"""

from .pipeline import BreakdownPipeline, BREAKDOWN

__all__ = ["BreakdownPipeline", "BREAKDOWN"]

