"""Pytest configuration.

The backend code is structured so that imports like `from layers...` work when
`backend/src` is on `PYTHONPATH`.

When running tests from repo root, we add it explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "backend" / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
