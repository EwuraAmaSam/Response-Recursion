from __future__ import annotations

from fastapi import FastAPI

from core.config import load_app_config
from core.logging import setup_logging
from layers.orchestration.controller import RunController
from app.api.routes_runs import router as runs_router

def create_app() -> FastAPI:
    setup_logging()
    cfg = load_app_config()
    ctrl = RunController(cfg)

    app = FastAPI(title="Response-Recursion Backend", version=cfg.version)
    runs_router.controller = ctrl  # type: ignore
    app.include_router(runs_router, prefix="/api", tags=["runs"])
    return app

app = create_app()