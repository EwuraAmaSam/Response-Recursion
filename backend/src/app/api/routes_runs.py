from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal

from layers.orchestration.controller import RunController

router = APIRouter()

class RunCreateRequest(BaseModel):
    description: Optional[str] = None
    mode: Literal["single", "loop"] = "loop"
    max_iterations: int = Field(default=5, ge=1, le=200)

class RunStatus(BaseModel):
    run_id: str
    state: Literal["created", "running", "completed", "failed"]
    iteration: int = 0
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    last_prompt: Optional[dict] = None

def _status_from_state(st) -> RunStatus:
    return RunStatus(
        run_id=st.run_id,
        state=st.status,
        iteration=st.iteration,
        last_error=st.error,
        created_at=st.created_at,
        last_prompt=st.last_prompt,
    )

@router.post("/runs", response_model=RunStatus)
def create_run(req: RunCreateRequest):
    ctrl: RunController = router.controller  # type: ignore
    st = ctrl.start_run(description=req.description)
    if req.mode == "single":
        st = ctrl.step(st.run_id)
        st.status = "completed" if st.status != "failed" else "failed"
    else:
        st = ctrl.run_loop(st.run_id, max_iterations=req.max_iterations)
    return _status_from_state(st)

@router.get("/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str):
    ctrl: RunController = router.controller  # type: ignore
    try:
        st = ctrl.get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run_id not found")
    return _status_from_state(st)

@router.post("/runs/{run_id}/step", response_model=RunStatus)
def step_run(run_id: str):
    ctrl: RunController = router.controller  # type: ignore
    try:
        st = ctrl.step(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run_id not found")
    return _status_from_state(st)