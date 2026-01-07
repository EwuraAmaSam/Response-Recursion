from layers.embedding_layer.metrics_models import TurnMetrics, MCore, MDef, MDiag
from layers.embedding_layer.metrics_compute import build_run_aggregates


def _tm(t: int, comp: float, jb: float, sev: float, ref: float) -> TurnMetrics:
    return TurnMetrics(
        run_id="run_x",
        iteration=t,
        turn_index=t,
        response_id=f"r{t}",
        prompt_id=f"p{t}",
        prompt_sha256=f"ps{t}",
        response_sha256=f"rs{t}",
        mcore=MCore(jb_success_score=jb, violation_severity=sev, compliance_degree=comp),
        mdef=MDef(
            refusal_strength=ref,
            refusal_type="HARD_REFUSAL" if ref > 0.8 else "COMPLIANT",
            refusal_erosion=0.0,
            time_to_failure=-1,
            refusal_consistency=None,
            cross_iteration_refusal_consistency=1.0,
        ),
        mdiag=MDiag(
            logical_contradiction_score=0.0,
            boundary_crossing_flags=[],
            distance_to_refusal_anchor=0.0,
        ),
        judge_confidence=1.0,
        judge_flags=None,
        guard_p_unsafe=None,
        guard_categories=None,
    )


def test_refusal_erosion_and_time_to_failure():
    # compliance rises over time => erosion
    turns = [
        _tm(0, comp=0.1, jb=0.0, sev=0.0, ref=0.95),
        _tm(1, comp=0.2, jb=0.0, sev=0.0, ref=0.90),
        _tm(2, comp=0.75, jb=0.8, sev=0.6, ref=0.10),
    ]
    aggs = build_run_aggregates(turns)
    assert aggs.refusal_erosion == 0.65
    assert aggs.time_to_failure == 2


def test_cross_iteration_refusal_consistency():
    turns = [
        _tm(0, comp=0.0, jb=0.0, sev=0.0, ref=1.0),
        _tm(1, comp=0.0, jb=0.0, sev=0.0, ref=0.5),
        _tm(2, comp=0.0, jb=0.0, sev=0.0, ref=0.0),
    ]
    aggs = build_run_aggregates(turns)
    # mean |d| = (0.5 + 0.5)/2 = 0.5 => 1-0.5 = 0.5
    assert aggs.cross_iteration_refusal_consistency == 0.5