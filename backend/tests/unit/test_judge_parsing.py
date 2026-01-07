from services.judge_client import RubricJudge
from services.llm_client import GenerationConfig, TextGenerationClient


class StubClient(TextGenerationClient):
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        return self._text


def test_rubric_judge_parses_embedded_json_and_clamps():
    raw = (
        "Here you go!\n"
        "{\n"
        "  \"jb_success_score\": 1.2,\n"
        "  \"violation_severity\": -0.5,\n"
        "  \"compliance_degree\": 0.33,\n"
        "  \"judge_confidence\": 0.9,\n"
        "  \"flags\": [\"x\", \"y\"],\n"
        "}\n"
        "extra text"
    )
    judge = RubricJudge(StubClient(raw))
    res = judge.score(prompt_text="p", response_text="r")
    assert res.jb_success_score == 1.0
    assert res.violation_severity == 0.0
    assert res.compliance_degree == 0.33
    assert res.judge_confidence == 0.9
    assert res.flags == ["x", "y"]


def test_rubric_judge_handles_no_json():
    judge = RubricJudge(StubClient("no json here"))
    res = judge.score(prompt_text="p", response_text="r")
    assert res.jb_success_score == 0.0
    assert res.violation_severity == 0.0
    assert res.compliance_degree == 0.0
    assert res.judge_confidence == 0.0
    assert res.flags is None
