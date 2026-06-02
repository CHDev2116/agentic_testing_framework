from eval.llm_judge import ReviewJudgeBudget, judge_review_row
from models.semantic_eval_settings import LLMJudgeSettings


def test_judge_review_row_respects_budget():
    settings = LLMJudgeSettings(enabled=True, max_calls_per_batch=1, mode="simulated")
    budget = ReviewJudgeBudget(1)
    row = {
        "file": "x.jpg",
        "contract": {"semantic_errors": ["semantic: bad"]},
        "metrics": {"avg_brightness": 80.0, "sharpness": 50.0},
        "decision": {"decision": "Blurry", "code": "ERR_OPTIC_SHRP_001"},
    }
    first = judge_review_row(row, settings=settings, judge_cfg={}, budget=budget)
    second = judge_review_row(row, settings=settings, judge_cfg={}, budget=budget)
    assert first is not None
    assert first.verdict == "NO_GO"
    assert second is None


def test_judge_disabled_returns_none():
    settings = LLMJudgeSettings(enabled=False)
    budget = ReviewJudgeBudget(5)
    verdict = judge_review_row(
        {"file": "x.jpg"},
        settings=settings,
        judge_cfg={},
        budget=budget,
    )
    assert verdict is None
