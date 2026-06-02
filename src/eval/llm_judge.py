"""LLM-as-judge for disputed REVIEW rows (P2.1, cost-capped)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Optional

from models.semantic_eval_settings import LLMJudgeSettings

logger = logging.getLogger(__name__)

_REQUESTS_MODULE = None


def _get_requests():
    global _REQUESTS_MODULE
    if _REQUESTS_MODULE is None:
        _REQUESTS_MODULE = import_module("requests")
    return _REQUESTS_MODULE


@dataclass
class JudgeVerdict:
    verdict: str
    rationale: str
    backend: str
    policy: str = "review_tiebreak"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "backend": self.backend,
            "policy": self.policy,
        }


class ReviewJudgeBudget:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max(0, max_calls)
        self.used = 0

    def consume(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True


def _simulated_verdict(row: Dict[str, Any], settings: LLMJudgeSettings) -> JudgeVerdict:
    raw_contract = row.get("contract")
    contract: Dict[str, Any] = raw_contract if isinstance(raw_contract, dict) else {}
    raw_arbitration = row.get("arbitration")
    arbitration: Dict[str, Any] = raw_arbitration if isinstance(raw_arbitration, dict) else {}
    semantic_errors = contract.get("semantic_errors") or arbitration.get("semantic_errors") or []

    if semantic_errors:
        return JudgeVerdict(
            verdict="NO_GO",
            rationale="Simulated judge: semantic contract errors present on REVIEW row.",
            backend="simulated_judge",
        )
    if contract.get("invalid_label") or contract.get("code_mismatch"):
        return JudgeVerdict(
            verdict="NO_GO",
            rationale="Simulated judge: invalid label or code mismatch.",
            backend="simulated_judge",
        )
    if settings.tie_break == "conservative":
        return JudgeVerdict(
            verdict="NO_GO",
            rationale="Simulated judge: conservative tie-break on unresolved REVIEW.",
            backend="simulated_judge",
        )
    return JudgeVerdict(
        verdict="REVIEW",
        rationale="Simulated judge: keep REVIEW when no hard semantic signal.",
        backend="simulated_judge",
    )


def _ollama_verdict(row: Dict[str, Any], judge_cfg: Dict[str, Any]) -> JudgeVerdict:
    host = str(judge_cfg.get("host", "http://localhost:11434")).rstrip("/")
    model = str(judge_cfg.get("model", "llama3.2"))
    timeout_s = float(judge_cfg.get("timeout_s", 30))
    metrics = row.get("metrics", {})
    decision = row.get("decision", {})
    prompt = (
        "You are a QA release judge. Given metrics and model inference JSON, "
        "return STRICT JSON: {\"verdict\": \"GO\"|\"NO_GO\"|\"REVIEW\", \"rationale\": \"...\"}.\n"
        f"metrics={json.dumps(metrics, ensure_ascii=False)}\n"
        f"inference={json.dumps(decision, ensure_ascii=False)}\n"
        f"contract={json.dumps(row.get('contract', {}), ensure_ascii=False)}\n"
    )
    response = _get_requests().post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    raw = str(response.json().get("response", "{}"))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    verdict = str(parsed.get("verdict", "REVIEW")).upper()
    if verdict not in {"GO", "NO_GO", "REVIEW"}:
        verdict = "REVIEW"
    return JudgeVerdict(
        verdict=verdict,
        rationale=str(parsed.get("rationale", "ollama judge")),
        backend="ollama_judge",
    )


def judge_review_row(
    row: Dict[str, Any],
    *,
    settings: LLMJudgeSettings,
    judge_cfg: Dict[str, Any],
    budget: ReviewJudgeBudget,
) -> Optional[JudgeVerdict]:
    if not settings.enabled or not budget.consume():
        return None
    logger.info(
        "llm_judge: reviewing row file=%s mode=%s (%s/%s)",
        row.get("file"),
        settings.mode,
        budget.used,
        budget.max_calls,
    )
    if settings.mode == "ollama":
        try:
            return _ollama_verdict(row, judge_cfg)
        except Exception as exc:
            logger.warning("llm_judge: ollama failed (%s), fallback simulated", exc)
            return _simulated_verdict(row, settings)
    return _simulated_verdict(row, settings)
