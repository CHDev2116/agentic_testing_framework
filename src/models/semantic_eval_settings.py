"""Eval-time semantic policy knobs (P2.1 companion settings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SemanticEvalSettings:
    enabled: bool = True
    invalid_label_release: str = "NO_GO"
    confidence_violation_policy: str = "review"
    inference_error_release: str = "NO_GO"

    @classmethod
    def from_policy_dict(
        cls,
        policy: Dict[str, Any],
        *,
        enabled: bool = True,
    ) -> "SemanticEvalSettings":
        return cls(
            enabled=enabled,
            invalid_label_release=str(
                policy.get("invalid_label_release", "NO_GO")
            ).upper(),
            confidence_violation_policy=str(
                policy.get("confidence_violation_policy", "review")
            ).lower(),
            inference_error_release=str(
                policy.get("inference_error_release", "NO_GO")
            ).upper(),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SemanticEvalSettings":
        eval_cfg = config.get("eval_settings", {})
        if not isinstance(eval_cfg, dict):
            eval_cfg = {}
        policy = eval_cfg.get("semantic_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        return cls.from_policy_dict(
            policy,
            enabled=bool(eval_cfg.get("semantic_asserts_enabled", True)),
        )

    @classmethod
    def from_oracle_case(cls, case: Dict[str, Any]) -> "SemanticEvalSettings":
        """Per-case overrides in oracle_cases.jsonl (optional semantic_policy block)."""
        if case.get("semantic_policy") is not None:
            policy = case.get("semantic_policy")
            if not isinstance(policy, dict):
                policy = {}
            enabled = case.get("semantic_asserts_enabled", True)
            return cls.from_policy_dict(policy, enabled=bool(enabled))
        return cls()


@dataclass(frozen=True)
class LLMJudgeSettings:
    enabled: bool = False
    max_calls_per_batch: int = 5
    mode: str = "simulated"
    tie_break: str = "conservative"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMJudgeSettings":
        eval_cfg = config.get("eval_settings", {})
        judge_cfg = eval_cfg.get("llm_judge", {}) if isinstance(eval_cfg, dict) else {}
        if not isinstance(judge_cfg, dict):
            judge_cfg = {}
        return cls(
            enabled=bool(judge_cfg.get("enabled", False)),
            max_calls_per_batch=max(0, int(judge_cfg.get("max_calls_per_batch", 5))),
            mode=str(judge_cfg.get("mode", "simulated")).lower(),
            tie_break=str(judge_cfg.get("tie_break", "conservative")).lower(),
        )
