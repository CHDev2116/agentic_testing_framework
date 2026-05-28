from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class InferenceOutput:
    decision: str
    code: str
    msg: str
    confidence: Optional[float] = None
    backend: Optional[str] = None

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        default_msg: str,
        default_decision: str = "Error",
        default_code: str = "ERR_MODEL_RESPONSE_422",
        backend: Optional[str] = None,
    ) -> "InferenceOutput":
        if not isinstance(payload, dict):
            return cls(
                decision=default_decision,
                code=default_code,
                msg=default_msg,
                backend=backend,
            )

        confidence: Optional[float] = None
        raw_confidence = payload.get("confidence")
        if raw_confidence is not None:
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = None

        return cls(
            decision=str(payload.get("decision", default_decision)),
            code=str(payload.get("code", default_code)),
            msg=str(payload.get("msg", default_msg)),
            confidence=confidence,
            backend=backend,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "decision": self.decision,
            "code": self.code,
            "msg": self.msg,
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        if self.backend:
            data["backend"] = self.backend
        return data


@dataclass(frozen=True)
class LoopbackPlan:
    action: Optional[str]
    stop_reason: str
    rationale: str
    fallback_used: bool = False
    planner_backend: str = "simulated"


class AgentStep(BaseModel):
    attempt: int = Field(..., description="Current retry round, starts from 1")
    signal: Literal["under", "over", "blurry", "other"] = Field(
        ..., description="Image signal emitted by evaluator"
    )
    action: Literal["brighten", "dim", "sharpen", "stop"] = Field(
        ..., description="Planner action taken for this step"
    )
    rationale: str = Field(..., description="Why planner selected this action")
    fallback_used: bool = Field(
        default=False,
        description="True when planner falls back from llm to simulated rules",
    )
    metrics_before: Dict[str, Any] = Field(
        default_factory=dict, description="Metrics before executing this step action"
    )
    metrics_after: Optional[Dict[str, Any]] = Field(
        default=None, description="Metrics observed after action is executed"
    )
    latency_ms: float = Field(..., description="Step latency in milliseconds")


class AgentInferenceOutput(BaseModel):
    image_path: str
    final_decision: Literal["GO", "REVIEW", "NO_GO"] = Field(
        ..., description="Final release decision for this image"
    )
    error_code: str = Field(
        default="SUCCESS_200", description="Machine-oriented decision/error code"
    )
    error_message: str = Field(
        default="Optimal", description="Human-oriented decision/error message"
    )
    steps: List[AgentStep] = Field(
        default_factory=list, description="Per-image agent decision trace"
    )
    total_latency_ms: float = Field(..., description="Total latency for the image")

    @model_validator(mode="after")
    def verify_latency_consistency(self) -> "AgentInferenceOutput":
        steps_latency = sum(step.latency_ms for step in self.steps)
        # Allow tiny floating-point rounding noise.
        if self.total_latency_ms + 1e-6 < steps_latency:
            raise ValueError(
                f"Total latency ({self.total_latency_ms}ms) cannot be less than "
                f"the sum of individual steps ({steps_latency}ms)."
            )
        return self
