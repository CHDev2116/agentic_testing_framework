"""Pure validation helpers for inference JSON contracts (P1 contract hardening)."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence, Tuple

# Keys the repair path must not silently default (see InferenceOutput.from_payload).
REQUIRED_INFERENCE_KEYS: Tuple[str, ...] = ("decision", "code", "msg")

# Aligned with Ollama/llama prompt templates and LlamaQuantizer outputs.
ALLOWED_DECISIONS = frozenset(
    {
        "Optimal",
        "Blurry",
        "Under-exposed",
        "Over-exposed",
        "Error",
    }
)

ContractErrorClass = Literal["parse", "validation"]

MAX_REPAIR_SNIPPET_CHARS = 2000

DEFAULT_REPAIR_PROMPT_SUFFIX = (
    "Return ONLY a single JSON object. No markdown fences or prose. "
    f"Required keys: {', '.join(REQUIRED_INFERENCE_KEYS)}. "
    f"Valid decisions: {', '.join(sorted(ALLOWED_DECISIONS))}."
)


def validate_inference_payload(payload: Any) -> List[str]:
    """
    Validate a parsed inference dict before lenient normalization.

    Returns a list of human-readable errors; empty list means the payload
    satisfies the strict contract.
    """
    errors: List[str] = []
    if not isinstance(payload, dict):
        errors.append("payload must be a JSON object (dict)")
        return errors
    if not payload:
        errors.append("payload is empty after parse")
        return errors

    for key in REQUIRED_INFERENCE_KEYS:
        if key not in payload:
            errors.append(f"missing required key: {key}")
            continue
        value = payload[key]
        if value is None:
            errors.append(f"required key {key} must not be null")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"required key {key} must be a non-empty string")
        elif key in ("decision", "code", "msg") and not isinstance(value, str):
            errors.append(f"required key {key} must be a string")

    decision = payload.get("decision")
    if isinstance(decision, str) and decision.strip():
        if decision.strip() not in ALLOWED_DECISIONS:
            allowed = ", ".join(sorted(ALLOWED_DECISIONS))
            errors.append(f"decision {decision!r} is not allowed (expected one of: {allowed})")

    code = payload.get("code")
    if isinstance(code, str) and not code.strip() and "code" in payload:
        errors.append("required key code must be a non-empty string")

    msg = payload.get("msg")
    if isinstance(msg, str) and not msg.strip() and "msg" in payload:
        errors.append("required key msg must be a non-empty string")

    raw_confidence = payload.get("confidence")
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            errors.append("confidence must be a number")
        else:
            if confidence < 0.0 or confidence > 1.0:
                errors.append("confidence must be in [0, 1]")

    return errors


def is_valid_inference_payload(payload: Any) -> bool:
    return not validate_inference_payload(payload)


def contract_error_class(
    payload: Any,
    errors: Sequence[str],
) -> ContractErrorClass:
    """Classify failures for logging and repair routing."""
    if not isinstance(payload, dict) or not payload:
        return "parse"
    if not errors:
        return "validation"
    parse_markers = (
        "payload must be a JSON object",
        "payload is empty after parse",
    )
    if any(any(marker in err for marker in parse_markers) for err in errors):
        return "parse"
    return "validation"


def format_validation_errors(errors: Sequence[str]) -> str:
    if not errors:
        return ""
    return "; ".join(errors)


def truncate_repair_snippet(raw_text: str, *, max_chars: int = MAX_REPAIR_SNIPPET_CHARS) -> str:
    text = raw_text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…[truncated]"


def build_repair_user_message(
    *,
    raw_snippet: str,
    errors: Sequence[str],
    required_keys: Sequence[str] = REQUIRED_INFERENCE_KEYS,
    suffix: str = DEFAULT_REPAIR_PROMPT_SUFFIX,
) -> str:
    """
    User-facing repair follow-up content (P1b adapters append to the base prompt).
    """
    error_text = format_validation_errors(errors) or "unknown contract violation"
    keys_text = ", ".join(required_keys)
    snippet = truncate_repair_snippet(raw_snippet)
    parts = [
        "The previous response failed contract validation.",
        f"Errors: {error_text}",
        f"Required JSON keys: {keys_text}.",
        suffix,
    ]
    if snippet.strip():
        parts.insert(1, f"Previous raw output (snippet):\n{snippet}")
    return "\n".join(parts)


def semantic_repair_drift_label(
    before_decision: Optional[str],
    after_decision: Optional[str],
) -> Optional[str]:
    """
    Hook surface for repair audit: label unstable decision changes across repair rounds.

    See models.repair_audit.build_repair_audit_entry (used when max_json_repair_attempts > 0).
    """
    from models.repair_audit import detect_semantic_repair_drift

    return detect_semantic_repair_drift(before_decision, after_decision)
