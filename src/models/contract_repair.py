"""JSON contract repair loop for vision inference backends (P1b)."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from models.contract_validator import (
    DEFAULT_REPAIR_PROMPT_SUFFIX,
    build_repair_user_message,
    contract_error_class,
    validate_inference_payload,
)
from models.contracts import InferenceOutput
from models.repair_audit import (
    audit_has_unstable_drift,
    build_repair_audit_entry,
    extract_parsed_semantic,
)

logger = logging.getLogger(__name__)

FetchRawText = Callable[[str], str]
FetchRawTextAsync = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class ContractRepairSettings:
    max_json_repair_attempts: int = 0
    strict_contract: bool = False
    repair_on_empty_dict: bool = True
    repair_prompt_suffix: str = DEFAULT_REPAIR_PROMPT_SUFFIX
    replay_mode: str = "off"

    @classmethod
    def from_inference_cfg(
        cls,
        inference_cfg: Dict[str, Any],
        *,
        replay_mode: str = "off",
    ) -> "ContractRepairSettings":
        contract_cfg = inference_cfg.get("contract", {})
        if not isinstance(contract_cfg, dict):
            contract_cfg = {}
        max_attempts = int(contract_cfg.get("max_json_repair_attempts", 0))
        replay = str(replay_mode or "off").lower()
        if replay == "replay":
            max_attempts = 0
        suffix = str(
            contract_cfg.get("repair_prompt_suffix", DEFAULT_REPAIR_PROMPT_SUFFIX)
        )
        return cls(
            max_json_repair_attempts=max(0, max_attempts),
            strict_contract=bool(contract_cfg.get("strict_contract", False)),
            repair_on_empty_dict=bool(contract_cfg.get("repair_on_empty_dict", True)),
            repair_prompt_suffix=suffix,
            replay_mode=replay,
        )

    @property
    def repair_enabled(self) -> bool:
        return self.max_json_repair_attempts > 0 and self.replay_mode != "replay"


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw_text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def attach_contract_meta(
    result: Dict[str, Any],
    *,
    repair_attempts: int,
    repair_audit: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    enriched = dict(result)
    meta: Dict[str, Any] = {"repair_attempts": repair_attempts}
    if repair_audit is not None:
        meta["repair_audit"] = repair_audit
        meta["unstable_repair"] = audit_has_unstable_drift(repair_audit)
    enriched["contract_meta"] = meta
    return enriched


def _parse_and_validate(raw_text: str) -> tuple[Dict[str, Any], List[str]]:
    parsed = extract_json_object(raw_text)
    errors = validate_inference_payload(parsed)
    return parsed, errors


def _should_attempt_repair(
    settings: ContractRepairSettings,
    parsed: Dict[str, Any],
    errors: List[str],
    repair_attempts: int,
) -> bool:
    if not settings.repair_enabled:
        return False
    if not errors:
        return False
    if repair_attempts >= settings.max_json_repair_attempts:
        return False
    if not parsed and not settings.repair_on_empty_dict:
        return False
    return True


def _lenient_result(
    parsed: Any,
    *,
    default_msg: str,
    backend: str,
    repair_attempts: int,
    repair_audit: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result = InferenceOutput.from_payload(
        parsed,
        default_msg=default_msg,
        backend=backend,
    ).to_dict()
    return attach_contract_meta(
        result, repair_attempts=repair_attempts, repair_audit=repair_audit
    )


def _repair_exhausted_result(
    *,
    backend: str,
    repair_attempts: int,
    errors: List[str],
    repair_audit: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    detail = "; ".join(errors) if errors else "contract validation failed"
    result = InferenceOutput(
        decision="Error",
        code="ERR_MODEL_RESPONSE_422",
        msg=f"repair_exhausted: {detail}",
        backend=backend,
    ).to_dict()
    return attach_contract_meta(
        result, repair_attempts=repair_attempts, repair_audit=repair_audit
    )


def _strict_valid_result(
    parsed: Dict[str, Any],
    *,
    backend: str,
    repair_attempts: int,
    repair_audit: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    confidence: Optional[float] = None
    raw_confidence = parsed.get("confidence")
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = None
    result = InferenceOutput(
        decision=str(parsed["decision"]),
        code=str(parsed["code"]),
        msg=str(parsed["msg"]),
        confidence=confidence,
        backend=backend,
    ).to_dict()
    return attach_contract_meta(
        result, repair_attempts=repair_attempts, repair_audit=repair_audit
    )


def _log_repair_attempt(
    *,
    backend: str,
    attempt: int,
    error_class: str,
    semantic_drift: Optional[str],
) -> None:
    logger.info(
        "contract_repair: backend=%s attempt=%s error_class=%s semantic_drift=%s",
        backend,
        attempt,
        error_class,
        semantic_drift or "none",
    )


def _record_audit_round(
    repair_audit: List[Dict[str, Any]],
    *,
    round_index: int,
    prompt: str,
    raw_text: str,
    parsed: Dict[str, Any],
    errors: List[str],
    previous_decision: Optional[str],
) -> Optional[str]:
    entry = build_repair_audit_entry(
        round_index=round_index,
        format_errors=errors,
        prompt_input_snapshot=prompt,
        raw_output_snapshot=raw_text,
        parsed=parsed,
        previous_decision=previous_decision,
    )
    repair_audit.append(entry)
    return entry.get("semantic_drift_from_previous")


async def _execute_contract_loop_async(
    settings: ContractRepairSettings,
    *,
    backend: str,
    default_msg: str,
    build_initial_prompt: Callable[[], str],
    fetch_model_text: FetchRawTextAsync,
) -> Dict[str, Any]:
    if not settings.repair_enabled:
        prompt = build_initial_prompt()
        raw_text = await fetch_model_text(prompt)
        parsed = extract_json_object(raw_text)
        return _lenient_result(
            parsed, default_msg=default_msg, backend=backend, repair_attempts=0
        )

    repair_attempts = 0
    repair_audit: List[Dict[str, Any]] = []
    prompt = build_initial_prompt()
    last_errors: List[str] = []
    last_decision: Optional[str] = None
    round_index = 0

    while True:
        raw_text = await fetch_model_text(prompt)
        parsed, errors = _parse_and_validate(raw_text)
        last_errors = errors

        drift = _record_audit_round(
            repair_audit,
            round_index=round_index,
            prompt=prompt,
            raw_text=raw_text,
            parsed=parsed,
            errors=errors,
            previous_decision=last_decision,
        )
        decision, _ = extract_parsed_semantic(parsed)
        if decision:
            last_decision = decision

        if not errors:
            if drift:
                _log_repair_attempt(
                    backend=backend,
                    attempt=repair_attempts,
                    error_class="validation",
                    semantic_drift=drift,
                )
            return _strict_valid_result(
                parsed,
                backend=backend,
                repair_attempts=repair_attempts,
                repair_audit=repair_audit,
            )

        if not _should_attempt_repair(settings, parsed, errors, repair_attempts):
            return _repair_exhausted_result(
                backend=backend,
                repair_attempts=repair_attempts,
                errors=last_errors,
                repair_audit=repair_audit,
            )

        repair_attempts += 1
        round_index += 1
        error_class = contract_error_class(parsed, errors)
        _log_repair_attempt(
            backend=backend,
            attempt=repair_attempts,
            error_class=error_class,
            semantic_drift=drift,
        )
        prompt = (
            f"{build_initial_prompt()}\n\n"
            f"{build_repair_user_message(raw_snippet=raw_text, errors=errors, suffix=settings.repair_prompt_suffix)}"
        )


def run_contract_inference_loop(
    settings: ContractRepairSettings,
    *,
    backend: str,
    default_msg: str,
    build_initial_prompt: Callable[[], str],
    fetch_model_text: FetchRawText,
) -> Dict[str, Any]:
    """Parse and optionally repair LLM JSON output before returning inference dict."""

    async def _async_fetch(prompt: str) -> str:
        return await asyncio.to_thread(fetch_model_text, prompt)

    return asyncio.run(
        _execute_contract_loop_async(
            settings,
            backend=backend,
            default_msg=default_msg,
            build_initial_prompt=build_initial_prompt,
            fetch_model_text=_async_fetch,
        )
    )


async def run_contract_inference_loop_async(
    settings: ContractRepairSettings,
    *,
    backend: str,
    default_msg: str,
    build_initial_prompt: Callable[[], str],
    fetch_model_text: FetchRawTextAsync,
) -> Dict[str, Any]:
    return await _execute_contract_loop_async(
        settings,
        backend=backend,
        default_msg=default_msg,
        build_initial_prompt=build_initial_prompt,
        fetch_model_text=fetch_model_text,
    )
