"""
Async inference helpers using httpx for I/O-bound backends.

CPU-only simulated inference runs in a thread pool via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

import httpx

from models.contract_repair import run_contract_inference_loop_async
from models.inference_adapter import (
    LlamaCppInferenceEngine,
    MockAPIInferenceEngine,
    OllamaVisionInferenceEngine,
    SimulatedInferenceEngine,
)
from models.contracts import InferenceOutput
from util.adaptive_backoff import (
    AdaptiveBackoffSettings,
    response_indicates_pressure,
    sleep_backoff,
)

logger = logging.getLogger(__name__)


def _backoff_settings(engine: Any) -> AdaptiveBackoffSettings:
    settings = getattr(engine, "adaptive_backoff_settings", None)
    if isinstance(settings, AdaptiveBackoffSettings):
        return settings
    return AdaptiveBackoffSettings()


async def _post_json_with_backoff(
    engine: Any,
    client: httpx.AsyncClient,
    url: str,
    *,
    payload: Dict[str, Any],
    timeout: httpx.Timeout,
) -> httpx.Response:
    settings = _backoff_settings(engine)
    response: httpx.Response | None = None
    attempts = settings.max_retries + 1 if settings.enabled else 1
    for attempt in range(attempts):
        response = await client.post(url, json=payload, timeout=timeout)
        if not settings.enabled or not response_indicates_pressure(response.status_code):
            return response
        if attempt >= settings.max_retries:
            return response
        await sleep_backoff(
            attempt,
            settings,
            reason=f"HTTP {response.status_code} url={url}",
        )
    assert response is not None
    return response
_WARNING_DEDUP_WINDOW_S = 5.0
_LAST_WARNING_AT: Dict[str, float] = {}


def _warning_dedup_key(scope: str, url: str, exc: Exception) -> str:
    return f"{scope}|{url}|{type(exc).__name__}|{exc}"


def _warn_with_dedup(scope: str, url: str, exc: Exception) -> None:
    key = _warning_dedup_key(scope, url, exc)
    now = time.monotonic()
    last_at = _LAST_WARNING_AT.get(key)
    if last_at is not None and (now - last_at) < _WARNING_DEDUP_WINDOW_S:
        return
    _LAST_WARNING_AT[key] = now
    logger.warning("%s: request failed url=%s error=%s", scope, url, exc)


async def predict_quality_async(
    engine: Any,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    """
    Async quality prediction mirroring sync inference_adapter engines.
    """
    backend = getattr(engine, "backend_name", type(engine).__name__)
    logger.debug(
        "predict_quality_async: start backend=%s photo_path=%s",
        backend,
        photo_path,
    )
    if isinstance(engine, SimulatedInferenceEngine):
        result = await asyncio.to_thread(engine.predict_quality, photo_path, metrics)
        logger.debug("predict_quality_async: done backend=%s (thread pool)", backend)
        return result
    if isinstance(engine, LlamaCppInferenceEngine):
        return await _llama_cpp_predict_async(engine, client, photo_path, metrics)
    if isinstance(engine, OllamaVisionInferenceEngine):
        return await _ollama_predict_async(engine, client, photo_path, metrics)
    if isinstance(engine, MockAPIInferenceEngine):
        return await _mock_api_predict_async(engine, client, photo_path, metrics)
    result = await asyncio.to_thread(engine.predict_quality, photo_path, metrics)
    logger.debug("predict_quality_async: done backend=%s (generic thread pool)", backend)
    return result


async def _fetch_llama_text_async(
    engine: LlamaCppInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
    prompt: str,
) -> str:
    url = f"{engine.host}{engine.endpoint}"
    timeout = httpx.Timeout(engine.timeout_s)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": engine._encode_image_data_uri(photo_path)},
                },
            ],
        }
    ]
    payload: Dict[str, Any] = {
        "model": engine.model,
        "messages": messages,
        "temperature": engine.temperature,
        "max_tokens": engine.max_tokens,
        "stream": False,
    }
    if engine.use_response_format:
        payload["response_format"] = {"type": "json_object"}

    response = await _post_json_with_backoff(
        engine, client, url, payload=payload, timeout=timeout
    )
    if response.status_code >= 400 and "response_format" in payload:
        payload_without_format = dict(payload)
        payload_without_format.pop("response_format", None)
        response = await _post_json_with_backoff(
            engine, client, url, payload=payload_without_format, timeout=timeout
        )
    response.raise_for_status()
    body = response.json()
    return str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))


async def _llama_cpp_predict_async(
    engine: LlamaCppInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    url = f"{engine.host}{engine.endpoint}"
    logger.debug(
        "predict_quality_async(llama_cpp): POST %s timeout_s=%.1f",
        url,
        engine.timeout_s,
    )
    try:
        result = await run_contract_inference_loop_async(
            engine._contract,
            backend=engine.backend_name,
            default_msg="llama.cpp returned unparsable response.",
            build_initial_prompt=lambda: engine._build_base_prompt(metrics),
            fetch_model_text=lambda prompt: _fetch_llama_text_async(
                engine, client, photo_path, metrics, prompt
            ),
        )
        return engine._apply_post_repair_fallback_policy(
            result, photo_path, metrics, label="llama.cpp"
        )
    except Exception as exc:
        _warn_with_dedup("predict_quality_async(llama_cpp)", url, exc)
        if engine.fallback_to_simulated and not engine._contract.strict_contract:
            fallback = await asyncio.to_thread(
                engine.simulated_fallback.predict_quality, photo_path, metrics
            )
            fallback["msg"] = f"llama.cpp fallback to simulated inference: {exc}"
            fallback["backend"] = f"{engine.backend_name}->simulated"
            return fallback
        return InferenceOutput(
            decision="Error",
            code="ERR_MODEL_BACKEND_503",
            msg=f"llama.cpp inference failed: {exc}",
            backend=engine.backend_name,
        ).to_dict()


async def _fetch_ollama_text_async(
    engine: OllamaVisionInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    prompt: str,
) -> str:
    url = f"{engine.host}/api/generate"
    timeout = httpx.Timeout(engine.timeout_s)
    payload = {
        "model": engine.model,
        "prompt": prompt,
        "stream": False,
        "images": [await asyncio.to_thread(engine._encode_image, photo_path)],
        "format": "json",
    }
    response = await _post_json_with_backoff(
        engine, client, url, payload=payload, timeout=timeout
    )
    response.raise_for_status()
    body = response.json()
    return str(body.get("response", ""))


async def _ollama_predict_async(
    engine: OllamaVisionInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    url = f"{engine.host}/api/generate"
    logger.debug(
        "predict_quality_async(ollama): POST %s timeout_s=%.1f",
        url,
        engine.timeout_s,
    )
    try:
        result = await run_contract_inference_loop_async(
            engine._contract,
            backend=engine.backend_name,
            default_msg="Ollama returned unparsable response.",
            build_initial_prompt=lambda: engine._build_base_prompt(metrics),
            fetch_model_text=lambda prompt: _fetch_ollama_text_async(
                engine, client, photo_path, prompt
            ),
        )
        return engine._apply_post_repair_fallback_policy(
            result, photo_path, metrics, label="Ollama"
        )
    except Exception as exc:
        _warn_with_dedup("predict_quality_async(ollama)", url, exc)
        if engine.fallback_to_simulated and not engine._contract.strict_contract:
            fallback = await asyncio.to_thread(
                engine.simulated_fallback.predict_quality, photo_path, metrics
            )
            fallback["msg"] = f"Ollama fallback to simulated inference: {exc}"
            fallback["backend"] = f"{engine.backend_name}->simulated"
            return fallback
        return InferenceOutput(
            decision="Error",
            code="ERR_MODEL_BACKEND_503",
            msg=f"Ollama inference failed: {exc}",
            backend=engine.backend_name,
        ).to_dict()


async def _mock_api_predict_async(
    engine: MockAPIInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    import os

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(engine.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "photo_path": photo_path,
        "metrics": metrics,
        "thresholds": engine.thresholds,
    }
    timeout = httpx.Timeout(engine.timeout_s)
    logger.debug(
        "predict_quality_async(mock_api): POST %s timeout_s=%.1f",
        engine.url,
        engine.timeout_s,
    )
    try:
        response = await client.post(
            engine.url, json=payload, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        body = response.json()
        result = body.get("result", body)
        return InferenceOutput.from_payload(
            result,
            default_msg="Mock API returned invalid response.",
            backend=engine.backend_name,
        ).to_dict()
    except Exception as exc:
        _warn_with_dedup("predict_quality_async(mock_api)", engine.url, exc)
        if engine.fallback_to_simulated:
            fallback = await asyncio.to_thread(
                engine.simulated_fallback.predict_quality, photo_path, metrics
            )
            fallback["msg"] = f"Mock API fallback to simulated inference: {exc}"
            fallback["backend"] = f"{engine.backend_name}->simulated"
            return fallback
        return InferenceOutput(
            decision="Error",
            code="ERR_MODEL_BACKEND_503",
            msg=f"Mock API inference failed: {exc}",
            backend=engine.backend_name,
        ).to_dict()
