"""
Async inference helpers using httpx for I/O-bound backends.

CPU-only simulated inference runs in a thread pool via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict

import httpx

from models.inference_adapter import (
    LlamaCppInferenceEngine,
    MockAPIInferenceEngine,
    OllamaVisionInferenceEngine,
    SimulatedInferenceEngine,
)
from models.contracts import InferenceOutput

logger = logging.getLogger(__name__)
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


async def _llama_cpp_predict_async(
    engine: LlamaCppInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    payload = {
        "model": engine.model,
        "messages": engine._build_messages(photo_path, metrics),
        "temperature": engine.temperature,
        "max_tokens": engine.max_tokens,
        "stream": False,
    }
    if engine.use_response_format:
        payload["response_format"] = {"type": "json_object"}

    url = f"{engine.host}{engine.endpoint}"
    timeout = httpx.Timeout(engine.timeout_s)
    logger.debug(
        "predict_quality_async(llama_cpp): POST %s timeout_s=%.1f",
        url,
        engine.timeout_s,
    )
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        if response.status_code >= 400 and "response_format" in payload:
            payload_without_format = dict(payload)
            payload_without_format.pop("response_format", None)
            response = await client.post(url, json=payload_without_format, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        model_text = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
        parsed = engine._extract_json_object(model_text)
        return InferenceOutput.from_payload(
            parsed,
            default_msg="llama.cpp returned unparsable response.",
            backend=engine.backend_name,
        ).to_dict()
    except Exception as exc:
        _warn_with_dedup("predict_quality_async(llama_cpp)", url, exc)
        if engine.fallback_to_simulated:
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


async def _ollama_predict_async(
    engine: OllamaVisionInferenceEngine,
    client: httpx.AsyncClient,
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    prompt = (
        f"{engine.prompt_template}\n"
        f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
        f"Thresholds: {json.dumps(engine.thresholds, ensure_ascii=False)}"
    )
    payload = {
        "model": engine.model,
        "prompt": prompt,
        "stream": False,
        "images": [await asyncio.to_thread(engine._encode_image, photo_path)],
        "format": "json",
    }
    url = f"{engine.host}/api/generate"
    timeout = httpx.Timeout(engine.timeout_s)
    logger.debug(
        "predict_quality_async(ollama): POST %s timeout_s=%.1f",
        url,
        engine.timeout_s,
    )
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        model_text = str(body.get("response", ""))
        parsed = engine._extract_json_object(model_text)
        return InferenceOutput.from_payload(
            parsed,
            default_msg="Ollama returned unparsable response.",
            backend=engine.backend_name,
        ).to_dict()
    except Exception as exc:
        _warn_with_dedup("predict_quality_async(ollama)", url, exc)
        if engine.fallback_to_simulated:
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
