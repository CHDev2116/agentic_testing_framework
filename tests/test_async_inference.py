"""
Async HTTP inference tests using httpx.MockTransport (no live servers).

Includes timeout/connect failure paths per project test conventions.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from PIL import Image

from models.async_inference import predict_quality_async
from models.inference_adapter import (
    LlamaCppInferenceEngine,
    MockAPIInferenceEngine,
    OllamaVisionInferenceEngine,
    SimulatedInferenceEngine,
)

THRESHOLDS = {
    "min_sharpness": 20.0,
    "min_brightness": 40.0,
    "max_brightness": 220.0,
}
GOOD_METRICS = {"sharpness": 50.0, "avg_brightness": 80.0}


def _make_test_image(path: Path) -> None:
    Image.new("L", (32, 32), color=120).save(path)


def _merge_inference_cfg(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
    nested_keys: tuple[str, ...],
) -> Dict[str, Any]:
    merged = dict(base)
    for key in nested_keys:
        if key in overrides:
            merged[key] = {**merged.get(key, {}), **overrides[key]}
    merged.update({k: v for k, v in overrides.items() if k not in nested_keys})
    return merged


def _llama_inference_cfg(**overrides: Any) -> Dict[str, Any]:
    return _merge_inference_cfg(
        {
            "fallback_to_simulated": True,
            "llama_cpp": {
                "host": "http://127.0.0.1:8080",
                "endpoint": "/v1/chat/completions",
                "model": "test-model",
                "timeout_s": 5.0,
                "use_response_format": True,
            },
        },
        overrides,
        ("llama_cpp",),
    )


def _ollama_inference_cfg(**overrides: Any) -> Dict[str, Any]:
    return _merge_inference_cfg(
        {
            "fallback_to_simulated": True,
            "ollama": {
                "host": "http://localhost:11434",
                "model": "llava:7b",
                "timeout_s": 5.0,
            },
        },
        overrides,
        ("ollama",),
    )


def _mock_api_inference_cfg(**overrides: Any) -> Dict[str, Any]:
    return _merge_inference_cfg(
        {
            "fallback_to_simulated": True,
            "mock_api": {
                "url": "http://localhost:9090/infer",
                "timeout_s": 3.0,
                "api_key_env": "MOCK_INFER_API_KEY",
            },
        },
        overrides,
        ("mock_api",),
    )


def _run_async(coro):
    return asyncio.run(coro)


async def _predict(
    engine: Any,
    handler: Callable[[httpx.Request], httpx.Response],
    photo_path: str,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await predict_quality_async(engine, client, photo_path, metrics)


def test_predict_quality_async_simulated():
    engine = SimulatedInferenceEngine(thresholds=THRESHOLDS)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"simulated backend should not call HTTP: {request.url}")

    result = _run_async(_predict(engine, handler, "dummy.jpg", GOOD_METRICS))
    assert result["backend"] == "simulated"
    assert result["decision"] in {"Optimal", "Blurry", "Under-exposed", "Over-exposed", "Error"}


def test_llama_cpp_async_success(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = LlamaCppInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_llama_inference_cfg(),
    )
    model_json = json.dumps(
        {"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok", "confidence": 0.91}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        assert body["model"] == "test-model"
        assert body.get("response_format") == {"type": "json_object"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": model_json}}]},
        )

    result = _run_async(_predict(engine, handler, str(image_path), GOOD_METRICS))
    assert result["backend"] == "llama_cpp"
    assert result["decision"] == "Optimal"
    assert result["code"] == "SUCCESS_200"
    assert result["confidence"] == pytest.approx(0.91)


def test_llama_cpp_async_retries_without_response_format_on_400(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = LlamaCppInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_llama_inference_cfg(),
    )
    calls: List[httpx.Request] = []
    model_json = json.dumps({"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"})

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content.decode())
        if len(calls) == 1:
            assert body.get("response_format") == {"type": "json_object"}
            return httpx.Response(400, json={"error": "response_format not supported"})
        assert "response_format" not in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": model_json}}]},
        )

    result = _run_async(_predict(engine, handler, str(image_path), GOOD_METRICS))
    assert len(calls) == 2
    assert result["backend"] == "llama_cpp"
    assert result["decision"] == "Optimal"


def test_llama_cpp_async_connect_timeout_falls_back(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = LlamaCppInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_llama_inference_cfg(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout(
            "connection timed out",
            request=request,
        )

    result = _run_async(_predict(engine, handler, str(image_path), GOOD_METRICS))
    assert result["backend"] == "llama_cpp->simulated"
    assert "fallback to simulated inference" in result["msg"]


def test_llama_cpp_async_error_without_fallback(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = LlamaCppInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_llama_inference_cfg(fallback_to_simulated=False),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "read timed out",
            request=request,
        )

    result = _run_async(_predict(engine, handler, str(image_path), GOOD_METRICS))
    assert result["backend"] == "llama_cpp"
    assert result["decision"] == "Error"
    assert result["code"] == "ERR_MODEL_BACKEND_503"
    assert "read timed out" in result["msg"]


def test_ollama_async_success(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = OllamaVisionInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_ollama_inference_cfg(),
    )
    model_json = json.dumps({"decision": "Blurry", "code": "ERR_IMG_BLUR_101", "msg": "soft"})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        body = json.loads(request.content.decode())
        assert body["model"] == "llava:7b"
        assert body["format"] == "json"
        assert len(body["images"]) == 1
        return httpx.Response(200, json={"response": model_json})

    result = _run_async(_predict(engine, handler, str(image_path), GOOD_METRICS))
    assert result["backend"] == "ollama_vision"
    assert result["decision"] == "Blurry"
    assert result["code"] == "ERR_IMG_BLUR_101"


def test_ollama_async_http_error_falls_back(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = OllamaVisionInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_ollama_inference_cfg(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model unavailable"})

    result = _run_async(_predict(engine, handler, str(image_path), GOOD_METRICS))
    assert result["backend"] == "ollama_vision->simulated"
    assert "Ollama fallback to simulated inference" in result["msg"]


def test_mock_api_async_success_with_result_wrapper():
    engine = MockAPIInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_mock_api_inference_cfg(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://localhost:9090/infer")
        body = json.loads(request.content.decode())
        assert body["photo_path"] == "shots/a.jpg"
        assert body["metrics"] == GOOD_METRICS
        return httpx.Response(
            200,
            json={
                "result": {
                    "decision": "Optimal",
                    "code": "SUCCESS_200",
                    "msg": "mock ok",
                }
            },
        )

    result = _run_async(_predict(engine, handler, "shots/a.jpg", GOOD_METRICS))
    assert result["backend"] == "mock_api"
    assert result["decision"] == "Optimal"
    assert result["msg"] == "mock ok"


def test_mock_api_async_success_flat_body():
    engine = MockAPIInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_mock_api_inference_cfg(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"decision": "Under-exposed", "code": "ERR_IMG_DARK_102", "msg": "dark"},
        )

    result = _run_async(_predict(engine, handler, "x.jpg", GOOD_METRICS))
    assert result["decision"] == "Under-exposed"
    assert result["code"] == "ERR_IMG_DARK_102"


def test_mock_api_async_sends_bearer_when_api_key_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOCK_INFER_API_KEY", "secret-token")
    engine = MockAPIInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_mock_api_inference_cfg(),
    )
    seen_auth: List[Optional[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization"))
        return httpx.Response(
            200,
            json={"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"},
        )

    _run_async(_predict(engine, handler, "x.jpg", GOOD_METRICS))
    assert seen_auth == ["Bearer secret-token"]


def test_mock_api_async_connect_timeout_without_fallback():
    engine = MockAPIInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_mock_api_inference_cfg(fallback_to_simulated=False),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout(
            "connect timed out",
            request=request,
        )

    result = _run_async(_predict(engine, handler, "x.jpg", GOOD_METRICS))
    assert result["backend"] == "mock_api"
    assert result["decision"] == "Error"
    assert result["code"] == "ERR_MODEL_BACKEND_503"
    assert "connect timed out" in result["msg"]


def test_mock_api_async_uses_configured_timeout_s():
    """Engine timeout_s is forwarded to httpx (API timeout handling)."""
    engine = MockAPIInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_mock_api_inference_cfg(mock_api={"timeout_s": 7.5}),
    )
    seen_timeouts: List[httpx.Timeout] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"},
        )

    async def _run_with_capture():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            original_post = client.post

            async def capturing_post(url, **kwargs):
                timeout = kwargs.get("timeout")
                if isinstance(timeout, httpx.Timeout):
                    seen_timeouts.append(timeout)
                return await original_post(url, **kwargs)

            client.post = capturing_post  # type: ignore[method-assign]
            return await predict_quality_async(engine, client, "x.jpg", GOOD_METRICS)

    _run_async(_run_with_capture())
    assert len(seen_timeouts) == 1
    assert seen_timeouts[0].connect == 7.5
    assert seen_timeouts[0].read == 7.5
