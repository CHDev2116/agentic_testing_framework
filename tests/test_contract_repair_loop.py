"""P1b contract repair loop tests (sync + async, mocked HTTP)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image

from models.async_inference import predict_quality_async
from models.contract_repair import ContractRepairSettings, run_contract_inference_loop
from models.inference_adapter import OllamaVisionInferenceEngine

THRESHOLDS = {
    "min_sharpness": 20.0,
    "min_brightness": 40.0,
    "max_brightness": 220.0,
}
GOOD_METRICS = {"sharpness": 50.0, "avg_brightness": 80.0}
VALID_JSON = json.dumps(
    {"decision": "Optimal", "code": "SUCCESS_200", "msg": "Quality ok."}
)


def _make_test_image(path: Path) -> None:
    Image.new("L", (16, 16), color=120).save(path)


def _ollama_cfg(**contract_overrides: Any) -> Dict[str, Any]:
    return {
        "fallback_to_simulated": False,
        "contract": {"max_json_repair_attempts": 2, **contract_overrides},
        "ollama": {
            "host": "http://localhost:11434",
            "model": "llava:7b",
            "timeout_s": 5.0,
        },
    }


def test_run_contract_loop_repair_disabled_uses_lenient_path():
    settings = ContractRepairSettings(max_json_repair_attempts=0)
    calls: List[str] = []

    def fetch(prompt: str) -> str:
        calls.append(prompt)
        return "not-json"

    result = run_contract_inference_loop(
        settings,
        backend="test",
        default_msg="unparsable",
        build_initial_prompt=lambda: "base",
        fetch_model_text=fetch,
    )
    assert len(calls) == 1
    assert result["decision"] == "Error"
    assert result["code"] == "ERR_MODEL_RESPONSE_422"
    assert result["contract_meta"]["repair_attempts"] == 0


def test_run_contract_loop_repairs_then_succeeds():
    settings = ContractRepairSettings(max_json_repair_attempts=2)
    responses = ['```json\n{"decision":', VALID_JSON]
    call_idx = {"n": 0}

    def fetch(prompt: str) -> str:
        raw = responses[call_idx["n"]]
        call_idx["n"] += 1
        return raw

    result = run_contract_inference_loop(
        settings,
        backend="test",
        default_msg="unparsable",
        build_initial_prompt=lambda: "base",
        fetch_model_text=fetch,
    )
    assert call_idx["n"] == 2
    assert result["decision"] == "Optimal"
    assert result["contract_meta"]["repair_attempts"] == 1
    assert "repair_audit" in result["contract_meta"]
    assert result["contract_meta"]["unstable_repair"] is False


def test_run_contract_loop_repair_exhausted():
    settings = ContractRepairSettings(max_json_repair_attempts=1)

    def fetch(prompt: str) -> str:
        return "not-json"

    result = run_contract_inference_loop(
        settings,
        backend="test",
        default_msg="unparsable",
        build_initial_prompt=lambda: "base",
        fetch_model_text=fetch,
    )
    assert "repair_exhausted" in result["msg"]
    assert result["code"] == "ERR_MODEL_RESPONSE_422"
    assert result["contract_meta"]["repair_attempts"] == 1


def test_ollama_sync_repair_second_response_valid(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = OllamaVisionInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_ollama_cfg(),
    )
    mock_responses = [
        MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"response": '```json\n{"decision":'}),
        ),
        MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"response": VALID_JSON}),
        ),
    ]
    with patch("models.inference_adapter._get_requests") as get_requests:
        session = MagicMock()
        session.post.side_effect = mock_responses
        get_requests.return_value = session
        result = engine.predict_quality(str(image_path), GOOD_METRICS)
    assert session.post.call_count == 2
    assert result["decision"] == "Optimal"
    assert result["contract_meta"]["repair_attempts"] == 1


def test_ollama_replay_mode_skips_repair(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = OllamaVisionInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_ollama_cfg(max_json_repair_attempts=3),
        replay_mode="replay",
    )
    mock_response = MagicMock(
        status_code=200,
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"response": "not-json"}),
    )
    with patch("models.inference_adapter._get_requests") as get_requests:
        session = MagicMock()
        session.post.return_value = mock_response
        get_requests.return_value = session
        result = engine.predict_quality(str(image_path), GOOD_METRICS)
    assert session.post.call_count == 1
    assert result["decision"] == "Error"
    assert result["contract_meta"]["repair_attempts"] == 0


def test_ollama_strict_contract_no_fallback_on_exhausted(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = OllamaVisionInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_ollama_cfg(
            max_json_repair_attempts=1,
            strict_contract=True,
        ),
    )
    engine.fallback_to_simulated = True
    bad = MagicMock(
        status_code=200,
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"response": "not-json"}),
    )
    with patch("models.inference_adapter._get_requests") as get_requests:
        session = MagicMock()
        session.post.return_value = bad
        get_requests.return_value = session
        result = engine.predict_quality(str(image_path), GOOD_METRICS)
    assert "repair_exhausted" in result["msg"]
    assert result["backend"] == "ollama_vision"
    assert "->simulated" not in result["backend"]


async def _predict_ollama_async(engine, handler, photo_path: str):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await predict_quality_async(engine, client, photo_path, GOOD_METRICS)


def test_ollama_async_repair_second_response_valid(tmp_path: Path):
    image_path = tmp_path / "photo.png"
    _make_test_image(image_path)
    engine = OllamaVisionInferenceEngine(
        thresholds=THRESHOLDS,
        inference_cfg=_ollama_cfg(),
    )
    calls: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json={"response": '```json\n{"decision":'})
        return httpx.Response(200, json={"response": VALID_JSON})

    result = asyncio.run(_predict_ollama_async(engine, handler, str(image_path)))
    assert len(calls) == 2
    assert result["decision"] == "Optimal"
    assert result["contract_meta"]["repair_attempts"] == 1
