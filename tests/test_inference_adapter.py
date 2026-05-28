from models.inference_adapter import _normalize_result
from models.contracts import InferenceOutput


def test_normalize_result_handles_invalid_payload():
    normalized = _normalize_result("invalid", "fallback")
    assert normalized["decision"] == "Error"
    assert normalized["code"] == "ERR_MODEL_RESPONSE_422"
    assert normalized["msg"] == "fallback"


def test_normalize_result_parses_confidence():
    normalized = _normalize_result(
        {"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok", "confidence": "0.88"},
        "fallback",
    )
    assert normalized["decision"] == "Optimal"
    assert normalized["code"] == "SUCCESS_200"
    assert normalized["msg"] == "ok"
    assert normalized["confidence"] == 0.88


def test_inference_output_from_payload_preserves_backend():
    output = InferenceOutput.from_payload(
        {"decision": "Optimal", "code": "SUCCESS_200", "msg": "ok"},
        default_msg="fallback",
        backend="mock_api",
    )
    assert output.backend == "mock_api"
    assert output.to_dict()["backend"] == "mock_api"
