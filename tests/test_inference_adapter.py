from models.inference_adapter import _normalize_result


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
