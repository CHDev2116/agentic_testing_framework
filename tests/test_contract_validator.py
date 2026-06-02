import pytest

from models.contract_validator import (
    ALLOWED_DECISIONS,
    REQUIRED_INFERENCE_KEYS,
    build_repair_user_message,
    contract_error_class,
    format_validation_errors,
    is_valid_inference_payload,
    truncate_repair_snippet,
    validate_inference_payload,
)


def _valid_payload() -> dict:
    return {
        "decision": "Optimal",
        "code": "SUCCESS_200",
        "msg": "Quality meets standards.",
        "confidence": 0.88,
    }


def test_valid_payload_returns_no_errors():
    assert validate_inference_payload(_valid_payload()) == []
    assert is_valid_inference_payload(_valid_payload())


@pytest.mark.parametrize(
    "payload,expected_substring",
    [
        ("not a dict", "JSON object"),
        ({}, "empty"),
        ({"decision": "Optimal"}, "missing required key: code"),
        ({"code": "X", "msg": "y"}, "missing required key: decision"),
        (
            {"decision": "Optimal", "code": "SUCCESS_200", "msg": ""},
            "msg must be a non-empty",
        ),
        (
            {"decision": "LoginSuccess", "code": "SUCCESS_200", "msg": "ok"},
            "not allowed",
        ),
        (
            {
                "decision": "Optimal",
                "code": "SUCCESS_200",
                "msg": "ok",
                "confidence": 1.5,
            },
            "confidence must be in [0, 1]",
        ),
        (
            {
                "decision": "Optimal",
                "code": "SUCCESS_200",
                "msg": "ok",
                "confidence": "high",
            },
            "confidence must be a number",
        ),
    ],
)
def test_invalid_payload_reports_errors(payload, expected_substring):
    errors = validate_inference_payload(payload)
    assert errors
    assert any(expected_substring in err for err in errors)


def test_allowed_decisions_match_quantizer_labels():
    assert "Under-exposed" in ALLOWED_DECISIONS
    assert "Over-exposed" in ALLOWED_DECISIONS
    assert "Underexposed" not in ALLOWED_DECISIONS


def test_contract_error_class_parse_vs_validation():
    assert contract_error_class({}, validate_inference_payload({})) == "parse"
    assert contract_error_class(None, validate_inference_payload(None)) == "parse"
    errors = validate_inference_payload({"decision": "Blurry", "code": "X", "msg": "y"})
    assert contract_error_class({"decision": "Blurry", "code": "X", "msg": "y"}, errors) == "validation"


def test_format_validation_errors_joins_messages():
    assert format_validation_errors(["a", "b"]) == "a; b"
    assert format_validation_errors([]) == ""


def test_truncate_repair_snippet():
    long_text = "x" * 3000
    truncated = truncate_repair_snippet(long_text, max_chars=100)
    assert len(truncated) < len(long_text)
    assert truncated.endswith("[truncated]")


def test_build_repair_user_message_includes_errors_and_keys():
    message = build_repair_user_message(
        raw_snippet='```json\n{"decision":',
        errors=["missing required key: code"],
    )
    assert "contract validation" in message
    assert "missing required key: code" in message
    assert "decision, code, msg" in message
    for key in REQUIRED_INFERENCE_KEYS:
        assert key in message
    assert "Optimal" in message
