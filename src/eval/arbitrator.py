from enum import Enum


class DecisionConflict(Enum):
    PHYSICAL_OK_MODEL_FAIL_HIGH_CONFIDENCE = "Engine OK; model negative verdict with high confidence (review)"
    PHYSICAL_OK_MODEL_FAIL_LOW_CONFIDENCE = "Engine OK; model negative verdict with low confidence (accept GO)"
    PHYSICAL_FAIL_MODEL_OK = "Engine Fails but Model OK (Potential Over-optimism)"
    ALL_PASS = "Consistent Pass"
    ALL_FAIL = "Consistent Fail"


def _is_physically_ok(engine_metrics, thresholds):
    thresholds = thresholds or {}
    brightness = float(engine_metrics.get("avg_brightness", engine_metrics.get("brightness", 0.0)))
    sharpness = float(engine_metrics.get("sharpness", 0.0))
    min_b = float(thresholds.get("min_brightness", 40.0))
    max_b = float(thresholds.get("max_brightness", 220.0))
    min_s = float(thresholds.get("min_sharpness", 20.0))
    if brightness < min_b or brightness > max_b:
        return False
    return sharpness >= min_s


def arbitrate_decision(engine_metrics, model_inference, thresholds=None):
    """
    Core arbitrator logic (single sample).
    engine_metrics: dict with avg_brightness (or brightness) and sharpness
    model_inference: dict with status or decision (e.g. \"Optimal\"), optional confidence in [0,1]
    thresholds: optional min_brightness, max_brightness, min_sharpness
    """
    thresholds = thresholds or {}
    is_physically_ok = _is_physically_ok(engine_metrics, thresholds)
    status = model_inference.get("status") or model_inference.get("decision")
    is_model_ok = status == "Optimal"
    raw_confidence = model_inference.get("confidence")
    model_confidence = float(raw_confidence) if raw_confidence is not None else 0.0

    if is_physically_ok and not is_model_ok:
        if model_confidence > 0.8:
            return "REVIEW", DecisionConflict.PHYSICAL_OK_MODEL_FAIL_HIGH_CONFIDENCE
        return "GO", DecisionConflict.PHYSICAL_OK_MODEL_FAIL_LOW_CONFIDENCE

    if not is_physically_ok and is_model_ok:
        return "NO_GO", DecisionConflict.PHYSICAL_FAIL_MODEL_OK

    if is_physically_ok and is_model_ok:
        return "GO", DecisionConflict.ALL_PASS

    return "NO_GO", DecisionConflict.ALL_FAIL


def aggregate_batch_decisions(per_image_decisions, conflict_strategy="conservative"):
    """
    Roll up per-image GO/REVIEW/NO_GO into one batch decision.
    """
    if not per_image_decisions:
        return "NO_GO"

    if conflict_strategy == "conservative":
        if "NO_GO" in per_image_decisions:
            return "NO_GO"
        if "REVIEW" in per_image_decisions:
            return "REVIEW"
        return "GO"

    # default: same as conservative
    if "NO_GO" in per_image_decisions:
        return "NO_GO"
    if "REVIEW" in per_image_decisions:
        return "REVIEW"
    return "GO"


def merge_gate_and_arbitration(gate_decision, arbitration_decision, conflict_strategy="conservative"):
    """
    Combine quality-gate decision with batch arbitration using the stricter outcome.
    """
    order = {"NO_GO": 3, "REVIEW": 2, "GO": 1}

    if conflict_strategy == "conservative":
        return max(
            [gate_decision, arbitration_decision],
            key=lambda d: order.get(d, 0),
        )

    return arbitration_decision
