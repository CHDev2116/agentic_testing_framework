from __future__ import annotations

import logging
import re
import random
import time
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple, cast

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from util.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)
configure_cli_logging()

QualityOrchestrator: Optional[type] = None
try:
    from agent.orchestrator import QualityOrchestrator as _QualityOrchestrator

    QualityOrchestrator = _QualityOrchestrator
except ImportError as exc:
    logger.warning(
        "QualityOrchestrator not importable (%s). From repo root run: "
        "PYTHONPATH=src streamlit run app.py",
        exc,
    )


class _PipelineRunner(Protocol):
    def run_pipeline(self, image_metrics: Dict[str, object]) -> Dict[str, object]: ...


def _get_orchestrator() -> Optional[_PipelineRunner]:
    if QualityOrchestrator is None:
        return None
    if "orchestrator" not in st.session_state:
        st.session_state["orchestrator"] = QualityOrchestrator()
    return cast(_PipelineRunner, st.session_state["orchestrator"])


def _build_sample_image() -> Image.Image:
    width, height = 640, 360
    x_gradient = np.linspace(40, 220, width, dtype=np.uint8)
    y_gradient = np.linspace(0, 25, height, dtype=np.uint8).reshape(height, 1)
    red = np.tile(x_gradient, (height, 1))
    green = np.clip(red.astype(np.int16) + y_gradient - 15, 0, 255).astype(np.uint8)
    blue = np.clip(240 - red // 2 + y_gradient, 0, 255).astype(np.uint8)
    rgb = np.dstack([red, green, blue])
    return Image.fromarray(rgb, mode="RGB")


def _load_sample_image() -> Tuple[Image.Image, str]:
    for candidate in (Path("sample.jpg"), Path("assets/sample.jpg")):
        if candidate.exists():
            return Image.open(candidate).convert("RGB"), str(candidate)
    return _build_sample_image(), "generated"


def _extract_metrics(image: Image.Image) -> Dict[str, object]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    noise_level = float(gray.std())
    gy, gx = np.gradient(gray)
    sharpness = float(np.var(gx) + np.var(gy))
    return {
        "id": "streamlit_uploaded_image",
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "noise_level": round(noise_level, 2),
    }


def _analyze_mock() -> Dict[str, object]:
    time.sleep(1.0)
    # Baseline mode intentionally simulates unstable manual-style judgments.
    score = random.randint(48, 76)
    confidence = round(random.uniform(0.45, 0.72), 2)
    label = "REVIEW" if score >= 60 else "FAIL"
    return {
        "score": score,
        "confidence": confidence,
        "label": label,
        "explanation": (
            "Manual-like baseline: subjective and less consistent; "
            "this mode is for contrast against AI pipeline stability."
        ),
        "raw": {
            "reviewer_note": random.choice(
                [
                    "Looks acceptable but uncertain under low light.",
                    "Borderline sharpness; might require human review.",
                    "Inconsistent judgement due to subjective threshold.",
                ]
            )
        },
    }


def _normalize_real_result(report: Dict[str, object]) -> Dict[str, object]:
    verdict = str(report.get("final_verdict", "REVIEW"))
    score_map = {"PASS": 90, "GO": 90, "REVIEW": 70, "FAIL": 40, "NO_GO": 30}
    confidence_map = {"PASS": 0.90, "GO": 0.90, "REVIEW": 0.75, "FAIL": 0.60, "NO_GO": 0.55}
    return {
        "score": score_map.get(verdict, 65),
        "confidence": confidence_map.get(verdict, 0.70),
        "label": verdict,
        "explanation": (
            f"Pipeline stage: {report.get('stage', 'Unknown')}; "
            f"engine: {report.get('engine', 'N/A')}"
        ),
        "raw": report,
    }


def run_analysis(image: Image.Image, mode: str) -> Dict[str, object]:
    if mode == "Manual Baseline (for contrast)":
        return _analyze_mock()

    orchestrator = _get_orchestrator()
    if orchestrator is None:
        return {
            "score": 65,
            "confidence": 0.7,
            "label": "Fallback",
            "explanation": "Real pipeline unavailable, fallback to stub output.",
            "raw": {"reason": "agent.orchestrator import failed; use PYTHONPATH=src from repo root"},
        }

    try:
        metrics = _extract_metrics(image)
        report = orchestrator.run_pipeline(metrics)
        return _normalize_real_result(report)
    except Exception as exc:
        logger.exception("AI pipeline run failed")
        return {
            "score": 60,
            "confidence": 0.6,
            "label": "Error",
            "explanation": "Real pipeline failed; check raw output for details.",
            "raw": {"error": str(exc)},
        }


def parse_llm_output(text: str) -> Dict[str, object]:
    """Demo parser for messy LLM text to normalized fields."""
    parsed: Dict[str, object] = {"score": None, "confidence": None, "label": "unknown"}
    lower_text = text.lower()

    score_match = re.search(r"(?:score|points?)\s*[:=]?\s*(\d{1,3})", lower_text)
    if score_match:
        score = int(score_match.group(1))
        parsed["score"] = max(0, min(score, 100))
    else:
        # Fallback: first plausible 0~100 integer in text
        generic = re.search(r"\b(\d{1,3})\b", lower_text)
        if generic:
            score = int(generic.group(1))
            if 0 <= score <= 100:
                parsed["score"] = score

    confidence_match = re.search(r"(?:confidence)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", lower_text)
    if confidence_match:
        parsed["confidence"] = float(confidence_match.group(1))

    if any(keyword in lower_text for keyword in ("excellent", "good", "great", "pass")):
        parsed["label"] = "positive"
    elif any(keyword in lower_text for keyword in ("bad", "poor", "fail")):
        parsed["label"] = "negative"

    return parsed


st.set_page_config(page_title="Agentic Testing Framework", layout="wide")

st.title("Agentic Testing Framework")
st.markdown("Replace manual image QA with a repeatable, config-driven pipeline.")
st.caption("From slow & inconsistent to fast & scalable")

if "result" not in st.session_state:
    st.session_state["result"] = None
if "compare_result" not in st.session_state:
    st.session_state["compare_result"] = None
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "source_name" not in st.session_state:
    st.session_state["source_name"] = ""

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Analysis mode",
        options=["Manual Baseline (for contrast)", "AI Pipeline (real)"],
        index=0,
    )
    show_raw = st.checkbox("Show raw output", value=True)
    show_latency = st.checkbox("Show latency", value=True)
    use_sample = st.button("Try sample image")
    if mode == "AI Pipeline (real)" and QualityOrchestrator is None:
        st.warning(
            "`agent.orchestrator` could not be imported. From the **repository root** run: "
            "`PYTHONPATH=src streamlit run app.py`"
        )

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if use_sample:
    sample_image, source_name = _load_sample_image()
    st.session_state["selected_image"] = sample_image
    st.session_state["source_name"] = source_name
    st.session_state["result"] = None
    st.session_state["compare_result"] = None
elif uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.session_state["selected_image"] = uploaded_image
        st.session_state["source_name"] = uploaded_file.name
        st.session_state["result"] = None
        st.session_state["compare_result"] = None
    except UnidentifiedImageError:
        logger.warning("Upload rejected: not a decodable image")
        st.error("Cannot decode this file as an image. Please upload PNG/JPG.")
    except Exception as exc:
        logger.exception("Failed to read uploaded image")
        st.error(f"Failed to read upload: {exc}")

image = st.session_state["selected_image"]

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.caption(f"Source: {st.session_state['source_name']}")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Analysis Result")
        if mode == "Manual Baseline (for contrast)":
            st.caption("Baseline mode: simulates subjective/manual-style checks.")
        else:
            st.caption("AI mode: uses orchestrator pipeline for reproducible decisions.")
        if st.button("Analyze", type="primary"):
            start = time.time()
            with st.spinner("Running AI + rules..."):
                result = run_analysis(image, mode)
            latency = time.time() - start
            st.session_state["result"] = {"payload": result, "latency": latency}
            st.session_state["compare_result"] = None

        if st.button("Compare Both Modes"):
            with st.spinner("Running baseline and AI pipeline..."):
                baseline_start = time.time()
                baseline_result = run_analysis(image, "Manual Baseline (for contrast)")
                baseline_latency = time.time() - baseline_start

                ai_start = time.time()
                ai_result = run_analysis(image, "AI Pipeline (real)")
                ai_latency = time.time() - ai_start

            st.session_state["compare_result"] = {
                "baseline": {"payload": baseline_result, "latency": baseline_latency},
                "ai": {"payload": ai_result, "latency": ai_latency},
            }
            st.session_state["result"] = None

        cached_result = st.session_state["result"]
        if cached_result:
            result = cached_result["payload"]
            st.metric("Score", f"{result['score']}/100")
            st.metric("Confidence", f"{result['confidence']:.2f}")
            st.write(f"**Label:** {result['label']}")
            if show_latency:
                st.write(f"Latency: {cached_result['latency']:.2f}s")
            st.markdown("### Explanation")
            st.write(result["explanation"])
            st.info("Robust parsing layer keeps model output structured for downstream QA.")
            if show_raw:
                with st.expander("Raw output"):
                    st.json(result["raw"])

        compare_result = st.session_state["compare_result"]
        if compare_result:
            st.markdown("### Side-by-side Comparison")
            compare_left, compare_right = st.columns(2)

            baseline = compare_result["baseline"]
            ai = compare_result["ai"]

            with compare_left:
                st.markdown("**Manual Baseline**")
                st.metric("Score", f"{baseline['payload']['score']}/100")
                st.metric("Confidence", f"{baseline['payload']['confidence']:.2f}")
                st.write(f"Label: {baseline['payload']['label']}")
                if show_latency:
                    st.write(f"Latency: {baseline['latency']:.2f}s")

            with compare_right:
                st.markdown("**AI Pipeline**")
                st.metric("Score", f"{ai['payload']['score']}/100")
                st.metric("Confidence", f"{ai['payload']['confidence']:.2f}")
                st.write(f"Label: {ai['payload']['label']}")
                if show_latency:
                    st.write(f"Latency: {ai['latency']:.2f}s")

            delta_score = ai["payload"]["score"] - baseline["payload"]["score"]
            st.info(f"AI minus Baseline score delta: {delta_score:+.0f} points")
else:
    st.info("Upload an image or click 'Try sample image' to start.")

st.divider()
st.subheader("Impact")
left, right = st.columns(2)
with left:
    st.markdown("### Before: Manual / Subjective Checks")
    st.write("- Human judgment varies by reviewer")
    st.write("- Hard to keep thresholds consistent")
    st.write("- Slower and less traceable decisions")
with right:
    st.markdown("### After: AI Pipeline Decisions")
    st.write("- Config-driven, repeatable decision policy")
    st.write("- Structured output for audit and CI")
    st.write("- Fast, scalable, and easier to govern")

st.divider()
st.subheader("How It Works")
st.markdown(
    """
1. Extract image features (blur, noise, exposure)
2. Apply rule-based validation
3. Use AI for semantic reasoning
4. Normalize output into structured format
"""
)

st.divider()
st.subheader("LLM Output Parsing Demo")
st.caption("Raw LLM output can be messy; parsing normalizes it into stable structured data.")

raw_outputs = [
    "Score: 85/100, confidence: 0.91, label: good",
    "I think this image is around 78 points with decent quality.",
    "Result => score=92; label=excellent; confidence=0.95",
    "This looks bad. Probably 60.",
]
raw_text = st.selectbox("Select LLM output example", raw_outputs)
demo_left, demo_right = st.columns(2)
with demo_left:
    st.markdown("### Raw LLM Output")
    st.code(raw_text)
with demo_right:
    st.markdown("### Parsed Output")
    st.json(parse_llm_output(raw_text))

st.info("Without parsing: unstable system. With parsing: reliable pipeline.")

st.divider()
st.caption("Demo for AI-powered testing / DevRel showcase")