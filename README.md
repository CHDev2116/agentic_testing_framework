# Agentic Testing Framework: Quantized Vision QA

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pillow](https://img.shields.io/badge/Library-Pillow-orange.svg)

## 📌 Project Overview
This project is an **automated testing framework** for mobile image-quality validation. It simulates how a **4-bit lightweight model (Quantized Vision Model)** can evaluate image quality in real time under resource-constrained environments such as phones.

The framework follows a **configuration-driven** design, fully decoupling quality thresholds from execution logic so it can quickly adapt to different quantized model standards.

All experiments are reproducible via fixed config profiles and a deterministic preprocessing pipeline.

## 🤖 AI Honesty Statement

Current state:
- **Real**: image metrics are computed from real files (brightness/sharpness).
- **Simulated**: model decision is currently rule-based (not a live LLM/VLM endpoint yet).

Next integration:
- Connect inference to **Ollama**.
- Swap in a **LLaMA-family vision-capable model** (or equivalent multimodal model).
- Keep the same three-layer architecture so ranking, decision, and benchmarking stay reusable.

## 🛠️ Technical Highlights

### 1. Modular Architecture and Config-Driven Design
* **Fully decoupled**: Uses `configs/*.json` to manage all test standards (sharpness/brightness thresholds), so strategies can be adjusted without code changes.
* **Engine layer**: feature extraction from input images (brightness/sharpness metrics).
* **Model layer**: inference abstraction (rule-based today, real-model adapter-ready).
* **Eval layer**: scoring, ranking, benchmark insights, and release decision.

### 2. Batch Processing and Performance Monitoring
* **Automated pipeline**: Scans the `test_images/` directory automatically, without manually specifying files.
* **Performance tracking**: Built-in **Latency Tracking** records per-image processing time for inference efficiency analysis.
* **Dashboard summary**: Automatically reports **Pass Rate** and **Average Latency** when testing completes.

### 3. Resilience and Error Handling
* **OOM stress simulation**: Includes a random memory-overflow simulator to validate system stability in extreme conditions.
* **Safety-net flow**: Uses `try-except-finally` to ensure the system still produces a context-rich **Crash Report (JSON)** even after failures.

## ✅ Delivery Targets

- **1. Clone repo and run immediately**: if no input images exist, the runner auto-generates sample images.
- **2. Produce comparable results**: run multiple profiles on the same image set and export a comparison report.
- **3. Provide ranking + decision**: every run outputs per-image ranking and a final release decision (`GO` / `REVIEW` / `NO_GO`).
- **4. Keep a clear three-layer architecture**: `engine` (feature extraction), `model` (inference abstraction), `eval` (scoring + decision).

## 📂 Directory Structure
```text
agentic_testing_framework/
├── configs/              # Environment-based configs (base/dev/benchmark)
├── src/
│   ├── engine/           # Feature extraction modules
│   ├── models/           # Inference abstraction layer
│   ├── eval/             # Scoring, ranking, and decision logic
│   └── ai_quality_agent.py # Orchestrator for batch flow and reporting
├── test_images/          # Input images for testing
├── results/
│   ├── dev/              # Per-run reports for dev profile
│   ├── benchmark/        # Per-run reports for benchmark profile
│   └── comparisons/      # Cross-profile comparison reports
└── README.md

## 🚀 Usage

Run from the project root:

```bash
# Install dependencies
pip install -r requirements.txt

# Use the development profile (configs/base.json + configs/dev.json)
python3 src/ai_quality_agent.py --profile dev

# Use the benchmark profile (configs/base.json + configs/benchmark.json)
python3 src/ai_quality_agent.py --profile benchmark

# Load a config file directly (applied on top of configs/base.json)
python3 src/ai_quality_agent.py --config configs/dev.json

# Compare multiple profiles and output a cross-profile ranking
python3 src/ai_quality_agent.py --compare-profiles dev benchmark

# Repeatability test: same image set, run 5 times, report variance
python3 src/ai_quality_agent.py --repeatability-test dev --repeatability-runs 5
```

Notes:
- `--profile` supports: `dev`, `benchmark`, `base`
- `--config` accepts either an absolute path or a project-root-relative path
- `--compare-profiles` runs each profile and creates `results/comparisons/profile_comparison_*.json`
- `--repeatability-test` runs the same profile repeatedly and writes `results/repeatability/repeatability_*.json`
- Reports are auto-cleaned when older than 14 days

## 📤 Output Example

Startup mode: PixelQA-Llama-4bit (4-bit)
Starting to process 3 image(s)...

Processed sample_good.png: [SUCCESS_200] Optimal (4.85ms)
Processed sample_dark.png: [ERR_LIGHT_DARK_002] Under-exposed (4.12ms)

=======================================================
Test Dashboard
  - Total tests: 3
  - Pass rate (Optimal): 66.7%
  - Average latency: 4.66 ms
  - Release decision: REVIEW
-------------------------------------------------------
Top ranking:
  #1 sample_good.png | score=84.2 | Optimal
  #2 sample_bright.png | score=31.6 | Over-exposed
=======================================================

## 👉 Benchmark Insights

1) **Trade-off**: stricter quality thresholds improve screening confidence but reduce pass rate.  
   **Observation**: benchmark profile usually keeps similar image ordering but can produce a stricter release outcome.  
   **Decision implication**: use benchmark profile for certification or final quality checks, and dev profile for faster iteration.

2) **Trade-off**: lower latency can hide quality risk if used alone.  
   **Observation**: one profile may be fastest while still returning `REVIEW` or `NO_GO`.  
   **Decision implication**: select profile by combined metrics (`pass_rate` + `avg_latency_ms` + `release_decision`), not speed only.

3) **Trade-off**: ranking is useful prioritization, not a release verdict by itself.  
   **Observation**: top-ranked images can coexist with failed/under-exposed samples in the same run.  
   **Decision implication**: keep `ranking` for debugging and prioritization, and keep `GO/REVIEW/NO_GO` as the final gate.

Project alignment:
- `--compare-profiles` now writes these insights into `results/comparisons/profile_comparison_*.json` under `benchmark_insights`.

## 🔁 Repeatability Example

Command used:

```bash
python3 src/ai_quality_agent.py --repeatability-test dev --repeatability-runs 5
```

Observed output (same image set, 5 runs):
- `same_image_set`: `True`
- `pass_rate_variance`: `0.0`
- `avg_latency_variance`: `0.8026`
- `max_per_image_score_variance`: `0.0`
- `decision_distribution`: `{"REVIEW": 5}`

Interpretation:
- The quality outputs are stable across repeated runs on the same image batch.
- Runtime latency varies slightly by environment/load, while ranking and release decision remain consistent in this run.

Threshold calibration is performed per profile using benchmark feedback to balance pass-rate targets and false-positive risk.
The architecture scales from small local test sets to larger benchmark batches by keeping feature extraction, inference abstraction, and eval logic independently extensible.

Roadmap

[ ] Multi-threading optimization: implement parallel test execution for larger datasets.

[ ] Real model integration: connect to a real Llama-3 Vision model through Ollama.

[ ] Visual analytics: add OpenCV support for finer color-shift and noise analysis.

👤 Author
Cheryl - AI Optimization & Testing Engineer