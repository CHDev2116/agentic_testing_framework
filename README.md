# Agentic Testing Framework: Quantized Vision QA

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pillow](https://img.shields.io/badge/Library-Pillow-orange.svg)
[![CI](https://github.com/CHDev2116/agentic_testing_framework/actions/workflows/ci.yml/badge.svg)](https://github.com/CHDev2116/agentic_testing_framework/actions/workflows/ci.yml)

Configuration-driven framework to evaluate image quality and make production release decisions: `GO` / `REVIEW` / `NO_GO`.

## Quick Start (CLI Pipeline)

```bash
git clone https://github.com/CHDev2116/agentic_testing_framework
cd agentic_testing_framework
python -m pip install -U pip
pip install -e ".[dev]"
python3 src/ai_quality_agent.py --profile dev
```

`requirements.txt` is a thin shim (`-e .[dev]`) for `pip install -r requirements.txt`; **all version pins live in `pyproject.toml`** (`[project.dependencies]`).

If no input images are present, sample images are auto-generated.

<details>
<summary><strong>DX: Built for Extensibility</strong></summary>

This repo optimizes for **integrators**: swap runtimes without rewriting the batch pipeline, keep a **fixed downstream contract**, and emit **auditable JSON** (not “score-only” blobs).

### Config-only inference backend selection

- Set `model_settings.inference.backend` in `configs/*.json` to one of: `simulated`, `ollama_vision`, `mock_api`, `llama_cpp`.
- For ad-hoc runs, the CLI can override without editing files: `python3 src/ai_quality_agent.py --profile dev --inference-backend mock_api` (see `--help`).
- **Connecting a live model** (llama.cpp server or Ollama vision): see [`docs/ModelInferenceSetup.md`](docs/ModelInferenceSetup.md).
- Composition root: `build_inference_engine()` in [`src/models/inference_adapter.py`](src/models/inference_adapter.py) selects the concrete engine class from config.

Same codebase path runs locally (simulated / Ollama / llama.cpp HTTP) or against a mock HTTP API—**no forked “deploy-only” branch** unless your infra truly requires it.

### Orchestrator contract: one method shape, normalized outputs

Engines are **not** tied to a shared ABC in this codebase. Each backend class implements the same surface:

`predict_quality(photo_path: str, metrics: dict) -> dict`

Return dicts are passed through `_normalize_result(...)` so downstream code sees a **stable schema**: at minimum `decision`, `code`, `msg`, plus optional `confidence`, and `backend` (including `provider->simulated` when fallback fires).

**Adding a new backend** today means: implement that method + normalize through `_normalize_result`, then add a branch in `build_inference_engine`. If you want static enforcement later, a `typing.Protocol` (or an ABC) is an incremental hardening step—the factory stays the single registry for CI/review friendliness.

### Actionable batch artifacts

- Per-inference payloads retain **`code`** (machine-oriented) and **`msg`** (human-oriented) after normalization—failures are classified, not opaque.
- Batch summaries include **`summary.decision_reason`**: a single string that records how **quality-gate** and **aggregated arbitration** were merged (`merge_gate_and_arbitration`), so **why** the merged outcome is `GO` / `REVIEW` / `NO_GO` is reproducible from the JSON without re-running the batch.

See also: [`docs/Architecture.md`](docs/Architecture.md) for the provider contract and fallback behavior.

</details>

## Demo UI (Streamlit)

From the **repository root** (so `src` is importable as top-level packages):

```bash
PYTHONPATH=src streamlit run app.py
```

The **AI Pipeline** mode imports `agent.orchestrator`; if imports fail, the UI shows a `PYTHONPATH=src` hint. **Manual Baseline** mode works without the orchestrator.

<details>
<summary><strong>Demo preview & optional assets</strong></summary>

Streamlit UI: generated sample input, **Manual Baseline** vs **AI Pipeline** side-by-side (score, confidence, label, latency), and score delta summary.

![Streamlit demo: baseline vs AI pipeline comparison](assets/streamlit-comparison.png)

Optional: add a short screen recording as `assets/demo.gif` and reference it here for motion (e.g. clicking **Analyze** / **Compare Both Modes**).

</details>

<details>
<summary><strong>Project identity</strong></summary>

| Item | Value |
|------|--------|
| **Display name** | Agentic Testing Framework |
| **Python package** (`pyproject.toml`) | `agentic_testing_framework` |
| **Default model profile label** (`configs/*.json` → `model_settings.name`) | `Agentic Testing Framework - Llama 4-bit` |
| **Docker image tag** (example) | `agentic-testing-framework:latest` |
| **Memory profiling** (`src/util/monitor_performance.py`) | Prefer `ATF_MONITOR_MEMORY=1`; legacy alias `PIXELQA_MONITOR_MEMORY` still works |

</details>

<details>
<summary><strong>Why this project</strong></summary>

- Automates repetitive image QA with consistent decision policy.
- Supports multiple inference backends (`simulated`, `ollama_vision`, `mock_api`, `llama_cpp`).
- Keeps results traceable with ranking, reports, and guardrail-driven recovery.

</details>

<details>
<summary><strong>Core guarantees (source of truth)</strong></summary>

- **Architecture**: `Engine -> Model -> Eval` with clear boundaries.
- **Decision policy**: conservative release gating (`GO` / `REVIEW` / `NO_GO`).
- **Loopback**: `NO_GO` recovery includes brighten/dim/sharpen strategies under retry limits.
- **Retention**: auto-clean for `batch_report_*.json` and `error_report_*.json` after 14 days.
- **CI scope**: Ruff on `src` + `tests` + `app.py` + `test_connection.py`; **mypy** on `src` then on `app.py` / `test_connection.py` with `MYPYPATH=src`; pytest with coverage (including **`--cov-fail-under=34`**). Tests emphasize the **release decision path** (arbitration, inference result normalization, loopback integration) and **golden checks** for batch ranking, release gates, log stability windows, Pillow-based vision metrics, and OpenCV exposure validation—see `tests/`.

</details>

<details>
<summary><strong>Pipeline flow</strong></summary>

```mermaid
flowchart LR
    A[Test Images] --> B[Engine Layer<br/>Brightness / Sharpness Metrics]
    B --> C[Model Layer<br/>Inference Backend]
    C --> D[Eval Layer<br/>Ranking / Arbitration / Release Decision]
    D --> E[Reports<br/>Batch / Comparison / Repeatability / Performance]
    D -- NO_GO: Guardrail Loopback --> B
```

</details>

## Usage

**Basic runs:**

```bash
python3 src/ai_quality_agent.py --profile dev
python3 src/ai_quality_agent.py --profile benchmark
python3 src/ai_quality_agent.py --config configs/dev.json
```

<details>
<summary><strong>Advanced CLI</strong></summary>

```bash
python3 src/ai_quality_agent.py --compare-profiles dev benchmark
python3 src/ai_quality_agent.py --repeatability-test dev --repeatability-runs 5
python3 src/ai_quality_agent.py --profile benchmark --inference-backend mock_api
python3 src/ai_quality_agent.py --profile dev --performance-analysis
python3 src/ai_quality_agent.py --profile dev --stress-test-100 --performance-analysis
python3 src/ai_quality_agent.py --profile dev --overhead-analysis
python3 src/ai_quality_agent.py --profile dev --parallel-metrics
python3 src/ai_quality_agent.py --profile dev --async-batch --async-concurrency 4
python3 src/ai_quality_agent.py --profile dev --async-batch --parallel-metrics
python3 src/test_failure_memory_retrieval.py
```

</details>

<details>
<summary><strong>Docker (optional)</strong></summary>

```bash
docker build -t agentic-testing-framework:latest .
docker run --rm \
  -v "$(pwd)/test_images:/app/test_images" \
  -v "$(pwd)/results:/app/results" \
  agentic-testing-framework:latest
```

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

- **Ollama not responding**: check `http://localhost:11434`
- **No images found**: samples are auto-generated
- **Slow performance**: try `--inference-backend simulated`

</details>

<details>
<summary><strong>CI / local tests</strong></summary>

```bash
python -m pip install -U pip
pip install -e ".[dev]"
ruff check src tests app.py test_connection.py
mypy --explicit-package-bases src
MYPYPATH=src mypy --explicit-package-bases app.py test_connection.py
PYTHONPATH=src pytest
```

Docker and other installs use **`pyproject.toml` only** for dependency pins (`pip install .` in the Dockerfile). The `requirements.txt` shim is optional for local workflows.

Workflow reference: `.github/workflows/ci.yml`

</details>

<details>
<summary><strong>Deeper documentation & roadmap</strong></summary>

**Docs**

- Architecture and provider details: [`docs/Architecture.md`](docs/Architecture.md)
- For benchmark, repeatability, and reliability narratives, use docs + report artifacts under `results/`.

**Roadmap (shipped)**

- [x] Multi-backend inference abstraction
- [x] Batch ranking + release arbitration
- [x] Repeatability / performance / overhead analysis
- [x] Automated JSON error reporting with retention

**Backlog (intentionally deferred)**

- **Multi-threading for very large batches**: not on the near-term roadmap so batch runs stay **single-threaded and easier to reproduce** in CI, benchmarks, and incident debugging. Revisit only after profiling shows preprocessing (not inference I/O) as the clear bottleneck.
- **Extended OpenCV visual diagnostics**: basic histogram-based exposure checks already live in `engine/image_validator.py`; richer diagnostics (e.g. saliency, segmentation-assisted QA) stay **out of scope** until there is a concrete partner or product requirement, to avoid scope creep ahead of a stable inference contract.

</details>

## Author

Cheryl - AI Optimization & Testing Engineer

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, tests, lint, and pull request expectations.

## License

This project is licensed under the [MIT License](LICENSE).
