# Real model inference setup

Checklist for moving from **simulated** / fallback inference to a live vision-capable backend. The framework already supports `llama_cpp` and `ollama_vision`; this doc is the operational path.

## Before you run a batch

- [ ] At least one image in `test_images/` (or your profile’s `folders.input`).
- [ ] Inference server is running and reachable (see options below).
- [ ] `configs/dev.json` (or your profile) sets `model_settings.inference.backend` to the backend you intend.
- [ ] Timeouts are realistic for your hardware (`timeout_s` in merged config from `configs/base.json`).

**Success signal in reports:** `decision.backend` is `llama_cpp` or `ollama_vision`, **not** `llama_cpp->simulated` or `ollama_vision->simulated`. The `msg` field should not mention “fallback to simulated inference”.

---

## Option A: llama.cpp (OpenAI-compatible HTTP server)

`dev` profile defaults to `llama_cpp` and merges host settings from `configs/base.json`:

| Setting | Default (base) |
|---------|----------------|
| Host | `http://127.0.0.1:8080` |
| Endpoint | `/v1/chat/completions` |
| Model name | `local-model` (must match server) |
| Timeout | `45` seconds |

### 1. Start the server

Use your usual llama.cpp / llama-server launch so it exposes **chat completions** on port `8080` (or change config to match). The model name in the server CLI must match `llama_cpp.model` in config.

### 2. Quick connectivity check

```bash
PYTHONPATH=src python test_connection.py
```

Adjust URL/model inside `test_connection.py` if your server differs. You should see `Connected.` and no connection error.

### 3. Optional dev overrides

`configs/dev.json` only needs the backend key today; add a block under `model_settings.inference` if you use a non-default port or model id:

```json
"inference": {
    "backend": "llama_cpp",
    "llama_cpp": {
        "host": "http://127.0.0.1:8080",
        "model": "your-gguf-model-id",
        "timeout_s": 60
    }
}
```

### 4. Run a small batch (real model)

```bash
python3 src/ai_quality_agent.py --profile dev \
  --inference-backend llama_cpp \
  --async-batch --async-concurrency 4 \
  --parallel-metrics
```

Start with a few images before `--stress-test-100`.

---

## Option B: Ollama (vision model)

### 1. Install and pull a vision model

```bash
ollama pull llava:7b
ollama serve   # if not already running
```

Default in `configs/base.json`: `http://localhost:11434`, model `llava:7b`.

### 2. Check the API

```bash
curl -s http://localhost:11434/api/tags
```

### 3. Point config at Ollama

CLI (no file edit):

```bash
python3 src/ai_quality_agent.py --profile dev --inference-backend ollama_vision
```

Or in config:

```json
"inference": {
    "backend": "ollama_vision",
    "ollama": {
        "host": "http://localhost:11434",
        "model": "llava:7b",
        "timeout_s": 45
    }
}
```

### 4. Run batch

Same as Option A step 4; use `--inference-backend ollama_vision` if not set in JSON.

---

## Which CLI flags when the model is live

| Flag | When to use |
|------|-------------|
| `--parallel-metrics` | Many images; CPU metrics (Pillow) are a large share of wall time. |
| `--async-batch` | Waiting on HTTP inference; limits in-flight requests with `--async-concurrency` (default 4). |
| `--repeatability-test dev --repeatability-runs 5` | Check model output stability across runs (meaningless for pure simulated). |

Simulated-only dev work does **not** need `--async-batch`; real model batches benefit from **both** async I/O and parallel metrics.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `404` on `127.0.0.1:8080` | llama server not running or wrong port/path |
| `backend`: `...->simulated` | Request failed; see `msg` for exception text |
| Same pass rate as before, very low latency | Still on simulated path |
| Timeouts / `ERR_MODEL_BACKEND_503` | Increase `timeout_s`; reduce `--async-concurrency` |
| Ollama errors | Model not pulled, wrong host, or non-vision model |

---

## CI vs local

CI continues to use **simulated** inference for deterministic, fast gates. Real model runs are **local or staging** until you add optional integration jobs with a pinned server image.

See also: [Architecture.md](Architecture.md), README § “Config-only inference backend selection”.
