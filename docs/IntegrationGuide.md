# Integration guide: ship your first inference backend in ~3 minutes

This page is written for **external integrators** the same way a DevRel team would onboard a partner: a **fast path**, a **stable contract**, and **support-style troubleshooting**—not only internal architecture notes.

For the full provider design rationale, see [`Architecture.md`](Architecture.md).

---

## Who this is for

- You are wiring **your own inference service** (HTTP) into the batch QA pipeline, **or**
- You want a **zero-risk first run** (`simulated`) before touching GPUs / remote APIs, **or**
- You are evaluating how this framework behaves when **LLM / multimodal APIs** fail, time out, or return non-JSON.

This aligns with roles that emphasize **sample integrations**, **API troubleshooting**, and **clear developer contracts** (stable schemas, timeouts, fallbacks).

---

## Prerequisites

- Python **3.9+**
- Project root as working directory (paths below assume you `cd` into the repo)

```bash
git clone https://github.com/CHDev2116/agentic_testing_framework.git
cd agentic_testing_framework
pip install -r requirements.txt
```

Run the CLI from the repo root:

```bash
python3 src/ai_quality_agent.py --help
```

---

## The integration contract (what your backend must satisfy)

The orchestrator calls **one method** on the selected engine:

```text
predict_quality(photo_path: str, metrics: dict) -> dict
```

After normalization, consumers expect at minimum:

| Field       | Meaning |
|------------|---------|
| `decision` | One of: `Optimal`, `Blurry`, `Under-exposed`, `Over-exposed`, `Error` |
| `code`     | Stable machine-oriented code (e.g. `SUCCESS_200`, `ERR_MODEL_BACKEND_503`) |
| `msg`      | Human-readable explanation |
| `backend`  | Provider id (e.g. `mock_api`; may show `ollama_vision->simulated` on fallback) |

Optional: `confidence` (float in `[0, 1]` when supported).

**Integration tip:** treat `code` + `msg` as what you would expose to **automations** vs **humans** in support queues—batch summaries and error reports in this repo preserve that split.

---

## Track A — ~3 minutes: first backend with zero external deps (`simulated`)

**Goal:** prove the pipeline, folders, and JSON reports work on your machine.

```bash
python3 src/ai_quality_agent.py --profile base --inference-backend simulated
```

What you should see:

- Log line similar to: `Inference backend: simulated`
- Outputs under `results/base/` (per `configs/base.json` → `folders.output`)

If `test_images/` is empty, the runner **auto-generates** sample inputs (see README).

---

## Track B — ~3 minutes: first **HTTP** backend (`mock_api`)

**Goal:** mirror how you would integrate a **proprietary or partner inference API** without adopting Ollama or llama.cpp yet.

### 1) Start a minimal compatible server (copy-paste)

Your server must accept **POST** JSON with:

- `photo_path` (string)
- `metrics` (object)
- `thresholds` (object)

and return JSON that is either:

- `{ "result": { "decision": "...", "code": "...", "msg": "..." } }`, **or**
- a bare object `{ "decision": "...", "code": "...", "msg": "..." }`.

Optional auth: if you set `MOCK_INFER_API_KEY` in the environment, the client sends `Authorization: Bearer <token>` (config key `model_settings.inference.mock_api.api_key_env`, default `MOCK_INFER_API_KEY`).

Example (stdlib only; suitable for local dev):

```python
#!/usr/bin/env python3
"""Minimal mock inference server for agentic_testing_framework mock_api backend."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

API_KEY = os.environ.get("MOCK_INFER_API_KEY")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/infer":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))

        auth = self.headers.get("Authorization", "")
        if API_KEY:
            if auth != f"Bearer {API_KEY}":
                self.send_response(401)
                self.end_headers()
                return

        metrics = body.get("metrics") or {}
        sharp = float(metrics.get("sharpness", metrics.get("laplacian_variance", 50)))
        decision = "Optimal" if sharp >= 30 else "Blurry"
        result = {
            "decision": decision,
            "code": "SUCCESS_200",
            "msg": "mock_api stub classification",
        }
        payload = json.dumps({"result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return  # quieter local server


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
```

Run it in a separate terminal:

```bash
python3 mock_server.py
```

### 2) Point the framework at it

Default URL in `configs/base.json` is `http://localhost:8080/infer`. Run:

```bash
python3 src/ai_quality_agent.py --profile base --inference-backend mock_api
```

You should see `Inference backend: mock_api` and per-image results with `backend: mock_api` when the server is healthy.

### 3) Override URL without editing files

Copy `configs/base.json` to `configs/local.mock.json`, adjust `model_settings.inference.mock_api.url`, then:

```bash
python3 src/ai_quality_agent.py --config configs/local.mock.json --inference-backend mock_api
```

---

## Track C — GenAI / multimodal style endpoints (Ollama & llama.cpp)

These backends are the closest analog to **“integrate an LLM / multimodal API”** in this repo: HTTP client, JSON parsing, timeouts, optional **fallback to `simulated`** so batches stay actionable.

| Backend        | Typical use case |
|----------------|------------------|
| `ollama_vision` | Local Ollama `/api/generate` with `images` + JSON-style response |
| `llama_cpp`   | OpenAI-compatible `POST /v1/chat/completions` (e.g. llama.cpp server) |

Configuration lives under `model_settings.inference` in your profile JSON. Defaults are documented in [`Architecture.md`](Architecture.md) and illustrated in [`configs/base.json`](../configs/base.json).

CLI override (no file edit):

```bash
python3 src/ai_quality_agent.py --profile base --inference-backend ollama_vision
python3 src/ai_quality_agent.py --profile dev --inference-backend llama_cpp
```

**Ollama quick checklist**

- Daemon reachable: `http://localhost:11434` (or your `ollama.host`)
- Model pulled: e.g. a vision-capable tag matching `ollama.model`
- Responses should be parseable JSON with keys `decision`, `code`, `msg` (the client enables `format: "json"`)

**llama.cpp server quick checklist**

- Base URL + `endpoint` default to `http://127.0.0.1:8080` + `/v1/chat/completions`
- If the server rejects `response_format`, the client **retries without** that field (see `LlamaCppInferenceEngine` in `src/models/inference_adapter.py`)

---

## Troubleshooting (support-queue style)

### 1) `Connection refused` / timeouts

| Symptom | Likely cause | What to try |
|--------|----------------|-------------|
| `mock_api` errors mentioning connection | Server not listening or wrong port/path | `curl -v http://127.0.0.1:8080/infer` (expect 404 on GET; test POST with `curl -d @payload.json`) |
| Ollama errors | Daemon down or wrong host | Open `host` in browser or `curl` `/api/tags` |
| Slow first call | Cold model load | Increase `timeout_s` in config; warm up model once |

### 2) `backend` shows `something->simulated`

This is **fallback**, not silent success: the remote path failed, and the framework returned a **normalized** simulated decision for continuity.

- Set `"fallback_to_simulated": false` under `model_settings.inference` if you want **hard failures** instead (useful when validating a new partner API).
- Read `msg`—it includes the original exception context.

### 3) `ERR_MODEL_RESPONSE_422` or “unparsable response”

The model returned text that is not JSON with the required keys.

- Tighten the prompt (`prompt_template`) to “return **only** JSON with keys decision, code, msg”.
- For `llama_cpp`, try `use_response_format: true` if the server supports JSON mode; otherwise rely on substring JSON extraction (already implemented).

### 4) 401 from `mock_api`

You set `MOCK_INFER_API_KEY` (or custom env via `api_key_env`) but the server and client disagree on the token.

### 5) Wrong profile / wrong output folder

`--profile dev|benchmark|base` selects `configs/<profile>.json` (`folders.output` differs per profile). Use `--config` for a custom file.

---

## How this maps to a DevRel-style interview narrative

When discussing **agentic / GenAI integrations**, you can point to this repo as:

1. **A stable downstream contract** (`decision` / `code` / `msg`) across heterogeneous backends.  
2. **A composition root** (`build_inference_engine`) that keeps the registry explicit and auditable.  
3. **Operational empathy**: timeouts, structured errors, optional simulated fallback so integrators are not blocked by a flaky API during batch QA.  
4. **A runnable “hello integration”** — Track A + Track B above — analogous to shipping **sample code** and a **minimal server** for partners.

---

## 繁體中文摘要（利害關係人溝通用）

- **約 3 分鐘首跑**：用 `--inference-backend simulated` 先驗證本機 pipeline 與報告輸出。  
- **約 3 分鐘接 HTTP**：用 `mock_api` 後端對照「合作夥伴／內部推理服務」的 JSON 契約；README 與本頁提供最小 server 範例與 `curl` 排查思路。  
- **對接多模態／LLM API**：`ollama_vision` 與 `llama_cpp` 展示 timeout、JSON 解析、可選 fallback；細節見 [`Architecture.md`](Architecture.md)。  
- **除錯習慣**：先看 `backend` 是否帶 `->simulated`（代表遠端失敗但已降級），再看 `code`／`msg` 分別服務自動化與人工支援流程。

---

## Related links

- Architecture & provider behavior: [`Architecture.md`](Architecture.md)  
- Project overview & Streamlit demo: [`README.md`](../README.md)
