# Interview narratives: Agentic Testing Framework

Use this as a **spoken script outline**, not a document to hand to interviewers. Your repo + `IntegrationGuide.md` + `Architecture.md` are the receipts; this file is for **rehearsal timing** and **story arc**.

Target roles: **Developer Relations / Developer Advocacy** with **GenAI / multimodal / agentic integrations** and **partner-style support** expectations.

---

## 版本一：90 秒電梯演講（繁中）

**計時目標：約 85–95 秒，正常語速。**

> 我做的是一個 **設定驅動的影像品質批次評估框架**，目的是在 release 前，用同一套 pipeline 產出 **GO／REVIEW／NO_GO** 這種可稽核的決策，而不是只有分數。
>
> 實務上最痛的是：**推理後端一直在換**——本機規則、Ollama、OpenAI 相容的 llama.cpp server、或合作方的 HTTP API。很多團隊會 fork 一支「部署版」程式，結果報告跟本機對不起來。
>
> 我的做法是抽一層 **inference provider**：對上只有一個方法 `predict_quality`，對下把各種回傳 **normalize 成固定 schema**——至少 `decision`、`code`、`msg`，再加上 `backend` 標記來源。這樣 **下游評分、仲裁、報告** 都不用因為換模型而改。
>
> 我也把 **整合者體驗** 當產品做：`configs` 裡換 backend，或 CLI 一個 flag 覆寫；遠端掛掉時可以選 **fallback 到 simulated**，批次還是跑得完，但 `msg` 會留下例外脈絡，方便支援與除錯，而不是靜默錯結果。
>
> 技術上這是 Python batch pipeline + 多後端 HTTP client + Streamlit demo；CI 有跑。若你問這跟 DevRel 有什麼關係：**我寫的是「別人接得進來的契約」**——我另外補了一篇對外整合指南，三分鐘可以從 simulated 接到 mock HTTP，再接到多模態推理，這就是我對 **sample integration + troubleshooting narrative** 的態度。

**一句收尾（可選，加 5 秒）：**

> 如果我在貴團隊，我會用同一套方法對 **公開 API**：文件、最小可跑範例、錯誤碼語意、以及論壇裡一則 issue 能複現的 repro。

---

## 版本二：10 分鐘深挖（繁中）

**結構：約 10 分鐘；每段附「若時間被壓縮要砍哪裡」。**

### 0:00–0:45 — 問題與誰會痛（Why）

- **問題**：影像／多模態 QA 在 release gate 要一致、可解釋、可重跑；但 **模型與推理基礎設施** 變動快。
- **誰是「開發者」**：在這個專案裡我把 **integrator** 當使用者——要接新後端的人、要讀 batch JSON 的人、要在 CI 重現的人。
- **壓縮時**：只留一句「後端可換、契約固定」。

### 0:45–2:30 — 你做了什麼（What），一句 demo 路徑

- **核心輸出**：批次報告 + 決策政策（含 gate 與仲裁合併理由，可在 JSON 追溯）。
- **兩條入口**（講清楚別誤導）：
  - **主路徑**：`ai_quality_agent.py` CLI → batch QA（面試主線講這個）。
  - **次要**：`agent/orchestrator.py` 是多階段實驗管線，**預設沒接到 CLI**；可誠實說「預留擴充／實驗」，避免被深挖時穿幫。
- **可視化**：Streamlit 並排對照 baseline vs pipeline（加分，30 秒帶過即可）。
- **壓縮時**：刪 Streamlit，只留 CLI。

### 2:30–5:30 — 架構與關鍵設計（How），對齊「API / 平台型 DevRel」

用白板或口頭 **三層** 即可：

1. **Engine**：影像指標（brightness / sharpness 等）。
2. **Model**：`build_inference_engine(config)` 選具體 backend；**normalize** 成統一 dict。
3. **Eval**：仲裁、批次彙總、release 決策與報告。

**深挖三個設計點（選你最有把握的 2 個講滿）：**

- **Explicit registry（工廠 if/elif）**：寧可寫清楚，方便 CI 與資安 review；呼應大廠對 **可審計整合點** 的偏好。
- **Stable machine vs human surface**：`code` 給自動化、`msg` 給人讀；呼應 JD 裡 **support queue / debug** 場景。
- **Fallback policy**：`backend` 可能出現 `ollama_vision->simulated`——**可觀測的降級**；並說你知道怎麼關掉 fallback 來驗證合作方 API（`fallback_to_simulated: false`）。

### 5:30–7:30 — 「若這是對外產品」你會怎麼做（DevRel 本體）

這段是把 **工程專案** 翻成 **advocacy 職能**，必講。

- **文件**：Integration guide（3 分鐘 simulated → mock HTTP → 多模態）；Architecture 講契約與限制。
- **範例優先級**：最小可跑 server stub、`curl` 排查、常見錯誤表（timeout、401、unparsable JSON）。
- **和 PM／Engineering 的 feedback loop**：你會從論壇 issue 歸納 **top failure modes**，回饋到 API 設計（錯誤碼、timeout 建議、JSON mode 相容）。
- **社群／活動**（若你履歷有再帶）：沒有就誠實說「這個 repo 是我對 **技術內容與整合故事** 的投資，活動經驗在 XXX」。

### 7:30–9:15 — 限制與下一步（Credibility）

- **誠實邊界**：這不是千萬級流量的線上服務；強在 **整合契約與可重現批次**。
- **你會怎麼演進**：例如 Protocol/ABC 靜態約束、更多 provider、指標與 SLO——**講 1 個具體即可**。

### 9:15–10:00 — 收束：為什麼是你

- 一句話：**我習慣把「接得人進來」當成和模型同等重要的 deliverable**——契約、範例、錯誤語意、可降級的運維故事。

**若面試官插問「講一個你幫開發者省時間的例子」**：  
→ 答 **mock_api 路徑 + normalize + 錯誤碼**，或答 **llama.cpp `response_format` 失敗自動重試**（依你實際讀 code 的熟度選）。

---

## English versions（雙語職缺備用）

### ~90 seconds (English)

> I built a **config-driven batch framework** for image-quality evaluation that outputs auditable release decisions—**GO / REVIEW / NO_GO**—not just opaque scores.
>
> The pain point is **backend churn**: teams swap between deterministic rules, local vision models, OpenAI-compatible servers, or partner HTTP APIs—and often fork “deploy-only” code, which breaks reproducibility.
>
> I abstracted inference behind a single **`predict_quality`** surface and **normalize** every backend into a stable schema: **`decision`, `code`, `msg`, plus `backend`**. Downstream ranking, arbitration, and reporting stay stable when the model changes.
>
> I also optimized for **integrator experience**: switch backends via config or a CLI override; remote failures can **fall back to simulated** so batches still complete, while preserving exception context in **`msg`**—useful for support-style debugging, not silent wrong answers.
>
> There’s a public-style **integration guide** for a 3-minute path from `simulated` to a mock HTTP backend to multimodal inference. That’s the mindset I bring to DevRel: **ship the contract, the sample, and the troubleshooting story—not only the model.**

### ~10 minutes (English outline)

1. **Problem & integrator persona** (45s)  
2. **What ships**: CLI batch pipeline, JSON artifacts, optional Streamlit demo; clarify orchestrator is experimental (45s)  
3. **Architecture**: Engine → Model factory + normalization → Eval / arbitration (2–3m)  
4. **Deep dives** (pick 2): explicit registry; `code` vs `msg`; fallback observability (2m)  
5. **If this were a public platform**: docs, minimal repro server, curl triage, feedback to API design (2m)  
6. **Honest limits + one roadmap item** (1m)  
7. **Close**: “I optimize for adoption and debuggability, not just accuracy.” (30s)

---

## 練習備忘

- **90 秒**：錄音計時；超時就刪形容詞，保留 *problem → contract → integrator DX → DevRel tie-in*。  
- **10 分鐘**：準備 **一張圖**（三層架構）或 **三個關鍵字** 在白板；深挖問題多半落在 **fallback、normalize、為何不用動態載入 plugin**。  
- **不要只丟連結**：開場說「我帶你走一遍主路徑」，**最後 20 秒**再給 repo 與 Integration guide 當 follow-up。
