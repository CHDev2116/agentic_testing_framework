# Advocacy case study（對照 DevRel JD）

假設目標產品線是 **Vision API** 或 **Agent / GenAI 平台**（多模態、HTTP API、開發者整合）。下面把本 repo README 裡 **兩個具體設計決策** 翻成：**若服務第三方開發者，我會怎麼寫文件、範例、版本策略**——對齊你貼的 JD：*sample code、client 整合、論壇／支援除錯、回饋產品與 API 設計、雙語利害關係人*。

---

## JD 對照（一句話）

| JD 元素 | 本案例怎麼示範 |
|--------|----------------|
| Sample code / client libraries | 最小可跑範例、mock server、與「官方回應 schema」對齊的 stub |
| Support queues / debug | `code` vs `msg`、`backend` 含 `->simulated` 的**可觀測降級**敘事 |
| Review API designs | 把 **normalize 層** 當成 public contract 的 pre-read；breaking change 清單 |
| Community / content | Quickstart 分層、錯誤碼表、遷移指南、中英雙語「整合者」用語 |

---

## 設計決策一：Config-only 後端選擇 + 單一 composition root

**README 原文要點**：`model_settings.inference.backend` 切換；CLI `--inference-backend` 覆寫；`build_inference_engine()` 是唯一 registry，避免「deploy-only fork」。

### 若這是對外 Vision / Agent 平台——我會怎麼寫 **文件**

1. **「Integration paths」分頁（三分層）**  
   - **Tier 0 — 無金鑰／離線**：對應 `simulated`——講清 *教學與 CI 用途*，避免開發者誤以為是正式模型品質。  
   - **Tier 1 — 自有服務 HTTP**：對應 `mock_api`——文件重心放在 **request/response JSON**、auth header、timeout。  
   - **Tier 2 — 託管多模態推理**：對應 `ollama_vision` / `llama_cpp`——文件寫 **我們保證的契約**（見決策二），與 **各 provider 專屬限制**（JSON mode、vision 模型、延遲）。

2. **Composition root 的「公開說法」**  
   - 對開發者：**「所有官方範例都經過同一個支援的設定路徑」**（像單一 `ClientOptions` / `InferenceBackend` enum），不在文件裡鼓勵複製內部分支。  
   - 對內部工程：**registry 變更 = release note 必載項目**（見版本策略）。

3. **中英並陳（JD：Mandarin + English）**  
   - 英文：API reference + 錯誤碼。  
   - 繁中：*快速入門*、常見整合坑、論壇置頂的「第一通支援先查這三項」。

### **範例** 我會怎麼排（優先順序）

| 順序 | 範例目的 | 內容 |
|------|-----------|------|
| 1 | 30 秒成功 | 單一 `curl` 或 10 行 Python：打 **mock／sandbox endpoint**，拿到 normalize 後的 JSON。 |
| 2 | 真實整合 | 同一支程式只改 **backend / endpoint 設定** 切到 staging production-like。 |
| 3 | 生產防呆 | 展示 **timeout、重試邊界、關閉 fallback** 的設定（對應 `fallback_to_simulated: false` 這類概念在對外 SDK 的 `fail_open` flag）。 |

*呼應 JD「協助開發者除錯」：範例裡刻意印出 `backend` 與 `code`，訓練使用者讀結構化錯誤，而不是只截圖整段 traceback。*

### **版本策略**

- **設定檔／SDK 的 schema 版本化**：`config.version`（你已有）對外會變成 **`apiVersion` 或 SDK major**。  
- **Breaking vs additive**：新增 backend **值** 預設 **additive**；若 `predict_quality` 回傳 **必填欄位** 有變 → **minor 文件 + major 版本**（或 deprecation window）。  
- **Registry 與審計**：對外說明「**不支援任意動態外掛**」的理由——**安全與 CI 可重現**（對 enterprise 開發者好交代）。若未來開 plugin：另走 **signed provider** + 明確支援矩陣，不混進預設 quickstart。

---

## 設計決策二：單一方法形狀 + `_normalize_result` 穩定 schema（含 `code` / `msg` / `backend`）

**README 原文要點**：`predict_quality(...) -> dict`；至少 `decision`, `code`, `msg`, optional `confidence`, `backend`；`backend` 可為 `provider->simulated` when fallback；`code` 機讀、`msg` 人讀；batch 有 `decision_reason` 可重現「為何 GO/REVIEW/NO_GO」。

### 若這是對外 API——我會怎麼寫 **文件**

1. **Response 規格頁（合約優先）**  
   - 表格列出每個欄位：**型別、是否穩定、是否可作為程式分支依據**。  
   - 明確寫：**`decision` 給業務標籤；`code` 給自動化；`msg` 可本地化、不保證機器穩定**（若對外要穩定訊息，另給 `detail_code` 或 doc link id）。

2. **Fallback 專章（支援隊伍必用）**  
   - 定義 `backend` 字串格式：`primary` vs `primary->fallback`。  
   - 開發者文件：**「看到 `->` 代表結果不是純 primary provider」**；營運／論壇 SOP：**先確認是否為預期降級**。  
   - 與 JD「論壇／queue 除錯」對齊：提供 **三個診斷問題**（endpoint? timeout? JSON parse?）。

3. **與 Vision API 的類比**  
   - 把 `metrics` + `thresholds` 進模型，類比 **「結構化前置特徵 + 使用者閾值」**；文件說明哪些欄位由 **平台保證**、哪些由 **呼叫端提供**，避免「同一份 JSON 在文件各處說法不一致」。

4. **與 Agent 平台的類比**  
   - 強調 **tool / step 輸出要是 machine-parseable**（呼應 agentic 整合）：我們在 normalize 層堅持 JSON 形狀，就是在產品層主張 **「可編排的代理步驟」** 而不是自由文字。

### **範例**

- **錯誤碼表 + 對應範例 response**（靜態頁，可搜尋）。  
- **「錯誤處理一頁紙」**：Python / Java / Go 各一段：`if code.startswith("ERR_")` vs 看 `backend` 是否含 `->`。  
- **Batch 報告**：若對外有 batch job API，文件說明 **`decision_reason`** 類欄位——**支援「為何過／沒過 gate」無需重跑**（對 enterprise 審計友善）。

### **版本策略**

- **`code` 的穩定性分級**：  
  - **Stable**：保證跨 minor 不變（例如 `SUCCESS_200`）。  
  - **Unstable / vendor**：前綴區隔（例如 `ERR_PROVIDER_*`），在 release note 明列。  
- **`msg` 不保證不變**；自動化**禁止** parse `msg`。  
- **Fallback 行為**：若從預設「容錯開」改為「嚴格失敗」，屬 **行為變更** → **文件 + changelog + 遷移期**（或新 API 版本）。

---

## 面試時 60 秒收口（可直接唸）

> 這個專案裡我做了兩個對外產品也會做的決定：**第一**，用設定與單一 factory 選後端，避免整合者複製 deploy fork——若平台化，我會用 **分層 quickstart、schema 版本、registry 變更＝release 義務** 來維護。  
> **第二**，強制 **normalize 後的穩定 schema**，機讀 `code`、人讀 `msg`、`backend` 標註是否降級——若平台化，我會寫 **合約頁、錯誤碼表、fallback 專章**，並訓練社群與支援用同一套診斷語言。  
> 這就是把工程決策轉成 **advocacy**：開發者省時間、產品收到可結構化的 feedback、API 設計有清楚邊界。

---

## 相關文件

- 整合者步驟與 troubleshooting：[`IntegrationGuide.md`](IntegrationGuide.md)  
- Provider 與 fallback 行為：[`Architecture.md`](Architecture.md)  
- 口條腳本：[`InterviewNarratives.md`](InterviewNarratives.md)
