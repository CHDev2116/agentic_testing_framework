# Agentic Testing Framework: Quantized Vision QA

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pillow](https://img.shields.io/badge/Library-Pillow-orange.svg)

## 📌 專案概述 (Project Overview)
本專案為一個專為移動端影像品質測試設計的 **自動化測試框架**。模擬在硬體資源受限（如手機）的環境下，如何利用 **4-bit 輕量化模型 (Quantized Vision Model)** 進行即時的影像品質判定。

框架核心採用 **「配置驅動（Configuration-Driven）」** 設計，將測試門檻與執行邏輯完全解耦，使其能快速適應不同的量化模型標準。

## 🛠️ 技術亮點 (Technical Highlights)

### 1. 模組化架構與配置驅動 (Modular & Config-Driven)
* **完全解耦**: 採用 `config.json` 管理所有測試標準（Sharpness/Brightness Thresholds），無需變更代碼即可調整測試策略。
* **Engine 層**: 整合 **Pillow** 進行影像預處理（降採樣、灰階轉換），將單張處理延遲優化至 **< 10ms**。
* **Models 層**: 模擬 4-bit 量化模型在精度損失下的決策行為，支援邊界條件判定。

### 2. 批次處理與效能監控 (Batch Processing & Metrics)
* **自動化流水線**: 支援 `test_images/` 資料夾全量自動掃描，無需手動指定檔案。
* **效能追蹤**: 內建 **Latency Tracking**，精確記錄每張測試樣本的處理時長，協助分析模型推論效率。
* **可視化看板**: 測試結束後自動匯總 **合格率 (Pass Rate)** 與 **平均延遲 (Avg Latency)**。

### 3. 異常容錯機制 (Resilience & Error Handling)
* **OOM 壓力模擬**: 實作隨機記憶體溢位模擬器，驗證系統在極端環境下的穩定性。
* **安全氣囊結構**: 使用 `try-except-finally` 確保在崩潰發生時，系統依然能產出包含環境上下文的 **Crash Report (JSON)**。

## 📂 檔案結構 (Directory Structure)
```text
agentic_testing_framework/
├── config.json           # 測試標準與路徑設定檔 (核心控制)
├── src/
│   ├── engine/           # 影像處理模組
│   ├── models/           # 4-bit AI 決策邏輯
│   └── ai_quality_agent.py # 指揮官：負責批次調度與統計看板
├── test_images/          # 待測試影像資料夾
├── results/              # 自動生成的 JSON 測試報告
└── README.md

執行範例 (Output Example)

🚀 啟動模式: PixelQA-Llama-4bit (4-bit)
📋 開始處理 4 張照片...

🔹 處理 image1.jpg: Optimal (5.22ms)
🔹 處理 image4.jpeg: Blurry (Below 20.0) (8.53ms)

========================================
📊 測試統計看板 (Threshold Mode)
  - 總測試量: 4
  - 合格率 (Optimal): 75.0%
  - 平均延遲: 5.86 ms
========================================

未來優化 (Roadmap)

[ ] 多執行緒優化: 實作並行測試 (Multi-threading) 以應對大規模數據集。

[ ] 真實模型接入: 透過 Ollama 整合真實 Llama-3 Vision 模型。

[ ] 視覺化分析: 增加 OpenCV 支援，進行更細緻的色差與雜訊分析。

👤 作者
Cheryl - AI Optimization & Testing Engineer