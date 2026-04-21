# Agentic Testing Framework: Quantized Vision QA

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pillow](https://img.shields.io/badge/Library-Pillow-orange.svg)

## 📌 Project Overview
This project is an **automated testing framework** for mobile image-quality validation. It simulates how a **4-bit lightweight model (Quantized Vision Model)** can evaluate image quality in real time under resource-constrained environments such as phones.

The framework follows a **configuration-driven** design, fully decoupling quality thresholds from execution logic so it can quickly adapt to different quantized model standards.

## 🛠️ Technical Highlights

### 1. Modular Architecture and Config-Driven Design
* **Fully decoupled**: Uses `configs/*.json` to manage all test standards (sharpness/brightness thresholds), so strategies can be adjusted without code changes.
* **Engine layer**: Uses **Pillow** for image preprocessing (downsampling, grayscale conversion), reducing per-image latency to **< 10ms**.
* **Models layer**: Simulates 4-bit quantized model decisions under precision loss and supports boundary-condition checks.

### 2. Batch Processing and Performance Monitoring
* **Automated pipeline**: Scans the `test_images/` directory automatically, without manually specifying files.
* **Performance tracking**: Built-in **Latency Tracking** records per-image processing time for inference efficiency analysis.
* **Dashboard summary**: Automatically reports **Pass Rate** and **Average Latency** when testing completes.

### 3. Resilience and Error Handling
* **OOM stress simulation**: Includes a random memory-overflow simulator to validate system stability in extreme conditions.
* **Safety-net flow**: Uses `try-except-finally` to ensure the system still produces a context-rich **Crash Report (JSON)** even after failures.

## 📂 Directory Structure
```text
agentic_testing_framework/
├── configs/              # Environment-based configs (base/dev/benchmark)
├── src/
│   ├── engine/           # Image-processing modules
│   ├── models/           # 4-bit AI decision logic
│   └── ai_quality_agent.py # Orchestrator for batch flow and dashboard stats
├── test_images/          # Input images for testing
├── results/              # Auto-generated JSON reports
└── README.md

## 🚀 Usage

Run from the project root:

```bash
# Use the development profile (configs/base.json + configs/dev.json)
python3 src/ai_quality_agent.py --profile dev

# Use the benchmark profile (configs/base.json + configs/benchmark.json)
python3 src/ai_quality_agent.py --profile benchmark

# Load a config file directly (applied on top of configs/base.json)
python3 src/ai_quality_agent.py --config configs/dev.json
```

Notes:
- `--profile` supports: `dev`, `benchmark`, `base`
- `--config` accepts either an absolute path or a project-root-relative path

## 📤 Output Example

🚀 Startup mode: PixelQA-Llama-4bit (4-bit)
📋 Starting to process 4 image(s)...

🔹 Processed image1.jpg: Optimal (5.22ms)
🔹 Processed image4.jpeg: Blurry (Below 20.0) (8.53ms)

========================================
📊 Test Dashboard (Threshold Mode)
  - Total tests: 4
  - Pass rate (Optimal): 75.0%
  - Average latency: 5.86 ms
========================================

Roadmap

[ ] Multi-threading optimization: implement parallel test execution for larger datasets.

[ ] Real model integration: connect to a real Llama-3 Vision model through Ollama.

[ ] Visual analytics: add OpenCV support for finer color-shift and noise analysis.

👤 Author
Cheryl - AI Optimization & Testing Engineer