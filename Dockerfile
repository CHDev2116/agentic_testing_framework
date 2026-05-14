# Linux container (default: amd64). llama.cpp here is a **CPU** build with OpenBLAS —
# not Apple Metal (Metal is macOS-only; use a host install if you need GPU on Apple Silicon).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency pins: pyproject.toml only. Runtime install (no [dev] extras).
COPY . .
RUN pip install --no-cache-dir llama-cpp-python \
    && pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/your-model-q4_k_m.gguf

CMD ["python", "src/ai_quality_agent.py", "--profile", "dev"]
