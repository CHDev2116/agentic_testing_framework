# Linux container (default: amd64). llama.cpp here is a **CPU** build with OpenBLAS —
# not Apple Metal (Metal is macOS-only; use a host install if you need GPU on Apple Silicon).
FROM python:3.11-slim

# System deps for optional llama-cpp-python source builds and OpenCV headers used by some stacks.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned runtime set (kept in sync with pyproject.toml [project.dependencies]).
COPY requirements.txt .

# CPU wheel / build for Linux (no CMAKE_ARGS for Metal).
RUN pip install --no-cache-dir llama-cpp-python
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/your-model-q4_k_m.gguf

CMD ["python", "src/ai_quality_agent.py", "--profile", "dev"]
