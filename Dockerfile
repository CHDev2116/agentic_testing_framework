# Use ARM64 Python base image for Apple Silicon compatibility
FROM --platform=linux/arm64 python:3.10-slim

# 1. Install system dependencies for llama.cpp compilation and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Copy and install dependencies
# Note: Ensure requirements.txt is optimized for your PixelQA project
COPY requirements.txt .

# CRITICAL: Build llama-cpp-python with Metal support for M4 hardware acceleration
# This ensures the model utilizes the Apple Silicon GPU instead of CPU only
RUN CMAKE_ARGS="-DLLAMA_METAL=on" pip install --no-cache-dir llama-cpp-python
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy source code 
# Note: Models should be mounted via volumes to keep image size minimal
COPY . .

# 5. Environment variables
# PYTHONUNBUFFERED=1 ensures logs are printed in real-time
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/your-model-q4_k_m.gguf

# 6. Entry point
CMD ["python", "src/ai_quality_agent.py", "--profile", "dev"]
