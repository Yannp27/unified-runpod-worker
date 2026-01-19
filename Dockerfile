# Unified Worker: Image Processing + SearXNG + Orchestration
# DeepSeek runs as separate RunPod vLLM endpoint
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System deps (including SearXNG native deps per upstream docs)
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg \
    build-essential python3-dev \
    libxml2-dev libxslt1-dev zlib1g-dev libffi-dev libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (core)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Real-ESRGAN (from git, not PyPI)
RUN pip install --no-cache-dir basicsr realesrgan

# Pre-cache image models (eliminates cold-start)
RUN python -c "from rembg import new_session; new_session('isnet-anime')"
RUN mkdir -p /models && \
    wget -q -P /models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

# SearXNG: Use external instance via SEARXNG_URL env var
# Deploy SearXNG separately: docker run -p 8080:8080 searxng/searxng

# Application code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV WORKER_TYPE=unified

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
