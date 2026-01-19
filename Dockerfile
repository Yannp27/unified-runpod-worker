# Unified Worker: Image Processing + SearXNG + Orchestration
# DeepSeek runs as separate RunPod vLLM endpoint
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache image models (eliminates cold-start)
RUN python -c "from rembg import new_session; new_session('isnet-anime')"
RUN mkdir -p /models && \
    wget -q -P /models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

# SearXNG (bundled)
RUN pip install searxng uvloop httpx[http2]
COPY searxng_settings.yml /etc/searxng/settings.yml

# Application code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV WORKER_TYPE=unified

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
