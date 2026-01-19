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

# SearXNG: clone, install deps only (setup.py has circular imports)
RUN git clone --depth 1 https://github.com/searxng/searxng.git /opt/searxng && \
    pip install --no-cache-dir -r /opt/searxng/requirements.txt && \
    mkdir -p /etc/searxng
COPY searxng_settings.yml /etc/searxng/settings.yml
ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
ENV PYTHONPATH="/opt/searxng:${PYTHONPATH}"

# Application code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV WORKER_TYPE=unified

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
