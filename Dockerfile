# Unified RunPod Worker with GPU support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download rembg model (eliminates cold-start for image_tools)
RUN python -c "from rembg import new_session; new_session('isnet-anime'); print('isnet-anime cached')"

# Copy all code
COPY . .

# RunPod handler entrypoint
CMD ["python", "-u", "handler.py"]
