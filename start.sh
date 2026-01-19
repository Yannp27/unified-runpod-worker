#!/bin/bash
# Unified Worker Startup
# Starts SearXNG in background, then runs handler

set -e

echo "[WORKER] Starting SearXNG..."
python -m searx.webapp &
SEARXNG_PID=$!

# Wait for SearXNG
for i in {1..30}; do
    if curl -s http://localhost:8888/healthz > /dev/null 2>&1; then
        echo "[WORKER] SearXNG ready!"
        break
    fi
    sleep 1
done

echo "[WORKER] Starting RunPod handler..."
exec python -u handler.py
