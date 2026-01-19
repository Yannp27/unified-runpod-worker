#!/bin/bash
# Unified Worker Startup
# Starts SearXNG in background, then runs handler

set -e

echo "[WORKER] Starting SearXNG..."
cd /opt/searxng && python -m searx.webapp 2>&1 &
SEARXNG_PID=$!

# Wait for SearXNG (check search endpoint, no /healthz)
for i in {1..30}; do
    if curl -s "http://localhost:8080/search?q=test&format=json" > /dev/null 2>&1; then
        echo "[WORKER] SearXNG ready!"
        break
    fi
    echo "[WORKER] Waiting for SearXNG... ($i/30)"
    sleep 1
done

echo "[WORKER] Starting RunPod handler..."
exec python -u handler.py
