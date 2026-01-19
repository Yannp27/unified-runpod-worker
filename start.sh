#!/bin/bash
# Unified Worker Startup

set -e

echo "[WORKER] Starting RunPod handler..."
echo "[WORKER] SearXNG: Use SEARXNG_URL env var for external instance"
exec python -u handler.py
