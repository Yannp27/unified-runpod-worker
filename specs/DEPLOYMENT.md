# 2-Endpoint Deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RunPod Serverless                       │
├─────────────────────────────┬───────────────────────────────┤
│  DeepSeek vLLM (Template)   │   Unified Worker (This repo)  │
│  ─────────────────────────  │   ───────────────────────────  │
│  • RunPod vLLM template     │   • SearXNG (bundled)          │
│  • deepseek-ai/DeepSeek-V3  │   • Real-ESRGAN + rembg        │
│  • A100 80GB                │   • LangGraph orchestration    │
│  • No build needed          │   • RTX 4090 (for images)      │
└─────────────────────────────┴───────────────────────────────┘
```

---

## Deployment Steps

### 1. Deploy DeepSeek vLLM (RunPod Template)
```bash
# Use RunPod UI or API
# Template: vLLM
# Model: deepseek-ai/DeepSeek-V3
# GPU: A100 80GB
# Get endpoint ID after deployment
```

### 2. Deploy Unified Worker
```bash
docker build -t unified-worker .
docker push your-registry/unified-worker

runpod endpoint create \
  --name unified-worker \
  --docker-image your-registry/unified-worker \
  --gpu RTX-4090 \
  --env DEEPSEEK_URL=https://api.runpod.ai/v2/{deepseek-endpoint-id}/openai/v1 \
  --env ANTHROPIC_BASE_URL=https://agproxy.pleasance.app
```

---

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `DEEPSEEK_URL` | `https://api.runpod.ai/v2/{id}/openai/v1` | DeepSeek vLLM endpoint |
| `RUNPOD_API_KEY` | Your RunPod API key | For calling DeepSeek endpoint |
| `ANTHROPIC_BASE_URL` | `https://agproxy.pleasance.app` | Claude via AG proxy |
| `SEARXNG_URL` | `http://localhost:8888` | Bundled SearXNG |

---

## Cost Estimate

| Component | GPU | Idle | Active |
|-----------|-----|------|--------|
| DeepSeek vLLM | A100 80GB | $0 | ~$1.89/hr |
| Unified Worker | RTX 4090 | $0 | ~$0.44/hr |

Both scale-to-zero. Pay only when active.
