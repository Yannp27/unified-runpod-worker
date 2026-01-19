# Unified RunPod Worker

[![RunPod](https://api.runpod.io/badge/Yannp27/unified-runpod-worker)](https://console.runpod.io/hub/Yannp27/unified-runpod-worker)

All-in-one AI serverless worker: LangGraph swarm orchestration, image processing, and SearXNG web search.

## Features

- **LangGraph Swarm**: Claude→DeepSeek→Claude delegation with checkpointing
- **Image Processing**: Real-ESRGAN upscaling, rembg background removal
- **Keyframe Interpolation**: SVD-based frame interpolation for animation
- **SearXNG Integration**: External web search via SEARXNG_URL
- **Provider Abstraction**: Hot-swap LLM/Search/Image providers via config

## Projects

| Project | Actions | Description |
|---------|---------|-------------|
| `swarm` | `execute`, `health` | LangGraph AI orchestration |
| `image_tools` | `rembg`, `upscale`, `pipeline`, `keyframe_interpolate` | GPU image/video processing |
| `pleasance` | `batch_generate`, `batch_review`, `health` | Content generation |
| `speedb04t` | `command`, `git_workflow` | Git + shell workflows |

## API Usage

```json
{
  "input": {
    "project": "image_tools",
    "action": "upscale",
    "image_base64": "...",
    "scale": 4
  }
}
```

## Actions

### image_tools

| Action | Description |
|--------|-------------|
| `rembg` | Background removal (isnet-anime) |
| `upscale` | Real-ESRGAN 4x upscaling |
| `pipeline` | Multi-step processing |
| `keyframe_interpolate` | SVD frame interpolation |

### swarm

| Action | Description |
|--------|-------------|
| `execute` | Run Claude→DeepSeek delegation task |
| `health` | Check LangGraph availability |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_BASE_URL` | Claude API endpoint (or proxy) |
| `DEEPSEEK_URL` | DeepSeek vLLM endpoint URL |
| `RUNPOD_API_KEY` | For inter-worker calls |
| `DEFAULT_LLM_PROVIDER` | `deepseek`, `claude`, or `gemini` |
| `SEARXNG_URL` | External SearXNG instance URL |

## Deployment

1. Deploy DeepSeek vLLM endpoint (RunPod template)
2. Deploy this worker from GitHub
3. Set `DEEPSEEK_URL` to point to DeepSeek endpoint

## Architecture

```
RunPod/
├── .runpod/           # Hub configuration
├── Dockerfile         # CUDA + models
├── handler.py         # Main dispatcher
├── projects/          # Project handlers
│   ├── swarm.py       # LangGraph orchestration
│   ├── image_tools.py # rembg, upscale, SVD
│   └── ...
├── utils/             # Provider abstractions
│   ├── providers.py   # Abstract interfaces
│   ├── llm_providers.py
│   ├── search_providers.py
│   └── worker_router.py
└── specs/             # Design documentation
```
