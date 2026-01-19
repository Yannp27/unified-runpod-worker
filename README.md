# Unified RunPod Worker

Multi-project serverless handler with GPU support.

## Projects

| Project | Actions | Description |
|---------|---------|-------------|
| `pleasance` | `batch_generate`, `batch_review`, `health` | Kink DB content gen |
| `speedb04t` | `command`, `git_workflow` | Git + shell workflows |
| `image_tools` | `rembg`, `ping` | GPU image processing |

## API Usage

```json
{
  "input": {
    "project": "image_tools",
    "action": "rembg",
    "image_base64": "..."
  }
}
```

## image_tools Actions

### rembg (Background Removal)
```json
{
  "project": "image_tools",
  "action": "rembg",
  "image_base64": "<base64>",
  "model": "isnet-anime",
  "alpha_matting": false
}
```

### ping
```json
{"project": "image_tools", "action": "ping"}
```

## Deployment

1. Push to: `Yannp27/unified-runpod-worker`
2. Create RunPod endpoint (GPU: RTX 3090/4090)
3. Set env vars: `ANTHROPIC_BASE_URL`, `AGENT_SECRET`, `PLEASANCE_API`

## Architecture

```
RunPod/
├── Dockerfile          # CUDA + rembg
├── requirements.txt
├── handler.py          # Main dispatcher
└── projects/
    ├── pleasance.py
    ├── speedb04t.py
    └── image_tools.py  # rembg, upscale (planned)
```
