# Image Providers

## Purpose
GPU-accelerated image processing for drawn/anime-style content. All tools hot-swappable via config.

---

## Available Actions

| Action | Provider | Description |
|--------|----------|-------------|
| `upscale` | Real-ESRGAN | 2-4x resolution enhancement |
| `rembg` | rembg | Background removal |
| `fix_artifacts` | Inpainting | Fix compression/generation artifacts |
| `enhance` | Aura-SR | Alternative fast upscaler |

---

## Upscaling: Real-ESRGAN Anime 6B

### Why This Model
- **Trained on anime**: Preserves flat colors, sharp lines, cel-shading
- **Handles artifacts**: Cleans JPEG compression, generation noise
- **Open source**: MIT license, no per-image costs

### Configuration
```yaml
realesrgan:
  model: "realesrgan-x4plus-anime"  # or "realesrgan-anime-6b"
  scale: 4
  denoise_strength: 0.5
  tile_size: 512  # For VRAM management
  gpu_id: 0
```

### Docker Setup
```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime

RUN pip install realesrgan opencv-python-headless pillow

# Download model weights
RUN wget -P /models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
```

---

## Background Removal: rembg

### Models (Selectable)
| Model | Best For |
|-------|----------|
| `isnet-anime` | Anime/drawn characters (default) |
| `u2net` | General photos |
| `isnet-general-use` | Mixed content |

### Configuration
```yaml
rembg:
  model: "isnet-anime"
  alpha_matting: true
  alpha_matting_foreground_threshold: 270
  alpha_matting_background_threshold: 5
```

---

## Artifact Fixing Pipeline

### Common Artifacts
1. **JPEG blocking**: 8x8 grid artifacts from compression
2. **AI hallucinations**: Extra fingers, merged features
3. **Color banding**: Gradient posterization

### Fix Strategy
```python
async def fix_artifacts(
    image: bytes,
    upscaler: ImageProvider,
    detector: Optional[ArtifactDetector] = None
) -> bytes:
    # 1. Detect artifact regions (optional AI detection)
    if detector:
        regions = await detector.detect(image)
    else:
        regions = None  # Full image enhancement
    
    # 2. Upscale (fixes most compression artifacts)
    enhanced = await upscaler.process(
        image,
        action="upscale",
        config={"scale": 2, "denoise": 0.7}
    )
    
    # 3. Optional: Inpaint specific regions
    if regions:
        enhanced = await inpaint_regions(enhanced, regions)
    
    return enhanced
```

---

## Provider Interface

```python
from enum import Enum
from typing import Protocol

class ImageAction(Enum):
    UPSCALE = "upscale"
    REMBG = "rembg"
    FIX_ARTIFACTS = "fix_artifacts"
    ENHANCE = "enhance"

class ImageProvider(Protocol):
    async def process(
        self,
        image: bytes,
        action: ImageAction,
        config: dict | None = None
    ) -> bytes: ...

class RealESRGANProvider:
    def __init__(self, model_path: str, scale: int = 4):
        self.model = RRDBNet(...)  # Load once
        self.upsampler = RealESRGANer(model=self.model, scale=scale)
    
    async def process(self, image: bytes, action: ImageAction, config=None):
        if action == ImageAction.UPSCALE:
            img = Image.open(io.BytesIO(image))
            output, _ = self.upsampler.enhance(np.array(img))
            # Convert back to bytes
            ...
```

---

## AuraSR (Alternative Upscaler)

### Pros
- Faster inference (0.25s for 1024px)
- Lower VRAM (<5GB)
- GAN-based, sharp output

### Cons
- Mixed results on heavily compressed input
- Less specialized for anime

### Config
```yaml
aurasr:
  model: "aura-sr-v2"
  scale: 4
  overlapping_tiles: true  # Reduces seams
```

---

## Hot-Swap Example
```python
# Switch from Real-ESRGAN to AuraSR:
config.providers.image.upscale.model = "aura-sr-v2"
# No code changes needed
```
