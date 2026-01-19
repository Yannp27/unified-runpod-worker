# Image Processing Pipeline

## Overview
GPU-accelerated image processing for anime/drawn-style content. Runs on RunPod GPU workers.

---

## Pipeline Stages

```
Input Image → [Background Removal] → [Upscaling] → [Artifact Fixing] → Output
                    (rembg)         (Real-ESRGAN)      (Optional)
```

---

## 1. Background Removal (rembg)

### Models
| Model | Best For | Speed |
|-------|----------|-------|
| `isnet-anime` | Anime characters ✓ | Fast |
| `u2net` | Photos | Medium |
| `isnet-general-use` | Mixed | Fast |

### Usage
```python
from rembg import remove, new_session

session = new_session("isnet-anime")

async def remove_background(image_bytes: bytes, alpha_matting: bool = True) -> bytes:
    from PIL import Image
    import io
    
    img = Image.open(io.BytesIO(image_bytes))
    
    if alpha_matting:
        result = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=5,
            alpha_matting_erode_size=3
        )
    else:
        result = remove(img, session=session)
    
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
```

---

## 2. Upscaling (Real-ESRGAN Anime 6B)

### Why This Model
- Trained specifically on anime/illustration
- Preserves flat colors, sharp lines
- Cleans compression artifacts

### Installation
```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime

RUN pip install realesrgan basicsr

# Download anime model
RUN wget -P /models \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
```

### Usage
```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
from PIL import Image

def get_upscaler(scale: int = 4) -> RealESRGANer:
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=6, num_grow_ch=32, scale=4
    )
    return RealESRGANer(
        scale=scale,
        model_path="/models/realesr-animevideov3.pth",
        model=model,
        tile=512,  # For large images
        tile_pad=10,
        half=True  # FP16 for speed
    )

upscaler = get_upscaler()

async def upscale_image(image_bytes: bytes, scale: int = 4) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    
    output, _ = upscaler.enhance(img_array, outscale=scale)
    
    result = Image.fromarray(output)
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
```

---

## 3. Artifact Fixing

### Common Issues
1. **JPEG blocking** → Upscaling fixes most
2. **AI hallucinations** → Manual inpainting needed
3. **Color banding** → Denoise + upscale

### Auto-Fix Pipeline
```python
async def fix_artifacts(
    image_bytes: bytes,
    denoise_strength: float = 0.5
) -> bytes:
    # Most artifacts fixed by high-quality upscale
    upscaled = await upscale_image(image_bytes, scale=2)
    
    # Optional: Apply additional denoising
    # (Real-ESRGAN already includes denoising)
    
    return upscaled
```

### Manual Review Gate
```python
async def image_review_node(state: SwarmState) -> SwarmState:
    """Optional Claude review for image quality."""
    claude = get_provider("llm", provider="claude", tier="fast")
    
    # Claude vision can check for obvious issues
    response = await claude.complete([
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": state["image_b64"]}},
                {"type": "text", "text": "Check this image for: extra limbs, merged features, text artifacts. Reply PASS or FAIL with reason."}
            ]
        }
    ])
    
    passed = "PASS" in response["text"]
    return {
        "results": {"image_review": {"passed": passed, "feedback": response["text"]}},
        "current_agent": "end" if passed else "manual_fix"
    }
```

---

## Full Pipeline Handler

```python
async def handle_image_pipeline(input_data: dict) -> dict:
    """
    Input:
        image_base64: str
        actions: list["rembg", "upscale", "fix_artifacts"]
        config: dict (optional)
    """
    image_b64 = input_data["image_base64"]
    actions = input_data.get("actions", ["upscale"])
    config = input_data.get("config", {})
    
    image_bytes = base64.b64decode(image_b64)
    
    for action in actions:
        if action == "rembg":
            image_bytes = await remove_background(
                image_bytes,
                alpha_matting=config.get("alpha_matting", True)
            )
        elif action == "upscale":
            image_bytes = await upscale_image(
                image_bytes,
                scale=config.get("scale", 4)
            )
        elif action == "fix_artifacts":
            image_bytes = await fix_artifacts(
                image_bytes,
                denoise_strength=config.get("denoise", 0.5)
            )
    
    result_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return {"image_base64": result_b64, "actions_applied": actions}
```

---

## Configuration

```yaml
image_pipeline:
  rembg:
    model: "isnet-anime"
    alpha_matting: true
  
  upscale:
    model: "realesrgan-anime-6b"
    scale: 4
    tile_size: 512
    half_precision: true
  
  fix_artifacts:
    denoise_strength: 0.5
    review_enabled: false  # Optional Claude review
```
