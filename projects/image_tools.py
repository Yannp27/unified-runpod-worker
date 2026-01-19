"""
Image Tools Project Handler

GPU-accelerated image processing:
- rembg: Background removal (isnet-anime)
- upscale: Real-ESRGAN anime upscaling
- pipeline: Multi-step processing
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Optional

# =============================================================================
# LAZY-LOADED MODELS
# =============================================================================

_rembg_session = None
_upscaler = None


def _get_rembg_session():
    """Lazy-load rembg model."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        print("[IMAGE] Loading isnet-anime model...")
        _rembg_session = new_session("isnet-anime")
        print("[IMAGE] rembg loaded!")
    return _rembg_session


def _get_upscaler(scale: int = 4):
    """Lazy-load Real-ESRGAN."""
    global _upscaler
    if _upscaler is None:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            print("[IMAGE] Loading Real-ESRGAN anime model...")
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6, num_grow_ch=32, scale=4
            )
            _upscaler = RealESRGANer(
                scale=scale,
                model_path="/models/realesr-animevideov3.pth",
                model=model,
                tile=512,
                tile_pad=10,
                half=True  # FP16 for speed
            )
            print("[IMAGE] Real-ESRGAN loaded!")
        except Exception as e:
            print(f"[IMAGE] Real-ESRGAN load failed: {e}")
            return None
    return _upscaler


# =============================================================================
# HANDLER
# =============================================================================

async def handle(input_data: dict) -> dict:
    """
    Main handler for image_tools project.
    
    Actions:
        ping: Health check
        rembg: Background removal
        upscale: Real-ESRGAN upscaling
        pipeline: Multi-step processing
    """
    action = input_data.get("action", "rembg")
    
    if action == "ping":
        return {
            "status": "ok",
            "project": "image_tools",
            "actions": ["rembg", "upscale", "pipeline", "ping"],
            "models": {
                "rembg": _rembg_session is not None,
                "upscaler": _upscaler is not None
            }
        }
    
    if action == "rembg":
        return await rembg_action(input_data)
    
    if action == "upscale":
        return await upscale_action(input_data)
    
    if action == "pipeline":
        return await pipeline_action(input_data)
    
    return {"error": f"Unknown action: {action}"}


# =============================================================================
# ACTIONS
# =============================================================================

async def rembg_action(input_data: dict) -> dict:
    """Remove background from image."""
    from rembg import remove, new_session
    
    image_b64 = input_data.get("image_base64")
    if not image_b64:
        return {"error": "No image_base64 provided"}
    
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes))
    
    model = input_data.get("model", "isnet-anime")
    use_matting = input_data.get("alpha_matting", False)
    
    session = _get_rembg_session() if model == "isnet-anime" else new_session(model)
    
    if use_matting:
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
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {"image_base64": result_b64, "action": "rembg"}


async def upscale_action(input_data: dict) -> dict:
    """Upscale image using Real-ESRGAN."""
    image_b64 = input_data.get("image_base64")
    if not image_b64:
        return {"error": "No image_base64 provided"}
    
    scale = input_data.get("scale", 4)
    upscaler = _get_upscaler(scale)
    
    if upscaler is None:
        return {"error": "Real-ESRGAN not available"}
    
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    
    try:
        output, _ = upscaler.enhance(img_array, outscale=scale)
        result = Image.fromarray(output)
        
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "image_base64": result_b64,
            "action": "upscale",
            "scale": scale,
            "original_size": list(img.size),
            "new_size": list(result.size)
        }
    except Exception as e:
        return {"error": f"Upscale failed: {e}"}


async def pipeline_action(input_data: dict) -> dict:
    """
    Multi-step image processing pipeline.
    
    Input:
        image_base64: Base64 image
        steps: ["rembg", "upscale"] - order of operations
        config: {step_name: {params}} - per-step config
    """
    image_b64 = input_data.get("image_base64")
    if not image_b64:
        return {"error": "No image_base64 provided"}
    
    steps = input_data.get("steps", ["upscale"])
    config = input_data.get("config", {})
    
    current_image = image_b64
    results = []
    
    for step in steps:
        step_config = config.get(step, {})
        step_input = {"image_base64": current_image, **step_config}
        
        if step == "rembg":
            result = await rembg_action(step_input)
        elif step == "upscale":
            result = await upscale_action(step_input)
        else:
            return {"error": f"Unknown pipeline step: {step}"}
        
        if "error" in result:
            return {"error": f"Pipeline failed at {step}: {result['error']}"}
        
        current_image = result["image_base64"]
        results.append({"step": step, "success": True})
    
    return {
        "image_base64": current_image,
        "action": "pipeline",
        "steps_completed": results
    }
