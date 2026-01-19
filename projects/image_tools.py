"""
Image Tools Project Handler

GPU-accelerated image processing tools:
- rembg: Background removal (isnet-anime)
- (future) upscale, face_detect, etc.
"""

import io
import base64
from PIL import Image

# Lazy-loaded rembg session
_rembg_session = None


def _get_rembg_session():
    """Lazy-load rembg model."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        print("Loading isnet-anime model...")
        _rembg_session = new_session("isnet-anime")
        print("rembg model loaded!")
    return _rembg_session


async def handle(input_data: dict) -> dict:
    """
    Main handler for image_tools project.
    
    Input:
        action: "rembg" | "ping"
        ... action-specific params
    """
    action = input_data.get("action", "rembg")
    
    if action == "ping":
        return {"status": "ok", "project": "image_tools", "actions": ["rembg", "ping"]}
    
    if action == "rembg":
        return await rembg_action(input_data)
    
    return {"error": f"Unknown action: {action}"}


async def rembg_action(input_data: dict) -> dict:
    """
    Remove background from image.
    
    Input:
        image_base64: Base64 encoded input image
        model: Optional model name (default: isnet-anime)
        alpha_matting: Optional bool for cleaner edges (default: False)
    
    Output:
        image_base64: Base64 encoded PNG with transparent background
    """
    from rembg import remove, new_session
    
    image_b64 = input_data.get("image_base64")
    if not image_b64:
        return {"error": "No image_base64 provided"}
    
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes))
    
    model = input_data.get("model", "isnet-anime")
    use_matting = input_data.get("alpha_matting", False)
    
    # Use cached session for default model
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
    
    return {"image_base64": result_b64}
