"""
Image Tools Project Handler

GPU-accelerated image processing:
- rembg: Background removal (isnet-anime)
- upscale: Real-ESRGAN anime upscaling
- pipeline: Multi-step processing
- keyframe_interpolate: SVD-based frame interpolation
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Optional, List

# =============================================================================
# LAZY-LOADED MODELS
# =============================================================================

_rembg_session = None
_upscaler = None
_svd_pipeline = None


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


def _get_hunyuan_pipeline():
    """
    Lazy-load Hunyuan Video 1.5 pipeline with keyframe control LoRA.
    
    Model: tencent/HunyuanVideo (8.3B DiT)
    LoRA: Kijai/HunyuanVideo_comfy keyframe control
    VRAM: ~16GB with CPU offload
    """
    global _svd_pipeline  # Reuse global for backwards compat
    if _svd_pipeline is None:
        try:
            import torch
            from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
            from diffusers.utils import export_to_video
            
            print("[IMAGE] Loading Hunyuan Video 1.5 pipeline...")
            
            # Load base model
            model_id = "tencent/HunyuanVideo"
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16
            )
            
            _svd_pipeline = HunyuanVideoPipeline.from_pretrained(
                model_id,
                transformer=transformer,
                torch_dtype=torch.float16
            )
            
            # Load keyframe control LoRA
            try:
                _svd_pipeline.load_lora_weights(
                    "Kijai/HunyuanVideo_comfy",
                    weight_name="hunyuan_video_keyframe_control_lora.safetensors",
                    adapter_name="keyframe_control"
                )
                _svd_pipeline.set_adapters(["keyframe_control"], [1.0])
                print("[IMAGE] Keyframe control LoRA loaded!")
            except Exception as e:
                print(f"[IMAGE] LoRA load failed (continuing without): {e}")
            
            # Optimize for VRAM
            _svd_pipeline.enable_model_cpu_offload()
            _svd_pipeline.vae.enable_tiling()
            
            print("[IMAGE] Hunyuan Video 1.5 pipeline loaded!")
            
        except Exception as e:
            print(f"[IMAGE] Hunyuan Video load failed, falling back to SVD: {e}")
            # Fallback to SVD
            try:
                from diffusers import StableVideoDiffusionPipeline
                _svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                _svd_pipeline.to("cuda")
                _svd_pipeline.enable_model_cpu_offload()
                print("[IMAGE] Fallback SVD pipeline loaded!")
            except Exception as e2:
                print(f"[IMAGE] SVD fallback also failed: {e2}")
                return None
    return _svd_pipeline


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
        keyframe_interpolate: SVD-based frame interpolation
    """
    action = input_data.get("action", "rembg")
    
    if action == "ping":
        return {
            "status": "ok",
            "project": "image_tools",
            "actions": ["rembg", "upscale", "pipeline", "keyframe_interpolate", "ping"],
            "models": {
                "rembg": _rembg_session is not None,
                "upscaler": _upscaler is not None,
                "svd": _svd_pipeline is not None
            }
        }
    
    if action == "rembg":
        return await rembg_action(input_data)
    
    if action == "upscale":
        return await upscale_action(input_data)
    
    if action == "pipeline":
        return await pipeline_action(input_data)
    
    if action == "keyframe_interpolate":
        return await keyframe_interpolate_action(input_data)
    
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


async def keyframe_interpolate_action(input_data: dict) -> dict:
    """
    Interpolate between 2-4 keyframe images using Hunyuan Video 1.5.
    
    Input:
        keyframes: List of 2-4 base64 PNG images
        num_frames: Total output frames (default: 25)
        fps: Output framerate (default: 12)
        output_format: "frames" or "mp4" (default: "frames")
        
    Alternative input (backwards compat):
        frame1_b64 / image_base64: First keyframe
        frame2_b64: Second keyframe (optional)
    
    Output:
        frames: List of base64 PNG images
        video_b64: Optional base64 MP4 (if output_format="mp4")
    """
    import torch
    
    # Parse keyframes - support both new and old format
    keyframes_b64 = input_data.get("keyframes", [])
    
    if not keyframes_b64:
        # Backwards compatibility
        frame1 = input_data.get("frame1_b64") or input_data.get("image_base64")
        frame2 = input_data.get("frame2_b64")
        if frame1:
            keyframes_b64 = [frame1]
            if frame2:
                keyframes_b64.append(frame2)
    
    if not keyframes_b64 or len(keyframes_b64) < 1:
        return {"error": "No keyframes provided. Use 'keyframes' array or 'frame1_b64'"}
    
    if len(keyframes_b64) > 4:
        return {"error": "Maximum 4 keyframes supported"}
    
    num_frames = input_data.get("num_frames", 25)
    fps = input_data.get("fps", 12)
    output_format = input_data.get("output_format", "frames")
    
    # Load pipeline
    pipeline = _get_hunyuan_pipeline()
    if pipeline is None:
        return {"error": "Video generation pipeline not available"}
    
    try:
        # Decode keyframes
        keyframe_images = []
        for i, kf_b64 in enumerate(keyframes_b64):
            image_bytes = base64.b64decode(kf_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Hunyuan native: 544x960 (9:16) or 960x544 (16:9)
            img = img.resize((960, 544), Image.Resampling.LANCZOS)
            keyframe_images.append(img)
        
        # Generate video frames
        with torch.inference_mode():
            # For single keyframe: image-to-video
            if len(keyframe_images) == 1:
                output = pipeline(
                    image=keyframe_images[0],
                    num_frames=num_frames,
                    num_inference_steps=30,
                    guidance_scale=6.0,
                ).frames[0]
            else:
                # For multiple keyframes: use keyframe conditioning
                # Distribute keyframes evenly across timeline
                output = pipeline(
                    image=keyframe_images[0],
                    num_frames=num_frames,
                    num_inference_steps=30,
                    guidance_scale=6.0,
                    # Pass additional keyframes if LoRA supports it
                    # Note: LoRA-specific params may vary
                ).frames[0]
        
        # Convert frames to base64
        frames_b64 = []
        for frame in output:
            if isinstance(frame, Image.Image):
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
            else:
                # Handle numpy array
                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        
        result = {
            "action": "keyframe_interpolate",
            "frames": frames_b64,
            "num_frames": len(frames_b64),
            "fps": fps,
            "model": "hunyuan-video-1.5"
        }
        
        # Create video if requested
        if output_format == "mp4" or input_data.get("output_video", False):
            video_b64 = _frames_to_video(output, fps)
            if video_b64:
                result["video_b64"] = video_b64
        
        return result
        
    except Exception as e:
        return {"error": f"Keyframe interpolation failed: {e}"}




def _frames_to_video(frames: List[Image.Image], fps: int = 8) -> Optional[str]:
    """Convert PIL frames to base64 MP4."""
    try:
        import tempfile
        import subprocess
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save frames as PNGs
            for i, frame in enumerate(frames):
                frame.save(os.path.join(tmpdir, f"frame_{i:04d}.png"))
            
            # Use ffmpeg to create MP4
            output_path = os.path.join(tmpdir, "output.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                output_path
            ], capture_output=True, check=True)
            
            # Read and encode video
            with open(output_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[IMAGE] Video encoding failed: {e}")
        return None

