"""
Image Provider Implementation

GPU-side image processing with provider abstraction.
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Optional
from .providers import ImageProvider, ImageAction, ImageConfig, ProviderRegistry


# =============================================================================
# LAZY-LOADED MODELS
# =============================================================================

_rembg_session = None
_upscaler = None


def _get_rembg_session(model: str = "isnet-anime"):
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        print(f"[IMAGE] Loading {model} model...")
        _rembg_session = new_session(model)
        print("[IMAGE] rembg loaded!")
    return _rembg_session


def _get_upscaler(scale: int = 4):
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
                model_path="/models/esrgan/realesr-animevideov3.pth",
                model=model,
                tile=512,
                tile_pad=10,
                half=True
            )
            print("[IMAGE] Real-ESRGAN loaded!")
        except Exception as e:
            print(f"[IMAGE] Real-ESRGAN load failed: {e}")
            return None
    return _upscaler


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================

class RealESRGANProvider:
    """Real-ESRGAN upscaling provider."""
    
    provider_name = "realesrgan"
    
    async def process(
        self,
        image: bytes,
        action: ImageAction,
        config: Optional[ImageConfig] = None
    ) -> bytes:
        if action != ImageAction.UPSCALE:
            raise ValueError(f"RealESRGAN only supports UPSCALE, got {action}")
        
        cfg = config or ImageConfig()
        upscaler = _get_upscaler(cfg.scale)
        
        if upscaler is None:
            raise RuntimeError("Real-ESRGAN not available")
        
        img = Image.open(io.BytesIO(image)).convert("RGB")
        img_array = np.array(img)
        
        output, _ = upscaler.enhance(img_array, outscale=cfg.scale)
        result = Image.fromarray(output)
        
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue()


class RembgProvider:
    """Background removal provider."""
    
    provider_name = "rembg"
    
    async def process(
        self,
        image: bytes,
        action: ImageAction,
        config: Optional[ImageConfig] = None
    ) -> bytes:
        if action != ImageAction.REMBG:
            raise ValueError(f"Rembg only supports REMBG, got {action}")
        
        from rembg import remove
        
        cfg = config or ImageConfig()
        session = _get_rembg_session(cfg.model)
        
        img = Image.open(io.BytesIO(image))
        
        if cfg.alpha_matting:
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


class UnifiedImageProvider:
    """
    Unified image provider that routes to specialized providers.
    """
    
    provider_name = "unified"
    
    def __init__(self):
        self._upscaler = RealESRGANProvider()
        self._rembg = RembgProvider()
    
    async def process(
        self,
        image: bytes,
        action: ImageAction,
        config: Optional[ImageConfig] = None
    ) -> bytes:
        if action == ImageAction.UPSCALE:
            return await self._upscaler.process(image, action, config)
        elif action == ImageAction.REMBG:
            return await self._rembg.process(image, action, config)
        elif action == ImageAction.FIX_ARTIFACTS:
            # Fix artifacts = upscale with denoise
            cfg = config or ImageConfig()
            cfg.scale = 2
            cfg.denoise_strength = 0.7
            return await self._upscaler.process(image, ImageAction.UPSCALE, cfg)
        elif action == ImageAction.ENHANCE:
            # Enhance = mild upscale
            cfg = config or ImageConfig()
            cfg.scale = 2
            return await self._upscaler.process(image, ImageAction.UPSCALE, cfg)
        else:
            raise ValueError(f"Unknown action: {action}")


# =============================================================================
# REGISTRATION
# =============================================================================

def register_image_providers():
    """Register image providers."""
    ProviderRegistry.register_image("realesrgan", RealESRGANProvider())
    ProviderRegistry.register_image("rembg", RembgProvider())
    ProviderRegistry.register_image("unified", UnifiedImageProvider())
