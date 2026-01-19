"""
Provider Abstraction Layer

All external dependencies (LLM, Search, Image, Storage) accessed through
abstract interfaces. Swap implementations via config, not code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProviderConfig:
    """Base config for all providers."""
    name: str
    endpoint: str
    timeout: int = 120
    retries: int = 3
    extra: dict = field(default_factory=dict)


# =============================================================================
# LLM PROVIDER
# =============================================================================

@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict
    success: bool = True
    error: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract interface for LLM providers."""
    provider_name: str
    
    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse: ...
    
    async def health_check(self) -> bool: ...


# =============================================================================
# SEARCH PROVIDER
# =============================================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None


@runtime_checkable
class SearchProvider(Protocol):
    """Abstract interface for search providers."""
    provider_name: str
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[list[str]] = None
    ) -> list[SearchResult]: ...


# =============================================================================
# IMAGE PROVIDER
# =============================================================================

class ImageAction(Enum):
    UPSCALE = "upscale"
    REMBG = "rembg"
    FIX_ARTIFACTS = "fix_artifacts"
    ENHANCE = "enhance"


@dataclass
class ImageConfig:
    scale: int = 4
    model: str = "realesrgan-anime-6b"
    denoise_strength: float = 0.5
    alpha_matting: bool = True


@runtime_checkable
class ImageProvider(Protocol):
    """Abstract interface for image processing."""
    provider_name: str
    
    async def process(
        self,
        image: bytes,
        action: ImageAction,
        config: Optional[ImageConfig] = None
    ) -> bytes: ...


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

class ProviderRegistry:
    """
    Central registry for all providers.
    Enables hot-swapping via config.
    """
    
    _llm_providers: dict[str, LLMProvider] = {}
    _search_providers: dict[str, SearchProvider] = {}
    _image_providers: dict[str, ImageProvider] = {}
    
    # Tier mappings
    _model_tiers: dict[str, list[str]] = {
        "fast": ["deepseek-v3.1-q4", "gemini-2.5-flash-lite"],
        "balanced": ["deepseek-v3.1-q8", "claude-sonnet-4-5"],
        "premium": ["claude-opus-4-5-thinking"],
    }
    
    # Fallback chains
    _fallback_chains: dict[str, list[str]] = {
        "default": ["deepseek", "gemini", "claude"],
        "premium": ["claude", "gemini", "deepseek"],
        "budget": ["deepseek", "gemini"],
    }
    
    @classmethod
    def register_llm(cls, name: str, provider: LLMProvider):
        cls._llm_providers[name] = provider
    
    @classmethod
    def register_search(cls, name: str, provider: SearchProvider):
        cls._search_providers[name] = provider
    
    @classmethod
    def register_image(cls, name: str, provider: ImageProvider):
        cls._image_providers[name] = provider
    
    @classmethod
    def get_llm(
        cls,
        provider: Optional[str] = None,
        tier: Optional[str] = None
    ) -> LLMProvider:
        """
        Get LLM provider by name or tier.
        
        Args:
            provider: Specific provider name (deepseek, claude, gemini)
            tier: Model tier (fast, balanced, premium)
        
        Returns:
            LLMProvider instance
        """
        if provider:
            if provider not in cls._llm_providers:
                raise ValueError(f"Unknown LLM provider: {provider}")
            return cls._llm_providers[provider]
        
        # Default provider
        default = os.environ.get("DEFAULT_LLM_PROVIDER", "deepseek")
        return cls._llm_providers.get(default) or next(iter(cls._llm_providers.values()))
    
    @classmethod
    def get_search(cls, provider: Optional[str] = None) -> SearchProvider:
        if provider:
            return cls._search_providers[provider]
        default = os.environ.get("DEFAULT_SEARCH_PROVIDER", "searxng")
        return cls._search_providers.get(default) or next(iter(cls._search_providers.values()))
    
    @classmethod
    def get_image(cls, provider: Optional[str] = None) -> ImageProvider:
        if provider:
            return cls._image_providers[provider]
        default = os.environ.get("DEFAULT_IMAGE_PROVIDER", "realesrgan")
        return cls._image_providers.get(default) or next(iter(cls._image_providers.values()))
    
    @classmethod
    def get_fallback_chain(cls, chain: str = "default") -> list[str]:
        return cls._fallback_chains.get(chain, cls._fallback_chains["default"])
    
    @classmethod
    def get_tier_models(cls, tier: str) -> list[str]:
        return cls._model_tiers.get(tier, cls._model_tiers["balanced"])


# Convenience function
def get_provider(
    provider_type: str,
    provider: Optional[str] = None,
    tier: Optional[str] = None
) -> Any:
    """
    Universal provider getter.
    
    Examples:
        get_provider("llm", provider="claude")
        get_provider("llm", tier="fast")
        get_provider("search")
        get_provider("image")
    """
    if provider_type == "llm":
        return ProviderRegistry.get_llm(provider=provider, tier=tier)
    elif provider_type == "search":
        return ProviderRegistry.get_search(provider=provider)
    elif provider_type == "image":
        return ProviderRegistry.get_image(provider=provider)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
