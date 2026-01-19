"""
Utils Package Initialization

Registers all providers on import.
"""

from .providers import (
    ProviderRegistry,
    get_provider,
    LLMProvider,
    LLMResponse,
    SearchProvider,
    SearchResult,
    ImageProvider,
    ImageAction,
    ImageConfig,
)

from .llm_providers import (
    DeepSeekProvider,
    ClaudeProvider,
    GeminiProvider,
    ResilientLLMClient,
    register_llm_providers,
)

from .search_providers import (
    SearXNGProvider,
    TavilyProvider,
    register_search_providers,
    resilient_search,
)

from .worker_router import (
    WorkerRouter,
    get_router,
)


def init_providers():
    """Initialize all provider registrations."""
    register_llm_providers()
    register_search_providers()
    # Image providers registered on GPU worker only
    try:
        from .image_providers import register_image_providers
        register_image_providers()
    except ImportError:
        pass  # No GPU packages on CPU worker


__all__ = [
    # Core
    "ProviderRegistry",
    "get_provider",
    "init_providers",
    
    # LLM
    "LLMProvider",
    "LLMResponse",
    "DeepSeekProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "ResilientLLMClient",
    
    # Search
    "SearchProvider",
    "SearchResult",
    "SearXNGProvider",
    "TavilyProvider",
    "resilient_search",
    
    # Image
    "ImageProvider",
    "ImageAction",
    "ImageConfig",
    
    # Router
    "WorkerRouter",
    "get_router",
]
