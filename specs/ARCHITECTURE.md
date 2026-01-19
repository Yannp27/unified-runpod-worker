# Unified RunPod Worker Architecture

## Overview
A modular, hot-swappable AI swarm system running on RunPod Serverless. All components (models, tools, sources) are abstracted behind provider interfaces for zero-code swapping.

---

## Core Principles

1. **Provider Abstraction**: Every external dependency (LLM, search, storage) implements a standard interface
2. **Config-Driven**: Model selection, routing rules, and fallbacks defined in config, not code
3. **Graceful Degradation**: Every provider has fallback chains; system never fully fails
4. **Observability**: Every step logs provider used, latency, token count

---

## System Layers

```
┌─────────────────────────────────────────────────────────┐
│                    RunPod Handler                       │
│              (Unified Dispatcher + Router)              │
├─────────────────────────────────────────────────────────┤
│                  LangGraph Orchestrator                 │
│    (StateGraph, Checkpointing, Conditional Routing)     │
├─────────────────────────────────────────────────────────┤
│                   Provider Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │   LLM    │  │  Search  │  │  Image   │  │ Storage │ │
│  │ Provider │  │ Provider │  │ Provider │  │Provider │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Concrete Backends                     │
│  Claude, DeepSeek, Gemini | SearXNG | Real-ESRGAN | S3 │
└─────────────────────────────────────────────────────────┘
```

---

## Provider Interface Contracts

### LLMProvider
```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        config: LLMConfig
    ) -> LLMResponse: ...
    
    async def health_check(self) -> bool: ...
```

### SearchProvider
```python
class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        max_results: int = 10
    ) -> list[SearchResult]: ...
```

### ImageProvider
```python
class ImageProvider(Protocol):
    async def process(
        self,
        image: bytes,
        action: ImageAction,
        config: ImageConfig
    ) -> bytes: ...
```

---

## Request Flow

1. **RunPod** receives job → extracts `project` + `action`
2. **Dispatcher** routes to project handler (swarm, image_tools, etc.)
3. **LangGraph** executes state machine with checkpointing
4. **Agents** call providers via abstraction layer
5. **Response** returned with full execution trace

---

## Configuration Schema

```yaml
# config.yaml
providers:
  llm:
    default: "deepseek-v3.1-q8"
    tiers:
      fast: ["deepseek-v3.1-q4", "gemini-2.5-flash-lite"]
      balanced: ["deepseek-v3.1-q8", "claude-sonnet-4-5"]
      premium: ["claude-opus-4-5-thinking"]
    fallback_chain: ["deepseek", "gemini", "claude"]
  
  search:
    default: "searxng"
    backends:
      searxng:
        url: "${SEARXNG_URL}"
        format: "json"
      tavily:
        api_key: "${TAVILY_API_KEY}"
  
  image:
    upscale:
      model: "realesrgan-anime-6b"
      scale: 4
    rembg:
      model: "isnet-anime"

routing:
  delegation:
    bulk_content: { primary: "deepseek", reviewer: "claude" }
    research: { primary: "deepseek+search", reviewer: "claude" }
    critical: { primary: "claude", reviewer: null }
```

---

## Related Specs
- [LLM_PROVIDERS.md](PROVIDERS/LLM_PROVIDERS.md)
- [SEARCH_PROVIDERS.md](PROVIDERS/SEARCH_PROVIDERS.md)
- [IMAGE_PROVIDERS.md](PROVIDERS/IMAGE_PROVIDERS.md)
- [LANGGRAPH_CORE.md](SWARM/LANGGRAPH_CORE.md)
- [DELEGATION.md](SWARM/DELEGATION.md)
