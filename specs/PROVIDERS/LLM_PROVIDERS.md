# LLM Providers

## Provider Abstraction
All LLM calls go through `LLMProvider` interface. Swapping models = config change only.

---

## Registered Providers

### DeepSeek-V3.1 (Primary for bulk work)
```yaml
deepseek:
  endpoint: "${DEEPSEEK_ENDPOINT}"  # vLLM or llama.cpp server
  models:
    - id: "deepseek-v3.1-q4"
      context: 128000
      speed: "fast"
      quality: "good"
    - id: "deepseek-v3.1-q8"
      context: 128000
      speed: "medium"
      quality: "excellent"
  features:
    - thinking_mode  # Enable/disable chain-of-thought
    - function_calling
  serving:
    runtime: "vllm"  # or "llama.cpp"
    quantization: "gguf-q4_k_m"
    tensor_parallel: 2
```

### Claude (Orchestrator + Reviewer)
```yaml
claude:
  endpoint: "${ANTHROPIC_BASE_URL}"  # AG Proxy
  models:
    - id: "claude-sonnet-4-5"
      context: 200000
      speed: "fast"
      quality: "excellent"
    - id: "claude-opus-4-5-thinking"
      context: 200000
      speed: "slow"
      quality: "frontier"
  features:
    - extended_thinking
    - tool_use
    - vision
```

### Gemini (Fallback)
```yaml
gemini:
  endpoint: "${GEMINI_ENDPOINT}"  # AG Proxy
  models:
    - id: "gemini-2.5-flash-lite"
      speed: "ultrafast"
      quality: "good"
    - id: "gemini-2.5-flash"
      speed: "fast"
      quality: "excellent"
    - id: "gemini-3-pro"
      speed: "medium"
      quality: "frontier"
```

---

## Model Tiers
Tiers enable routing by task criticality:

| Tier | Use Case | Models |
|------|----------|--------|
| `fast` | Bulk generation, drafts | deepseek-q4, gemini-flash-lite |
| `balanced` | Standard tasks | deepseek-q8, claude-sonnet |
| `premium` | Critical reasoning | claude-opus-thinking |

---

## Fallback Chains
```python
FALLBACK_CHAINS = {
    "default": ["deepseek", "gemini", "claude"],
    "premium": ["claude", "gemini-pro", "deepseek"],
    "budget": ["deepseek", "gemini-flash-lite"],
}
```

---

## Implementation

```python
from abc import ABC, abstractmethod
from typing import Protocol

class LLMProvider(Protocol):
    provider_name: str
    
    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> dict:
        """Returns: {"text": str, "model": str, "usage": dict}"""
        ...

class DeepSeekProvider(LLMProvider):
    provider_name = "deepseek"
    
    def __init__(self, endpoint: str, default_model: str = "deepseek-v3.1-q8"):
        self.endpoint = endpoint
        self.default_model = default_model
    
    async def complete(self, messages, model=None, **kwargs):
        # OpenAI-compatible API call to vLLM/llama.cpp
        ...

class ClaudeProvider(LLMProvider):
    provider_name = "claude"
    
    async def complete(self, messages, model=None, **kwargs):
        # Anthropic API via AG Proxy
        ...
```

---

## Hot-Swap Example
```python
# Swap DeepSeek for Gemini without code changes:
config.providers.llm.default = "gemini-2.5-flash"
# All bulk work now uses Gemini instead of DeepSeek
```
