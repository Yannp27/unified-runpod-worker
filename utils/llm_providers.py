"""
Concrete LLM Provider Implementations

DeepSeek, Claude, Gemini - all implementing LLMProvider protocol.
"""

import os
import aiohttp
from typing import Optional
from .providers import LLMProvider, LLMResponse, ProviderRegistry


# =============================================================================
# DEEPSEEK PROVIDER
# =============================================================================

class DeepSeekProvider:
    """
    DeepSeek-V3.1 provider via OpenAI-compatible API.
    Connects to vLLM or llama.cpp server.
    """
    
    provider_name = "deepseek"
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_model: str = "deepseek-v3.1"
    ):
        self.endpoint = endpoint or os.environ.get("DEEPSEEK_URL", "http://localhost:8000")
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        try:
            session = await self._get_session()
            
            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            async with session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choice = data.get("choices", [{}])[0]
                    return LLMResponse(
                        text=choice.get("message", {}).get("content", ""),
                        model=data.get("model", model or self.default_model),
                        usage=data.get("usage", {}),
                        success=True
                    )
                else:
                    error = await resp.text()
                    return LLMResponse(
                        text="",
                        model=model or self.default_model,
                        usage={},
                        success=False,
                        error=f"HTTP {resp.status}: {error}"
                    )
        except Exception as e:
            return LLMResponse(
                text="",
                model=model or self.default_model,
                usage={},
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.endpoint}/health") as resp:
                return resp.status == 200
        except:
            return False
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# CLAUDE PROVIDER
# =============================================================================

class ClaudeProvider:
    """
    Claude provider via Anthropic API (or AG Proxy).
    """
    
    provider_name = "claude"
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_model: str = "claude-sonnet-4-5"
    ):
        self.endpoint = endpoint or os.environ.get(
            "ANTHROPIC_BASE_URL",
            "https://api.anthropic.com"
        )
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        try:
            session = await self._get_session()
            
            # Convert to Anthropic format if needed
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Anthropic uses system as top-level param
                    continue
                formatted_messages.append(msg)
            
            system_prompt = next(
                (m["content"] for m in messages if m["role"] == "system"),
                None
            )
            
            payload = {
                "model": model or self.default_model,
                "messages": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                payload["system"] = system_prompt
            
            async with session.post(
                f"{self.endpoint}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("content", [{}])
                    text = content[0].get("text", "") if content else ""
                    return LLMResponse(
                        text=text,
                        model=data.get("model", model or self.default_model),
                        usage=data.get("usage", {}),
                        success=True
                    )
                else:
                    error = await resp.text()
                    return LLMResponse(
                        text="",
                        model=model or self.default_model,
                        usage={},
                        success=False,
                        error=f"HTTP {resp.status}: {error}"
                    )
        except Exception as e:
            return LLMResponse(
                text="",
                model=model or self.default_model,
                usage={},
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.endpoint}/health") as resp:
                return resp.status == 200
        except:
            return False
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# GEMINI PROVIDER
# =============================================================================

class GeminiProvider:
    """
    Gemini provider via OpenAI-compatible endpoint (AG Proxy).
    """
    
    provider_name = "gemini"
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_model: str = "gemini-2.5-flash"
    ):
        self.endpoint = endpoint or os.environ.get(
            "GEMINI_ENDPOINT",
            os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:8080")
        )
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        try:
            session = await self._get_session()
            
            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            async with session.post(
                f"{self.endpoint}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("content", [{}])
                    text = content[0].get("text", "") if content else ""
                    return LLMResponse(
                        text=text,
                        model=data.get("model", model or self.default_model),
                        usage=data.get("usage", {}),
                        success=True
                    )
                else:
                    error = await resp.text()
                    return LLMResponse(
                        text="",
                        model=model or self.default_model,
                        usage={},
                        success=False,
                        error=f"HTTP {resp.status}: {error}"
                    )
        except Exception as e:
            return LLMResponse(
                text="",
                model=model or self.default_model,
                usage={},
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.endpoint}/health") as resp:
                return resp.status == 200
        except:
            return False
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# OPENROUTER PROVIDER (DeepSeek V3.2 Speciale - Cheaper Output)
# =============================================================================

class OpenRouterProvider:
    """
    OpenRouter provider - access to DeepSeek V3.2 Speciale with cheaper output.
    
    Pricing (per 1M tokens):
    - Input: $0.27
    - Output: $0.41 (63% cheaper than direct API)
    """
    
    provider_name = "openrouter"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "deepseek/deepseek-v3.2-speciale"
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.endpoint = "https://openrouter.ai/api/v1"
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=180)  # Longer for reasoning
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="",
                model=model or self.default_model,
                usage={},
                success=False,
                error="OPENROUTER_API_KEY not set"
            )
        
        try:
            session = await self._get_session()
            
            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Yannp27/unified-runpod-worker",
                "X-Title": "Unified RunPod Worker"
            }
            
            async with session.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choice = data.get("choices", [{}])[0]
                    return LLMResponse(
                        text=choice.get("message", {}).get("content", ""),
                        model=data.get("model", model or self.default_model),
                        usage=data.get("usage", {}),
                        success=True
                    )
                else:
                    error = await resp.text()
                    return LLMResponse(
                        text="",
                        model=model or self.default_model,
                        usage={},
                        success=False,
                        error=f"HTTP {resp.status}: {error}"
                    )
        except Exception as e:
            return LLMResponse(
                text="",
                model=model or self.default_model,
                usage={},
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        return self.api_key is not None
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# REGISTRATION
# =============================================================================

def register_llm_providers():
    """Register all LLM providers with the registry."""
    # OpenRouter for DeepSeek V3.2 Speciale (cheaper output)
    if os.environ.get("OPENROUTER_API_KEY"):
        ProviderRegistry.register_llm("deepseek", OpenRouterProvider())
        print("[LLM] Registered DeepSeek via OpenRouter (V3.2 Speciale)")
    else:
        # Fallback to direct DeepSeek endpoint
        ProviderRegistry.register_llm("deepseek", DeepSeekProvider())
        print("[LLM] Registered DeepSeek (direct endpoint)")
    
    # Also register as explicit "openrouter" provider
    ProviderRegistry.register_llm("openrouter", OpenRouterProvider())
    
    # Claude and Gemini via AG proxy
    ProviderRegistry.register_llm("claude", ClaudeProvider())
    ProviderRegistry.register_llm("gemini", GeminiProvider())


# =============================================================================
# RESILIENT CLIENT WITH FALLBACK
# =============================================================================

class ResilientLLMClient:
    """
    LLM client with automatic fallback chain.
    Uses provider registry for hot-swapping.
    """
    
    def __init__(self, fallback_chain: str = "default"):
        self.chain = ProviderRegistry.get_fallback_chain(fallback_chain)
    
    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Try each provider in chain until one succeeds."""
        last_error = None
        
        for provider_name in self.chain:
            try:
                provider = ProviderRegistry.get_llm(provider=provider_name)
                result = await provider.complete(messages, model=model, **kwargs)
                
                if result.success:
                    return result
                
                last_error = result.error
                print(f"[FALLBACK] {provider_name} failed: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                print(f"[FALLBACK] {provider_name} exception: {e}")
        
        return LLMResponse(
            text="",
            model="",
            usage={},
            success=False,
            error=f"All providers failed. Last error: {last_error}"
        )
