"""
DeepSeek Local Provider

Connects to llama.cpp server running locally in GPU worker container.
"""

import os
import aiohttp
from typing import Optional
from .providers import LLMResponse, ProviderRegistry


class DeepSeekLocalProvider:
    """
    DeepSeek provider for local llama.cpp server.
    Used on GPU worker where DeepSeek runs in-container.
    """
    
    provider_name = "deepseek-local"
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        default_model: str = "deepseek-v3.1"
    ):
        # Default to localhost (llama.cpp running in same container)
        self.endpoint = endpoint or os.environ.get("DEEPSEEK_URL", "http://localhost:8000")
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 min for long generations
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
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                **kwargs
            }
            
            # llama.cpp uses /v1/chat/completions (OpenAI compatible)
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
                        model=self.default_model,
                        usage=data.get("usage", {}),
                        success=True
                    )
                else:
                    error = await resp.text()
                    return LLMResponse(
                        text="",
                        model=self.default_model,
                        usage={},
                        success=False,
                        error=f"HTTP {resp.status}: {error}"
                    )
        except Exception as e:
            return LLMResponse(
                text="",
                model=self.default_model,
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


def register_deepseek_local():
    """Register local DeepSeek provider (GPU worker only)."""
    ProviderRegistry.register_llm("deepseek-local", DeepSeekLocalProvider())
