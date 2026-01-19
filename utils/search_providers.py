"""
Search Provider Implementations

SearXNG (primary, free) and Tavily (fallback).
"""

import os
import aiohttp
from typing import Optional
from .providers import SearchProvider, SearchResult, ProviderRegistry


# =============================================================================
# SEARXNG PROVIDER
# =============================================================================

class SearXNGProvider:
    """
    SearXNG metasearch engine.
    Free, self-hosted on RunPod CPU pod.
    """
    
    provider_name = "searxng"
    
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("SEARXNG_URL", "http://searxng:8080")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[list[str]] = None
    ) -> list[SearchResult]:
        try:
            session = await self._get_session()
            
            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
            }
            if categories:
                params["categories"] = ",".join(categories)
            
            async with session.get(f"{self.url}/search", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])[:max_results]
                    return [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            snippet=r.get("content", ""),
                            content=r.get("content")
                        )
                        for r in results
                    ]
                else:
                    print(f"[SEARXNG] Error: HTTP {resp.status}")
                    return []
        except Exception as e:
            print(f"[SEARXNG] Exception: {e}")
            return []
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# TAVILY PROVIDER
# =============================================================================

class TavilyProvider:
    """
    Tavily search API.
    Optimized for LLM consumption.
    """
    
    provider_name = "tavily"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.url = "https://api.tavily.com/search"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        categories: Optional[list[str]] = None
    ) -> list[SearchResult]:
        if not self.api_key:
            print("[TAVILY] No API key configured")
            return []
        
        try:
            session = await self._get_session()
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
            }
            
            async with session.post(self.url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    return [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            snippet=r.get("content", ""),
                            content=r.get("raw_content")
                        )
                        for r in results
                    ]
                else:
                    print(f"[TAVILY] Error: HTTP {resp.status}")
                    return []
        except Exception as e:
            print(f"[TAVILY] Exception: {e}")
            return []
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# REGISTRATION
# =============================================================================

def register_search_providers():
    """Register all search providers."""
    ProviderRegistry.register_search("searxng", SearXNGProvider())
    
    # Only register Tavily if API key available
    if os.environ.get("TAVILY_API_KEY"):
        ProviderRegistry.register_search("tavily", TavilyProvider())


# =============================================================================
# RESILIENT SEARCH
# =============================================================================

SEARCH_FALLBACK = ["searxng", "tavily"]

async def resilient_search(
    query: str,
    max_results: int = 10
) -> list[SearchResult]:
    """Search with fallback chain."""
    for provider_name in SEARCH_FALLBACK:
        try:
            provider = ProviderRegistry.get_search(provider=provider_name)
            results = await provider.search(query, max_results=max_results)
            if results:
                return results
        except Exception as e:
            print(f"[SEARCH FALLBACK] {provider_name} failed: {e}")
    
    return []
