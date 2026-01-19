# Search Providers

## Purpose
Enable LLMs to access real-time web information. Primary: SearXNG (free, self-hosted). Fallback: Tavily API.

---

## SearXNG (Primary)

### Why SearXNG
- **Free**: No per-query costs
- **Private**: Self-hosted, no data leaves your infra
- **Aggregates**: Pulls from Google, Bing, DuckDuckGo, etc.
- **JSON API**: Native structured output

### Configuration
```yaml
searxng:
  url: "${SEARXNG_URL}"  # e.g., http://searxng:8080
  format: "json"
  engines:
    - google
    - bing
    - duckduckgo
    - wikipedia
  categories:
    - general
    - news
    - science
  timeout: 10
  max_results: 10
```

### Docker Compose (Self-Hosted)
```yaml
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8080:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080
```

### settings.yml (Enable JSON API)
```yaml
server:
  secret_key: "your-secret-key"
  
search:
  formats:
    - html
    - json  # Required for API access
```

---

## Tavily (Fallback)

### Why Tavily
- Optimized for LLM consumption
- Returns pre-summarized results
- Higher quality for research tasks

### Configuration
```yaml
tavily:
  api_key: "${TAVILY_API_KEY}"
  search_depth: "advanced"  # or "basic"
  include_raw_content: false
  max_results: 5
```

---

## Provider Interface

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str | None = None

class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        max_results: int = 10,
        categories: list[str] | None = None
    ) -> list[SearchResult]: ...

class SearXNGProvider:
    def __init__(self, url: str):
        self.url = url
    
    async def search(self, query: str, max_results: int = 10, **kwargs):
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
            }
            async with session.get(f"{self.url}/search", params=params) as resp:
                data = await resp.json()
                return [
                    SearchResult(
                        title=r["title"],
                        url=r["url"],
                        snippet=r.get("content", ""),
                    )
                    for r in data.get("results", [])[:max_results]
                ]
```

---

## Integration with LLM

### Search-Augmented Generation
```python
async def search_augmented_complete(
    query: str,
    llm: LLMProvider,
    search: SearchProvider
) -> str:
    # 1. Search for context
    results = await search.search(query, max_results=5)
    context = "\n".join([
        f"[{r.title}]({r.url}): {r.snippet}"
        for r in results
    ])
    
    # 2. Inject into prompt
    prompt = f"""Answer based on these search results:

{context}

Question: {query}
"""
    
    # 3. Generate response
    response = await llm.complete([{"role": "user", "content": prompt}])
    return response["text"]
```

---

## Fallback Logic
```python
SEARCH_FALLBACK = ["searxng", "tavily"]

async def resilient_search(query: str) -> list[SearchResult]:
    for provider_name in SEARCH_FALLBACK:
        provider = get_search_provider(provider_name)
        try:
            results = await provider.search(query)
            if results:
                return results
        except Exception as e:
            log(f"[SEARCH FALLBACK] {provider_name} failed: {e}")
    return []
```
