# DeepSeek + SearXNG Integration

## Pipeline Overview
DeepSeek-V3.1 + SearXNG enables free, private, real-time web search for AI agents.

```
Query → SearXNG → Results → Inject into DeepSeek prompt → Response
```

---

## SearXNG Setup

### Docker Deployment
```yaml
# docker-compose.yml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888
    restart: unless-stopped
```

### Enable JSON API
```yaml
# searxng/settings.yml
server:
  secret_key: "generate-a-random-key"
  limiter: false  # Disable rate limiting for internal use

search:
  safe_search: 0
  formats:
    - html
    - json  # REQUIRED for API access
```

---

## DeepSeek Serving

### vLLM (Recommended)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3.1 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --quantization gguf \
  --port 8000
```

### llama.cpp (Lower VRAM)
```bash
./llama-server \
  -m deepseek-v3.1-q4_k_m.gguf \
  -c 32768 \
  --port 8000 \
  -ot ".ffn_.*_exps.=CPU"  # Offload MoE to RAM
```

---

## Integration Code

```python
import aiohttp
from typing import Optional

class DeepSeekSearchAgent:
    def __init__(
        self,
        deepseek_url: str = "http://localhost:8000",
        searxng_url: str = "http://localhost:8888"
    ):
        self.deepseek_url = deepseek_url
        self.searxng_url = searxng_url
    
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Query SearXNG for web results."""
        async with aiohttp.ClientSession() as session:
            params = {"q": query, "format": "json"}
            async with session.get(f"{self.searxng_url}/search", params=params) as resp:
                data = await resp.json()
                return data.get("results", [])[:max_results]
    
    async def search_augmented_query(
        self,
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Search web, inject results, query DeepSeek."""
        # 1. Search
        results = await self.search(query)
        
        # 2. Format context
        context = "\n\n".join([
            f"**{r['title']}** ({r['url']})\n{r.get('content', '')}"
            for r in results
        ])
        
        # 3. Build prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": f"""Based on these search results:

{context}

Answer: {query}"""
        })
        
        # 4. Query DeepSeek
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "deepseek-v3.1",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048
            }
            async with session.post(
                f"{self.deepseek_url}/v1/chat/completions",
                json=payload
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
```

---

## LangGraph Integration

```python
async def researcher_with_search(state: SwarmState) -> SwarmState:
    agent = DeepSeekSearchAgent(
        deepseek_url=os.environ["DEEPSEEK_URL"],
        searxng_url=os.environ["SEARXNG_URL"]
    )
    
    response = await agent.search_augmented_query(
        query=state["task"],
        system_prompt="You are a research assistant. Provide accurate, cited information."
    )
    
    return {
        "results": {"research": response},
        "current_agent": "writer"
    }
```

---

## Configuration

```yaml
integrations:
  deepseek_searxng:
    deepseek:
      url: "${DEEPSEEK_URL}"
      model: "deepseek-v3.1"
      max_tokens: 4096
    searxng:
      url: "${SEARXNG_URL}"
      max_results: 5
      engines: ["google", "bing", "duckduckgo"]
    
    # When to use search
    search_triggers:
      - "latest"
      - "current"
      - "recent news"
      - "search for"
      - "look up"
```
