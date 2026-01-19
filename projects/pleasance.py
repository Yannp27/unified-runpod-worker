"""
Pleasance Project Handler

Content generation and review for Kink Database.
All LLM calls run in parallel via asyncio.
"""

import os
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

PLEASANCE_API = os.environ.get("PLEASANCE_API", "https://api.pleasance.app")
AGENT_SECRET = os.environ.get("AGENT_SECRET")
PROXY_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://agproxy12461249316123.pleasance.app")

FAST_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "claude-sonnet-4-5",
]

SECTION_PROMPTS = {
    "appeal": """Write a compelling 2-3 paragraph description of why people find '{name}' appealing. 
Focus on psychological and sensory aspects. Be educational and non-judgmental.""",

    "howTo": """Write a practical guide for safely exploring '{name}' as beginners. 
Include safety considerations, communication tips, and gradual progression suggestions.""",

    "variations": """Describe 3-5 common variations or related practices to '{name}'. 
Be specific but tasteful. Include intensity levels.""",
}

# =============================================================================
# ASYNC LLM CLIENT
# =============================================================================

class AsyncLLMClient:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def call_model(self, prompt: str, model: str, max_tokens: int = 2048) -> Optional[str]:
        try:
            session = await self.get_session()
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            
            async with session.post(
                f"{PROXY_URL}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "content" in data and len(data["content"]) > 0:
                        return data["content"][0].get("text", "")
                    if "text" in data:
                        return data["text"]
            return None
        except Exception as e:
            print(f"[ERROR] {model}: {e}")
            return None
    
    async def complete(self, prompt: str) -> Dict[str, Any]:
        for i, model in enumerate(FAST_CHAIN):
            result = await self.call_model(prompt, model)
            if result:
                return {"text": result, "model": model, "success": True}
            if i < len(FAST_CHAIN) - 1:
                print(f"[FALLBACK] {model} failed, trying {FAST_CHAIN[i+1]}")
        return {"text": None, "model": None, "success": False, "error": "All models failed"}
    
    async def health_check(self) -> bool:
        try:
            session = await self.get_session()
            async with session.get(f"{PROXY_URL}/health") as resp:
                return resp.status == 200
        except:
            return False


client = AsyncLLMClient()

# =============================================================================
# GENERATION
# =============================================================================

async def generate_section(kink: dict, section_key: str) -> dict:
    prompt_template = SECTION_PROMPTS.get(section_key)
    if not prompt_template:
        return {"kinkId": kink.get("id"), "sectionKey": section_key, "content": None, "error": "Unknown section"}
    
    prompt = prompt_template.format(name=kink.get("name", "Unknown"))
    print(f"[GEN] {kink.get('name')} → {section_key}")
    
    result = await client.complete(prompt)
    
    return {
        "kinkId": kink["id"],
        "sectionKey": section_key,
        "content": result.get("text"),
        "model": result.get("model"),
        "error": result.get("error") if not result["success"] else None
    }


async def generate_batch(kinks: List[dict]) -> List[dict]:
    tasks = []
    for kink in kinks:
        for section_key in SECTION_PROMPTS.keys():
            tasks.append(generate_section(kink, section_key))
    
    print(f"[BATCH] Starting {len(tasks)} parallel LLM calls...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    sections = []
    for r in results:
        if isinstance(r, Exception):
            sections.append({"error": str(r)})
        else:
            sections.append(r)
    
    print(f"[BATCH] Completed {len(sections)} sections")
    await client.close()
    return sections

# =============================================================================
# HANDLER
# =============================================================================

async def handle(input_data: dict) -> dict:
    """
    Pleasance project handler.
    
    Actions:
    - health: Check proxy health
    - batch_generate: Generate sections for multiple kinks
    - batch_review: Review sections
    """
    action = input_data.get("action", "health")
    
    if action == "health":
        proxy_ok = await client.health_check()
        await client.close()
        return {"status": "ok", "proxy": proxy_ok, "project": "pleasance"}
    
    if action == "batch_generate":
        kinks = input_data.get("kinks", [])
        sections = await generate_batch(kinks)
        return {"sections": sections, "count": len(sections)}
    
    if action == "batch_review":
        # TODO: Implement review
        return {"error": "batch_review not yet implemented"}
    
    return {"error": f"Unknown action: {action}"}
