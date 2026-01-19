"""
Worker Router - Inter-Worker Communication

Routes requests between CPU and GPU workers.
CPU worker calls GPU worker for heavy tasks.
"""

import os
import aiohttp
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class WorkerConfig:
    """Configuration for a RunPod worker endpoint."""
    endpoint_id: str
    api_key: str
    timeout: int = 300  # 5 min default for GPU tasks


class WorkerRouter:
    """
    Routes requests to appropriate worker endpoint.
    
    Usage:
        router = WorkerRouter()
        result = await router.call_gpu("image_tools", {"action": "upscale", ...})
    """
    
    def __init__(self):
        self.runpod_api_key = os.environ.get("RUNPOD_API_KEY")
        self.gpu_endpoint_id = os.environ.get("GPU_WORKER_ENDPOINT")
        self.cpu_endpoint_id = os.environ.get("CPU_WORKER_ENDPOINT")
        self.worker_type = os.environ.get("WORKER_TYPE", "gpu")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_endpoint_url(self, endpoint_id: str) -> str:
        return f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    
    async def _call_endpoint(
        self,
        endpoint_id: str,
        payload: dict,
        timeout: int = 300
    ) -> dict:
        """Call a RunPod serverless endpoint."""
        if not self.runpod_api_key:
            return {"error": "RUNPOD_API_KEY not configured"}
        
        if not endpoint_id:
            return {"error": "Endpoint ID not configured"}
        
        session = await self._get_session()
        url = self._get_endpoint_url(endpoint_id)
        
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(url, json={"input": payload}, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "COMPLETED":
                        return data.get("output", {})
                    else:
                        return {"error": f"Job failed: {data.get('status')}", "details": data}
                else:
                    error = await resp.text()
                    return {"error": f"HTTP {resp.status}: {error}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def call_gpu(self, project: str, input_data: dict) -> dict:
        """
        Route request to GPU worker.
        
        Args:
            project: Project handler name (image_tools, swarm, etc.)
            input_data: Action-specific input data
        """
        if self.worker_type == "gpu":
            # Already on GPU, handle locally
            from projects import image_tools, swarm
            handlers = {
                "image_tools": image_tools.handle,
                "swarm": swarm.handle,
            }
            if project in handlers:
                return await handlers[project](input_data)
            return {"error": f"Unknown project: {project}"}
        
        # Route to GPU endpoint
        payload = {"project": project, **input_data}
        return await self._call_endpoint(self.gpu_endpoint_id, payload)
    
    async def call_cpu(self, project: str, input_data: dict) -> dict:
        """
        Route request to CPU worker.
        
        Args:
            project: Project handler name (search, orchestrator, etc.)
            input_data: Action-specific input data
        """
        if self.worker_type == "cpu":
            # Already on CPU, handle locally
            from utils import resilient_search
            if project == "search":
                query = input_data.get("query", "")
                results = await resilient_search(query)
                return {"results": [r.__dict__ for r in results]}
            return {"error": f"Unknown CPU project: {project}"}
        
        # Route to CPU endpoint
        payload = {"project": project, **input_data}
        return await self._call_endpoint(self.cpu_endpoint_id, payload)
    
    async def search(self, query: str, max_results: int = 10) -> list:
        """Convenience method for search."""
        if self.worker_type == "cpu":
            from utils import resilient_search
            return await resilient_search(query, max_results)
        else:
            result = await self.call_cpu("search", {
                "action": "search",
                "query": query,
                "max_results": max_results
            })
            return result.get("results", [])
    
    async def upscale(self, image_b64: str, scale: int = 4) -> dict:
        """Convenience method for upscaling."""
        return await self.call_gpu("image_tools", {
            "action": "upscale",
            "image_base64": image_b64,
            "scale": scale
        })
    
    async def rembg(self, image_b64: str, alpha_matting: bool = True) -> dict:
        """Convenience method for background removal."""
        return await self.call_gpu("image_tools", {
            "action": "rembg",
            "image_base64": image_b64,
            "alpha_matting": alpha_matting
        })


# Global router instance
_router: Optional[WorkerRouter] = None


def get_router() -> WorkerRouter:
    global _router
    if _router is None:
        _router = WorkerRouter()
    return _router
