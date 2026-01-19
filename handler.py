"""
Unified RunPod Serverless Handler

Multi-project dispatcher - routes actions to project handlers.
All LLM calls run in parallel via asyncio.
"""

import os
import asyncio
import runpod
from typing import Dict, Any

# Startup logging for RunPod Hub visibility
print("=" * 60)
print("[UNIFIED WORKER] Starting...")
print(f"[UNIFIED WORKER] LLM Provider: {os.environ.get('DEFAULT_LLM_PROVIDER', 'claude')}")
print(f"[UNIFIED WORKER] SearXNG URL: {os.environ.get('SEARXNG_URL', 'not set')}")
print(f"[UNIFIED WORKER] Worker Type: {os.environ.get('WORKER_TYPE', 'unified')}")
print("=" * 60)

# =============================================================================
# PROJECT HANDLERS
# =============================================================================

from projects import pleasance, speedb04t, image_tools, swarm

PROJECT_HANDLERS = {
    "pleasance": pleasance.handle,
    "speedb04t": speedb04t.handle,
    "image_tools": image_tools.handle,
    "swarm": swarm.handle,
}

# =============================================================================
# MAIN DISPATCHER
# =============================================================================

# GPU-heavy actions that should NOT run concurrently within same worker
GPU_BOUND_ACTIONS = {"rembg", "upscale", "pipeline", "keyframe_interpolate"}

async def handler(job: dict) -> dict:
    """
    Unified async handler - routes to project handlers.
    
    Supports in-worker concurrency for I/O-bound operations.
    GPU-bound operations are still queued properly by RunPod.
    
    Input:
    {
        "input": {
            "project": "pleasance" | "speedb04t",
            "action": "...",
            ...action-specific data...
        }
    }
    """
    try:
        input_data = job.get("input", {})
        project = input_data.get("project", "pleasance")  # Default to pleasance
        action = input_data.get("action", "")
        
        print(f"[UNIFIED WORKER] Request: project={project}, action={action}")
        
        # Health check for dispatcher
        if action == "ping":
            return {
                "status": "ok",
                "projects": list(PROJECT_HANDLERS.keys()),
                "mode": "unified",
                "concurrency": "async"
            }
        
        # Route to project handler
        if project not in PROJECT_HANDLERS:
            return {"error": f"Unknown project: {project}. Available: {list(PROJECT_HANDLERS.keys())}"}
        
        handler_fn = PROJECT_HANDLERS[project]
        
        # All project handlers are already async
        result = await handler_fn(input_data)
        return result
            
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    """
    Dynamic concurrency adjustment.
    
    Returns max concurrent requests this worker should handle.
    - I/O-bound ops (LLM calls, search): up to 4 concurrent
    - GPU-bound ops: RunPod handles queuing at worker level
    """
    # Allow up to 4 concurrent requests per worker
    # GPU-heavy ops will naturally serialize on the GPU
    return 4


# RunPod Serverless entry point with concurrency support
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})

