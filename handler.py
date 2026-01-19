"""
Unified RunPod Serverless Handler

Multi-project dispatcher - routes actions to project handlers.
All LLM calls run in parallel via asyncio.
"""

import os
import asyncio
import runpod
from typing import Dict, Any

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

def handler(job: dict) -> dict:
    """
    Unified handler - routes to project handlers.
    
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
        
        # Health check for dispatcher
        if input_data.get("action") == "ping":
            return {
                "status": "ok",
                "projects": list(PROJECT_HANDLERS.keys()),
                "mode": "unified"
            }
        
        # Route to project handler
        if project not in PROJECT_HANDLERS:
            return {"error": f"Unknown project: {project}. Available: {list(PROJECT_HANDLERS.keys())}"}
        
        handler_fn = PROJECT_HANDLERS[project]
        
        # Run in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(handler_fn(input_data))
            return result
        finally:
            loop.close()
            
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# RunPod Serverless entry point
runpod.serverless.start({"handler": handler})
