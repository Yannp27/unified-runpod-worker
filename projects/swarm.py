"""
LangGraph Swarm Project Handler

Uses LangGraph StateGraph for orchestration with Claude→DeepSeek delegation.
Supports cross-worker routing (CPU→GPU) for distributed execution.
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, Optional, Literal
from operator import add
from dataclasses import dataclass

# LangGraph - lazy import to avoid hanging on startup
LANGGRAPH_AVAILABLE = None  # Checked lazily in build_swarm_graph()


# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import get_provider, init_providers, resilient_search, get_router


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SwarmState(TypedDict):
    """Shared state across all swarm nodes."""
    task: str
    messages: Annotated[list, add]  # Accumulates
    results: dict
    errors: list
    current_agent: str
    iteration: int
    config: dict


# =============================================================================
# AGENT NODES
# =============================================================================

async def delegator_node(state: SwarmState) -> dict:
    """
    Claude orchestrator - breaks task into subtasks for DeepSeek.
    """
    claude = get_provider("llm", provider="claude")
    
    delegation_prompt = f"""You are a task delegator. Break this task into subtasks.

Task: {state["task"]}

For each subtask, specify:
- subtask_id: unique identifier
- description: what to do
- model_tier: "fast" (bulk work) or "balanced" (quality)
- needs_search: true if requires web search

Return ONLY a JSON array, no other text:
[{{"subtask_id": "1", "description": "...", "model_tier": "fast", "needs_search": false}}]
"""
    
    response = await claude.complete([{"role": "user", "content": delegation_prompt}])
    
    if not response.success:
        return {
            "errors": [f"Delegator failed: {response.error}"],
            "current_agent": "end"
        }
    
    try:
        # Extract JSON from response
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        subtasks = json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "errors": [f"Failed to parse subtasks: {e}"],
            "results": {"raw_delegation": response.text},
            "current_agent": "end"
        }
    
    return {
        "results": {"subtasks": subtasks},
        "messages": [f"Delegated {len(subtasks)} subtasks"],
        "current_agent": "swarmer"
    }


async def swarmer_node(state: SwarmState) -> dict:
    """
    DeepSeek swarmers - execute subtasks in parallel with optional search.
    Uses worker router for cross-worker calls.
    """
    deepseek = get_provider("llm", provider="deepseek")
    router = get_router()
    subtasks = state["results"].get("subtasks", [])
    
    if not subtasks:
        return {
            "errors": ["No subtasks to execute"],
            "current_agent": "end"
        }
    
    async def execute_subtask(subtask: dict) -> dict:
        subtask_id = subtask.get("subtask_id", "unknown")
        description = subtask.get("description", "")
        needs_search = subtask.get("needs_search", False)
        
        context = ""
        if needs_search:
            # Use router for search (routes to CPU worker if on GPU)
            results = await router.search(description, max_results=5)
            if results:
                if isinstance(results[0], dict):
                    context = "\n\nSearch results:\n" + "\n".join([
                        f"- [{r.get('title', '')}]({r.get('url', '')}): {r.get('snippet', '')}"
                        for r in results
                    ])
                else:
                    context = "\n\nSearch results:\n" + "\n".join([
                        f"- [{r.title}]({r.url}): {r.snippet}"
                        for r in results
                    ])
        
        prompt = f"{description}{context}"
        response = await deepseek.complete([{"role": "user", "content": prompt}])
        
        return {
            "subtask_id": subtask_id,
            "output": response.text if response.success else None,
            "error": response.error if not response.success else None,
            "search_used": needs_search and bool(context)
        }
    
    # Parallel execution
    outputs = await asyncio.gather(*[execute_subtask(s) for s in subtasks])
    
    successful = sum(1 for o in outputs if o.get("output"))
    
    return {
        "results": {"subtask_outputs": outputs},
        "messages": [f"Executed {successful}/{len(subtasks)} subtasks"],
        "current_agent": "reviewer"
    }


async def reviewer_node(state: SwarmState) -> dict:
    """
    Claude reviewer - approves or requests revision.
    """
    claude = get_provider("llm", provider="claude")
    outputs = state["results"].get("subtask_outputs", [])
    
    review_prompt = f"""Review these subtask outputs for the original task.

Original task: {state["task"]}

Subtask outputs:
{json.dumps(outputs, indent=2)}

Evaluate:
1. Is each output accurate and complete?
2. Does it answer the original task?
3. Any factual errors or missing information?

Return ONLY JSON:
{{
  "approved": true/false,
  "feedback": "overall assessment",
  "revisions_needed": [{{"subtask_id": "...", "issue": "..."}}]
}}
"""
    
    response = await claude.complete([{"role": "user", "content": review_prompt}])
    
    if not response.success:
        # On review failure, approve anyway to avoid infinite loops
        return {
            "results": {"final": outputs, "review_error": response.error},
            "messages": ["Review failed, accepting outputs"],
            "current_agent": "end"
        }
    
    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        review = json.loads(text)
    except json.JSONDecodeError:
        return {
            "results": {"final": outputs, "review_raw": response.text},
            "messages": ["Review parse failed, accepting outputs"],
            "current_agent": "end"
        }
    
    if review.get("approved", True):
        return {
            "results": {"final": outputs, "review": review},
            "messages": ["Review approved"],
            "current_agent": "end"
        }
    
    # Check iteration limit
    if state["iteration"] >= 3:
        return {
            "results": {"final": outputs, "review": review},
            "messages": ["Max iterations reached, accepting outputs"],
            "current_agent": "end"
        }
    
    # Request revision
    return {
        "results": {"revision_feedback": review},
        "messages": [f"Revision requested: {review.get('feedback', 'No feedback')}"],
        "iteration": state["iteration"] + 1,
        "current_agent": "swarmer"
    }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_swarm_graph():
    """Build the LangGraph swarm with Claude→DeepSeek→Claude flow."""
    global LANGGRAPH_AVAILABLE
    
    # Lazy import LangGraph
    if LANGGRAPH_AVAILABLE is None:
        try:
            from langgraph.graph import StateGraph as LG_StateGraph, END as LG_END
            from langgraph.checkpoint.memory import MemorySaver as LG_MemorySaver
            LANGGRAPH_AVAILABLE = True
        except ImportError:
            LANGGRAPH_AVAILABLE = False
            print("[SWARM] LangGraph not installed, using fallback mode")
    
    if not LANGGRAPH_AVAILABLE:
        return None
    
    # Import again for use (already cached by Python)
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    
    graph = StateGraph(SwarmState)
    
    # Add nodes
    graph.add_node("delegator", delegator_node)
    graph.add_node("swarmer", swarmer_node)
    graph.add_node("reviewer", reviewer_node)
    
    # Add edges
    graph.add_edge("delegator", "swarmer")
    graph.add_edge("swarmer", "reviewer")
    
    # Conditional edge from reviewer
    def route_after_review(state: SwarmState) -> Literal["swarmer", "__end__"]:
        if state.get("current_agent") == "end":
            return END
        if state.get("current_agent") == "swarmer":
            return "swarmer"
        return END
    
    graph.add_conditional_edges("reviewer", route_after_review)
    
    # Entry point
    graph.set_entry_point("delegator")
    
    # Compile with checkpointing
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# FALLBACK (NO LANGGRAPH)
# =============================================================================

async def fallback_swarm(task: str, config: dict) -> dict:
    """Simple sequential execution when LangGraph unavailable."""
    init_providers()
    
    state = {
        "task": task,
        "messages": [],
        "results": {},
        "errors": [],
        "current_agent": "delegator",
        "iteration": 0,
        "config": config
    }
    
    # Delegator
    result = await delegator_node(state)
    state.update(result)
    
    if state["current_agent"] == "end":
        return state
    
    # Swarmer
    result = await swarmer_node(state)
    state.update(result)
    
    if state["current_agent"] == "end":
        return state
    
    # Reviewer
    result = await reviewer_node(state)
    state.update(result)
    
    return state


# =============================================================================
# HANDLER
# =============================================================================

_swarm_graph = None

async def handle(input_data: dict) -> dict:
    """
    Swarm project handler.
    
    Input:
        action: "execute" | "health"
        task: str (for execute)
        config: dict (optional)
    """
    global _swarm_graph
    
    action = input_data.get("action", "execute")
    
    if action == "health":
        return {
            "status": "ok",
            "project": "swarm",
            "langgraph_available": LANGGRAPH_AVAILABLE
        }
    
    if action == "execute":
        task = input_data.get("task")
        if not task:
            return {"error": "No task provided"}
        
        config = input_data.get("config", {})
        job_id = input_data.get("job_id", "default")
        
        # Initialize providers
        init_providers()
        
        if LANGGRAPH_AVAILABLE:
            # Build graph if needed
            if _swarm_graph is None:
                _swarm_graph = build_swarm_graph()
            
            # Execute with checkpointing
            result = await _swarm_graph.ainvoke(
                {
                    "task": task,
                    "messages": [],
                    "results": {},
                    "errors": [],
                    "current_agent": "delegator",
                    "iteration": 0,
                    "config": config
                },
                {"configurable": {"thread_id": job_id}}
            )
        else:
            result = await fallback_swarm(task, config)
        
        return {
            "task": task,
            "results": result.get("results", {}),
            "messages": result.get("messages", []),
            "errors": result.get("errors", []),
            "iterations": result.get("iteration", 0)
        }
    
    return {"error": f"Unknown action: {action}"}
