# LangGraph Swarm Core

## Why LangGraph
Replaces manual `orchestrator.py` logic with a formal state machine.

| Manual Pattern | LangGraph Equivalent |
|----------------|---------------------|
| `if/else` routing | Conditional edges |
| `try/except` loops | Graph cycles |
| Passing args between functions | Shared TypedDict state |
| Manual error recovery | Checkpointing + resume |

---

## Core Concepts

### StateGraph
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class SwarmState(TypedDict):
    messages: Annotated[list, add]  # Accumulates
    task: str
    results: dict
    errors: list
    current_agent: str
    iteration: int
```

### Nodes (Agents)
```python
async def researcher_node(state: SwarmState) -> SwarmState:
    """Search web + gather context."""
    search = get_provider("search")
    llm = get_provider("llm", tier="fast")
    
    results = await search.search(state["task"])
    summary = await llm.complete(f"Summarize: {results}")
    
    return {
        "results": {"research": summary},
        "current_agent": "writer"
    }

async def writer_node(state: SwarmState) -> SwarmState:
    """Generate content based on research."""
    llm = get_provider("llm", tier="balanced")
    
    content = await llm.complete(
        f"Write about {state['task']} using: {state['results']['research']}"
    )
    
    return {
        "results": {"draft": content},
        "current_agent": "reviewer"
    }

async def reviewer_node(state: SwarmState) -> SwarmState:
    """Review and approve/reject."""
    llm = get_provider("llm", tier="premium")  # Claude for review
    
    review = await llm.complete(
        f"Review this draft: {state['results']['draft']}"
    )
    
    return {
        "results": {"review": review},
        "current_agent": "end" if "APPROVED" in review else "writer"
    }
```

### Edges
```python
def route_after_review(state: SwarmState) -> str:
    if "APPROVED" in state["results"].get("review", ""):
        return END
    if state["iteration"] > 3:
        return END  # Max retries
    return "writer"  # Revise

graph = StateGraph(SwarmState)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges("reviewer", route_after_review)

graph.set_entry_point("researcher")
swarm = graph.compile()
```

---

## Checkpointing

### Enable Persistence
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string(":memory:")  # Or file/postgres
swarm = graph.compile(checkpointer=checkpointer)
```

### Resume from Failure
```python
# Get last state
config = {"configurable": {"thread_id": job_id}}
state = await swarm.aget_state(config)

# Resume from checkpoint
result = await swarm.ainvoke(state.values, config)
```

### Time Travel
```python
# List all checkpoints
history = list(swarm.get_state_history(config))

# Rewind to specific checkpoint
old_state = history[3]
result = await swarm.ainvoke(old_state.values, config)
```

---

## Human-in-the-Loop

```python
# Pause at reviewer for human approval
graph.add_node("human_gate", lambda s: s)  # No-op, just pause
graph.add_edge("reviewer", "human_gate")

# Interrupt before human_gate
swarm = graph.compile(interrupt_before=["human_gate"])

# Resume after human approval
await swarm.ainvoke(None, config)  # Continues from checkpoint
```

---

## Execution

```python
async def run_swarm(task: str, job_id: str) -> dict:
    config = {"configurable": {"thread_id": job_id}}
    
    result = await swarm.ainvoke(
        {"task": task, "messages": [], "results": {}, "errors": [], "iteration": 0},
        config
    )
    
    return result["results"]
```
