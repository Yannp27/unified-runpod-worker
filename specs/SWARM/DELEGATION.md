# Claude→DeepSeek Delegation

## Pattern Overview
Claude acts as orchestrator/reviewer. DeepSeek handles bulk work + search. Every decision is config-driven.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│     Claude      │────▶│    DeepSeek      │────▶│     Claude      │
│  (Orchestrator) │     │   (Swarmer)      │     │   (Reviewer)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
       │                        │                        │
       ▼                        ▼                        ▼
  Route task            Execute + Search           Approve/Reject
```

---

## Delegation Rules

### Task Routing Matrix
```yaml
delegation:
  rules:
    - task_type: "research"
      primary: { provider: "deepseek", tier: "fast", search: true }
      reviewer: { provider: "claude", tier: "balanced" }
    
    - task_type: "bulk_content"
      primary: { provider: "deepseek", tier: "balanced" }
      reviewer: { provider: "claude", tier: "balanced" }
    
    - task_type: "critical_reasoning"
      primary: { provider: "claude", tier: "premium" }
      reviewer: null  # No review needed
    
    - task_type: "code_review"
      primary: { provider: "claude", tier: "premium" }
      reviewer: null
    
    - task_type: "image_processing"
      primary: { provider: "gpu_worker" }
      reviewer: { provider: "claude", tier: "fast", optional: true }
```

### Task Type Detection
```python
def classify_task(task: str) -> str:
    """Auto-classify task type for routing."""
    indicators = {
        "research": ["search", "find", "look up", "what is", "latest"],
        "bulk_content": ["write", "generate", "create", "draft"],
        "critical_reasoning": ["analyze", "decide", "strategy", "plan"],
        "code_review": ["review code", "check this", "bug"],
        "image_processing": ["upscale", "remove background", "enhance"],
    }
    
    task_lower = task.lower()
    for task_type, keywords in indicators.items():
        if any(kw in task_lower for kw in keywords):
            return task_type
    
    return "bulk_content"  # Default
```

---

## Implementation

### Delegator Node
```python
async def delegator_node(state: SwarmState) -> SwarmState:
    """Claude decides what to delegate."""
    claude = get_provider("llm", provider="claude", tier="balanced")
    
    # Ask Claude how to break down the task
    delegation_prompt = f"""
    Task: {state["task"]}
    
    Break this into subtasks. For each, specify:
    - subtask_id
    - description
    - model_tier: "fast" | "balanced" | "premium"
    - needs_search: true | false
    
    Return JSON array.
    """
    
    response = await claude.complete([{"role": "user", "content": delegation_prompt}])
    subtasks = json.loads(response["text"])
    
    return {
        "results": {"subtasks": subtasks},
        "current_agent": "swarmer"
    }
```

### Swarmer Node (DeepSeek)
```python
async def swarmer_node(state: SwarmState) -> SwarmState:
    """DeepSeek executes subtasks in parallel."""
    deepseek = get_provider("llm", provider="deepseek")
    search = get_provider("search")
    
    subtasks = state["results"]["subtasks"]
    
    async def execute_subtask(subtask):
        context = ""
        if subtask.get("needs_search"):
            results = await search.search(subtask["description"])
            context = f"\nSearch results:\n{format_results(results)}"
        
        tier = subtask.get("model_tier", "balanced")
        llm = get_provider("llm", provider="deepseek", tier=tier)
        
        response = await llm.complete([
            {"role": "user", "content": subtask["description"] + context}
        ])
        return {"id": subtask["subtask_id"], "output": response["text"]}
    
    # Parallel execution
    outputs = await asyncio.gather(*[execute_subtask(s) for s in subtasks])
    
    return {
        "results": {"subtask_outputs": outputs},
        "current_agent": "reviewer"
    }
```

### Reviewer Node (Claude)
```python
async def reviewer_node(state: SwarmState) -> SwarmState:
    """Claude reviews DeepSeek's work."""
    claude = get_provider("llm", provider="claude", tier="balanced")
    
    outputs = state["results"]["subtask_outputs"]
    
    review_prompt = f"""
    Original task: {state["task"]}
    
    Subtask outputs:
    {json.dumps(outputs, indent=2)}
    
    Review each output:
    1. Is it accurate and complete?
    2. Any factual errors?
    3. Does it match the task requirements?
    
    Return JSON:
    {{
      "approved": true/false,
      "feedback": "...",
      "revisions_needed": [{{subtask_id, issue}}]
    }}
    """
    
    response = await claude.complete([{"role": "user", "content": review_prompt}])
    review = json.loads(response["text"])
    
    if review["approved"]:
        return {"results": {"final": outputs, "review": review}, "current_agent": "end"}
    else:
        return {
            "results": {"revision_feedback": review},
            "current_agent": "swarmer",
            "iteration": state["iteration"] + 1
        }
```

---

## Graph Assembly

```python
graph = StateGraph(SwarmState)

graph.add_node("delegator", delegator_node)
graph.add_node("swarmer", swarmer_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge("delegator", "swarmer")
graph.add_edge("swarmer", "reviewer")
graph.add_conditional_edges(
    "reviewer",
    lambda s: END if s["results"].get("final") else "swarmer"
)

graph.set_entry_point("delegator")
delegation_swarm = graph.compile(checkpointer=checkpointer)
```

---

## Cost Optimization

| Model | Cost/1M tokens | Use For |
|-------|----------------|---------|
| DeepSeek-Q4 | ~$0 (self-hosted) | Bulk drafts |
| DeepSeek-Q8 | ~$0 (self-hosted) | Quality content |
| Claude Sonnet | $3-15 | Orchestration, review |
| Claude Opus | $15-75 | Critical only |

**Result**: 90% of tokens go through DeepSeek (free). Claude only for routing/review.
