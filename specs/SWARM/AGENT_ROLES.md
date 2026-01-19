# Agent Roles

## Overview
Swarm agents are specialized by function. Each role maps to LangGraph nodes.

---

## Core Roles

### Orchestrator
- **Provider**: Claude (balanced)
- **Function**: Break tasks into subtasks, route to specialists
- **Typical Prompts**: Task decomposition, dependency analysis

### Researcher
- **Provider**: DeepSeek + SearXNG
- **Function**: Gather context, search web, summarize findings
- **Tools**: Search, URL fetch, document parsing

### Writer
- **Provider**: DeepSeek (balanced)
- **Function**: Generate content based on research
- **Style**: Follows brand voice, format guidelines

### Reviewer
- **Provider**: Claude (balanced/premium)
- **Function**: Quality gate - approve, reject, or request revisions
- **Checks**: Accuracy, completeness, style, safety

### Editor
- **Provider**: Claude (fast)
- **Function**: Polish approved content - grammar, formatting
- **Non-blocking**: Runs after approval

### Image Processor
- **Provider**: GPU worker (Real-ESRGAN, rembg)
- **Function**: Upscale, remove backgrounds, fix artifacts
- **Async**: Can run parallel to text generation

---

## Role Configuration

```yaml
agent_roles:
  orchestrator:
    provider: claude
    tier: balanced
    max_retries: 2
  
  researcher:
    provider: deepseek
    tier: fast
    tools: [search, fetch_url]
    max_results: 10
  
  writer:
    provider: deepseek
    tier: balanced
    temperature: 0.7
    max_tokens: 4096
  
  reviewer:
    provider: claude
    tier: balanced
    temperature: 0.3  # More deterministic
    approval_threshold: 0.8
  
  editor:
    provider: claude
    tier: fast
    temperature: 0.3
  
  image_processor:
    provider: gpu_worker
    actions: [upscale, rembg, fix_artifacts]
```

---

## Role Swapping

Roles are abstract. Swap underlying model without changing flow:

```python
# Production: Claude reviews
config.agent_roles.reviewer.provider = "claude"

# Budget mode: Gemini reviews
config.agent_roles.reviewer.provider = "gemini"

# Same graph, different execution
```

---

## Multi-Agent Flows

### Content Pipeline
```
Orchestrator → Researcher → Writer → Reviewer → [Writer loop] → Editor → Done
```

### Image Pipeline
```
Orchestrator → Image Processor → [Optional] Reviewer → Done
```

### Research Pipeline
```
Orchestrator → Researcher (parallel x5) → Synthesizer → Reviewer → Done
```

---

## Adding Custom Roles

```python
# Define new role
async def fact_checker_node(state: SwarmState) -> SwarmState:
    claude = get_provider("llm", provider="claude", tier="premium")
    
    # Verify claims against sources
    claims = extract_claims(state["results"]["draft"])
    verified = await verify_claims(claims, claude)
    
    return {
        "results": {"fact_check": verified},
        "current_agent": "reviewer" if all(v["verified"] for v in verified) else "writer"
    }

# Register in graph
graph.add_node("fact_checker", fact_checker_node)
graph.add_edge("writer", "fact_checker")
graph.add_edge("fact_checker", "reviewer")
```

---

## Role Capabilities Matrix

| Role | Claude | DeepSeek | GPU | Search |
|------|--------|----------|-----|--------|
| Orchestrator | ✓ | | | |
| Researcher | | ✓ | | ✓ |
| Writer | | ✓ | | |
| Reviewer | ✓ | | | |
| Editor | ✓ | | | |
| Image Processor | | | ✓ | |
