# Example 01 · LLM with Tools (ReAct Agent)

> **Source:** [LangGraph Tutorial — Build a ReAct Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

## What This Example Demonstrates

This example shows how to build a **ReAct (Reasoning + Acting) Agent** using the LangGraph Graph API. The agent solves multi-step math problems by reasoning about which tool to call, executing it, observing the result, and repeating until it has a final answer.

## Key Concepts

### ReAct Pattern
ReAct is a prompting strategy where the model alternates between:
1. **Reasoning** — deciding what to do next
2. **Acting** — calling a tool
3. **Observing** — reading the tool's result

This loop continues until the model determines it has enough information to give a final answer.

### LangGraph StateGraph
Instead of a simple chain, this agent is modeled as a **graph**:

```
START → llm_call ──(has tool calls?)──► tool_node ──► llm_call
                 └──(no tool calls)──► END
```

- **`llm_call` node** — sends the conversation to the LLM and appends its response
- **`tool_node` node** — executes every tool call in the latest AI message
- **Conditional edge** — checks whether the LLM wants to call more tools or is done

### State Accumulation
`AgentState` holds a `messages` list annotated with `operator.add`, so each node *appends* messages rather than replacing them. The full conversation history (human → AI → tool results → AI → …) is always available to the LLM.

## Tools Available

| Tool | Description |
|------|-------------|
| `add(a, b)` | Returns `a + b` |
| `multiply(a, b)` | Returns `a * b` |
| `divide(a, b)` | Returns `a / b` |

## Running the Example

```bash
uv run python -m examples.s_01_llm_with_tools
```

Expected output (abbreviated):

```
==================================================
Question: What is 12345 multiplied by 17, then divided by 3?
==================================================
================================ Human Message =================================
What is 12345 multiplied by 17, then divided by 3?
================================== Ai Message ==================================
Tool Calls:
  multiply (call_xxx)
  Args: a=12345, b=17
================================= Tool Message =================================
Name: multiply
209865
================================== Ai Message ==================================
Tool Calls:
  divide (call_yyy)
  Args: a=209865, b=3
================================= Tool Message =================================
Name: divide
69955.0
================================== Ai Message ==================================
The result is 69955.0.
```

## Tracing with Langfuse

Each run creates a new Langfuse trace capturing the full call graph — LLM calls, tool calls, token counts, and latency. Open [http://localhost:3000](http://localhost:3000) after running to inspect the trace.

## Files

| File | Purpose |
|------|---------|
| `main.py` | All agent logic — tools, state, nodes, graph, entry point |
| `__main__.py` | Thin wrapper so `python -m examples.s_01_llm_with_tools` works |
