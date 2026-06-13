# Example 02 · Research Agent (Deep Agent + Multi-Turn Memory)

> **Source:** [LangChain Quickstart — LangChain Agents](https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents)

## What This Example Demonstrates

A **document research agent** built with `deepagents.create_deep_agent`. It fetches a full novel from Project Gutenberg and answers factual questions about it across two conversation turns — demonstrating that the agent remembers what it fetched in turn 1 when answering the follow-up in turn 2.

## Key Concepts

### `create_deep_agent`
A high-level agent factory from the `deepagents` package. Compared to `create_agent`, deep agents come with planning, sub-agent orchestration, and file-system tools already built in — maximum capability with minimal setup.

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=...,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)
```

### Multi-Turn Memory with `thread_id`
The `InMemorySaver` checkpointer persists conversation state keyed by `thread_id`. As long as both `.invoke()` calls use the same `thread_id`, the agent retains full context across calls:

```python
config = {"configurable": {"thread_id": "great-gatsby"}}

# Turn 1: agent fetches the document and answers initial questions
agent.invoke({"messages": [{"role": "user", "content": TURN_1}]}, config=config)

# Turn 2: agent already has the full document in context — no re-fetch needed
agent.invoke({"messages": [{"role": "user", "content": TURN_2}]}, config=config)
```

This is the key advantage of a stateful agent over a stateless chain: you can ask follow-up questions naturally, and the agent builds on previous results instead of starting from scratch.

### Grounding via System Prompt
The system prompt tells the agent not to guess line counts or positions — it must use `fetch_text_from_url` to retrieve the document first. This is a standard technique for reducing hallucination in retrieval-heavy agents.

## Conversation Flow

```
Turn 1 ──► fetch document (~75 KB) ──► answer: line counts, synopsis
Turn 2 ──► (document already in context) ──► answer: narrator background
```

## Tool

| Tool | Description |
|------|-------------|
| `fetch_text_from_url(url)` | Downloads the raw UTF-8 text of a document from any public URL |

## Running the Example

```bash
uv run python -m examples.s_02_research_agent
```

Expected output (abbreviated):

```
============================================================
Research Agent (Deep Agent) — The Great Gatsby
============================================================

────────────────────────────────────────────────────────────
[Turn 1] Project Gutenberg hosts a plain-text copy of ...
────────────────────────────────────────────────────────────
1) Lines containing "Gatsby": ...
2) First line containing "Daisy": line ...
3) Synopsis: ...

────────────────────────────────────────────────────────────
[Turn 2 (follow-up, no re-fetch needed)] Who is the narrator ...
────────────────────────────────────────────────────────────
The narrator is Nick Carraway ...
```

> **Note:** Turn 1 downloads ~75 KB and may take 10–30 seconds. Turn 2 is fast — the document is already in the agent's memory.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Tool, agent setup, two-turn conversation, entry point |
| `__main__.py` | Thin wrapper so `python -m examples.s_02_research_agent` works |
