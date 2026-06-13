# Example 02 · Research Agent

> **Source:** [LangChain Quickstart — LangChain Agents](https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents)

## What This Example Demonstrates

A **document research agent** built with `langchain.agents.create_agent`. The agent is given a single tool — fetching raw text from a URL — and uses it to answer detailed factual questions about the document's content. It cannot guess; it must fetch and verify.

## Key Concepts

### `create_agent`
LangChain's high-level agent factory. It wires together a model, a list of tools, a system prompt, and an optional checkpointer into a runnable agent without requiring you to manually define a graph.

```python
agent = create_agent(
    model=...,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)
```

Under the hood this uses LangGraph, so the agent is a `StateGraph` with the same ReAct loop as Example 01 — but the graph is built for you.

### Checkpointer and Multi-Turn Memory
`InMemorySaver` persists the conversation state keyed by `thread_id`. This means:
- The agent remembers previous messages within the same thread.
- You can ask follow-up questions in a second `.invoke()` call with the same `thread_id` and the agent retains full context.
- State is in-process only and is lost when the program exits.

### Grounding Agent Behavior via System Prompt
The system prompt instructs the agent **not to guess** — it must use the tool to retrieve the document before making any claims about line numbers or counts. This is a key technique for reducing hallucination in research agents.

## Tool

| Tool | Description |
|------|-------------|
| `fetch_text_from_url(url)` | Downloads the raw UTF-8 text of a document from any public URL |

## Running the Example

```bash
uv run python -m examples.s_02_research_agent
```

The agent will:
1. Receive a question about *The Great Gatsby* on Project Gutenberg
2. Call `fetch_text_from_url` to download the full text (~100K tokens)
3. Analyze the text and return verified answers

> **Note:** This example downloads ~75 KB of text and may make several LLM calls. Expect it to take 10–30 seconds depending on your model.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Tool definition, agent setup, research question, entry point |
| `__main__.py` | Thin wrapper so `python -m examples.s_02_research_agent` works |
