# Example 03 · Agent Harness Patterns

> **Source:** [LangChain Agents — The Harness](https://docs.langchain.com/oss/python/langchain/agents)

## Core Idea: Agent = Model + Harness

The LangChain Agents documentation introduces a simple mental model:

```
Agent = Model + Harness
```

The **model** does the reasoning. The **harness** is everything surrounding it — the system prompt, tools, structured output schema, per-run context, checkpointer, and middleware. Getting the harness right is what makes an agent actually useful.

This example demonstrates three harness features that go beyond a basic ReAct loop:

---

## Demo A — Streaming

By default, `agent.invoke()` waits for the full response. `agent.stream()` yields state snapshots after every node execution, so you can print tool calls and partial replies as they arrive.

```python
for chunk in agent.stream(input, config=config, stream_mode="values"):
    latest = chunk["messages"][-1]
    if isinstance(latest, AIMessage) and latest.tool_calls:
        print(f"[tool call] → {[tc['name'] for tc in latest.tool_calls]}")
    elif isinstance(latest, AIMessage) and latest.content:
        print(f"[reply] {latest.content}")
```

`stream_mode="values"` yields the full state (all messages so far) after each step. Other modes (`"updates"`, `"messages"`) yield deltas or individual message chunks instead.

---

## Demo B — Structured Output (`response_format`)

Pass a Pydantic model as `response_format` to get a validated object back instead of free text. The agent populates every field by combining tool results with its own reasoning.

```python
class BookReport(BaseModel):
    title: str
    author: str
    one_line_summary: str
    themes: list[str]
    recommended_for: str
    rating: float  # 1.0–5.0

agent = create_agent(
    model=...,
    tools=[lookup_book],
    response_format=BookReport,
)

result = agent.invoke(input, config=config)
report: BookReport = result["structured_response"]  # fully validated Pydantic object
```

This is useful when downstream code needs to parse the agent's output — the harness guarantees the schema, so you never need to regex-parse free text.

---

## Demo C — Context Schema (`context_schema`)

`context_schema` lets you pass **per-run user data** into the agent at invocation time, without baking it into the system prompt or a tool parameter.

```python
@dataclass
class ReaderContext:
    reader_name: str
    preferred_style: str  # "academic", "casual", "bullet points", …

agent = create_agent(
    model=...,
    tools=[lookup_book],
    system_prompt="Address the reader by name and match their preferred style.",
    context_schema=ReaderContext,
)

result = agent.invoke(
    input,
    config=config,
    context=ReaderContext(reader_name="Alice", preferred_style="bullet points"),
)
```

Typical uses: user ID, locale, feature flags, API credentials, tenant config — anything that changes per request but shouldn't be hardcoded into the prompt.

---

## Running the Example

```bash
uv run python -m examples.s_03_agent_harness
```

Expected output (abbreviated):

```
============================================================
DEMO A — Streaming
============================================================
Q: Tell me about the book '1984' by George Orwell.

  [tool call] → lookup_book
  [reply] '1984' is a dystopian novel by George Orwell (1949)...

============================================================
DEMO B — Structured Output (response_format=BookReport)
============================================================
  Title           : Dune
  Author          : Frank Herbert
  One-line summary: A young nobleman's journey on a desert planet ...
  Themes          : ecology, religion, politics
  Recommended for : Readers who enjoy political intrigue and world-building
  Rating          : 4.8 / 5.0

============================================================
DEMO C — Context Schema (context_schema=ReaderContext)
============================================================
  Reader: Alice | Style: bullet points
  Response:
  • Nick Carraway narrates the story of...

  Reader: Bob | Style: casual and conversational
  Response:
  So here's the deal with Gatsby — he's this mysterious rich guy...
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | All three demos: streaming, structured output, context schema |
| `__main__.py` | Entry point for `python -m examples.s_03_agent_harness` |
