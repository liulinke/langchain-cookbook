# Example 02 · Research Agent — `create_agent` vs `create_deep_agent`

> **Source:** [LangChain Quickstart — LangChain Agents](https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents)

## What This Example Demonstrates

This example runs the **same research task** with two different agent factories and prints both results for direct comparison. It then shows a **multi-turn follow-up** that proves the deep agent remembers prior context without re-fetching the document.

## `create_agent` vs `create_deep_agent`

Both functions take the same arguments (`model`, `tools`, `system_prompt`, `checkpointer`) so they're drop-in alternatives. The difference is what they bring to the table out of the box:

| | `create_agent` | `create_deep_agent` |
|---|---|---|
| **Source** | `langchain.agents` | `deepagents` |
| **ReAct loop** | Yes | Yes |
| **Built-in planning** | No | Yes |
| **Sub-agent orchestration** | No | Yes |
| **Built-in file-system tools** | No | Yes (`ls`, `read_file`, `write_file`, `grep`, …) |
| **Best for** | Simple, focused tasks | Complex, multi-step research |

### Different counting strategies

Because `create_deep_agent` includes `FilesystemMiddleware` automatically, each agent can take a different — and more idiomatic — approach to line counting:

**`create_agent` (standard)** — no built-in file tools, so we provide a custom `count_lines_containing(url, substring)` tool that fetches and counts in Python:

```
fetch_text_from_url  →  count_lines_containing  →  answer
```

**`create_deep_agent` (deep)** — has `write_file` and `grep` built in, so we instruct it to use them. The agent saves the fetched document to a virtual file and runs `grep` against it — accurate and without needing a custom counting tool:

```
fetch_text_from_url  →  write_file("/gatsby.txt")  →  grep("Gatsby", "/gatsby.txt")  →  answer
```

This also highlights a key design principle: with `create_deep_agent` you wire in the *capability* (filesystem access), not a hand-rolled implementation of every operation.

## Demo Structure

### Part A — Side-by-side comparison

Both agents receive the identical research question:

> Fetch *The Great Gatsby* from Project Gutenberg and answer:
> 1. How many distinct lines contain "Gatsby"?
> 2. Line number of the first line containing "Daisy"?
> 3. A two-sentence synopsis.

Run both, print both answers, and observe how each agent reasons through the task.

### Part B — Multi-turn memory (`create_deep_agent`)

After Part A, a follow-up question is sent to the deep agent **on the same `thread_id`**:

> "Who is the narrator of the novel, and how does he know Gatsby?"

Because the `InMemorySaver` checkpointer stores conversation history keyed by `thread_id`, the agent already has the full document text in its context window. It answers instantly — no re-fetch.

```python
# Both invocations share the same thread_id
config = {"configurable": {"thread_id": "gatsby-deep"}}

# Turn 1: agent fetches ~75 KB and answers research questions
agent.invoke({"messages": [{"role": "user", "content": RESEARCH_QUESTION}]}, config=config)

# Turn 2: document is still in context — agent answers without calling the tool again
agent.invoke({"messages": [{"role": "user", "content": FOLLOWUP_QUESTION}]}, config=config)
```

## Running the Example

```bash
uv run python -m examples.s_02_research_agent
```

Expected structure:

```
============================================================
PART A — Side-by-side comparison
Same question, same model, same tools. Only the agent differs.
============================================================

────────────────────────────────────────────────────────────
[create_agent (standard)]
Q: Project Gutenberg hosts a plain-text copy of ...
────────────────────────────────────────────────────────────
1) Lines containing "Gatsby": ...
2) First "Daisy" line: ...
3) Synopsis: ...

────────────────────────────────────────────────────────────
[create_deep_agent (deep)]
Q: Project Gutenberg hosts a plain-text copy of ...
────────────────────────────────────────────────────────────
1) Lines containing "Gatsby": ...
2) First "Daisy" line: ...
3) Synopsis: ...

============================================================
PART B — Multi-turn memory (create_deep_agent)
Follow-up question reuses the document already in context.
============================================================

────────────────────────────────────────────────────────────
[Turn 2 follow-up (no re-fetch)]
Q: Who is the narrator of the novel, and how does he know Gatsby?
────────────────────────────────────────────────────────────
The narrator is Nick Carraway ...
```

> **Note:** Part A fetches the document twice (once per agent), so expect ~20–60 seconds. Part B is fast.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Tool definition, both agents, comparison run, multi-turn follow-up |
| `__main__.py` | Thin wrapper so `python -m examples.s_02_research_agent` works |
