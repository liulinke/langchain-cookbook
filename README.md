# LangChain Cookbook

A collection of runnable examples covering [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), and Deep Agents patterns. All examples are sourced from the [LangChain official documentation](https://python.langchain.com/docs/).

> 中文版说明请见 [README_CN.md](README_CN.md)

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Quick Start

```bash
# Install dependencies (uv creates the virtual environment automatically)
uv sync
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-openai-key

# Langfuse tracing (optional, local service)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000
```

## List All Examples

```bash
uv run python main.py
```

## Examples

- **[01 · LLM with Tools (ReAct Agent)](examples/s_01_llm_with_tools/README.md)** — Build a ReAct Agent with LangGraph that calls arithmetic tools to solve math problems step by step.
  `uv run python -m examples.s_01_llm_with_tools`

- **[02 · Research Agent](examples/s_02_research_agent/README.md)** — A document research agent using `create_deep_agent` (from the `deepagents` package) that fetches a full novel and answers questions across two conversation turns, demonstrating persistent multi-turn memory via `thread_id`.
  `uv run python -m examples.s_02_research_agent`

- **[03 · Agent Harness Patterns](examples/s_03_agent_harness/README.md)** — Demonstrates the `Agent = Model + Harness` concept with three harness features: streaming tool calls in real time, structured Pydantic output via `response_format`, and per-run user data via `context_schema`.
  `uv run python -m examples.s_03_agent_harness`

---

## Project Structure

```
langchain-cookbook/
├── common/
│   ├── env.py          # Load .env environment variables
│   ├── llm.py          # LLM factory (default: gpt-4o-mini)
│   └── tracing.py      # Langfuse tracing integration
├── examples/
│   └── s_01_llm_with_tools/
│       ├── README.md       # Explanation of the example
│       ├── __main__.py     # Package entry point (python -m ...)
│       └── main.py         # Example logic
├── main.py             # Lists all available examples
└── .env.example        # Template for environment variables
```

## Adding a New Example

1. Create a directory under `examples/`, e.g. `examples/s_02_rag/`
2. Add `__init__.py`, `__main__.py`, and `main.py`; import shared utilities from `common/`
3. Add a `README.md` in the new directory explaining the example
4. Register the example in the root `main.py` `EXAMPLES` list
5. Run: `uv run python -m examples.s_02_rag`
