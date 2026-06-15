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

## Try Examples in Jupyter Lab

Each example is a self-contained Jupyter notebook. Start Jupyter Lab with:

```bash
uv run jupyter lab
```

Then open one of the notebooks in the browser and run cells top to bottom:

- `examples/notebooks/s_01_llm_with_tools.ipynb`
- `examples/notebooks/s_02_research_agent.ipynb`
- `examples/notebooks/s_03_agent_harness.ipynb`

## Examples

- **01 · LLM with Tools (ReAct Agent)** — Build a ReAct Agent with LangGraph that calls arithmetic tools to solve math problems step by step.
  `examples/notebooks/s_01_llm_with_tools.ipynb`

- **02 · Research Agent** — A document research agent using `create_deep_agent` (from `deepagents`) that fetches a full novel and answers questions across two conversation turns, demonstrating persistent multi-turn memory via `thread_id`.
  `examples/notebooks/s_02_research_agent.ipynb`

- **03 · Agent Harness Patterns** — Demonstrates the `Agent = Model + Harness` concept with three harness features: streaming tool calls in real time, structured Pydantic output via `response_format`, and per-run user data via `context_schema`.
  `examples/notebooks/s_03_agent_harness.ipynb`

---

## Project Structure

```
langchain-cookbook/
├── common/
│   ├── env.py          # Load .env environment variables
│   ├── llm.py          # LLM factory (default: gpt-4o-mini)
│   └── tracing.py      # Langfuse tracing integration
├── examples/
│   └── notebooks/
│       ├── s_01_llm_with_tools.ipynb
│       ├── s_02_research_agent.ipynb
│       └── s_03_agent_harness.ipynb
├── main.py             # Lists all available examples
└── .env.example        # Template for environment variables
```

## Adding a New Example

1. Create a directory under `examples/`, e.g. `examples/s_04_rag/`
2. Add `notebook.ipynb` and optionally `main.py`; import shared utilities from `common/`
3. Register the example in the root `main.py` `EXAMPLES` list
