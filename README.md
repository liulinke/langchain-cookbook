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
# Model (default: gpt-4o-mini via OpenAI)
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-key

# To use Qwen via DashScope instead, comment out the above and uncomment:
# LLM_MODEL=qwen-plus
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# DASHSCOPE_API_KEY=your-dashscope-key

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
- `examples/notebooks/s_04_rag.ipynb`
- `examples/notebooks/s_05_human_in_the_loop.ipynb`
- `examples/notebooks/s_06_multi_agent_subagents.ipynb`

## Examples

- **01 · LLM with Tools (ReAct Agent)** — Build a ReAct Agent with LangGraph that calls arithmetic tools to solve math problems step by step.
  `examples/notebooks/s_01_llm_with_tools.ipynb`

- **02 · Research Agent** — A document research agent using `create_deep_agent` (from `deepagents`) that fetches a full novel and answers questions across two conversation turns, demonstrating persistent multi-turn memory via `thread_id`.
  `examples/notebooks/s_02_research_agent.ipynb`

- **03 · Agent Harness Patterns** — Demonstrates the `Agent = Model + Harness` concept with three harness features: streaming tool calls in real time, structured Pydantic output via `response_format`, and per-run user data via `context_schema`.
  `examples/notebooks/s_03_agent_harness.ipynb`

- **04 · Retrieval-Augmented Generation (RAG)** — Full RAG pipeline: load a local HTML document → split into chunks → embed with OpenAI → index in a vector store. Then two retrieval patterns: a RAG Agent (LLM decides when to retrieve) and a RAG Chain (always retrieves first). Includes source-document auditing and prompt-injection defense.
  `examples/notebooks/s_04_rag.ipynb`

- **05 · Human-in-the-Loop (HITL)** — Two patterns for human oversight of agent actions: LangGraph's native `interrupt()` for pausing and resuming graph execution, and `HumanInTheLoopMiddleware` for policy-based tool-call approvals. Covers all four decision types (approve / edit / reject / respond) and conditional interrupts that only fire for risky operations.
  `examples/notebooks/s_05_human_in_the_loop.ipynb`

- **06 · Multi-Agent: Subagents** — Supervisor architecture where a main agent coordinates specialized subagents as tools. Demonstrates four patterns: Tool-Per-Agent (one wrapper per subagent), Single Dispatch Tool (one `task()` routes to any agent), Enum-constrained type-safe discovery, and Command-based structured output that writes subagent results directly into supervisor state.
  `examples/notebooks/s_06_multi_agent_subagents.ipynb`

---

## Project Structure

```
langchain-cookbook/
├── common/
│   ├── env.py          # Load .env environment variables
│   ├── llm.py          # LLM factory (default: gpt-4o-mini)
│   └── tracing.py      # Langfuse tracing integration
├── examples/
│   ├── data/
│   │   └── lilian_weng_agent_post.html   # pre-downloaded source doc for s_04
│   └── notebooks/
│       ├── s_01_llm_with_tools.ipynb
│       ├── s_02_research_agent.ipynb
│       ├── s_03_agent_harness.ipynb
│       ├── s_04_rag.ipynb
│       ├── s_05_human_in_the_loop.ipynb
│       └── s_06_multi_agent_subagents.ipynb
├── main.py             # Lists all available examples
└── .env.example        # Template for environment variables
```

## Adding a New Example

1. Create a directory under `examples/`, e.g. `examples/s_04_rag/`
2. Add `notebook.ipynb` and optionally `main.py`; import shared utilities from `common/`
3. Register the example in the root `main.py` `EXAMPLES` list
