"""
Example 02: Research Agent — create_agent vs create_deep_agent
==============================================================
Runs the same research task with two different agent factories side by side
so you can compare their outputs and reasoning depth:

  create_agent      — standard ReAct loop: LLM + your tools, nothing more.
                      Only has fetch_text_from_url — no way to count lines
                      accurately, so line-count questions will fail or hallucinate.

  create_deep_agent — same interface, but adds built-in planning, sub-agent
                      orchestration, and file-system tools (ls, read_file,
                      write_file, grep, …) automatically.
                      Counting strategy: fetch → write_file → grep (built-in).

After the comparison, the demo continues with create_deep_agent to show
multi-turn memory: turn 2 is a follow-up question that reuses the document
already loaded in turn 1 — no re-fetch needed.

Source: https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents

Run:
    uv run python -m examples.s_02_research_agent
"""
import urllib.error
import urllib.request

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common.env import get_env  # noqa: F401  — ensures .env is loaded
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host

# ──────────────────────────────────────────────
# 1. System prompts
# Each agent gets its own prompt explaining the counting strategy it should use.
# ──────────────────────────────────────────────

# Standard agent only has fetch_text_from_url — it has no file tools and no grep.
# It must try to count lines from in-context text, which LLMs do poorly.
# This limitation is the key contrast with create_deep_agent.
SYSTEM_PROMPT_STANDARD = """You are a literary data assistant.

## Tools

- `fetch_text_from_url`: loads document text from a URL into the conversation.

Answer line-count questions as best you can using the fetched text."""

# Deep agent has built-in write_file and grep (from FilesystemMiddleware).
# Counting strategy: fetch the document → save it with write_file → grep for the pattern.
# This is more accurate than in-context counting by LLM sub-agents.
SYSTEM_PROMPT_DEEP = """You are a literary data assistant.

## Tools

- `fetch_text_from_url`: fetches the full text of a document from a URL.
- Built-in file tools: `write_file`, `grep`, `read_file`, etc.

## How to answer line-count questions

1. Call `fetch_text_from_url` to download the document.
2. Call `write_file` to save the content to a local file, e.g. `/gatsby.txt`.
3. Call `grep` with the pattern and the file path to count matching lines.
   Use the grep result for your answer — do NOT count manually."""

# ──────────────────────────────────────────────
# 2. Define tools
# ──────────────────────────────────────────────

def _fetch_raw(url: str) -> str:
    """Download text from a URL (shared helper)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the full text of a document from a URL.

    Returns the first 50 000 characters to stay within model context limits.
    """
    try:
        text = _fetch_raw(url)
    except Exception as e:
        return f"Fetch failed: {e}"
    # Truncate to avoid hitting the model's context length limit;
    # 50 000 chars covers the bulk of most novels while staying within token budgets.
    return text[:50_000]


# ──────────────────────────────────────────────
# 3. Build agents
# ──────────────────────────────────────────────

# Separate checkpointers so the two agents' memories never cross-contaminate
_checkpointer_standard = InMemorySaver()
_checkpointer_deep = InMemorySaver()


def build_standard_agent():
    """Plain ReAct agent — only fetch_text_from_url, no file tools, no grep."""
    return create_agent(
        model=create_llm(temperature=0.5, max_tokens=4096),
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT_STANDARD,
        checkpointer=_checkpointer_standard,
    )


def build_deep_agent():
    """Deep agent that uses its built-in write_file + grep for line counting."""
    return create_deep_agent(
        model=create_llm(temperature=0.5, max_tokens=4096),
        # Only fetch_text_from_url is needed — write_file and grep come from
        # FilesystemMiddleware which create_deep_agent includes automatically.
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT_DEEP,
        checkpointer=_checkpointer_deep,
    )

# ──────────────────────────────────────────────
# 4. Research questions
# ──────────────────────────────────────────────

RESEARCH_QUESTION = """\
Project Gutenberg hosts a plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Please fetch the document and answer:
1) How many lines contain the substring "Gatsby"? (Count distinct lines, not occurrences.)
2) What is the 1-based line number of the first line containing "Daisy"?
3) A two-sentence neutral synopsis of the novel."""

# The follow-up reuses what create_deep_agent already loaded in turn 1 —
# no re-fetch, instant answer from memory.
FOLLOWUP_QUESTION = "Who is the narrator of the novel, and how does he know Gatsby?"

# ──────────────────────────────────────────────
# 5. Helpers
# ──────────────────────────────────────────────

def invoke_and_print(agent, label: str, message: str, config: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"[{label}]")
    print(f"Q: {message[:120]}{'...' if len(message) > 120 else ''}")
    print("─"*60)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )

    answer = result["messages"][-1]
    # content_blocks is the newer API; fall back to .content for compatibility
    output = getattr(answer, "content_blocks", None) or answer.content
    print(f"\n{output}")

# ──────────────────────────────────────────────
# 6. Entry point
# ──────────────────────────────────────────────

def main():
    standard_agent = build_standard_agent()
    deep_agent = build_deep_agent()

    langfuse_handler = create_langfuse_handler()

    standard_config = build_langfuse_config(
        langfuse_handler,
        session_id="s_02_research_agent",
        user_id="demo-user",
        trace_name="Standard Agent — Great Gatsby",
    )
    # thread_id must be a top-level "configurable" key for LangGraph's checkpointer —
    # it cannot live inside "metadata"
    standard_config["configurable"] = {"thread_id": "gatsby-standard"}

    deep_config = build_langfuse_config(
        langfuse_handler,
        session_id="s_02_research_agent",
        user_id="demo-user",
        trace_name="Deep Agent — Great Gatsby",
    )
    deep_config["configurable"] = {"thread_id": "gatsby-deep"}

    # ── Part A: side-by-side comparison ──────────────────
    print(f"\n{'='*60}")
    print("PART A — Side-by-side comparison")
    print("Same question, same model, same fetch tool. Only the agent differs.")
    print("  create_agent:      no file tools — LLM must count from context (unreliable)")
    print("  create_deep_agent: built-in write_file + grep — accurate")
    print("="*60)

    invoke_and_print(standard_agent, "create_agent (standard)", RESEARCH_QUESTION, standard_config)
    invoke_and_print(deep_agent,     "create_deep_agent (deep)", RESEARCH_QUESTION, deep_config)

    # ── Part B: multi-turn memory (deep agent only) ──────
    print(f"\n{'='*60}")
    print("PART B — Multi-turn memory (create_deep_agent)")
    print("Follow-up question reuses the document already in context.")
    print("The agent does NOT re-fetch the URL.")
    print("="*60)

    invoke_and_print(deep_agent, "Turn 2 follow-up (no re-fetch)", FOLLOWUP_QUESTION, deep_config)

    print(f"\n{'='*60}")
    print(f"Traces uploaded to Langfuse: {get_langfuse_host()}")


if __name__ == "__main__":
    main()
