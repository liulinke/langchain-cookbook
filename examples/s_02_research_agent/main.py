"""
Example 02: Research Agent (create_deep_agent + multi-turn memory)
===================================================================
A document research agent built with deepagents.create_deep_agent.
Deep agents come with planning, sub-agent, and file-system capabilities
built in — minimal setup, maximum capability.

This example also demonstrates multi-turn conversation: the agent fetches
and analyzes a document in turn 1, then answers a follow-up in turn 2
using the same thread_id — no need to re-fetch the document.

Source: https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents

Run:
    uv run python -m examples.s_02_research_agent
"""
import urllib.error
import urllib.request

from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common.env import get_env  # noqa: F401  — ensures .env is loaded
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config

# ──────────────────────────────────────────────
# 1. System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions — ground them in tool results from the fetched file."""

# ──────────────────────────────────────────────
# 2. Define tools
# ──────────────────────────────────────────────

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the full text of a document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    return raw.decode("utf-8", errors="replace")

# ──────────────────────────────────────────────
# 3. Build agent
# ──────────────────────────────────────────────

# A single InMemorySaver shared across all invocations — this is what
# makes multi-turn memory work. Each thread_id gets its own conversation history.
_checkpointer = InMemorySaver()


def build_agent():
    """Create a deep research agent with URL-fetching capability and persistent memory."""
    return create_deep_agent(
        model=create_llm(temperature=0.5, max_tokens=4096),
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )

# ──────────────────────────────────────────────
# 4. Conversation turns
# ──────────────────────────────────────────────

# Turn 1: ask the agent to fetch the document and answer initial questions
TURN_1 = """\
Project Gutenberg hosts a plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Please fetch the document and answer:
1) How many lines contain the substring "Gatsby"? (Count distinct lines, not occurrences.)
2) What is the 1-based line number of the first line containing "Daisy"?
3) A two-sentence neutral synopsis of the novel."""

# Turn 2: follow-up — the agent already has the full document in context,
# so it answers instantly without re-fetching
TURN_2 = "Who is the narrator of the novel, and how does he know Gatsby?"


def invoke_and_print(agent, turn_label: str, message: str, config: dict) -> None:
    """Send one message to the agent and print the response."""
    print(f"\n{'─'*60}")
    print(f"[{turn_label}] {message}")
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
# 5. Entry point
# ──────────────────────────────────────────────

def main():
    agent = build_agent()

    print(f"\n{'='*60}")
    print("Research Agent (Deep Agent) — The Great Gatsby")
    print("="*60)

    langfuse_handler = create_langfuse_handler()

    # Both turns share the same thread_id so the agent accumulates memory
    # across calls — turn 2 has full context from turn 1 without re-fetching.
    config = build_langfuse_config(
        langfuse_handler,
        session_id="s_02_research_agent",
        user_id="demo-user",
        trace_name="Great Gatsby Research",
        extra_metadata={"configurable": {"thread_id": "great-gatsby"}},
    )

    invoke_and_print(agent, "Turn 1", TURN_1, config)
    invoke_and_print(agent, "Turn 2 (follow-up, no re-fetch needed)", TURN_2, config)

    print(f"\n{'='*60}")
    print("Traces uploaded to Langfuse: http://localhost:3000")


if __name__ == "__main__":
    main()
