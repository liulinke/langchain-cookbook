"""
Example 02: Research Agent
===========================
A document research agent built with LangChain's create_agent API.
The agent fetches text from a URL and answers detailed questions about the content.

Source: https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents

Run:
    uv run python -m examples.s_02_research_agent
"""
import urllib.error
import urllib.request

from langchain.agents import create_agent
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

def build_agent():
    """Create a research agent with URL-fetching capability and in-memory conversation state."""
    return create_agent(
        model=create_llm(temperature=0.5, max_tokens=4096),
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

# ──────────────────────────────────────────────
# 4. Entry point
# ──────────────────────────────────────────────

RESEARCH_QUESTION = """\
Project Gutenberg hosts a plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Please answer:

1) How many lines in the file contain the substring "Gatsby"? (Count distinct lines, not occurrences.)
2) What is the 1-based line number of the first line containing "Daisy"?
3) A two-sentence neutral synopsis of the novel.

If you cannot verify an exact count using your tools, use null for that field and explain why."""


def main():
    agent = build_agent()

    print(f"\n{'='*60}")
    print("Research Agent — The Great Gatsby")
    print("="*60)
    print(f"\nQuestion:\n{RESEARCH_QUESTION}\n")

    langfuse_handler = create_langfuse_handler()
    config = build_langfuse_config(
        langfuse_handler,
        session_id="s_02_research_agent",
        user_id="demo-user",
        trace_name="Great Gatsby Research",
        # Thread ID is required by InMemorySaver for multi-turn conversation state
        extra_metadata={"configurable": {"thread_id": "great-gatsby"}},
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": RESEARCH_QUESTION}]},
        config=config,
    )

    answer = result["messages"][-1]
    # content_blocks is available on newer LangChain; fall back to .content
    output = getattr(answer, "content_blocks", None) or answer.content
    print(f"\nAnswer:\n{output}")
    print("\nTraces uploaded to Langfuse: http://localhost:3000")


if __name__ == "__main__":
    main()
