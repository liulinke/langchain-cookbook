"""
Example 03: Agent Harness Patterns
====================================
Demonstrates the core concept from the LangChain Agents documentation:

    Agent = Model + Harness

The "harness" is everything surrounding the model — tools, system prompt,
structured output schema, per-run context, checkpointer, and middleware.
This example shows three harness features that go beyond a plain ReAct loop:

  Demo A — Streaming      : watch tool calls and replies arrive in real time
  Demo B — Structured output : agent returns a validated Pydantic object
  Demo C — Context schema  : pass per-run user data into the agent loop

Source: https://docs.langchain.com/oss/python/langchain/agents

Run:
    uv run python -m examples.s_03_agent_harness
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from common.env import get_env  # noqa: F401  — ensures .env is loaded
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host

# ──────────────────────────────────────────────
# 1. Context schema — defined first so the tool can reference it
# ──────────────────────────────────────────────

@dataclass
class ReaderContext:
    """Per-run user data injected into the agent at invocation time.

    The key idea: context reaches the tool via ToolRuntime, NOT via the LLM.
    The LLM never sees reader_name or preferred_style as message text —
    they flow through the harness directly into the tool function.
    """
    reader_name: str
    preferred_style: str  # e.g. "academic", "casual", "bullet points"

# ──────────────────────────────────────────────
# 2. Shared tool — a mock book database
# ──────────────────────────────────────────────

# In a real app this would call an API or database; here we return static data
# so the example runs without external dependencies.
_BOOK_DB = {
    "the great gatsby": {
        "author": "F. Scott Fitzgerald",
        "year": 1925,
        "genre": "Literary fiction",
        "themes": ["The American Dream", "class and wealth", "obsession", "the past"],
        "plot": (
            "Nick Carraway moves to Long Island and becomes neighbours with the "
            "mysterious millionaire Jay Gatsby, who throws lavish parties hoping to "
            "reunite with his lost love Daisy Buchanan."
        ),
    },
    "1984": {
        "author": "George Orwell",
        "year": 1949,
        "genre": "Dystopian fiction",
        "themes": ["totalitarianism", "surveillance", "propaganda", "freedom of thought"],
        "plot": (
            "Winston Smith lives in a totalitarian society ruled by Big Brother. "
            "He secretly rebels against the Party, falling in love with Julia and "
            "seeking contact with a rumoured underground resistance."
        ),
    },
    "dune": {
        "author": "Frank Herbert",
        "year": 1965,
        "genre": "Science fiction",
        "themes": ["ecology", "religion", "politics", "power"],
        "plot": (
            "Paul Atreides and his family are sent to govern the desert planet Arrakis, "
            "the only source of the most valuable substance in the universe. After a "
            "betrayal, Paul joins the native Fremen and begins a journey toward destiny."
        ),
    },
}


@tool
def lookup_book(title: str, runtime: ToolRuntime[ReaderContext]) -> str:
    """Look up facts about a book by title. Returns content formatted for the current reader."""
    # runtime.context is the ReaderContext passed via context= in agent.invoke().
    # The LLM never sees these values as message text — they arrive here directly
    # through the harness, bypassing the conversation entirely.
    reader = runtime.context
    if reader is not None:
        print(f"  [tool received context] reader={reader.reader_name!r}, style={reader.preferred_style!r}")

    key = title.lower().strip()
    book = _BOOK_DB.get(key)
    if not book:
        available = ", ".join(f'"{t.title()}"' for t in _BOOK_DB)
        return f'Book not found. Available titles: {available}'

    # Format the output differently based on the reader's preferred style so
    # the LLM receives pre-shaped content and produces visibly different responses.
    if reader and reader.preferred_style == "bullet points":
        return (
            f"Book info for {reader.reader_name} (bullet-point style):\n"
            f"• Title: {title.title()}\n"
            f"• Author: {book['author']} ({book['year']})\n"
            f"• Genre: {book['genre']}\n"
            f"• Key themes: {', '.join(book['themes'])}\n"
            f"• Plot: {book['plot']}"
        )
    else:
        style_note = f" (style: {reader.preferred_style})" if reader else ""
        return (
            f"Book info for {reader.reader_name if reader else 'reader'}{style_note}:\n"
            f"Title: {title.title()}\n"
            f"Author: {book['author']} ({book['year']})\n"
            f"Genre: {book['genre']}\n"
            f"Themes: {', '.join(book['themes'])}\n"
            f"Plot: {book['plot']}"
        )


# ──────────────────────────────────────────────
# 2. Structured output schema (Demo B)
# ──────────────────────────────────────────────

class BookReport(BaseModel):
    """Structured book report returned by the agent."""
    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    one_line_summary: str = Field(description="One sentence that captures the book's essence")
    themes: list[str] = Field(description="Up to 3 main themes")
    recommended_for: str = Field(description="Who would enjoy this book most")
    rating: float = Field(description="Rating from 1.0 to 5.0", ge=1.0, le=5.0)


# ──────────────────────────────────────────────
# 4. Helpers
# ──────────────────────────────────────────────

def new_config(langfuse_handler, trace_name: str) -> dict:
    """Build a LangChain config with a fresh thread_id for each demo."""
    cfg = build_langfuse_config(
        langfuse_handler,
        session_id="s_03_agent_harness",
        user_id="demo-user",
        trace_name=trace_name,
    )
    cfg["configurable"] = {"thread_id": str(uuid7())}
    return cfg


# ──────────────────────────────────────────────
# 5. Demo A — Streaming
# ──────────────────────────────────────────────

def demo_streaming(langfuse_handler):
    """Stream the agent's tool calls and replies as they arrive."""
    print(f"\n{'=' * 60}")
    print("DEMO A — Streaming")
    print("Watch tool calls and responses arrive in real time.")
    print("=" * 60)

    agent = create_agent(
        model=create_llm(),
        tools=[lookup_book],
        system_prompt="You are a knowledgeable book assistant. Use the lookup_book tool to answer questions.",
        checkpointer=InMemorySaver(),
    )

    question = "Tell me about the book '1984' by George Orwell."
    print(f"\nQ: {question}\n")

    config = new_config(langfuse_handler, "Demo A — Streaming")

    # stream_mode="values" yields the full state after every node execution,
    # so we can inspect the latest message at each step.
    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            stream_mode="values",
    ):
        latest = chunk["messages"][-1]
        if isinstance(latest, HumanMessage):
            pass  # skip echoing the user's own message
        elif isinstance(latest, AIMessage) and latest.tool_calls:
            names = [tc["name"] for tc in latest.tool_calls]
            print(f"  [tool call] → {', '.join(names)}")
        elif isinstance(latest, AIMessage) and latest.content:
            print(f"  [reply] {latest.content}")


# ──────────────────────────────────────────────
# 6. Demo B — Structured output
# ──────────────────────────────────────────────

def demo_structured_output(langfuse_handler):
    """Agent returns a validated Pydantic BookReport instead of free text."""
    print(f"\n{'=' * 60}")
    print("DEMO B — Structured Output (response_format=BookReport)")
    print("The agent returns a validated Pydantic object, not free text.")
    print("=" * 60)

    # response_format instructs the agent to populate a Pydantic model.
    # The result is available under result["structured_response"].
    agent = create_agent(
        model=create_llm(),
        tools=[lookup_book],
        system_prompt=(
            "You are a literary critic. Use the lookup_book tool to research the book, "
            "then fill in every field of the structured report accurately."
        ),
        response_format=BookReport,
        checkpointer=InMemorySaver(),
    )

    question = "Write a structured report on 'Dune'."
    print(f"\nQ: {question}\n")

    config = new_config(langfuse_handler, "Demo B — Structured Output")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    report: BookReport = result["structured_response"]
    print(f"  Title           : {report.title}")
    print(f"  Author          : {report.author}")
    print(f"  One-line summary: {report.one_line_summary}")
    print(f"  Themes          : {', '.join(report.themes)}")
    print(f"  Recommended for : {report.recommended_for}")
    print(f"  Rating          : {report.rating} / 5.0")


# ──────────────────────────────────────────────
# 7. Demo C — Context schema
# ──────────────────────────────────────────────

def demo_context(langfuse_handler):
    """Pass per-run user data to the agent without embedding it in the prompt."""
    print(f"\n{'=' * 60}")
    print("DEMO C — Context Schema (context_schema=ReaderContext)")
    print("Per-run user data (name, style preference) is injected at invocation")
    print("time without hardcoding it into the system prompt.")
    print("=" * 60)

    # context_schema declares the shape of per-run data.
    # The actual data is passed via context= in agent.invoke().
    # Tools and the system prompt can reference it through the harness.
    agent = create_agent(
        model=create_llm(),
        tools=[lookup_book],
        system_prompt=(
            "You are a personal book assistant. "
            "Use lookup_book to fetch the book info, then summarize it for the reader "
            "following the format and style already indicated in the tool's output."
        ),
        context_schema=ReaderContext,
        checkpointer=InMemorySaver(),
    )

    readers = [
        ReaderContext(reader_name="Alice", preferred_style="bullet points"),
        ReaderContext(reader_name="Bob", preferred_style="casual and conversational"),
    ]

    for reader in readers:
        print(f"\n  Reader: {reader.reader_name} | Style: {reader.preferred_style}")
        config = new_config(langfuse_handler, f"Demo C — Context ({reader.reader_name})")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Summarize 'The Great Gatsby' for me."}]},
            config=config,
            context=reader,
        )
        answer = result["messages"][-1].content
        print(f"  Response:\n{answer}\n")


# ──────────────────────────────────────────────
# 8. Entry point
# ──────────────────────────────────────────────

def main():
    print(f"\n{'=' * 60}")
    print("Example 03: Agent Harness Patterns")
    print("Agent = Model + Harness")
    print("=" * 60)

    langfuse_handler = create_langfuse_handler()

    demo_streaming(langfuse_handler)
    demo_structured_output(langfuse_handler)
    demo_context(langfuse_handler)

    print(f"\n{'=' * 60}")
    print(f"Traces uploaded to Langfuse: {get_langfuse_host()}")


if __name__ == "__main__":
    main()
