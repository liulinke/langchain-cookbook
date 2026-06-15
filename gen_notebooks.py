"""Generate Jupyter notebooks for all examples. Run once then delete."""
import nbformat as nbf
from pathlib import Path


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def save(path: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.cells = cells
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  written → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 · LLM with Tools (ReAct Agent)
# ─────────────────────────────────────────────────────────────────────────────
NB1 = [
    md("""
# Example 01 · LLM with Tools (ReAct Agent)

**Source:** [LangGraph Tutorial — Build a ReAct Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

This notebook builds a **ReAct (Reasoning + Acting) Agent** using the LangGraph Graph API.
The agent solves multi-step arithmetic problems by calling tools one at a time and reasoning about the results.

Run each cell from top to bottom — every cell builds on the previous one.
"""),

    md("""
## Key Concepts

### The ReAct Loop
ReAct agents alternate between *reasoning* (calling the LLM) and *acting* (running a tool):

```
START → llm_call ──(has tool calls?)──► tool_node ──► llm_call
                 └──(no tool calls)──► END
```

### LangGraph StateGraph
The agent is a directed graph with two nodes:
- **`llm_call`** — sends the message history to the LLM, appends its response
- **`tool_node`** — executes every tool call in the latest AI message, appends results

**State accumulation:** `AgentState.messages` uses `operator.add` as a reducer,
so each node *appends* rather than replaces. The LLM always sees the full history.
"""),

    code("""
import sys
from pathlib import Path

# Add project root to sys.path so the 'common' package is importable
_root = Path().resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

from common.env import get_env  # noqa: F401 — triggers .env loading
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host

print("✓ All imports loaded")
"""),

    md("""
## Step 1 · Define Tools

Tools are plain Python functions decorated with `@tool`.
LangChain reads the function's type annotations and docstring to generate the
JSON schema the LLM uses when deciding to call a tool.
"""),

    code("""
@tool
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply a by b.\"\"\"
    return a * b

@tool
def add(a: int, b: int) -> int:
    \"\"\"Add a and b.\"\"\"
    return a + b

@tool
def divide(a: int, b: float) -> float:
    \"\"\"Divide a by b.\"\"\"
    return a / b

# Name → tool mapping used by tool_node to dispatch calls
_tools = [add, multiply, divide]
_tools_by_name = {t.name: t for t in _tools}

print(f"✓ {len(_tools)} tools defined: {[t.name for t in _tools]}")
"""),

    md("""
## Step 2 · Build the Agent Graph

Four parts assembled in a single cell for clarity:
1. **`AgentState`** — TypedDict holding the full message history
2. **Node functions** — `llm_call` and `tool_node`
3. **Routing function** — `should_continue` decides the next node
4. **`StateGraph`** — wires everything together and compiles the runnable
"""),

    code("""
# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # operator.add means each node's output is appended, not replaced
    messages: Annotated[list[AnyMessage], operator.add]

# ── Model bound to tools (done once to avoid re-initialising on every call) ──
_model = create_llm().bind_tools(_tools)

_SYSTEM_PROMPT = (
    "You are a math assistant that uses tools for arithmetic. "
    "Call one tool at a time and wait for the result before deciding the next step."
)

# ── Nodes ─────────────────────────────────────────────────────────────────────
def llm_call(state: AgentState) -> dict:
    \"\"\"Invoke the LLM with the full message history.\"\"\"
    response = _model.invoke([SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    \"\"\"Execute every tool call in the latest AI message.\"\"\"
    results = []
    for tc in state["messages"][-1].tool_calls:
        observation = _tools_by_name[tc["name"]].invoke(tc["args"])
        # Attach name so the LLM can match results to calls during parallel execution
        results.append(ToolMessage(content=str(observation), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": results}

# ── Routing ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    \"\"\"If the latest message has pending tool calls, run them. Otherwise end.\"\"\"
    return "tool_node" if state["messages"][-1].tool_calls else END

# ── Graph ─────────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("llm_call", llm_call)
    g.add_node("tool_node", tool_node)
    g.add_edge(START, "llm_call")
    g.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    g.add_edge("tool_node", "llm_call")
    return g.compile()

print("✓ Agent graph defined — ready to compile")
"""),

    md("""
## Step 3 · Run the Agent

The agent will reason through the problem step by step:
1. LLM → call `multiply(12345, 17)` → result `209865`
2. LLM → call `divide(209865, 3)` → result `69955.0`
3. LLM → no more tool calls → returns final answer

Each `.pretty_print()` below shows one message in the chain.
"""),

    code("""
agent = build_agent()
question = "What is 12345 multiplied by 17, then divided by 3?"

print(f"Question: {question}")
print("=" * 60)

handler = create_langfuse_handler()
config = build_langfuse_config(
    handler,
    session_id="s_01_notebook",
    user_id="demo-user",
    trace_name="Notebook: Math Question",
)
config["configurable"] = {"thread_id": "nb-s01"}

result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

for msg in result["messages"]:
    msg.pretty_print()

print(f"\\nTrace: {get_langfuse_host()}")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 · Research Agent — create_agent vs create_deep_agent
# ─────────────────────────────────────────────────────────────────────────────
NB2 = [
    md("""
# Example 02 · Research Agent — `create_agent` vs `create_deep_agent`

**Source:** [LangChain Quickstart — LangChain Agents](https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents)

This notebook runs the **same research task** on two agent factories and compares outputs.
It then demonstrates **multi-turn memory**: the deep agent answers a follow-up question
without re-fetching the document.
"""),

    md("""
## `create_agent` vs `create_deep_agent`

Both accept the same arguments (`model`, `tools`, `system_prompt`, `checkpointer`)
but differ in what they include automatically:

| Feature | `create_agent` | `create_deep_agent` |
|---------|---------------|---------------------|
| ReAct loop | ✓ | ✓ |
| Built-in planning | — | ✓ |
| Sub-agent orchestration | — | ✓ |
| File-system tools (`write_file`, `grep`, …) | — | ✓ (FilesystemMiddleware) |

### Counting strategy differences

- **`create_agent`**: no file tools → LLM must count lines from in-context text → unreliable
- **`create_deep_agent`**: has `write_file` + `grep` built in → fetch → save → grep → accurate
"""),

    code("""
import sys, urllib.error, urllib.request
from pathlib import Path

_root = Path().resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common.env import get_env  # noqa: F401
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host

# ── Shared fetch tool ──────────────────────────────────────────────────────────
@tool
def fetch_text_from_url(url: str) -> str:
    \"\"\"Fetch the full text of a document from a URL (first 50 000 chars).\"\"\"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    # Truncate to avoid hitting model context limits
    return text[:50_000]

# Separate checkpointers — memories must never cross-contaminate
_ckpt_standard = InMemorySaver()
_ckpt_deep     = InMemorySaver()

handler = create_langfuse_handler()

def make_config(trace_name: str, thread_id: str) -> dict:
    cfg = build_langfuse_config(handler, session_id="s02-nb", trace_name=trace_name)
    cfg["configurable"] = {"thread_id": thread_id}
    return cfg

print("✓ Setup complete")
"""),

    code("""
# ── System prompts — each agent is told its counting strategy ─────────────────
PROMPT_STANDARD = \"\"\"You are a literary data assistant.
Tools: fetch_text_from_url.
Answer line-count questions as best you can from the fetched text.\"\"\"

# Deep agent has write_file and grep built-in via FilesystemMiddleware
PROMPT_DEEP = \"\"\"You are a literary data assistant.
Tools: fetch_text_from_url, write_file (built-in), grep (built-in).

For line-count questions:
1. Call fetch_text_from_url to download the document.
2. Call write_file to save it locally, e.g. /gatsby.txt.
3. Call grep to count matching lines. Do NOT count manually.\"\"\"

# ── Research question (identical for both agents) ─────────────────────────────
RESEARCH_QUESTION = \"\"\"\\
Project Gutenberg hosts a plain-text copy of The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Please fetch and answer:
1) How many distinct lines contain the substring "Gatsby"?
2) What is the 1-based line number of the first line containing "Daisy"?
3) A two-sentence neutral synopsis.\"\"\"

FOLLOWUP = "Who is the narrator of the novel, and how does he know Gatsby?"

print("✓ Prompts and questions defined")
"""),

    md("""
## Part A · Standard Agent (`create_agent`)

Only has `fetch_text_from_url`. The LLM must try to count from in-context text — which LLMs do poorly on large documents. Expect imprecise or hallucinated line counts.
"""),

    code("""
standard_agent = create_agent(
    model=create_llm(temperature=0.5, max_tokens=4096),
    tools=[fetch_text_from_url],
    system_prompt=PROMPT_STANDARD,
    checkpointer=_ckpt_standard,
)

print("[create_agent] Running research question …")
print("=" * 60)

std_result = standard_agent.invoke(
    {"messages": [{"role": "user", "content": RESEARCH_QUESTION}]},
    config=make_config("Std Agent — Turn 1", "gatsby-std"),
)
print(std_result["messages"][-1].content)
"""),

    md("""
## Part A · Deep Agent (`create_deep_agent`)

Has `write_file` and `grep` from `FilesystemMiddleware` (included automatically).
After fetching, it saves the document to a virtual file and uses `grep` for accurate counting.
"""),

    code("""
deep_agent = create_deep_agent(
    model=create_llm(temperature=0.5, max_tokens=4096),
    tools=[fetch_text_from_url],
    system_prompt=PROMPT_DEEP,
    checkpointer=_ckpt_deep,
)

print("[create_deep_agent] Running same research question …")
print("=" * 60)

deep_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": RESEARCH_QUESTION}]},
    config=make_config("Deep Agent — Turn 1", "gatsby-deep"),
)
print(deep_result["messages"][-1].content)
"""),

    md("""
## Part B · Multi-Turn Memory

The deep agent already has the document in its context (stored under `thread_id="gatsby-deep"`).
A follow-up question is answered instantly — no re-fetch.

This is the value of `InMemorySaver`: state persists across `.invoke()` calls as long as
the same `thread_id` is reused.
"""),

    code("""
print(f"[Turn 2 — follow-up, no re-fetch] {FOLLOWUP}")
print("=" * 60)

followup_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": FOLLOWUP}]},
    config=make_config("Deep Agent — Turn 2", "gatsby-deep"),
)
print(followup_result["messages"][-1].content)

print(f"\\nTraces: {get_langfuse_host()}")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 3 · Agent Harness Patterns
# ─────────────────────────────────────────────────────────────────────────────
NB3 = [
    md("""
# Example 03 · Agent Harness Patterns

**Source:** [LangChain Agents — The Harness](https://docs.langchain.com/oss/python/langchain/agents)

## Core Idea: Agent = Model + Harness

The **model** reasons. The **harness** is everything surrounding it — system prompt, tools,
structured output schema, per-run context, checkpointer, and middleware.

This notebook demonstrates three harness features that go beyond a plain ReAct loop:

| Demo | Feature | Key API |
|------|---------|---------|
| A | **Streaming** — watch tool calls arrive in real time | `agent.stream(..., stream_mode="values")` |
| B | **Structured output** — agent returns a validated Pydantic object | `response_format=BookReport` |
| C | **Context schema** — per-run user data injected into tools | `context_schema=ReaderContext` + `ToolRuntime` |
"""),

    code("""
import sys
from dataclasses import dataclass
from pathlib import Path

_root = Path().resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from common.env import get_env  # noqa: F401
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host

# ── Mock book database ────────────────────────────────────────────────────────
_BOOKS = {
    "the great gatsby": {
        "author": "F. Scott Fitzgerald (1925)", "genre": "Literary fiction",
        "themes": ["The American Dream", "class and wealth", "obsession"],
        "plot": "Nick Carraway befriends millionaire Jay Gatsby, who throws lavish parties hoping to win back his lost love Daisy.",
    },
    "1984": {
        "author": "George Orwell (1949)", "genre": "Dystopian fiction",
        "themes": ["totalitarianism", "surveillance", "propaganda"],
        "plot": "Winston Smith secretly rebels against the all-seeing Party in a totalitarian society ruled by Big Brother.",
    },
    "dune": {
        "author": "Frank Herbert (1965)", "genre": "Science fiction",
        "themes": ["ecology", "religion", "politics", "power"],
        "plot": "Paul Atreides is sent to govern the desert planet Arrakis and is ultimately betrayed, beginning a journey toward destiny.",
    },
}

# ── Context schema ─────────────────────────────────────────────────────────────
@dataclass
class ReaderContext:
    \"\"\"Per-run user data. Passed via context= in agent.invoke(); reaches tools via ToolRuntime.\"\"\"
    reader_name: str
    preferred_style: str  # "bullet points" | "casual" | "academic"

# ── Tool that accesses context via ToolRuntime ────────────────────────────────
@tool
def lookup_book(title: str, runtime: ToolRuntime[ReaderContext]) -> str:
    \"\"\"Look up a book and return content formatted for the current reader's preferred style.\"\"\"
    # runtime.context is the ReaderContext passed via context= — it arrives here
    # through the harness without appearing in the LLM's message history.
    reader = runtime.context
    if reader:
        print(f"  [tool] context → name={reader.reader_name!r}, style={reader.preferred_style!r}")

    key = title.lower().strip()
    book = _BOOKS.get(key)
    if not book:
        return f"Not found. Available: {', '.join(t.title() for t in _BOOKS)}"

    # Format output based on reader style so the LLM produces visibly different responses
    if reader and reader.preferred_style == "bullet points":
        return (
            f"For {reader.reader_name} (bullet-point format):\\n"
            f"• Title: {title.title()}\\n"
            f"• Author: {book['author']}\\n"
            f"• Genre: {book['genre']}\\n"
            f"• Themes: {', '.join(book['themes'])}\\n"
            f"• Plot: {book['plot']}"
        )
    name  = reader.reader_name if reader else "reader"
    style = f" ({reader.preferred_style} style)" if reader else ""
    return (
        f"For {name}{style}:\\n"
        f"Title: {title.title()} by {book['author']}\\n"
        f"Genre: {book['genre']}\\n"
        f"Themes: {', '.join(book['themes'])}\\n"
        f"Plot: {book['plot']}"
    )

handler = create_langfuse_handler()

def new_config(trace_name: str, thread_id: str = None) -> dict:
    cfg = build_langfuse_config(handler, session_id="s03-nb", trace_name=trace_name)
    cfg["configurable"] = {"thread_id": thread_id or str(uuid7())}
    return cfg

print("✓ Setup, book DB, and tools ready")
"""),

    md("""
## Demo A · Streaming

`agent.stream()` with `stream_mode="values"` yields the **full agent state** after every
node execution. Inspect `chunk["messages"][-1]` at each step to see tool calls and replies
as they arrive — no waiting for the final answer.
"""),

    code("""
streaming_agent = create_agent(
    model=create_llm(),
    tools=[lookup_book],
    system_prompt="You are a knowledgeable book assistant. Use lookup_book to answer questions.",
    checkpointer=InMemorySaver(),
)

question_a = "Tell me about '1984' by George Orwell."
print(f"Q: {question_a}\\n")

for chunk in streaming_agent.stream(
    {"messages": [{"role": "user", "content": question_a}]},
    config=new_config("Demo A — Streaming"),
    stream_mode="values",
):
    latest = chunk["messages"][-1]
    if isinstance(latest, HumanMessage):
        pass  # skip echoing the user's own message
    elif isinstance(latest, AIMessage) and latest.tool_calls:
        names = [tc["name"] for tc in latest.tool_calls]
        print(f"  [tool call] → {', '.join(names)}")
    elif isinstance(latest, AIMessage) and latest.content:
        print(f"  [reply]\\n{latest.content}")
"""),

    md("""
## Demo B · Structured Output (`response_format`)

Pass a Pydantic model as `response_format` to receive a **validated object** instead of free text.
The agent fills every field by combining tool results with its own reasoning.
The result lives under `result["structured_response"]` — guaranteed schema, no text parsing needed.
"""),

    code("""
class BookReport(BaseModel):
    \"\"\"Structured book report — every field populated by the agent.\"\"\"
    title: str           = Field(description="Book title")
    author: str          = Field(description="Author name and publication year")
    one_line_summary: str = Field(description="One sentence capturing the book's essence")
    themes: list[str]   = Field(description="Up to 3 main themes")
    recommended_for: str = Field(description="Who would enjoy this book most")
    rating: float       = Field(description="Your rating from 1.0 to 5.0", ge=1.0, le=5.0)

structured_agent = create_agent(
    model=create_llm(),
    tools=[lookup_book],
    system_prompt=(
        "You are a literary critic. "
        "Use lookup_book to research the book, then fill every field of the report accurately."
    ),
    response_format=BookReport,
    checkpointer=InMemorySaver(),
)

question_b = "Write a structured report on 'Dune'."
print(f"Q: {question_b}\\n")

result_b = structured_agent.invoke(
    {"messages": [{"role": "user", "content": question_b}]},
    config=new_config("Demo B — Structured Output"),
)

report: BookReport = result_b["structured_response"]
print(f"Title           : {report.title}")
print(f"Author          : {report.author}")
print(f"One-line summary: {report.one_line_summary}")
print(f"Themes          : {', '.join(report.themes)}")
print(f"Recommended for : {report.recommended_for}")
print(f"Rating          : {report.rating} / 5.0")
"""),

    md("""
## Demo C · Context Schema (`context_schema` + `ToolRuntime`)

`context_schema` declares the shape of per-run user data. The data is passed via `context=`
at invocation time and reaches tool functions through `ToolRuntime` — injected by the harness.

**Why this is different from just adding context to the system prompt:**
- The LLM never sees `reader_name` or `preferred_style` as message text
- Context flows directly into the tool function via dependency injection
- Sensitive values (user IDs, API keys, feature flags) stay out of the LLM conversation

In this demo the tool formats its output differently for each reader, so both the tool output
and the final LLM response are visibly different — proving the context arrived in the tool.
"""),

    code("""
context_agent = create_agent(
    model=create_llm(),
    tools=[lookup_book],
    system_prompt=(
        "You are a personal book assistant. "
        "Use lookup_book to fetch book info, then summarize following the format in the tool output."
    ),
    context_schema=ReaderContext,
    checkpointer=InMemorySaver(),
)

question_c = "Summarize 'The Great Gatsby' for me."

readers = [
    ReaderContext(reader_name="Alice", preferred_style="bullet points"),
    ReaderContext(reader_name="Bob",   preferred_style="casual and conversational"),
]

for reader in readers:
    print(f"\\n{'─'*50}")
    print(f"Reader: {reader.reader_name}  |  Style: {reader.preferred_style}")
    print("─" * 50)
    result_c = context_agent.invoke(
        {"messages": [{"role": "user", "content": question_c}]},
        config=new_config(f"Demo C — Context ({reader.reader_name})"),
        context=reader,
    )
    print(result_c["messages"][-1].content)

print(f"\\nTraces: {get_langfuse_host()}")
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────────
print("Generating notebooks …")
save("examples/s_01_llm_with_tools/notebook.ipynb", NB1)
save("examples/s_02_research_agent/notebook.ipynb",  NB2)
save("examples/s_03_agent_harness/notebook.ipynb",   NB3)
print("Done.")
