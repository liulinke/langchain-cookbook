"""
Example 01: LangGraph Quick Start
==================================
Build a ReAct Agent using the LangGraph Graph API.
The agent calls arithmetic tools (add, multiply, divide) to answer math questions.
Each invocation trace is uploaded to a local Langfuse server.

Source: https://langchain-ai.github.io/langgraph/tutorials/introduction/

Run:
    uv run python -m examples.s_01_llm_with_tools
"""
import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

from common.env import get_env  # noqa: F401  — ensures .env is loaded
from common.llm import create_llm
from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host

# ──────────────────────────────────────────────
# 1. Define tools
# ──────────────────────────────────────────────

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a by b."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


@tool
def divide(a: int, b: float) -> float:
    """Divide a by b."""
    return a / b


_tools = [add, multiply, divide]
_tools_by_name = {t.name: t for t in _tools}

# ──────────────────────────────────────────────
# 2. Define agent state
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    # operator.add appends new messages rather than overwriting the list
    messages: Annotated[list[AnyMessage], operator.add]


# ──────────────────────────────────────────────
# 3. Define node functions
# ──────────────────────────────────────────────

# Bind tools to the model once at module level to avoid repeated initialization
_model_with_tools = create_llm().bind_tools(_tools)

_SYSTEM_PROMPT = (
    "You are a math assistant that uses tools to perform arithmetic operations. "
    "Call one tool at a time and wait for the result before deciding the next step. "
    "Do not issue multiple tool calls simultaneously."
)


def llm_call(state: AgentState) -> dict:
    """Invoke the LLM and return its response message."""
    response = _model_with_tools.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls in the latest message and return ToolMessage results."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        chosen_tool = _tools_by_name[tool_call["name"]]
        observation = chosen_tool.invoke(tool_call["args"])
        # Include name so the LLM can match results to calls when parallel calls are used
        results.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        ))
    return {"messages": results}


# ──────────────────────────────────────────────
# 4. Define routing logic
# ──────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Route to tool_node if the latest message has tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# ──────────────────────────────────────────────
# 5. Build and compile the LangGraph
# ──────────────────────────────────────────────

def build_agent():
    """Construct and compile the StateGraph agent."""
    builder = StateGraph(AgentState)

    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")

    return builder.compile()


# ──────────────────────────────────────────────
# 6. Entry point
# ──────────────────────────────────────────────

def main():
    agent = build_agent()

    questions = [
        "What is 12345 multiplied by 17, then divided by 3?",
    ]

    for i, question in enumerate(questions):
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print("="*50)

        # Create a new handler per invocation so each gets its own Langfuse trace
        langfuse_handler = create_langfuse_handler()

        config = build_langfuse_config(
            langfuse_handler,
            session_id="s_01_llm_with_tools",
            user_id="demo-user",
            trace_name=f"Question {i+1}",
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        for msg in result["messages"]:
            msg.pretty_print()

    print(f"\nTraces uploaded to Langfuse: {get_langfuse_host()}")


if __name__ == "__main__":
    main()
