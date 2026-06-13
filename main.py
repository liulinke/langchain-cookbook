"""
LangChain / LangGraph Cookbook
===============================
Each example is independently runnable and shares utilities from common/.

Available examples:
  s_01_llm_with_tools  —— ReAct Agent with tool calling (add / multiply / divide)
      uv run python -m examples.s_01_llm_with_tools

Adding a new example:
  1. Create a directory under examples/, e.g. examples/s_02_rag/
  2. Add __init__.py, __main__.py, and main.py
  3. Register it in the EXAMPLES list below
"""

EXAMPLES = [
    {
        "name": "s_01_llm_with_tools",
        "desc": "ReAct Agent with tool calling (add / multiply / divide)",
        "cmd": "uv run python -m examples.s_01_llm_with_tools",
    },
]


def main():
    print("LangChain / LangGraph Cookbook\n")
    for ex in EXAMPLES:
        print(f"  [{ex['name']}]  {ex['desc']}")
        print(f"    Run: {ex['cmd']}\n")


if __name__ == "__main__":
    main()
