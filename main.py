"""
LangChain / LangGraph Cookbook
===============================
Each example lives in examples/<name>/ as a Jupyter notebook.
Open with: uv run jupyter lab
"""

EXAMPLES = [
    {
        "name": "s_01_llm_with_tools",
        "desc": "ReAct Agent with tool calling (add / multiply / divide)",
        "notebook": "examples/notebooks/s_01_llm_with_tools.ipynb",
    },
    {
        "name": "s_02_research_agent",
        "desc": "Research Agent — fetch and analyze documents from the web",
        "notebook": "examples/notebooks/s_02_research_agent.ipynb",
    },
    {
        "name": "s_03_agent_harness",
        "desc": "Agent Harness Patterns — streaming, structured output, context schema",
        "notebook": "examples/notebooks/s_03_agent_harness.ipynb",
    },
]


def main():
    print("LangChain / LangGraph Cookbook")
    print("Start Jupyter Lab:  uv run jupyter lab\n")
    for ex in EXAMPLES:
        print(f"  [{ex['name']}]  {ex['desc']}")
        print(f"    Notebook: {ex['notebook']}\n")


if __name__ == "__main__":
    main()
