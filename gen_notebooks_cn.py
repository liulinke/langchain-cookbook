"""生成所有示例的中文 Jupyter Notebook。运行一次后可删除。"""
import nbformat as nbf
from pathlib import Path


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def save(path: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python (langchain-cookbook)",
        "language": "python",
        "name": "langchain-cookbook",
    }
    nb.cells = cells
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  已生成 → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 笔记本一：LLM 工具调用（ReAct 智能体）
# ─────────────────────────────────────────────────────────────────────────────
NB1 = [
    md("""
# 示例 01 · LLM 工具调用（ReAct 智能体）

**来源：** [LangGraph 教程 — 构建 ReAct 智能体](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

本笔记本使用 LangGraph Graph API 构建一个 **ReAct（推理 + 行动）智能体**。
智能体通过逐步调用工具来解决多步骤算术问题，并对每步结果进行推理。

请从上到下依次运行每个单元格——每个单元格都依赖上一个的执行结果。
"""),

    md("""
## 核心概念

### ReAct 循环
ReAct 智能体在"推理"（调用 LLM）和"行动"（执行工具）之间交替运行：

```
开始 → llm_call ──（有工具调用？）──► tool_node ──► llm_call
               └──（无工具调用）──► 结束
```

### LangGraph StateGraph（状态图）
智能体是一个包含两个节点的有向图：
- **`llm_call`** — 将完整消息历史发送给 LLM，追加其响应
- **`tool_node`** — 执行最新 AI 消息中的所有工具调用，追加结果

**状态累积机制：** `AgentState.messages` 使用 `operator.add` 作为归约器，
每个节点的输出会*追加*而非替换原有消息。LLM 始终能看到完整的对话历史。
"""),

    code(
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        '# 将项目根目录加入 sys.path，使 common 包可被导入\n'
        '_root = Path().resolve().parent.parent.parent\n'
        'if str(_root) not in sys.path:\n'
        '    sys.path.insert(0, str(_root))\n'
        '\n'
        'import operator\n'
        'from typing import Literal\n'
        'from typing_extensions import TypedDict, Annotated\n'
        '\n'
        'from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage\n'
        'from langchain.tools import tool\n'
        'from langgraph.graph import StateGraph, START, END\n'
        '\n'
        'from common.env import get_env  # noqa: F401 — 触发 .env 加载\n'
        'from common.llm import create_llm\n'
        'from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host\n'
        '\n'
        'print("✓ 所有依赖导入完成")\n'
    ),

    md("""
## 第一步 · 定义工具

工具是用 `@tool` 装饰器修饰的普通 Python 函数。
LangChain 会读取函数的类型注解和文档字符串，自动生成 LLM 决策调用时所需的 JSON Schema。
"""),

    code(
        '@tool\n'
        'def multiply(a: int, b: int) -> int:\n'
        '    """将 a 乘以 b。"""\n'
        '    return a * b\n'
        '\n'
        '@tool\n'
        'def add(a: int, b: int) -> int:\n'
        '    """将 a 与 b 相加。"""\n'
        '    return a + b\n'
        '\n'
        '@tool\n'
        'def divide(a: int, b: float) -> float:\n'
        '    """将 a 除以 b。"""\n'
        '    return a / b\n'
        '\n'
        '# 工具名称到工具对象的映射，供 tool_node 按名称分发调用\n'
        '_tools = [add, multiply, divide]\n'
        '_tools_by_name = {t.name: t for t in _tools}\n'
        '\n'
        'print(f"✓ 已定义 {len(_tools)} 个工具：{[t.name for t in _tools]}")\n'
    ),

    md("""
## 第二步 · 构建智能体图

以下单元格将四个部分组装在一起：
1. **`AgentState`** — 保存完整消息历史的 TypedDict
2. **节点函数** — `llm_call` 和 `tool_node`
3. **路由函数** — `should_continue` 决定下一个节点
4. **`StateGraph`** — 将所有部分连接并编译为可运行对象
"""),

    code(
        '# ── 状态定义 ───────────────────────────────────────────────────────────────\n'
        'class AgentState(TypedDict):\n'
        '    # operator.add 表示每个节点的输出追加到列表，而非替换\n'
        '    messages: Annotated[list[AnyMessage], operator.add]\n'
        '\n'
        '# ── 绑定工具的模型（只初始化一次，避免重复创建）─────────────────────────\n'
        '_model = create_llm().bind_tools(_tools)\n'
        '\n'
        '_SYSTEM_PROMPT = (\n'
        '    "You are a math assistant that uses tools for arithmetic. "\n'
        '    "Call one tool at a time and wait for the result before deciding the next step."\n'
        ')\n'
        '\n'
        '# ── 节点函数 ───────────────────────────────────────────────────────────────\n'
        'def llm_call(state: AgentState) -> dict:\n'
        '    """将完整消息历史传给 LLM 并获取响应。"""\n'
        '    response = _model.invoke([SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"])\n'
        '    return {"messages": [response]}\n'
        '\n'
        'def tool_node(state: AgentState) -> dict:\n'
        '    """执行最新 AI 消息中的所有工具调用。"""\n'
        '    results = []\n'
        '    for tc in state["messages"][-1].tool_calls:\n'
        '        observation = _tools_by_name[tc["name"]].invoke(tc["args"])\n'
        '        # 附带工具名称，便于 LLM 在并行调用时匹配结果\n'
        '        results.append(ToolMessage(content=str(observation), tool_call_id=tc["id"], name=tc["name"]))\n'
        '    return {"messages": results}\n'
        '\n'
        '# ── 路由函数 ───────────────────────────────────────────────────────────────\n'
        'def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:\n'
        '    """若最新消息包含待执行的工具调用则继续，否则结束。"""\n'
        '    return "tool_node" if state["messages"][-1].tool_calls else END\n'
        '\n'
        '# ── 图构建 ─────────────────────────────────────────────────────────────────\n'
        'def build_agent():\n'
        '    g = StateGraph(AgentState)\n'
        '    g.add_node("llm_call", llm_call)\n'
        '    g.add_node("tool_node", tool_node)\n'
        '    g.add_edge(START, "llm_call")\n'
        '    g.add_conditional_edges("llm_call", should_continue, ["tool_node", END])\n'
        '    g.add_edge("tool_node", "llm_call")\n'
        '    return g.compile()\n'
        '\n'
        'print("✓ 智能体图定义完成，准备编译")\n'
    ),

    md("""
## 第三步 · 运行智能体

智能体将逐步推理：
1. LLM → 调用 `multiply(12345, 17)` → 结果 `209865`
2. LLM → 调用 `divide(209865, 3)` → 结果 `69955.0`
3. LLM → 无更多工具调用 → 返回最终答案

每个 `.pretty_print()` 展示对话链中的一条消息。
"""),

    code(
        'agent = build_agent()\n'
        'question = "What is 12345 multiplied by 17, then divided by 3?"\n'
        '\n'
        'print(f"问题：{question}")\n'
        'print("=" * 60)\n'
        '\n'
        'handler = create_langfuse_handler()\n'
        'config = build_langfuse_config(\n'
        '    handler,\n'
        '    session_id="s_01_notebook_cn",\n'
        '    user_id="demo-user",\n'
        '    trace_name="笔记本：数学计算",\n'
        ')\n'
        'config["configurable"] = {"thread_id": "nb-s01-cn"}\n'
        '\n'
        'result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)\n'
        '\n'
        'for msg in result["messages"]:\n'
        '    msg.pretty_print()\n'
        '\n'
        'print(f"\\n追踪记录：{get_langfuse_host()}")\n'
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 笔记本二：研究智能体 — create_agent 与 create_deep_agent 对比
# ─────────────────────────────────────────────────────────────────────────────
NB2 = [
    md("""
# 示例 02 · 研究智能体 — `create_agent` 与 `create_deep_agent` 对比

**来源：** [LangChain 快速入门 — LangChain 智能体](https://docs.langchain.com/oss/python/langchain/quickstart#langchain-agents)

本笔记本在同一研究任务上运行两种智能体工厂并对比输出结果。
同时演示**多轮记忆**功能：深度智能体无需重新抓取文档，即可回答后续问题。
"""),

    md("""
## `create_agent` 与 `create_deep_agent` 对比

两者接受相同的参数（`model`、`tools`、`system_prompt`、`checkpointer`），
但内置能力不同：

| 功能 | `create_agent` | `create_deep_agent` |
|------|---------------|---------------------|
| ReAct 循环 | ✓ | ✓ |
| 内置规划能力 | — | ✓ |
| 子智能体编排 | — | ✓ |
| 文件系统工具（`write_file`、`grep` 等） | — | ✓（FilesystemMiddleware）|

### 行数统计策略的差异

- **`create_agent`**：无文件工具 → LLM 只能从上下文中手动统计 → 结果不可靠
- **`create_deep_agent`**：内置 `write_file` + `grep` → 抓取 → 保存 → grep 统计 → 精确
"""),

    code(
        'import sys, urllib.error, urllib.request\n'
        'from pathlib import Path\n'
        '\n'
        '# 将项目根目录加入 sys.path\n'
        '_root = Path().resolve().parent.parent.parent\n'
        'if str(_root) not in sys.path:\n'
        '    sys.path.insert(0, str(_root))\n'
        '\n'
        'from langchain.agents import create_agent\n'
        'from deepagents import create_deep_agent\n'
        'from langchain.tools import tool\n'
        'from langgraph.checkpoint.memory import InMemorySaver\n'
        '\n'
        'from common.env import get_env  # noqa: F401\n'
        'from common.llm import create_llm\n'
        'from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host\n'
        '\n'
        '# ── 共享抓取工具 ────────────────────────────────────────────────────────\n'
        '@tool\n'
        'def fetch_text_from_url(url: str) -> str:\n'
        '    """从 URL 抓取文档全文（最多 50000 个字符）。"""\n'
        '    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})\n'
        '    try:\n'
        '        with urllib.request.urlopen(req, timeout=120) as resp:\n'
        '            text = resp.read().decode("utf-8", errors="replace")\n'
        '    except urllib.error.URLError as e:\n'
        '        return f"抓取失败：{e}"\n'
        '    # 截断以避免超出模型上下文长度限制\n'
        '    return text[:50_000]\n'
        '\n'
        '# 使用独立的检查点器，避免两个智能体的记忆相互污染\n'
        '_ckpt_standard = InMemorySaver()\n'
        '_ckpt_deep     = InMemorySaver()\n'
        '\n'
        'handler = create_langfuse_handler()\n'
        '\n'
        'def make_config(trace_name: str, thread_id: str) -> dict:\n'
        '    cfg = build_langfuse_config(handler, session_id="s02-nb-cn", trace_name=trace_name)\n'
        '    cfg["configurable"] = {"thread_id": thread_id}\n'
        '    return cfg\n'
        '\n'
        'print("✓ 初始化完成")\n'
    ),

    code(
        '# ── 系统提示词（各智能体的统计策略不同）──────────────────────────────────\n'
        'PROMPT_STANDARD = (\n'
        '    "You are a literary data assistant.\\n"\n'
        '    "Tools: fetch_text_from_url.\\n"\n'
        '    "Answer line-count questions as best you can from the fetched text."\n'
        ')\n'
        '\n'
        '# 深度智能体通过 FilesystemMiddleware 内置了 write_file 和 grep\n'
        'PROMPT_DEEP = (\n'
        '    "You are a literary data assistant.\\n"\n'
        '    "Tools: fetch_text_from_url, write_file (built-in), grep (built-in).\\n\\n"\n'
        '    "For line-count questions:\\n"\n'
        '    "1. Call fetch_text_from_url to download the document.\\n"\n'
        '    "2. Call write_file to save it locally, e.g. /gatsby.txt.\\n"\n'
        '    "3. Call grep to count matching lines. Do NOT count manually."\n'
        ')\n'
        '\n'
        '# ── 研究问题（两个智能体使用完全相同的问题）────────────────────────────\n'
        'RESEARCH_QUESTION = (\n'
        '    "Project Gutenberg hosts a plain-text copy of The Great Gatsby.\\n"\n'
        '    "URL: https://www.gutenberg.org/files/64317/64317-0.txt\\n\\n"\n'
        '    "Please fetch and answer:\\n"\n'
        '    \'1) How many distinct lines contain the substring "Gatsby"?\\n\'\n'
        '    \'2) What is the 1-based line number of the first line containing "Daisy"?\\n\'\n'
        '    "3) A two-sentence neutral synopsis."\n'
        ')\n'
        '\n'
        '# 追问问题（用于演示多轮记忆）\n'
        'FOLLOWUP = "Who is the narrator of the novel, and how does he know Gatsby?"\n'
        '\n'
        'print("✓ 提示词和问题定义完成")\n'
    ),

    md("""
## 第 A 部分 · 标准智能体（`create_agent`）

仅有 `fetch_text_from_url` 工具。LLM 需要从上下文文本中手动统计行数——
大型文档的统计对 LLM 来说非常不准确，预计会出现错误或幻觉。
"""),

    code(
        'standard_agent = create_agent(\n'
        '    model=create_llm(temperature=0.5, max_tokens=4096),\n'
        '    tools=[fetch_text_from_url],\n'
        '    system_prompt=PROMPT_STANDARD,\n'
        '    checkpointer=_ckpt_standard,\n'
        ')\n'
        '\n'
        'print("[create_agent] 正在执行研究问题……")\n'
        'print("=" * 60)\n'
        '\n'
        'std_result = standard_agent.invoke(\n'
        '    {"messages": [{"role": "user", "content": RESEARCH_QUESTION}]},\n'
        '    config=make_config("标准智能体 — 第1轮", "gatsby-std-cn"),\n'
        ')\n'
        'print(std_result["messages"][-1].content)\n'
    ),

    md("""
## 第 A 部分 · 深度智能体（`create_deep_agent`）

通过 `FilesystemMiddleware` 自动内置了 `write_file` 和 `grep` 工具。
抓取文档后保存到虚拟文件，再用 `grep` 精确统计行数。
"""),

    code(
        'deep_agent = create_deep_agent(\n'
        '    model=create_llm(temperature=0.5, max_tokens=4096),\n'
        '    tools=[fetch_text_from_url],\n'
        '    system_prompt=PROMPT_DEEP,\n'
        '    checkpointer=_ckpt_deep,\n'
        ')\n'
        '\n'
        'print("[create_deep_agent] 执行相同的研究问题……")\n'
        'print("=" * 60)\n'
        '\n'
        'deep_result = deep_agent.invoke(\n'
        '    {"messages": [{"role": "user", "content": RESEARCH_QUESTION}]},\n'
        '    config=make_config("深度智能体 — 第1轮", "gatsby-deep-cn"),\n'
        ')\n'
        'print(deep_result["messages"][-1].content)\n'
    ),

    md("""
## 第 B 部分 · 多轮记忆

深度智能体已将文档保存在 `thread_id="gatsby-deep-cn"` 对应的上下文中。
追问问题将被立即回答——无需重新抓取文档。

这就是 `InMemorySaver` 的价值：只要复用相同的 `thread_id`，
状态就会在多次 `.invoke()` 调用之间持久保存。
"""),

    code(
        'print(f"[第 2 轮 — 追问，无需重新抓取] {FOLLOWUP}")\n'
        'print("=" * 60)\n'
        '\n'
        'followup_result = deep_agent.invoke(\n'
        '    {"messages": [{"role": "user", "content": FOLLOWUP}]},\n'
        '    config=make_config("深度智能体 — 第2轮", "gatsby-deep-cn"),\n'
        ')\n'
        'print(followup_result["messages"][-1].content)\n'
        '\n'
        'print(f"\\n追踪记录：{get_langfuse_host()}")\n'
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 笔记本三：智能体 Harness 模式
# ─────────────────────────────────────────────────────────────────────────────
NB3 = [
    md("""
# 示例 03 · 智能体 Harness 模式

**来源：** [LangChain 智能体 — Harness 概念](https://docs.langchain.com/oss/python/langchain/agents)

## 核心思想：智能体 = 模型 + Harness

**模型**负责推理。**Harness** 是围绕模型的一切——系统提示、工具、
结构化输出 Schema、每次运行的上下文、检查点器和中间件。

本笔记本演示三种超越普通 ReAct 循环的 Harness 功能：

| 演示 | 功能 | 核心 API |
|------|------|---------|
| A | **流式输出** — 实时观察工具调用过程 | `agent.stream(..., stream_mode="values")` |
| B | **结构化输出** — 智能体返回经过验证的 Pydantic 对象 | `response_format=BookReport` |
| C | **上下文 Schema** — 每次运行的用户数据注入工具 | `context_schema=ReaderContext` + `ToolRuntime` |
"""),

    code(
        'import sys\n'
        'from dataclasses import dataclass\n'
        'from pathlib import Path\n'
        '\n'
        '# 将项目根目录加入 sys.path\n'
        '_root = Path().resolve().parent.parent.parent\n'
        'if str(_root) not in sys.path:\n'
        '    sys.path.insert(0, str(_root))\n'
        '\n'
        'from langchain.agents import create_agent\n'
        'from langchain.tools import tool, ToolRuntime\n'
        'from langchain_core.messages import AIMessage, HumanMessage\n'
        'from langchain_core.utils.uuid import uuid7\n'
        'from langgraph.checkpoint.memory import InMemorySaver\n'
        'from pydantic import BaseModel, Field\n'
        '\n'
        'from common.env import get_env  # noqa: F401\n'
        'from common.llm import create_llm\n'
        'from common.tracing import create_langfuse_handler, build_langfuse_config, get_langfuse_host\n'
        '\n'
        '# ── 模拟图书数据库 ────────────────────────────────────────────────────\n'
        '_BOOKS = {\n'
        '    "the great gatsby": {\n'
        '        "author": "F. Scott Fitzgerald (1925)", "genre": "文学小说",\n'
        '        "themes": ["美国梦", "阶级与财富", "执念"],\n'
        '        "plot": "尼克·卡拉韦与百万富翁杰伊·盖茨比结为朋友，盖茨比举办奢华派对，只为赢回昔日恋人黛西。",\n'
        '    },\n'
        '    "1984": {\n'
        '        "author": "George Orwell (1949)", "genre": "反乌托邦小说",\n'
        '        "themes": ["极权主义", "监控", "宣传"],\n'
        '        "plot": "温斯顿·史密斯在大哥监视一切的极权社会中秘密反抗党的统治。",\n'
        '    },\n'
        '    "dune": {\n'
        '        "author": "Frank Herbert (1965)", "genre": "科幻小说",\n'
        '        "themes": ["生态", "宗教", "政治", "权力"],\n'
        '        "plot": "保罗·厄崔迪被派往沙漠星球厄拉科斯执政，遭遇背叛后踏上命运之旅。",\n'
        '    },\n'
        '}\n'
        '\n'
        '# ── 上下文 Schema 定义 ────────────────────────────────────────────────\n'
        '@dataclass\n'
        'class ReaderContext:\n'
        '    """每次运行的用户数据。通过 context= 传入智能体，经由 ToolRuntime 注入工具函数。"""\n'
        '    reader_name: str\n'
        '    preferred_style: str  # "bullet points"（要点列表）| "casual"（随意）| "academic"（学术）\n'
        '\n'
        '# ── 通过 ToolRuntime 获取上下文的工具 ──────────────────────────────────\n'
        '@tool\n'
        'def lookup_book(title: str, runtime: ToolRuntime[ReaderContext]) -> str:\n'
        '    """按书名查询图书信息，并按当前读者的偏好格式化返回内容。"""\n'
        '    # runtime.context 是通过 agent.invoke() 的 context= 参数传入的 ReaderContext\n'
        '    # 它经由 Harness 直接注入工具函数，不会出现在 LLM 的消息历史中\n'
        '    reader = runtime.context\n'
        '    if reader:\n'
        '        print(f"  [工具] 收到上下文 → 读者={reader.reader_name!r}, 风格={reader.preferred_style!r}")\n'
        '\n'
        '    key = title.lower().strip()\n'
        '    book = _BOOKS.get(key)\n'
        '    if not book:\n'
        '        return f"未找到该书。可用书目：{\', \'.join(t.title() for t in _BOOKS)}"\n'
        '\n'
        '    # 根据读者风格格式化输出，使 LLM 产生明显不同的响应\n'
        '    if reader and reader.preferred_style == "bullet points":\n'
        '        return (\n'
        '            f"为 {reader.reader_name} 整理（要点列表格式）:\\n"\n'
        '            f"• 书名：{title.title()}\\n"\n'
        '            f"• 作者：{book[\'author\']}\\n"\n'
        '            f"• 类型：{book[\'genre\']}\\n"\n'
        '            f"• 主题：{\', \'.join(book[\'themes\'])}\\n"\n'
        '            f"• 简介：{book[\'plot\']}"\n'
        '        )\n'
        '    name  = reader.reader_name if reader else "读者"\n'
        '    style = f"（{reader.preferred_style} 风格）" if reader else ""\n'
        '    return (\n'
        '        f"为 {name}{style}：\\n"\n'
        '        f"书名：{title.title()} — {book[\'author\']}\\n"\n'
        '        f"类型：{book[\'genre\']}\\n"\n'
        '        f"主题：{\', \'.join(book[\'themes\'])}\\n"\n'
        '        f"简介：{book[\'plot\']}"\n'
        '    )\n'
        '\n'
        'handler = create_langfuse_handler()\n'
        '\n'
        'def new_config(trace_name: str, thread_id: str = None) -> dict:\n'
        '    """为每次演示构建带有唯一 thread_id 的 LangChain 配置。"""\n'
        '    cfg = build_langfuse_config(handler, session_id="s03-nb-cn", trace_name=trace_name)\n'
        '    cfg["configurable"] = {"thread_id": thread_id or str(uuid7())}\n'
        '    return cfg\n'
        '\n'
        'print("✓ 初始化、图书数据库和工具准备完成")\n'
    ),

    md("""
## 演示 A · 流式输出

`agent.stream()` 配合 `stream_mode="values"` 在每个节点执行后产出**完整的智能体状态**。
通过检查每步的 `chunk["messages"][-1]`，可以实时看到工具调用和回复——无需等待最终答案。
"""),

    code(
        'streaming_agent = create_agent(\n'
        '    model=create_llm(),\n'
        '    tools=[lookup_book],\n'
        '    system_prompt="You are a knowledgeable book assistant. Use lookup_book to answer questions.",\n'
        '    checkpointer=InMemorySaver(),\n'
        ')\n'
        '\n'
        'question_a = "Tell me about \'1984\' by George Orwell."\n'
        'print(f"问题：{question_a}\\n")\n'
        '\n'
        'for chunk in streaming_agent.stream(\n'
        '    {"messages": [{"role": "user", "content": question_a}]},\n'
        '    config=new_config("演示 A — 流式输出"),\n'
        '    stream_mode="values",\n'
        '):\n'
        '    latest = chunk["messages"][-1]\n'
        '    if isinstance(latest, HumanMessage):\n'
        '        pass  # 跳过用户自身的消息\n'
        '    elif isinstance(latest, AIMessage) and latest.tool_calls:\n'
        '        names = [tc["name"] for tc in latest.tool_calls]\n'
        '        print(f"  [工具调用] → {\', \'.join(names)}")\n'
        '    elif isinstance(latest, AIMessage) and latest.content:\n'
        '        print(f"  [回复]\\n{latest.content}")\n'
    ),

    md("""
## 演示 B · 结构化输出（`response_format`）

将 Pydantic 模型作为 `response_format` 传入，即可获得**经过验证的对象**而非自由文本。
智能体通过结合工具结果和自身推理来填充每个字段。
结果存储在 `result["structured_response"]` 中——保证 Schema 正确，无需解析文本。
"""),

    code(
        'class BookReport(BaseModel):\n'
        '    """结构化书评报告——每个字段均由智能体填写。"""\n'
        '    title: str            = Field(description="书名")\n'
        '    author: str           = Field(description="作者姓名及出版年份")\n'
        '    one_line_summary: str = Field(description="一句话概括本书精髓")\n'
        '    themes: list[str]     = Field(description="最多 3 个核心主题")\n'
        '    recommended_for: str  = Field(description="最适合哪类读者")\n'
        '    rating: float         = Field(description="评分，1.0 到 5.0", ge=1.0, le=5.0)\n'
        '\n'
        'structured_agent = create_agent(\n'
        '    model=create_llm(),\n'
        '    tools=[lookup_book],\n'
        '    system_prompt=(\n'
        '        "You are a literary critic. "\n'
        '        "Use lookup_book to research the book, then fill every field of the report accurately."\n'
        '    ),\n'
        '    response_format=BookReport,\n'
        '    checkpointer=InMemorySaver(),\n'
        ')\n'
        '\n'
        'question_b = "Write a structured report on \'Dune\'."\n'
        'print(f"问题：{question_b}\\n")\n'
        '\n'
        'result_b = structured_agent.invoke(\n'
        '    {"messages": [{"role": "user", "content": question_b}]},\n'
        '    config=new_config("演示 B — 结构化输出"),\n'
        ')\n'
        '\n'
        'report: BookReport = result_b["structured_response"]\n'
        'print(f"书名     ：{report.title}")\n'
        'print(f"作者     ：{report.author}")\n'
        'print(f"一句话概括：{report.one_line_summary}")\n'
        'print(f"核心主题 ：{\', \'.join(report.themes)}")\n'
        'print(f"推荐读者 ：{report.recommended_for}")\n'
        'print(f"评分     ：{report.rating} / 5.0")\n'
    ),

    md("""
## 演示 C · 上下文 Schema（`context_schema` + `ToolRuntime`）

`context_schema` 声明每次运行的用户数据结构。数据通过调用时的 `context=` 参数传入，
经由 `ToolRuntime` 注入工具函数——整个过程由 Harness 完成。

**与直接在系统提示中添加上下文的区别：**
- LLM 的消息历史中**看不到** `reader_name` 或 `preferred_style`
- 上下文通过依赖注入直接流入工具函数
- 敏感数据（用户 ID、API 密钥、功能开关）不会进入 LLM 对话

本演示中工具针对不同读者格式化输出，因此工具输出和 LLM 最终回复都会明显不同——
证明上下文确实到达了工具函数。
"""),

    code(
        'context_agent = create_agent(\n'
        '    model=create_llm(),\n'
        '    tools=[lookup_book],\n'
        '    system_prompt=(\n'
        '        "You are a personal book assistant. "\n'
        '        "Use lookup_book to fetch book info, then summarize following the format in the tool output."\n'
        '    ),\n'
        '    context_schema=ReaderContext,\n'
        '    checkpointer=InMemorySaver(),\n'
        ')\n'
        '\n'
        'question_c = "Summarize \'The Great Gatsby\' for me."\n'
        '\n'
        'readers = [\n'
        '    ReaderContext(reader_name="爱丽丝", preferred_style="bullet points"),\n'
        '    ReaderContext(reader_name="鲍勃",   preferred_style="casual and conversational"),\n'
        ']\n'
        '\n'
        'for reader in readers:\n'
        '    print(f"\\n{\\"─\\"*50}")\n'
        '    print(f"读者：{reader.reader_name}  |  风格：{reader.preferred_style}")\n'
        '    print("─" * 50)\n'
        '    result_c = context_agent.invoke(\n'
        '        {"messages": [{"role": "user", "content": question_c}]},\n'
        '        config=new_config(f"演示 C — 上下文（{reader.reader_name}）"),\n'
        '        context=reader,\n'
        '    )\n'
        '    print(result_c["messages"][-1].content)\n'
        '\n'
        'print(f"\\n追踪记录：{get_langfuse_host()}")\n'
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 写入所有中文笔记本
# ─────────────────────────────────────────────────────────────────────────────
print("正在生成中文笔记本……")
save("examples/notebooks_cn/s_01_llm_with_tools.ipynb", NB1)
save("examples/notebooks_cn/s_02_research_agent.ipynb",  NB2)
save("examples/notebooks_cn/s_03_agent_harness.ipynb",   NB3)
print("全部完成。")
