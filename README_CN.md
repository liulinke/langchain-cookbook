# LangChain Cookbook（中文说明）

用于学习和实验 LangChain、LangGraph 的代码示例。每个示例独立可运行，共享公共工具代码。所有示例均来自 [LangChain 官方文档](https://python.langchain.com/docs/)。

> English README: [README.md](README.md)

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）

## 快速开始

```bash
# 克隆项目后，安装依赖（uv 会自动创建虚拟环境）
uv sync
```

复制 `.env.example` 为 `.env`，填入 API 密钥：

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=你的 OpenAI Key

# Langfuse 追踪（可选，本地服务）
LANGFUSE_PUBLIC_KEY=你的 Langfuse 公钥
LANGFUSE_SECRET_KEY=你的 Langfuse 私钥
LANGFUSE_HOST=http://localhost:3000
```

## 查看所有示例

```bash
uv run python main.py
```

## 示例列表

### 01 · 带工具调用的 LLM（ReAct Agent）

基于 LangGraph Graph API 构建的 ReAct Agent，能调用加减乘除工具逐步完成数学问题。调用链路自动上传到本地 Langfuse 服务。

> 来源：[LangGraph 教程 — 构建 ReAct Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

---

### 02 · Research Agent（调研 Agent）

使用 `create_deep_agent`（来自 `deepagents`）构建的文档调研 Agent，能读取完整小说并跨两轮对话回答问题，通过 `thread_id` 实现持久化多轮记忆。

`examples/notebooks_cn/s_02_research_agent.ipynb`

---

### 03 · Agent Harness 模式

演示 `Agent = 模型 + Harness` 的核心概念，涵盖三种 Harness 特性：实时流式输出工具调用、通过 `response_format` 生成结构化 Pydantic 输出、以及通过 `context_schema` 注入每次运行的用户数据。

`examples/notebooks_cn/s_03_agent_harness.ipynb`

---

### 04 · RAG（检索增强生成）

完整 RAG 流水线：加载本地 HTML 文档 → 分块 → 用 OpenAI 嵌入 → 存入向量数据库。然后演示两种检索模式：RAG Agent（LLM 自决何时检索）和 RAG Chain（始终先检索）。包含来源文档审计和提示注入防御。

`examples/notebooks_cn/s_04_rag.ipynb`

---

### 05 · Human-in-the-Loop（人工介入）

两种人工监督 Agent 操作的模式：LangGraph 原生 `interrupt()` 用于暂停和恢复图执行；`HumanInTheLoopMiddleware` 用于基于策略的工具调用审批。涵盖四种决策类型（批准 / 编辑 / 拒绝 / 直接回复）以及只在高风险操作时触发的条件中断。

`examples/notebooks_cn/s_05_human_in_the_loop.ipynb`

---

### 06 · 多 Agent：子 Agent（Subagents）

监督者架构：主 Agent 以工具的形式协调各专业子 Agent。演示四种模式：每个 Agent 一个工具（Tool-Per-Agent）、单一调度工具（一个 `task()` 路由到任意 Agent）、Enum 约束的类型安全 Agent 发现、以及基于 `Command` 的结构化输出（将子 Agent 的结果直接写入监督者状态）。

`examples/notebooks_cn/s_06_multi_agent_subagents.ipynb`

---

## 项目结构

```
langchain-cookbook/
├── common/
│   ├── env.py          # 加载 .env 环境变量
│   ├── llm.py          # 创建 LLM 实例（默认 gpt-4o-mini）
│   └── tracing.py      # Langfuse 追踪集成
├── examples/
│   ├── notebooks/
│   │   ├── s_01_llm_with_tools.ipynb
│   │   ├── s_02_research_agent.ipynb
│   │   ├── s_03_agent_harness.ipynb
│   │   ├── s_04_rag.ipynb
│   │   ├── s_05_human_in_the_loop.ipynb
│   │   └── s_06_multi_agent_subagents.ipynb
│   └── notebooks_cn/
│       ├── s_01_llm_with_tools.ipynb
│       ├── s_02_research_agent.ipynb
│       ├── s_03_agent_harness.ipynb
│       ├── s_04_rag.ipynb
│       ├── s_05_human_in_the_loop.ipynb
│       └── s_06_multi_agent_subagents.ipynb
├── main.py             # 示例列表
└── .env.example        # 环境变量模板
```

## 添加新示例

1. 在 `examples/` 下新建目录，如 `examples/s_02_rag/`
2. 创建 `__init__.py`、`__main__.py` 和 `main.py`，从 `common/` 导入公共工具
3. 在新目录中添加 `README.md` 说明示例
4. 在根目录 `main.py` 的 `EXAMPLES` 列表中登记
5. 运行：`uv run python -m examples.s_02_rag`
