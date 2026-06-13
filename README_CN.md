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

```bash
uv run python -m examples.s_01_llm_with_tools
```

详细说明见 [examples/s_01_llm_with_tools/README.md](examples/s_01_llm_with_tools/README.md)。

---

## 项目结构

```
langchain-cookbook/
├── common/
│   ├── env.py          # 加载 .env 环境变量
│   ├── llm.py          # 创建 LLM 实例（默认 gpt-4o-mini）
│   └── tracing.py      # Langfuse 追踪集成
├── examples/
│   └── s_01_llm_with_tools/
│       ├── README.md       # 示例说明
│       ├── __main__.py     # 包入口（python -m ...）
│       └── main.py         # 示例逻辑
├── main.py             # 示例列表
└── .env.example        # 环境变量模板
```

## 添加新示例

1. 在 `examples/` 下新建目录，如 `examples/s_02_rag/`
2. 创建 `__init__.py`、`__main__.py` 和 `main.py`，从 `common/` 导入公共工具
3. 在新目录中添加 `README.md` 说明示例
4. 在根目录 `main.py` 的 `EXAMPLES` 列表中登记
5. 运行：`uv run python -m examples.s_02_rag`
