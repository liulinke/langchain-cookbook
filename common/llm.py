"""LLM factory: create ChatModel instances from a single place.

Model is resolved in this order:
  1. Explicit `model` argument passed to create_llm()
  2. LLM_MODEL environment variable
  3. Hard-coded default: gpt-4o-mini

Provider selection (checked in order):
  1. DASHSCOPE_API_KEY set  → langchain-qwq ChatQwen (native DashScope)
  2. LLM_BASE_URL set       → OpenAI-compatible custom endpoint
  3. Otherwise              → init_chat_model (OpenAI / Anthropic / etc.)
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from common.env import get_env

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_QWEN_MODEL = "qwen-plus"
# DashScope China-region endpoint (default for keys issued on aliyun.com)
_DASHSCOPE_CN_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_llm(model: str | None = None, temperature: float = 0, **kwargs) -> BaseChatModel:
    """Return a ChatModel configured from .env or explicit arguments.

    .env quick-start
    ----------------
    OpenAI:
        LLM_MODEL=gpt-4o-mini
        OPENAI_API_KEY=sk-...

    Qwen (China region key, langchain-qwq):
        LLM_MODEL=qwen-plus          # qwen-max / qwen-turbo / qwen3-32b / qwq-32b …
        DASHSCOPE_API_KEY=sk-...
        # LLM_BASE_URL defaults to the China endpoint; set it only to override.
    """
    resolved_model = model or get_env("LLM_MODEL")
    dashscope_key = get_env("DASHSCOPE_API_KEY")
    base_url = get_env("LLM_BASE_URL")

    # ── Qwen via native langchain-qwq ─────────────────────────────────────
    if dashscope_key:
        from langchain_qwq import ChatQwen

        # LLM_BASE_URL lets users override the endpoint (e.g. intl vs CN).
        # Fall back to the China-region endpoint for keys issued on aliyun.com.
        api_base = base_url or _DASHSCOPE_CN_BASE
        return ChatQwen(
            model_name=resolved_model or _DEFAULT_QWEN_MODEL,
            api_key=dashscope_key,
            api_base=api_base,
            temperature=temperature,
            **kwargs,
        )

    # ── OpenAI-compatible custom endpoint ─────────────────────────────────
    if base_url:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolved_model or _DEFAULT_MODEL,
            temperature=temperature,
            base_url=base_url,
            api_key=get_env("OPENAI_API_KEY"),
            **kwargs,
        )

    # ── Standard: provider inferred from model name ────────────────────────
    return init_chat_model(
        resolved_model or _DEFAULT_MODEL,
        temperature=temperature,
        **kwargs,
    )
