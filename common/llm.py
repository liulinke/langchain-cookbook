"""LLM factory: create ChatModel instances from a single place.

Model is resolved in this order:
  1. Explicit `model` argument passed to create_llm()
  2. LLM_MODEL environment variable
  3. Hard-coded default: gpt-4o-mini

When LLM_BASE_URL is set (e.g. DashScope compatible endpoint for Qwen),
ChatOpenAI is used directly with that base URL and the API key is read from
DASHSCOPE_API_KEY (falling back to OPENAI_API_KEY).  This lets you switch
between OpenAI and Qwen by changing just two lines in .env.
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from common.env import get_env

_DEFAULT_MODEL = "gpt-4o-mini"


def create_llm(model: str | None = None, temperature: float = 0, **kwargs) -> BaseChatModel:
    """Return a ChatModel configured from .env or explicit arguments.

    Examples
    --------
    OpenAI (default):
        OPENAI_API_KEY=sk-...
        LLM_MODEL=gpt-4o-mini          # or omit to use the default

    Qwen via DashScope compatible mode:
        DASHSCOPE_API_KEY=sk-...
        LLM_MODEL=qwen-plus
        LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    """
    # Resolve model name: argument → env var → default
    resolved_model = model or get_env("LLM_MODEL") or _DEFAULT_MODEL

    base_url = get_env("LLM_BASE_URL")

    if base_url:
        # Use OpenAI-compatible client pointed at a custom endpoint (e.g. DashScope)
        from langchain_openai import ChatOpenAI

        # DashScope key takes priority; fall back to OPENAI_API_KEY
        api_key = get_env("DASHSCOPE_API_KEY") or get_env("OPENAI_API_KEY")
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

    # Standard path: provider inferred from model name by init_chat_model
    return init_chat_model(resolved_model, temperature=temperature, **kwargs)
