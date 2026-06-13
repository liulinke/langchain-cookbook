"""LLM factory: create ChatModel instances in a single place."""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Default to gpt-4o-mini for a good cost/quality balance
DEFAULT_MODEL = "gpt-4o-mini"


def create_llm(model: str = DEFAULT_MODEL, temperature: float = 0, **kwargs) -> BaseChatModel:
    """
    Create and return a ChatModel instance.

    Model name examples:
      - "gpt-4o-mini"       → OpenAI  (requires OPENAI_API_KEY)
      - "claude-sonnet-4-6" → Anthropic (requires ANTHROPIC_API_KEY)

    Provider is inferred automatically from the model name.
    """
    return init_chat_model(model, temperature=temperature, **kwargs)
