"""Langfuse tracing: forward LangChain / LangGraph call traces to a local Langfuse server.

Langfuse v4.x reads credentials from environment variables:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

Session / user metadata is passed via the LangChain config dict using these keys:
  langfuse_session_id, langfuse_user_id, langfuse_trace_name, langfuse_tags
"""
from langfuse.langchain import CallbackHandler

# Ensure .env is loaded before Langfuse reads its env vars
import common.env  # noqa: F401


def create_langfuse_handler() -> CallbackHandler:
    """
    Create a Langfuse callback handler (credentials read from env vars).

    Usage with LangGraph:
        handler = create_langfuse_handler()
        result = agent.invoke(
            input,
            config={
                "callbacks": [handler],
                "metadata": {
                    "langfuse_session_id": "my-session",
                    "langfuse_user_id": "alice",
                    "langfuse_trace_name": "my-trace",
                },
            },
        )
    """
    return CallbackHandler()


def build_langfuse_config(
    handler: CallbackHandler,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Build a LangChain config dict containing Langfuse metadata.

    Pass the returned dict directly to agent.invoke():
        config = build_langfuse_config(handler, session_id="s1", user_id="alice")
        agent.invoke(input, config=config)
    """
    metadata: dict = extra_metadata.copy() if extra_metadata else {}

    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if trace_name:
        metadata["langfuse_trace_name"] = trace_name
    if tags:
        metadata["langfuse_tags"] = tags

    return {
        "callbacks": [handler],
        "metadata": metadata,
    }
