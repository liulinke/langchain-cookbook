"""Load .env and provide a unified interface for reading environment variables."""
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root from this file's location (two levels up)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# LangChain auto-injects lc_versions (a dict) into run metadata; LangGraph warns
# when it can't propagate non-string metadata values to child nodes. Drop it silently.
warnings.filterwarnings(
    "ignore",
    message=".*Propagated attribute.*lc_versions.*not a string.*",
)


def get_env(key: str, default: str | None = None) -> str | None:
    """Return the value of an environment variable, or default if not set."""
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """Return the value of a required environment variable; raise if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}. Check your .env file.")
    return value
