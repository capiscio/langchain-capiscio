"""langchain-capiscio — Trust enforcement adapter for LangChain and LangGraph.

from langchain_capiscio import CapiscioGuard
from langchain_openai import ChatOpenAI

secured = CapiscioGuard() | ChatOpenAI()
result = secured.invoke("Summarise quarterly earnings")
"""

from langchain_capiscio._context import (
    CapiscioRequestContext,
    clear_capiscio_context,
    extract_badge_token,
    get_capiscio_context,
    set_capiscio_context,
)
from langchain_capiscio.callbacks import CapiscioCallbackHandler
from langchain_capiscio.guard import (
    CapiscioConfigError,
    CapiscioGuard,
    CapiscioTool,
    CapiscioTrustError,
    capiscio_guard,
)
from langchain_capiscio.tools import resolve_agent_card, verify_badge

__all__ = [
    # Core enforcement
    "CapiscioGuard",
    "CapiscioTool",
    "capiscio_guard",
    # Callbacks
    "CapiscioCallbackHandler",
    # Tools
    "verify_badge",
    "resolve_agent_card",
    # Context
    "CapiscioRequestContext",
    "set_capiscio_context",
    "get_capiscio_context",
    "clear_capiscio_context",
    "extract_badge_token",
    # Errors
    "CapiscioTrustError",
    "CapiscioConfigError",
]

__version__ = "0.1.0"
