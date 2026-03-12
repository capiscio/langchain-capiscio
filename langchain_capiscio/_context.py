"""Badge token extraction from contextvar, RunnableConfig, and input dict.

Priority order:
1. Context variable (set by A2A server middleware at the HTTP perimeter)
2. RunnableConfig configurable (for programmatic multi-agent orchestration)
3. Input dict key (fallback for simple cases)
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

CAPISCIO_BADGE_KEY = "capiscio_badge"

_capiscio_context: ContextVar[CapiscioRequestContext | None] = ContextVar(
    "capiscio_context", default=None
)


@dataclass(frozen=True)
class CapiscioRequestContext:
    """Verified request context set by perimeter middleware."""

    badge_token: str
    caller_did: str | None = None
    claims: dict[str, Any] | None = None


def set_capiscio_context(ctx: CapiscioRequestContext) -> None:
    """Set the CapiscIO request context for the current async/thread context.

    Call this from your A2A server middleware after verifying the badge at the
    HTTP perimeter. CapiscioGuard will read from this automatically.
    """
    _capiscio_context.set(ctx)


def get_capiscio_context() -> CapiscioRequestContext | None:
    """Get the current CapiscIO request context, if any."""
    return _capiscio_context.get()


def clear_capiscio_context() -> None:
    """Clear the CapiscIO request context."""
    _capiscio_context.set(None)


def extract_badge_token(
    input_data: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Extract badge token using the priority chain.

    Priority:
    1. Context variable (perimeter middleware)
    2. RunnableConfig configurable
    3. Input dict key

    Returns:
        Badge token string, or None if not found.
    """
    # 1. Context variable (highest priority — set by perimeter middleware)
    ctx = _capiscio_context.get()
    if ctx is not None:
        return ctx.badge_token

    # 2. RunnableConfig configurable
    if config is not None:
        configurable = config.get("configurable", {})
        if configurable and CAPISCIO_BADGE_KEY in configurable:
            return configurable[CAPISCIO_BADGE_KEY]

    # 3. Input dict fallback
    if input_data is not None and CAPISCIO_BADGE_KEY in input_data:
        return input_data[CAPISCIO_BADGE_KEY]

    return None
