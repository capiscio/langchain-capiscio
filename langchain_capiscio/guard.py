"""CapiscioGuard — Runnable trust enforcement for LangChain and LangGraph.

Primary security boundary: verifies caller trust badges before downstream
chain execution. Composable via LCEL pipe (|) operator.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_capiscio._context import extract_badge_token

logger = logging.getLogger(__name__)

# Mode-to-SecurityConfig mapping (lazy import to avoid import-time SDK dependency)
_MODE_MAP = {
    "block": "production",
    "monitor": None,  # custom
    "log": "development",
}


class CapiscioTrustError(Exception):
    """Raised when trust verification fails in block mode."""

    def __init__(self, message: str, *, claims: dict[str, Any] | None = None):
        super().__init__(message)
        self.claims = claims


class CapiscioConfigError(Exception):
    """Raised for configuration errors during guard setup."""


class CapiscioGuard(RunnableSerializable[dict, dict]):
    """Runnable that verifies CapiscIO trust badges before downstream execution.

    Composes with any LangChain Runnable via the pipe operator:

        secured = CapiscioGuard() | my_chain
        result = secured.invoke({"input": "..."})

    Badge token is extracted from (in priority order):
    1. Context variable (set by A2A server perimeter middleware)
    2. RunnableConfig configurable["capiscio_badge"]
    3. Input dict key "capiscio_badge"
    """

    mode: str = Field(default="block", description="Enforcement mode: block, monitor, or log")
    api_key: str | None = Field(
        default=None,
        description="CapiscIO API key (reads CAPISCIO_API_KEY env if not set)",
    )
    name: str | None = Field(default=None, description="Agent name for CapiscIO registration")
    server_url: str | None = Field(default=None, description="CapiscIO registry URL override")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _identity: Any = PrivateAttr(default=None)
    _guard: Any = PrivateAttr(default=None)
    _config: Any = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        *,
        identity: Any | None = None,
        config: Any | None = None,
        mode: str = "block",
        api_key: str | None = None,
        name: str | None = None,
        server_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(mode=mode, api_key=api_key, name=name, server_url=server_url, **kwargs)
        if mode not in ("block", "monitor", "log"):
            raise CapiscioConfigError(f"Invalid mode '{mode}'. Must be one of: block, monitor, log")

        if config is not None:
            self._config = config
        else:
            self._config = self._make_config(mode)

        if identity is not None:
            self._identity = identity
            self._guard = getattr(identity, "_guard", None)
            self._wire_badge_renewal(identity)
            self._initialized = True

    @staticmethod
    def _make_config(mode: str) -> Any:
        """Create a SecurityConfig from a mode string."""
        from capiscio_sdk import SecurityConfig

        if mode == "block":
            return SecurityConfig.production()
        elif mode == "log":
            return SecurityConfig.development()
        elif mode == "monitor":
            return SecurityConfig(fail_mode="monitor")
        return SecurityConfig.production()

    @staticmethod
    def _wire_badge_renewal(identity: Any) -> None:
        """Wire BadgeKeeper.on_renew -> SimpleGuard.set_badge_token.

        Workaround for capiscio-sdk-python#40 where CapiscIO.connect() does not
        wire badge renewal to the guard.
        """
        keeper = getattr(identity, "_keeper", None)
        guard = getattr(identity, "_guard", None)
        if keeper is not None and guard is not None:
            keeper.on_renew = lambda token: guard.set_badge_token(token)

    def _ensure_initialized(self) -> None:
        """Lazy initialization — calls CapiscIO.connect() on first use."""
        if self._initialized:
            return

        from capiscio_sdk import CapiscIO

        connect_kwargs: dict[str, Any] = {}
        api_key = self.api_key or os.environ.get("CAPISCIO_API_KEY")
        if not api_key:
            raise CapiscioConfigError(
                "No API key provided. Set CAPISCIO_API_KEY env var"
                " or pass api_key= to CapiscioGuard."
            )

        if self.name:
            connect_kwargs["name"] = self.name
        if self.server_url:
            connect_kwargs["server_url"] = self.server_url

        self._identity = CapiscIO.connect(api_key, **connect_kwargs)
        self._guard = getattr(self._identity, "_guard", None)
        self._wire_badge_renewal(self._identity)
        self._initialized = True

    @property
    def identity(self) -> Any:
        """The AgentIdentity backing this guard. Triggers connect() if needed."""
        self._ensure_initialized()
        return self._identity

    def invoke(self, input: dict, config: RunnableConfig | None = None) -> dict:
        """Verify trust badge and pass through with injected verification metadata.

        On success: returns input with capiscio_verified=True and capiscio_claims.
        On block failure: raises CapiscioTrustError.
        On monitor/log failure: returns input with capiscio_verified=False and capiscio_warnings.
        """
        self._ensure_initialized()

        badge_token = extract_badge_token(input, config)
        fail_mode = self._config.fail_mode if self._config else "block"

        if badge_token is None:
            return self._handle_missing_badge(input, fail_mode)

        return self._verify_and_enforce(input, badge_token, fail_mode)

    async def ainvoke(
        self,
        input: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        """Async verification — offloads sync gRPC call to thread pool."""
        return await asyncio.to_thread(self.invoke, input, config)

    def _handle_missing_badge(self, input: dict, fail_mode: str) -> dict:
        """Handle case where no badge token is found."""
        msg = "No badge token found in context, config, or input"

        if fail_mode == "block":
            raise CapiscioTrustError(msg)

        logger.warning("CapiscioGuard: %s (mode=%s, continuing)", msg, fail_mode)
        return {
            **input,
            "capiscio_verified": False,
            "capiscio_warnings": [msg],
        }

    def _verify_and_enforce(self, input: dict, badge_token: str, fail_mode: str) -> dict:
        """Verify the badge token and enforce policy."""
        from capiscio_sdk.errors import VerificationError

        try:
            claims = self._guard.verify_inbound(badge_token)
            logger.debug("CapiscioGuard: verification succeeded, issuer=%s", claims.get("iss"))
            return {
                **input,
                "capiscio_verified": True,
                "capiscio_claims": claims,
            }
        except VerificationError as e:
            return self._handle_verification_failure(input, fail_mode, str(e))

    def _handle_verification_failure(self, input: dict, fail_mode: str, error: str) -> dict:
        """Apply enforcement policy after a verification failure."""
        msg = f"Badge verification failed: {error}"

        if fail_mode == "block":
            raise CapiscioTrustError(msg)

        logger.warning("CapiscioGuard: %s (mode=%s, continuing)", msg, fail_mode)
        return {
            **input,
            "capiscio_verified": False,
            "capiscio_warnings": [msg],
        }


class CapiscioTool:
    """Wraps a LangChain Tool with trust enforcement.

    Verifies the caller's badge before allowing tool execution.
    Tool calls are the highest-risk point — where external actions happen.
    """

    def __init__(
        self,
        tool: Any,
        *,
        identity: Any | None = None,
        config: Any | None = None,
        mode: str = "block",
        api_key: str | None = None,
    ):
        self._tool = tool
        self._guard = CapiscioGuard(identity=identity, config=config, mode=mode, api_key=api_key)

        # Preserve tool metadata
        self.name = getattr(tool, "name", "unknown")
        self.description = getattr(tool, "description", "")

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        """Verify trust, then execute the wrapped tool."""
        tool_input = input if isinstance(input, dict) else {"input": input}
        self._guard.invoke(tool_input, config)
        return self._tool.invoke(input, config)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Async verify trust, then execute the wrapped tool."""
        tool_input = input if isinstance(input, dict) else {"input": input}
        await self._guard.ainvoke(tool_input, config)
        return await self._tool.ainvoke(input, config)


def capiscio_guard(
    mode: str = "block",
    *,
    identity: Any | None = None,
    config: Any | None = None,
    api_key: str | None = None,
):
    """Decorator for LangGraph function-based nodes.

    Verifies trust badge before the decorated function executes.

        @capiscio_guard(mode="block")
        def call_external_agent(state: dict) -> dict:
            ...
    """
    guard = CapiscioGuard(identity=identity, config=config, mode=mode, api_key=api_key)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: dict, config: RunnableConfig | None = None) -> dict:
            guard.invoke(state, config)
            return fn(state) if config is None else fn(state, config)

        @functools.wraps(fn)
        async def async_wrapper(state: dict, config: RunnableConfig | None = None) -> dict:
            await guard.ainvoke(state, config)
            if inspect.iscoroutinefunction(fn):
                return await fn(state) if config is None else await fn(state, config)
            return fn(state) if config is None else fn(state, config)

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    return decorator
