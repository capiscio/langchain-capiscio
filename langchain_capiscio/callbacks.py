"""CapiscioCallbackHandler — LangChain callback handler wired to CapiscIO EventEmitter.

Emits structured audit events for chain, tool, and verification lifecycle events.
Observability layer — not enforcement.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class CapiscioCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that emits events to the CapiscIO EventEmitter.

    Wires LangChain lifecycle callbacks to CapiscIO's event telemetry for
    enterprise audit trails and dashboard visibility.

        from langchain_capiscio import CapiscioCallbackHandler

        handler = CapiscioCallbackHandler()
        result = chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        *,
        emitter: Any | None = None,
        identity: Any | None = None,
    ):
        """Initialize the callback handler.

        Args:
            emitter: An existing EventEmitter instance. If not provided, one will
                     be obtained from the identity's _emitter attribute.
            identity: An AgentIdentity from CapiscIO.connect(). Used to obtain an
                      emitter if none is provided.
        """
        super().__init__()
        self._emitter = emitter
        if self._emitter is None and identity is not None:
            self._emitter = getattr(identity, "_emitter", None)

    def _emit(self, event_type: str, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Emit an event if the emitter is available."""
        if self._emitter is None:
            return
        try:
            self._emitter.emit(event_type, data, **kwargs)
        except Exception:
            logger.debug("Failed to emit event %s", event_type, exc_info=True)

    # -- Chain lifecycle --

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        self._emit(
            "task_started",
            {"chain": chain_name, "run_id": str(run_id)},
            task_id=str(run_id),
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "task_completed",
            {"run_id": str(run_id)},
            task_id=str(run_id),
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "task_failed",
            {"error": str(error), "run_id": str(run_id)},
            task_id=str(run_id),
        )

    # -- Tool lifecycle --

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        self._emit(
            "tool_call",
            {"tool": tool_name, "run_id": str(run_id)},
            task_id=str(parent_run_id) if parent_run_id else None,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "tool_result",
            {"run_id": str(run_id)},
            task_id=str(parent_run_id) if parent_run_id else None,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "task_failed",
            {"error": str(error), "tool_run_id": str(run_id)},
            task_id=str(parent_run_id) if parent_run_id else None,
        )
