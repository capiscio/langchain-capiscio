"""Tests for CapiscioCallbackHandler."""

from __future__ import annotations

from uuid import uuid4

from langchain_capiscio.callbacks import CapiscioCallbackHandler


class FakeEmitter:
    """Records emitted events for assertion."""

    def __init__(self):
        self.events: list[tuple[str, dict | None, dict]] = []

    def emit(self, event_type: str, data=None, **kwargs):
        self.events.append((event_type, data, kwargs))
        return True


class TestCapiscioCallbackHandler:
    def test_chain_start_emits_task_started(self):
        emitter = FakeEmitter()
        handler = CapiscioCallbackHandler(emitter=emitter)
        run_id = uuid4()

        handler.on_chain_start({"name": "MyChain"}, {"input": "test"}, run_id=run_id)

        assert len(emitter.events) == 1
        event_type, data, kwargs = emitter.events[0]
        assert event_type == "task_started"
        assert data["chain"] == "MyChain"
        assert data["run_id"] == str(run_id)

    def test_chain_end_emits_task_completed(self):
        emitter = FakeEmitter()
        handler = CapiscioCallbackHandler(emitter=emitter)
        run_id = uuid4()

        handler.on_chain_end({"output": "result"}, run_id=run_id)

        assert len(emitter.events) == 1
        event_type, data, _ = emitter.events[0]
        assert event_type == "task_completed"

    def test_chain_error_emits_task_failed(self):
        emitter = FakeEmitter()
        handler = CapiscioCallbackHandler(emitter=emitter)
        run_id = uuid4()

        handler.on_chain_error(ValueError("boom"), run_id=run_id)

        assert len(emitter.events) == 1
        event_type, data, _ = emitter.events[0]
        assert event_type == "task_failed"
        assert "boom" in data["error"]

    def test_tool_start_emits_tool_call(self):
        emitter = FakeEmitter()
        handler = CapiscioCallbackHandler(emitter=emitter)
        run_id = uuid4()
        parent_id = uuid4()

        handler.on_tool_start({"name": "search"}, "query", run_id=run_id, parent_run_id=parent_id)

        assert len(emitter.events) == 1
        event_type, data, kwargs = emitter.events[0]
        assert event_type == "tool_call"
        assert data["tool"] == "search"

    def test_tool_end_emits_tool_result(self):
        emitter = FakeEmitter()
        handler = CapiscioCallbackHandler(emitter=emitter)
        run_id = uuid4()

        handler.on_tool_end("result", run_id=run_id)

        assert len(emitter.events) == 1
        event_type, _, _ = emitter.events[0]
        assert event_type == "tool_result"

    def test_no_emitter_does_not_raise(self):
        handler = CapiscioCallbackHandler()
        run_id = uuid4()
        # Should not raise even with no emitter
        handler.on_chain_start({"name": "X"}, {}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)
        handler.on_chain_error(ValueError("x"), run_id=run_id)
        handler.on_tool_start({"name": "x"}, "", run_id=run_id)
        handler.on_tool_end("", run_id=run_id)
        handler.on_tool_error(ValueError("x"), run_id=run_id)

    def test_emitter_from_identity(self):
        emitter = FakeEmitter()

        class FakeIdentity:
            _emitter = emitter

        handler = CapiscioCallbackHandler(identity=FakeIdentity())
        handler.on_chain_start({"name": "test"}, {}, run_id=uuid4())
        assert len(emitter.events) == 1

    def test_chain_name_fallback_to_id(self):
        emitter = FakeEmitter()
        handler = CapiscioCallbackHandler(emitter=emitter)
        run_id = uuid4()

        handler.on_chain_start({"id": ["langchain", "MyChain"]}, {"input": "test"}, run_id=run_id)

        _, data, _ = emitter.events[0]
        assert data["chain"] == "MyChain"

    def test_emitter_exception_swallowed(self):
        class BrokenEmitter:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        handler = CapiscioCallbackHandler(emitter=BrokenEmitter())
        # Should not raise
        handler.on_chain_start({"name": "X"}, {}, run_id=uuid4())
