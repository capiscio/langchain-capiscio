"""Tests for CapiscioGuard — trust enforcement Runnable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_capiscio._context import (
    CapiscioRequestContext,
    clear_capiscio_context,
    set_capiscio_context,
)
from langchain_capiscio.guard import (
    CapiscioConfigError,
    CapiscioGuard,
    CapiscioTool,
    CapiscioTrustError,
    capiscio_guard,
)

# -- Fixtures: mock SDK objects without importing SDK --


@dataclass
class FakeIdentity:
    agent_id: str = "agent-1"
    did: str = "did:web:example.com"
    name: str = "test-agent"
    _guard: Any = field(default=None, repr=False)
    _keeper: Any = field(default=None, repr=False)
    _emitter: Any = field(default=None, repr=False)


class FakeGuard:
    """Mock SimpleGuard."""

    def __init__(self, *, should_fail: bool = False, error_msg: str = "bad badge"):
        self.should_fail = should_fail
        self.error_msg = error_msg
        self.badge_token: str | None = None

    def verify_inbound(self, jws: str, body: bytes | None = None) -> dict[str, Any]:
        if self.should_fail:
            # Simulate VerificationError — we raise a generic Exception here
            # because importing VerificationError requires the SDK.
            # The guard catches VerificationError specifically, so in tests with
            # a real identity we need to patch. For these unit tests we test
            # the _handle_verification_failure path directly.
            raise _FakeVerificationError(self.error_msg)
        return {"iss": "did:web:issuer.com", "sub": "did:web:subject.com", "tl": 2}

    def set_badge_token(self, token: str) -> None:
        self.badge_token = token


class _FakeVerificationError(Exception):
    """Stand-in for capiscio_sdk.errors.VerificationError in tests."""


class FakeKeeper:
    on_renew: Any = None


class FakeConfig:
    def __init__(self, fail_mode: str = "block"):
        self.fail_mode = fail_mode


def _make_guard(
    *, mode: str = "block", should_fail: bool = False, error_msg: str = "bad badge"
) -> CapiscioGuard:
    """Create a CapiscioGuard with fake identity — no SDK import needed."""
    fake_guard_obj = FakeGuard(should_fail=should_fail, error_msg=error_msg)
    fake_keeper = FakeKeeper()
    identity = FakeIdentity(_guard=fake_guard_obj, _keeper=fake_keeper)
    return CapiscioGuard(identity=identity, config=FakeConfig(fail_mode=mode), mode=mode)


# ---- Constructor tests ----


class TestCapiscioGuardInit:
    def test_valid_modes(self):
        for mode in ("block", "monitor", "log"):
            g = _make_guard(mode=mode)
            assert g.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(CapiscioConfigError, match="Invalid mode"):
            identity = FakeIdentity(_guard=FakeGuard(), _keeper=FakeKeeper())
            CapiscioGuard(identity=identity, config=FakeConfig(), mode="invalid")

    def test_wires_badge_renewal(self):
        fake_guard_obj = FakeGuard()
        fake_keeper = FakeKeeper()
        identity = FakeIdentity(_guard=fake_guard_obj, _keeper=fake_keeper)
        CapiscioGuard(identity=identity, config=FakeConfig())
        # on_renew should be wired
        assert fake_keeper.on_renew is not None
        # Calling it should set badge token on the guard
        fake_keeper.on_renew("new_token")
        assert fake_guard_obj.badge_token == "new_token"

    def test_no_api_key_raises_on_lazy_init(self):
        import os

        old = os.environ.pop("CAPISCIO_API_KEY", None)
        try:
            guard = CapiscioGuard.__new__(CapiscioGuard)
            CapiscioGuard.__init__(guard, mode="block")
            with pytest.raises(CapiscioConfigError, match="No API key"):
                guard._ensure_initialized()
        finally:
            if old is not None:
                os.environ["CAPISCIO_API_KEY"] = old


# ---- Invoke tests (with fake identity, no real SDK) ----


class TestCapiscioGuardInvoke:
    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    def test_block_mode_no_badge_raises(self):
        guard = _make_guard(mode="block")
        with pytest.raises(CapiscioTrustError, match="No badge token"):
            guard.invoke({"input": "test"})

    def test_monitor_mode_no_badge_passes_with_warning(self):
        guard = _make_guard(mode="monitor")
        result = guard.invoke({"input": "test"})
        assert result["capiscio_verified"] is False
        assert "No badge token" in result["capiscio_warnings"][0]
        assert result["input"] == "test"

    def test_log_mode_no_badge_passes_with_warning(self):
        guard = _make_guard(mode="log")
        result = guard.invoke({"input": "test"})
        assert result["capiscio_verified"] is False

    def test_successful_verification_from_input(self):
        guard = _make_guard(mode="block")
        result = guard.invoke({"input": "test", "capiscio_badge": "valid_token"})
        assert result["capiscio_verified"] is True
        assert result["capiscio_claims"]["iss"] == "did:web:issuer.com"
        assert result["input"] == "test"

    def test_successful_verification_from_config(self):
        guard = _make_guard(mode="block")
        config = {"configurable": {"capiscio_badge": "valid_token"}}
        result = guard.invoke({"input": "test"}, config=config)
        assert result["capiscio_verified"] is True

    def test_successful_verification_from_contextvar(self):
        guard = _make_guard(mode="block")
        ctx = CapiscioRequestContext(badge_token="valid_token")
        set_capiscio_context(ctx)
        result = guard.invoke({"input": "test"})
        assert result["capiscio_verified"] is True

    def test_verification_failure_block_mode_raises(self):
        guard = _make_guard(mode="block", should_fail=True, error_msg="expired")
        with pytest.raises(CapiscioTrustError, match="expired"):
            guard._handle_verification_failure({"input": "test"}, "block", "expired")

    def test_verification_failure_monitor_mode(self):
        guard = _make_guard(mode="monitor", should_fail=True)
        result = guard._handle_verification_failure({"input": "test"}, "monitor", "expired")
        assert result["capiscio_verified"] is False
        assert "expired" in result["capiscio_warnings"][0]

    def test_verification_failure_log_mode(self):
        guard = _make_guard(mode="log", should_fail=True)
        result = guard._handle_verification_failure({"input": "test"}, "log", "expired")
        assert result["capiscio_verified"] is False

    def test_passthrough_preserves_input_keys(self):
        guard = _make_guard(mode="block")
        result = guard.invoke({"input": "test", "extra_key": 42, "capiscio_badge": "tok"})
        assert result["input"] == "test"
        assert result["extra_key"] == 42
        assert result["capiscio_verified"] is True


# ---- Async tests ----


class TestCapiscioGuardAsync:
    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    @pytest.mark.asyncio
    async def test_ainvoke_success(self):
        guard = _make_guard(mode="block")
        result = await guard.ainvoke({"input": "test", "capiscio_badge": "tok"})
        assert result["capiscio_verified"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_block_no_badge(self):
        guard = _make_guard(mode="block")
        with pytest.raises(CapiscioTrustError):
            await guard.ainvoke({"input": "test"})


# ---- Identity property ----


class TestCapiscioGuardIdentity:
    def test_identity_property_returns_injected(self):
        guard = _make_guard()
        assert guard.identity.agent_id == "agent-1"
        assert guard.identity.did == "did:web:example.com"


# ---- CapiscioTool tests ----


class TestCapiscioTool:
    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    def test_tool_invoke_verifies_then_runs(self):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Web search"
        mock_tool.invoke.return_value = "result"

        fake_guard_obj = FakeGuard()
        fake_keeper = FakeKeeper()
        identity = FakeIdentity(_guard=fake_guard_obj, _keeper=fake_keeper)

        secured = CapiscioTool(tool=mock_tool, identity=identity, mode="block")
        assert secured.name == "search"

        # With badge in input
        result = secured.invoke(
            {"query": "test", "capiscio_badge": "tok"},
        )
        mock_tool.invoke.assert_called_once()
        assert result == "result"

    def test_tool_invoke_blocks_without_badge(self):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        fake_guard_obj = FakeGuard()
        identity = FakeIdentity(_guard=fake_guard_obj, _keeper=FakeKeeper())

        secured = CapiscioTool(tool=mock_tool, identity=identity, mode="block")
        with pytest.raises(CapiscioTrustError):
            secured.invoke({"query": "test"})
        mock_tool.invoke.assert_not_called()


# ---- @capiscio_guard decorator tests ----


class TestCapiscioGuardDecorator:
    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    def test_decorator_sync(self):
        fake_guard_obj = FakeGuard()
        identity = FakeIdentity(_guard=fake_guard_obj, _keeper=FakeKeeper())

        @capiscio_guard(mode="block", identity=identity)
        def my_node(state: dict) -> dict:
            return {**state, "processed": True}

        # Should pass with badge in input
        result = my_node({"input": "test", "capiscio_badge": "tok"})
        assert result["processed"] is True

    def test_decorator_blocks_without_badge(self):
        fake_guard_obj = FakeGuard()
        identity = FakeIdentity(_guard=fake_guard_obj, _keeper=FakeKeeper())

        @capiscio_guard(mode="block", identity=identity)
        def my_node(state: dict) -> dict:
            return state

        with pytest.raises(CapiscioTrustError):
            my_node({"input": "test"})

    @pytest.mark.asyncio
    async def test_decorator_async(self):
        fake_guard_obj = FakeGuard()
        identity = FakeIdentity(_guard=fake_guard_obj, _keeper=FakeKeeper())

        @capiscio_guard(mode="block", identity=identity)
        async def my_node(state: dict) -> dict:
            return {**state, "processed": True}

        result = await my_node({"input": "test", "capiscio_badge": "tok"})
        assert result["processed"] is True
