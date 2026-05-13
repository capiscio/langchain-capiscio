"""Tests for CapiscioGuard — trust enforcement Runnable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_capiscio._context import (
    CapiscioRequestContext,
    clear_capiscio_context,
    get_capiscio_context,
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
        import sys
        from unittest.mock import MagicMock, patch

        old = os.environ.pop("CAPISCIO_API_KEY", None)
        try:
            guard = CapiscioGuard.__new__(CapiscioGuard)
            CapiscioGuard.__init__(guard, mode="block")
            # Mock SDK so test doesn't depend on protobuf stubs being installed
            with patch.dict(sys.modules, {"capiscio_sdk": MagicMock()}):
                with pytest.raises(CapiscioConfigError, match="No API key"):
                    guard._ensure_initialized()
        finally:
            if old is not None:
                os.environ["CAPISCIO_API_KEY"] = old


# ---- connect() tests ----


class TestCapiscioGuardConnect:
    def test_connect_returns_initialized_guard(self):
        """connect() should eagerly initialize and return a ready guard."""
        import os
        import sys
        from unittest.mock import MagicMock, patch

        fake_guard_obj = FakeGuard()
        fake_keeper = FakeKeeper()
        fake_identity = FakeIdentity(_guard=fake_guard_obj, _keeper=fake_keeper)

        mock_sdk = MagicMock()
        mock_sdk.CapiscIO.connect.return_value = fake_identity
        mock_sdk.SecurityConfig.production.return_value = FakeConfig("block")

        old_key = os.environ.get("CAPISCIO_API_KEY")
        os.environ["CAPISCIO_API_KEY"] = "cap_test_key"
        try:
            with patch.dict(sys.modules, {"capiscio_sdk": mock_sdk}):
                guard = CapiscioGuard.connect()
        finally:
            if old_key is not None:
                os.environ["CAPISCIO_API_KEY"] = old_key
            else:
                os.environ.pop("CAPISCIO_API_KEY", None)

        assert guard._initialized is True
        assert guard.mode == "block"
        mock_sdk.CapiscIO.connect.assert_called_once_with("cap_test_key")

    def test_connect_explicit_params(self):
        """Explicit params should be forwarded to CapiscIO.connect()."""
        import sys
        from unittest.mock import MagicMock, patch

        fake_guard_obj = FakeGuard()
        fake_keeper = FakeKeeper()
        fake_identity = FakeIdentity(_guard=fake_guard_obj, _keeper=fake_keeper)

        mock_sdk = MagicMock()
        mock_sdk.CapiscIO.connect.return_value = fake_identity
        mock_sdk.SecurityConfig.development.return_value = FakeConfig("warn")

        with patch.dict(sys.modules, {"capiscio_sdk": mock_sdk}):
            guard = CapiscioGuard.connect(
                api_key="cap_explicit",
                mode="log",
                name="my-agent",
                server_url="https://dev.registry.capisc.io",
            )

        assert guard._initialized is True
        assert guard.mode == "log"
        mock_sdk.CapiscIO.connect.assert_called_once_with(
            "cap_explicit",
            name="my-agent",
            server_url="https://dev.registry.capisc.io",
        )

    def test_connect_dev_mode(self):
        """dev_mode=True should be forwarded to CapiscIO.connect()."""
        import sys
        from unittest.mock import MagicMock, patch

        fake_identity = FakeIdentity(_guard=FakeGuard(), _keeper=FakeKeeper())
        mock_sdk = MagicMock()
        mock_sdk.CapiscIO.connect.return_value = fake_identity
        mock_sdk.SecurityConfig.production.return_value = FakeConfig("block")

        with patch.dict(sys.modules, {"capiscio_sdk": mock_sdk}):
            guard = CapiscioGuard.connect(api_key="cap_test", dev_mode=True)

        mock_sdk.CapiscIO.connect.assert_called_once_with(
            "cap_test",
            dev_mode=True,
        )
        assert guard._initialized is True

    def test_connect_no_api_key_raises(self):
        """connect() with no api_key and no env var should raise immediately."""
        import os

        old = os.environ.pop("CAPISCIO_API_KEY", None)
        try:
            with pytest.raises(CapiscioConfigError, match="No API key"):
                CapiscioGuard.connect()
        finally:
            if old is not None:
                os.environ["CAPISCIO_API_KEY"] = old

    def test_connect_env_dev_mode(self):
        """CAPISCIO_DEV_MODE env var should be respected."""
        import os
        import sys
        from unittest.mock import MagicMock, patch

        fake_identity = FakeIdentity(_guard=FakeGuard(), _keeper=FakeKeeper())
        mock_sdk = MagicMock()
        mock_sdk.CapiscIO.connect.return_value = fake_identity
        mock_sdk.SecurityConfig.production.return_value = FakeConfig("block")

        old_key = os.environ.get("CAPISCIO_API_KEY")
        old_dev = os.environ.get("CAPISCIO_DEV_MODE")
        os.environ["CAPISCIO_API_KEY"] = "cap_test"
        os.environ["CAPISCIO_DEV_MODE"] = "true"
        try:
            with patch.dict(sys.modules, {"capiscio_sdk": mock_sdk}):
                CapiscioGuard.connect()
        finally:
            if old_key is not None:
                os.environ["CAPISCIO_API_KEY"] = old_key
            else:
                os.environ.pop("CAPISCIO_API_KEY", None)
            if old_dev is not None:
                os.environ["CAPISCIO_DEV_MODE"] = old_dev
            else:
                os.environ.pop("CAPISCIO_DEV_MODE", None)

        mock_sdk.CapiscIO.connect.assert_called_once_with(
            "cap_test",
            dev_mode=True,
        )

    def test_from_env_delegates_to_connect(self):
        """from_env() should delegate to connect()."""
        import os
        import sys
        from unittest.mock import MagicMock, patch

        fake_identity = FakeIdentity(_guard=FakeGuard(), _keeper=FakeKeeper())
        mock_sdk = MagicMock()
        mock_sdk.CapiscIO.connect.return_value = fake_identity
        mock_sdk.SecurityConfig.development.return_value = FakeConfig("warn")

        old_key = os.environ.get("CAPISCIO_API_KEY")
        os.environ["CAPISCIO_API_KEY"] = "cap_test"
        try:
            with patch.dict(sys.modules, {"capiscio_sdk": mock_sdk}):
                guard = CapiscioGuard.from_env(mode="log")
        finally:
            if old_key is not None:
                os.environ["CAPISCIO_API_KEY"] = old_key
            else:
                os.environ.pop("CAPISCIO_API_KEY", None)

        assert guard._initialized is True
        assert guard.mode == "log"


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
        original = {"input": "test"}
        result = guard.invoke(original)
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is False
        assert "No badge token" in ctx.warnings[0]

    def test_log_mode_no_badge_passes_with_warning(self):
        guard = _make_guard(mode="log")
        original = {"input": "test"}
        result = guard.invoke(original)
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is False

    def test_successful_verification_from_input(self):
        guard = _make_guard(mode="block")
        original = {"input": "test", "capiscio_badge": "valid_token"}
        result = guard.invoke(original)
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is True
        assert ctx.claims["iss"] == "did:web:issuer.com"

    def test_successful_verification_from_config(self):
        guard = _make_guard(mode="block")
        config = {"configurable": {"capiscio_badge": "valid_token"}}
        original = {"input": "test"}
        result = guard.invoke(original, config=config)
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is True

    def test_successful_verification_from_contextvar(self):
        guard = _make_guard(mode="block")
        set_capiscio_context(CapiscioRequestContext(badge_token="valid_token"))
        original = {"input": "test"}
        result = guard.invoke(original)
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is True

    def test_verification_failure_block_mode_raises(self):
        guard = _make_guard(mode="block", should_fail=True, error_msg="expired")
        with pytest.raises(CapiscioTrustError, match="expired"):
            guard._handle_verification_failure({"input": "test"}, "block", "expired")

    def test_verification_failure_monitor_mode(self):
        guard = _make_guard(mode="monitor", should_fail=True)
        original = {"input": "test"}
        result = guard._handle_verification_failure(original, "monitor", "expired")
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is False
        assert "expired" in ctx.warnings[0]

    def test_verification_failure_log_mode(self):
        guard = _make_guard(mode="log", should_fail=True)
        original = {"input": "test"}
        result = guard._handle_verification_failure(original, "log", "expired")
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is False

    def test_passthrough_preserves_input_keys(self):
        guard = _make_guard(mode="block")
        original = {"input": "test", "extra_key": 42, "capiscio_badge": "tok"}
        result = guard.invoke(original)
        assert result is original
        assert result["input"] == "test"
        assert result["extra_key"] == 42
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is True

    def test_string_passthrough_with_badge_in_config(self):
        guard = _make_guard(mode="block")
        config = {"configurable": {"capiscio_badge": "valid_token"}}
        result = guard.invoke("Summarise quarterly earnings", config=config)
        assert result == "Summarise quarterly earnings"
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is True
        assert ctx.claims["iss"] == "did:web:issuer.com"


# ---- Async tests ----


class TestCapiscioGuardAsync:
    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    @pytest.mark.asyncio
    async def test_ainvoke_success(self):
        guard = _make_guard(mode="block")
        original = {"input": "test", "capiscio_badge": "tok"}
        result = await guard.ainvoke(original)
        assert result is original
        ctx = get_capiscio_context()
        assert ctx is not None
        assert ctx.verified is True

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
