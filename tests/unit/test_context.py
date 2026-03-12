"""Tests for badge token extraction from contextvar / config / input."""

from __future__ import annotations

from langchain_capiscio._context import (
    CapiscioRequestContext,
    clear_capiscio_context,
    extract_badge_token,
    get_capiscio_context,
    set_capiscio_context,
)


class TestExtractBadgeToken:
    """Test the priority chain: contextvar > config > input."""

    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    def test_returns_none_when_no_source(self):
        assert extract_badge_token() is None

    def test_returns_none_with_empty_input(self):
        assert extract_badge_token(input_data={}) is None

    def test_returns_none_with_empty_config(self):
        assert extract_badge_token(config={}) is None

    def test_extracts_from_input_dict(self):
        token = extract_badge_token(input_data={"capiscio_badge": "tok_input"})
        assert token == "tok_input"

    def test_extracts_from_config_configurable(self):
        config = {"configurable": {"capiscio_badge": "tok_config"}}
        token = extract_badge_token(config=config)
        assert token == "tok_config"

    def test_extracts_from_contextvar(self):
        ctx = CapiscioRequestContext(badge_token="tok_ctx")
        set_capiscio_context(ctx)
        token = extract_badge_token()
        assert token == "tok_ctx"

    def test_contextvar_takes_priority_over_config(self):
        ctx = CapiscioRequestContext(badge_token="tok_ctx")
        set_capiscio_context(ctx)
        config = {"configurable": {"capiscio_badge": "tok_config"}}
        token = extract_badge_token(config=config)
        assert token == "tok_ctx"

    def test_contextvar_takes_priority_over_input(self):
        ctx = CapiscioRequestContext(badge_token="tok_ctx")
        set_capiscio_context(ctx)
        token = extract_badge_token(input_data={"capiscio_badge": "tok_input"})
        assert token == "tok_ctx"

    def test_config_takes_priority_over_input(self):
        config = {"configurable": {"capiscio_badge": "tok_config"}}
        input_data = {"capiscio_badge": "tok_input"}
        token = extract_badge_token(input_data=input_data, config=config)
        assert token == "tok_config"

    def test_all_three_sources_returns_contextvar(self):
        ctx = CapiscioRequestContext(badge_token="tok_ctx")
        set_capiscio_context(ctx)
        config = {"configurable": {"capiscio_badge": "tok_config"}}
        input_data = {"capiscio_badge": "tok_input"}
        token = extract_badge_token(input_data=input_data, config=config)
        assert token == "tok_ctx"


class TestCapiscioRequestContext:
    def setup_method(self):
        clear_capiscio_context()

    def teardown_method(self):
        clear_capiscio_context()

    def test_set_and_get_context(self):
        ctx = CapiscioRequestContext(
            badge_token="tok", caller_did="did:web:example.com", claims={"iss": "test"}
        )
        set_capiscio_context(ctx)
        result = get_capiscio_context()
        assert result is not None
        assert result.badge_token == "tok"
        assert result.caller_did == "did:web:example.com"
        assert result.claims == {"iss": "test"}

    def test_clear_context(self):
        ctx = CapiscioRequestContext(badge_token="tok")
        set_capiscio_context(ctx)
        clear_capiscio_context()
        assert get_capiscio_context() is None

    def test_context_defaults(self):
        ctx = CapiscioRequestContext(badge_token="tok")
        assert ctx.caller_did is None
        assert ctx.claims is None

    def test_context_is_frozen(self):
        ctx = CapiscioRequestContext(badge_token="tok")
        try:
            ctx.badge_token = "new"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass
