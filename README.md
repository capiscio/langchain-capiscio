# CapiscIO LangChain Guard

Trust enforcement for LangChain and LangGraph agents.

[![PyPI version](https://badge.fury.io/py/langchain-capiscio.svg)](https://badge.fury.io/py/langchain-capiscio)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**LangChain Guard** is the CapiscIO trust enforcement adapter for [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). It verifies caller trust badges, enforces security policies, and emits audit events — all composable via LangChain's LCEL pipe (`|`) operator with **zero configuration**.

## Installation

```bash
pip install langchain-capiscio
```

## Quick Start

Turn any LangChain agent into a trust-verified agent in 2 lines:

```python
from langchain_capiscio import CapiscioGuard

# Reads CAPISCIO_API_KEY from env, connects to registry
secured = CapiscioGuard.connect() | my_chain
result = secured.invoke({"input": "Summarize this ticket"})
```

`CapiscioGuard.connect()` reads your environment, connects to the CapiscIO registry, and returns a guard that verifies caller trust badges before every invocation — consistent with `CapiscIO.connect()` and `CapiscioMCPServer.connect()` across the ecosystem.

## Why LangChain Guard?

LangChain agents orchestrate powerful tools — search, databases, code execution. But LangChain itself doesn't define how to:

- **Authenticate** which agent is calling your chain
- **Authorize** whether that agent meets your trust requirements
- **Audit** what happened for post-incident review

LangChain Guard solves this with:

| Feature | Description |
|---------|-------------|
| **`CapiscioGuard`** | `Runnable[dict, dict]` — verifies trust badges before downstream execution. Composable with `\|`. |
| **`CapiscioTool`** | Wraps a LangChain `Tool` with trust enforcement at the tool-call boundary. |
| **`CapiscioCallbackHandler`** | Audit trail — emits chain/tool lifecycle events to the CapiscIO EventEmitter. |
| **`@capiscio_guard`** | Decorator for LangGraph function-based nodes. |
| **`verify_badge` / `resolve_agent_card`** | Convenience `@tool`-decorated functions for agent-driven trust checks. |

## Enforcement Modes

Control enforcement behavior per guard instance:

```python
guard = CapiscioGuard.connect(mode="block")    # Fail closed (production default)
guard = CapiscioGuard.connect(mode="monitor")  # Warn but continue
guard = CapiscioGuard.connect(mode="log")      # Log only
```

## LCEL Pipe Composition

`CapiscioGuard` is a LangChain `Runnable` — compose it with any chain or agent via the `|` operator:

> **Note**: The example below uses [LangGraph](https://langchain-ai.github.io/langgraph/).
> Install it separately: `pip install langgraph`

```python
from langchain_capiscio import CapiscioGuard
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
secured = CapiscioGuard.connect(mode="log") | agent
result = secured.invoke({"input": "What's 42 * 17?"})
```

## Callback Handler

Emit structured audit events (task lifecycle, tool calls) to the CapiscIO dashboard:

```python
from langchain_capiscio import CapiscioCallbackHandler

handler = CapiscioCallbackHandler(emitter=my_event_emitter)
result = chain.invoke(
    {"input": "..."},
    config={"callbacks": [handler]},
)
```

Events emitted: `task_started`, `task_completed`, `task_failed`, `tool_call`, `tool_result`.

## LangGraph Integration

```python
from langchain_capiscio import CapiscioGuard, capiscio_guard

# Option 1: Runnable as graph node
graph.add_node("verify", CapiscioGuard.connect())

# Option 2: Decorator
@capiscio_guard(mode="block")
def call_agent(state: dict) -> dict:
    ...
```

## "Let's Encrypt" Style Setup

### Zero-config (recommended)

Set environment variables and connect with no arguments:

```bash
export CAPISCIO_API_KEY="cap_..."
export CAPISCIO_SERVER_URL="https://dev.registry.capisc.io"  # optional
export CAPISCIO_AGENT_NAME="my-agent"                        # optional
export CAPISCIO_DEV_MODE="true"                              # optional
```

```python
guard = CapiscioGuard.connect()  # reads env vars, connects eagerly
```

### Explicit configuration

```python
guard = CapiscioGuard.connect(
    api_key="cap_...",
    mode="block",
    name="my-agent",
    server_url="https://dev.registry.capisc.io",
)
```

### Extra connect kwargs

Pass additional keyword arguments through to `CapiscIO.connect()`:

```python
guard = CapiscioGuard.connect(
    mode="log",
    dev_mode=True,
    keys_dir="capiscio_keys/",
    agent_card=my_card_dict,
)
```

## Using Environment Variables

`CapiscioGuard.connect()` reads environment variables automatically. `from_env()` is kept as a convenience alias:

```python
guard = CapiscioGuard.connect(mode="log")
```

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `CAPISCIO_API_KEY` | Yes* | Registry API key | — |
| `CAPISCIO_SERVER_URL` | No | Registry URL override | `https://registry.capisc.io` |
| `CAPISCIO_AGENT_NAME` | No | Agent name for registration | — |
| `CAPISCIO_DEV_MODE` | No | Enable dev mode (`true`/`1`/`yes`) | `false` |
| `CAPISCIO_AGENT_PRIVATE_KEY_JWK` | No | JSON-encoded Ed25519 private JWK for ephemeral environments | — |

*Required if not passed explicitly via constructor.

Priority: explicit constructor args > `connect_kwargs` > env vars > SDK defaults.

## Deploying to Containers / Serverless

In ephemeral environments (Docker, Lambda, Cloud Run) the local `~/.capiscio/keys/` directory doesn't survive restarts. Without a persisted key, the SDK generates a **new keypair on every start**, which means a new DID and invalidated badges.

### Key Persistence via Environment Variable

On first run the SDK generates a keypair and logs a capture hint:

```
╔══════════════════════════════════════════════════════════════════╗
║  New agent identity generated — save key for persistence         ║
╚══════════════════════════════════════════════════════════════════╝

  Add to your secrets manager / .env:

    CAPISCIO_AGENT_PRIVATE_KEY_JWK='{"kty":"OKP","crv":"Ed25519","d":"...","x":"...","kid":"did:key:z6Mk..."}'
```

Copy that value into your secrets manager and set it as an environment variable.
On subsequent starts the SDK recovers the same DID without generating a new identity.

**Key resolution priority:** env var → local file → generate new.

```yaml
# docker-compose.yml
services:
  langchain-agent:
    image: my-langchain-agent
    environment:
      CAPISCIO_API_KEY: ${CAPISCIO_API_KEY}
      CAPISCIO_AGENT_PRIVATE_KEY_JWK: ${AGENT_KEY_JWK}  # from secrets manager
      CAPISCIO_DEV_MODE: "false"
```

```python
# No code changes needed — CapiscioGuard.connect() reads env vars automatically
secured = CapiscioGuard.connect(mode="block") | my_agent
```

> **Warning:** Never bake private keys into container images. Inject them at runtime via environment variables or mounted secrets.

See the [Ephemeral Deployment Guide](https://docs.capisc.io/how-to/security/ephemeral-deployment/) for secrets manager examples and volume-mount alternatives.

## Badge Token Extraction

`CapiscioGuard` extracts the caller's badge token from (in priority order):

1. **Context variable** — set by A2A server middleware via `set_capiscio_context()`
2. **RunnableConfig** — `config={"configurable": {"capiscio_badge": token}}`
3. **Input dict** — `{"capiscio_badge": token, ...}`

For A2A server integrations, set the context at the HTTP perimeter:

```python
from langchain_capiscio import CapiscioRequestContext, set_capiscio_context

set_capiscio_context(CapiscioRequestContext(
    badge_token=badge_jwt,
    caller_did="did:web:caller.example.com",
))
```

## Trust Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Self-Signed (SS) | No external validation, `did:key` issuer |
| 1 | Registered (REG) | Account registration with CapiscIO Registry |
| 2 | Domain Validated (DV) | Domain ownership verified via DNS/HTTP challenge |
| 3 | Organization Validated (OV) | Organization existence verified (DUNS, legal entity) |
| 4 | Extended Validated (EV) | Manual review + legal agreement with CapiscIO |

## API Reference

### Guard

- `CapiscioGuard.connect(api_key, *, mode, name, server_url, dev_mode, **kwargs)` — Connect to registry and return a ready-to-use guard (recommended)
- `CapiscioGuard.from_env(mode, **kwargs)` — Alias for `connect()` (reads env vars)
- `CapiscioGuard(*, identity, config, mode, api_key, name, server_url, connect_kwargs)` — Low-level constructor (keyword-only, lazy init on first invoke)
- `CapiscioGuard.invoke(input, config)` — Verify badge and pass through to downstream
- `CapiscioGuard.ainvoke(input, config)` — Async version

### Callbacks

- `CapiscioCallbackHandler(emitter, identity)` — Emit chain/tool lifecycle events to CapiscIO

### Tool Enforcement

- `CapiscioTool(tool, *, identity=None, config=None, mode="block", api_key=None)` — Wrap a LangChain `Tool` with trust enforcement
- `verify_badge` — `@tool`-decorated function for agent-driven badge verification
- `resolve_agent_card` — `@tool`-decorated function for agent card resolution

### LangGraph

- `@capiscio_guard(mode, identity, config, api_key)` — Decorator for LangGraph function-based nodes

### Context

- `set_capiscio_context(ctx)` — Set request context (badge token, caller DID) for the current invocation
- `get_capiscio_context()` — Retrieve current request context
- `CapiscioRequestContext` — Dataclass holding badge token and caller DID

## Documentation

- [LangChain Integration Guide](https://docs.capisc.io/how-to/integrations/langchain/)
- [Trust Badges Overview](https://docs.capisc.io/concepts/trust-badges/)
- [A2A Protocol](https://github.com/google/A2A)

## Development

```bash
# Clone repository
git clone https://github.com/capiscio/langchain-capiscio.git
cd langchain-capiscio

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run tests with coverage
pytest --cov=langchain_capiscio --cov-report=html
```

## License

Apache License 2.0

## Contributing

Contributions welcome! Please open an issue or pull request on [GitHub](https://github.com/capiscio/langchain-capiscio).
