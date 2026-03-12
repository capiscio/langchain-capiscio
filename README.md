# langchain-capiscio

Trust enforcement adapter for [LangChain](https://python.langchain.com/) and
[LangGraph](https://langchain-ai.github.io/langgraph/) — powered by
[CapiscIO](https://capisc.io).

Verify agent trust badges, enforce security policies, and emit audit events —
all composable via LangChain's LCEL pipe (`|`) operator.

## Install

```bash
pip install langchain-capiscio
```

## Quick Start

```python
from langchain_capiscio import CapiscioGuard

# Guard before chain execution — reads CAPISCIO_API_KEY from env
secured = CapiscioGuard() | my_chain
result = secured.invoke({"input": "Summarize this ticket"})
```

## Features

- **CapiscioGuard** — `Runnable[dict, dict]` that verifies trust badges before
  downstream execution. Composable with `|`.
- **CapiscioTool** — Enforce trust before individual tool calls.
- **CapiscioCallbackHandler** — Audit trail via CapiscIO EventEmitter.
- **@capiscio_guard** — Decorator for LangGraph function-based nodes.

## Enforcement Modes

```python
from langchain_capiscio import CapiscioGuard

guard = CapiscioGuard(mode="block")    # Fail closed (production default)
guard = CapiscioGuard(mode="monitor")  # Warn but continue
guard = CapiscioGuard(mode="log")      # Log only
```

## LangGraph

```python
from langchain_capiscio import CapiscioGuard, capiscio_guard

# Option 1: Runnable as graph node
graph.add_node("verify", CapiscioGuard())

# Option 2: Decorator
@capiscio_guard(mode="block")
def call_agent(state: dict) -> dict:
    ...
```

## License

Apache-2.0
