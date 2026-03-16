# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-16

### Added
- **`CapiscioGuard`**: LCEL runnable for trust badge enforcement in LangChain chains (#1, #5)
  - Typed passthrough preserving input type through the guard
  - Zero-config initialization via `from_env()`
- **LangChain Callback Handler**: Trust context propagation through LangChain callbacks (#1)
- **Trust-Aware Tool Wrapper**: Wrap LangChain tools with CapiscIO badge enforcement (#1)
- **Context Management**: `set_capiscio_context` / `get_capiscio_context` utilities (#1, #3)
- **PyPI Publish Workflow**: Automated publishing on tag push (#2, #4)
