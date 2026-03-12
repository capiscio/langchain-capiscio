"""LangChain Tool wrappers for CapiscIO badge verification and agent card validation.

Tertiary integration — convenience tools for demos and agent-driven trust checks.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def verify_badge(
    token: str,
    trusted_issuers: list[str] | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    """Verify a CapiscIO trust badge token.

    Returns a dict with 'valid', 'claims', 'error', and 'warnings' keys.

    Args:
        token: The badge JWT token to verify.
        trusted_issuers: Optional list of trusted issuer DIDs.
        audience: Optional expected audience claim.
    """
    from capiscio_sdk.badge import VerifyResult
    from capiscio_sdk.badge import verify_badge as sdk_verify_badge

    kwargs: dict[str, Any] = {}
    if trusted_issuers:
        kwargs["trusted_issuers"] = trusted_issuers
    if audience:
        kwargs["audience"] = audience

    result: VerifyResult = sdk_verify_badge(token, **kwargs)
    output: dict[str, Any] = {
        "valid": result.valid,
        "warnings": result.warnings,
    }
    if result.error:
        output["error"] = result.error
    if result.claims:
        output["claims"] = {
            "issuer": result.claims.issuer,
            "subject": result.claims.subject,
            "trust_level": str(result.claims.trust_level.value),
            "domain": result.claims.domain,
            "agent_name": result.claims.agent_name,
            "expires_at": result.claims.expires_at.isoformat(),
        }
    return output


@tool
async def resolve_agent_card(agent_url: str) -> dict[str, Any]:
    """Fetch and validate an A2A Agent Card by URL.

    Returns a dict with 'success', 'issues', and 'metadata' keys.

    Args:
        agent_url: The URL of the agent to validate (e.g. https://agent.example.com).
    """
    from capiscio_sdk.validators import CoreValidator

    validator = CoreValidator()
    result = await validator.fetch_and_validate(agent_url)
    output: dict[str, Any] = {
        "success": result.success,
        "issues": [
            {
                "severity": issue.severity.value,
                "code": issue.code,
                "message": issue.message,
            }
            for issue in result.issues
        ],
    }
    if result.metadata:
        output["metadata"] = result.metadata
    return output
