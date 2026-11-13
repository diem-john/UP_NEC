"""Backend-enforced guardrails for PA-LAW.

The LLM is never an authorization boundary. Call these checks in application
services before protected operations and before policy-answer generation.
"""
from __future__ import annotations

ALLOWED_ROLES = {"EMPLOYEE", "REVIEWER", "ADMIN"}
ROLE_PERMISSIONS = {
    "EMPLOYEE": {"submit_concern", "view_own_ticket"},
    "REVIEWER": {"submit_concern", "view_own_ticket", "view_all_tickets", "review_ticket"},
    "ADMIN": {"submit_concern", "view_own_ticket", "view_all_tickets", "review_ticket", "manage_users"},
}


def validate_concern_input(nature_choice: str, description: str) -> tuple[str, str]:
    """Validate untrusted concern input before it reaches LangGraph."""
    choice = str(nature_choice or "").strip()
    text = str(description or "").strip()
    if choice not in {"1", "2", "3", "4", "5", "6"}:
        raise ValueError("Invalid nature-of-concern selection.")
    if len(text) < 10:
        raise ValueError("Description must contain at least 10 characters.")
    if len(text) > 10000:
        raise ValueError("Description must not exceed 10,000 characters.")
    return choice, text


def require_permission(role: str, permission: str) -> None:
    """Fail closed when the caller has no recognized role or permission."""
    normalized = str(role or "").upper()
    if normalized not in ALLOWED_ROLES or permission not in ROLE_PERMISSIONS[normalized]:
        raise PermissionError(f"Role is not authorized to perform: {permission}")


def evidence_is_sufficient(retrieved: list[dict], min_score: float = 0.15) -> bool:
    """Require at least one non-empty evidence chunk with a usable relevance score."""
    if not retrieved:
        return False
    for item in retrieved:
        text = str(item.get("text", "")).strip()
        score = item.get("score")
        if text and isinstance(score, (int, float)) and score >= min_score:
            return True
    return False


def format_untrusted_evidence(retrieved: list[dict]) -> str:
    """Clearly delimit retrieved handbook content as data, never instructions."""
    blocks = []
    for item in retrieved:
        source = str(item.get("source", "Unknown source"))
        body = str(item.get("text", "")).strip()
        if body:
            blocks.append(f"<handbook_excerpt source={source!r}>\n{body}\n</handbook_excerpt>")
    return "\n\n".join(blocks)
