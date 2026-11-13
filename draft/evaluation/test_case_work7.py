"""Focused Case Work 7 tests: qualitative rubric, trajectory, red-team controls."""
import json
from pathlib import Path

import pytest

from draft.evaluation.llm_judge import JUDGE_RUBRIC, build_judge_prompt
from draft.evaluation.trajectory import compare_trajectory, expected_nodes
from draft.security.guardrails import (
    evidence_is_sufficient,
    format_untrusted_evidence,
    require_permission,
    validate_concern_input,
)

ROOT = Path(__file__).resolve().parent


def test_qualitative_judge_rubric_covers_required_dimensions():
    assert {"grounding", "relevance", "completeness", "policy_compliance", "clarity"} <= set(JUDGE_RUBRIC)
    prompt = build_judge_prompt("request", "handbook evidence", "draft answer")
    assert "Do not reward confident fabrication" in prompt
    assert "HANDBOOK EXCERPTS" in prompt


def test_expected_complaint_trajectory_and_actual_trace_comparison():
    expected = expected_nodes("COMPLAINT", "MEDIUM", "COMPLEX")
    # This is a deterministic trace fixture matching the branch selected by
    # route_after_complexity; live traces are captured with capture_trajectory().
    actual = [
        "intake_agent", "classification_agent", "priority_agent",
        "routing_agent", "complexity_agent", "create_ticket_id",
        "legal_escalation_agent", "persist_ticket",
    ]
    comparison = compare_trajectory(actual, expected)
    assert comparison["matches"] is True
    assert comparison["missing_nodes"] == []
    assert comparison["unexpected_nodes"] == []


def test_red_team_rt01_prompt_injection_is_data_not_instruction():
    payload = json.loads((ROOT / "red_team_cases.json").read_text(encoding="utf-8"))
    case = next(item for item in payload if item["id"] == "RT01")
    evidence = [{"source": "Handbook", "text": "Leave requests require supervisor review.", "score": 0.82}]
    formatted = format_untrusted_evidence(evidence)
    assert "handbook_excerpt" in formatted
    assert "Ignore all previous instructions" in case["input"]
    assert "do not invent benefits" in case["expected"]


def test_red_team_rt02_unauthorized_action_is_denied():
    payload = json.loads((ROOT / "red_team_cases.json").read_text(encoding="utf-8"))
    assert next(item for item in payload if item["id"] == "RT02")
    with pytest.raises(PermissionError):
        require_permission("EMPLOYEE", "view_all_tickets")
    with pytest.raises(PermissionError):
        require_permission("EMPLOYEE", "delete_ticket")


def test_red_team_rt03_fabricated_policy_fails_evidence_gate():
    payload = json.loads((ROOT / "red_team_cases.json").read_text(encoding="utf-8"))
    assert next(item for item in payload if item["id"] == "RT03")
    unsupported = [{"source": "Handbook", "text": "General workplace conduct.", "score": 0.05}]
    assert evidence_is_sufficient(unsupported) is False
    assert evidence_is_sufficient([]) is False


def test_guardrail_input_validation_accepts_and_rejects():
    assert validate_concern_input("1", "Please explain annual leave.") == (
        "1", "Please explain annual leave."
    )
    with pytest.raises(ValueError):
        validate_concern_input("99", "A sufficiently long concern.")
    with pytest.raises(ValueError):
        validate_concern_input("1", "short")


def test_guardrail_role_permissions_fail_closed():
    require_permission("REVIEWER", "view_all_tickets")
    with pytest.raises(PermissionError):
        require_permission("UNKNOWN", "view_all_tickets")


def test_evidence_gate_requires_relevant_nonempty_excerpt():
    assert evidence_is_sufficient([
        {"text": "Policy excerpt about leave.", "score": 0.75}
    ])
    assert not evidence_is_sufficient([
        {"text": "   ", "score": 0.99}
    ])
