"""Expected-vs-actual LangGraph trajectory capture for PA-LAW."""
from __future__ import annotations


def expected_nodes(request_type: str, priority: str, complexity: str) -> list[str]:
    nodes = [
        "intake_agent", "classification_agent", "priority_agent",
        "routing_agent", "complexity_agent", "create_ticket_id",
    ]
    if priority in {"URGENT", "CRITICAL"} or complexity == "COMPLEX":
        branch = "legal_escalation_agent"
    elif request_type == "COMPLAINT":
        branch = "complaint_review_agent"
    else:
        branch = "policy_knowledge_agent"
    return nodes + [branch, "persist_ticket"]


def capture_trajectory(graph, initial_state: dict, config: dict | None = None) -> dict:
    """Run a real graph invocation and record the actual node update sequence."""
    actual_nodes: list[str] = []
    final_state = {}
    for update in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in update.items():
            actual_nodes.append(node_name)
            if isinstance(node_output, dict):
                final_state.update(node_output)
    return {"actual_nodes": actual_nodes, "final_state": final_state}


def compare_trajectory(actual_nodes: list[str], expected: list[str]) -> dict:
    return {
        "expected_nodes": expected,
        "actual_nodes": actual_nodes,
        "matches": actual_nodes == expected,
        "missing_nodes": [node for node in expected if node not in actual_nodes],
        "unexpected_nodes": [node for node in actual_nodes if node not in expected],
    }
