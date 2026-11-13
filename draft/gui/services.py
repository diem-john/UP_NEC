"""Object-oriented application services for the PA-LAW Dash interface."""
from __future__ import annotations

import sqlite3
import sys
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

DRAFT_DIR = Path(__file__).resolve().parents[1]
if str(DRAFT_DIR) not in sys.path:
    sys.path.insert(0, str(DRAFT_DIR))
import Palaw_agentic as legacy  # noqa: E402


class TicketRepository:
    """Owns SQLite access for concern tickets."""
    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path or legacy.DB_NAME)

    def list_priority_sorted(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """SELECT ticket_id, date_created, classification, priority,
                          complexity, primary_department, status, assigned_to
                   FROM tickets ORDER BY CASE priority
                       WHEN 'CRITICAL' THEN 1 WHEN 'URGENT' THEN 2
                       WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 4
                       WHEN 'LOW' THEN 5 ELSE 6 END, date_created ASC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, ticket_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.strip(),)
            ).fetchone()
        return dict(row) if row else None


class PaLawAgentSuite:
    """Class-based adapters for each specialized LangGraph agent node.

    Agent behavior is delegated to the established, tested implementation in
    Palaw_agentic.py; this class owns node composition and graph construction.
    """
    def intake(self, state):
        return legacy.intake_agent(state)

    def classify(self, state):
        return legacy.classification_agent(state)

    def assess_priority(self, state):
        return legacy.priority_assessment_agent(state)

    def route_department(self, state):
        return legacy.department_routing_agent(state)

    def assess_complexity(self, state):
        return legacy.complexity_assessment_agent(state)

    def create_ticket_id(self, state):
        return legacy.create_ticket_id_node(state)

    def retrieve_policy(self, state):
        return legacy.policy_knowledge_agent(state)

    def escalate(self, state):
        return legacy.legal_escalation_agent(state)

    def route_complaint_for_review(self, state):
        return legacy.complaint_human_review_agent(state)

    def persist_ticket(self, state):
        return legacy.persist_ticket_node(state)

    def build_graph(self):
        from langgraph.graph import END, START, StateGraph
        graph = StateGraph(legacy.ConcernState)
        graph.add_node("intake_agent", self.intake)
        graph.add_node("classification_agent", self.classify)
        graph.add_node("priority_agent", self.assess_priority)
        graph.add_node("routing_agent", self.route_department)
        graph.add_node("complexity_agent", self.assess_complexity)
        graph.add_node("create_ticket_id", self.create_ticket_id)
        graph.add_node("policy_knowledge_agent", self.retrieve_policy)
        graph.add_node("legal_escalation_agent", self.escalate)
        graph.add_node("complaint_review_agent", self.route_complaint_for_review)
        graph.add_node("persist_ticket", self.persist_ticket)
        graph.add_edge(START, "intake_agent")
        graph.add_edge("intake_agent", "classification_agent")
        graph.add_edge("classification_agent", "priority_agent")
        graph.add_edge("priority_agent", "routing_agent")
        graph.add_edge("routing_agent", "complexity_agent")
        graph.add_edge("complexity_agent", "create_ticket_id")
        graph.add_conditional_edges(
            "create_ticket_id", legacy.route_after_complexity,
            {
                "legal_escalation": "legal_escalation_agent",
                "complaint_review": "complaint_review_agent",
                "policy_knowledge": "policy_knowledge_agent",
            },
        )
        graph.add_edge("legal_escalation_agent", "persist_ticket")
        graph.add_edge("complaint_review_agent", "persist_ticket")
        graph.add_edge("policy_knowledge_agent", "persist_ticket")
        graph.add_edge("persist_ticket", END)
        return graph.compile()


class PaLawEngine:
    """Application service that owns the LangGraph workflow lifecycle."""
    def __init__(self, repository: TicketRepository | None = None,
                 initialize_knowledge: bool = True) -> None:
        self.repository = repository or TicketRepository()
        legacy.initialize_database()
        if initialize_knowledge:
            # Existing RAG layer falls back to keyword retrieval if embeddings fail.
            legacy.load_handbook_and_index()
        self.agents = PaLawAgentSuite()\n        self.graph = self.agents.build_graph()

    @staticmethod
    def validate_submission(nature_choice: str, description: str) -> tuple[str, str]:
        choice, text = str(nature_choice).strip(), str(description or "").strip()
        if choice not in legacy.NATURE_OPTIONS:
            raise ValueError("Select a valid nature of concern.")
        if len(text) < 10:
            raise ValueError("Please describe the concern using at least 10 characters.")
        if len(text) > 10000:
            raise ValueError("The description must not exceed 10,000 characters.")
        return choice, text

    def process_concern(self, nature_choice: str, description: str) -> dict[str, Any]:
        choice, text = self.validate_submission(nature_choice, description)
        initial_state: legacy.ConcernState = {
            "nature_choice": choice,
            "nature_of_concern": legacy.NATURE_OPTIONS[choice],
            "description": text,
        }
        return dict(self.graph.invoke(initial_state))


# Dask uses an in-process LocalCluster. Keeping the engine in this process avoids
# serializing the compiled LangGraph graph or opening a second copy of Ollama.
_SHARED_ENGINE: PaLawEngine | None = None

def _run_engine_task(nature_choice: str, description: str) -> dict[str, Any]:
    if _SHARED_ENGINE is None:
        raise RuntimeError("PA-LAW engine has not been initialized.")
    return _SHARED_ENGINE.process_concern(nature_choice, description)


class DaskTaskManager:
    """Runs graph invocations asynchronously using Dask Distributed."""
    def __init__(self, engine: PaLawEngine, workers: int = 2) -> None:
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as exc:
            raise RuntimeError(
                "Dask Distributed is missing. Install draft/requirements_gui.txt."
            ) from exc
        global _SHARED_ENGINE
        _SHARED_ENGINE = engine
        self.cluster = LocalCluster(
            n_workers=max(1, workers), threads_per_worker=1,
            processes=False, dashboard_address=None,
        )
        self.client = Client(self.cluster, set_as_default=False)
        self._futures: dict[str, Any] = {}
        self._lock = RLock()

    def submit(self, nature_choice: str, description: str) -> str:
        choice, text = PaLawEngine.validate_submission(nature_choice, description)
        job_id = uuid.uuid4().hex
        future = self.client.submit(
            _run_engine_task, choice, text,
            key=f"palaw-concern-{job_id}", pure=False,
        )
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return {"state": "UNKNOWN", "message": "Processing task was not found."}
        if future.status == "finished":
            try:
                return {"state": "FINISHED", "result": future.result()}
            except Exception as exc:
                return {"state": "ERROR", "message": str(exc)}
        if future.status in {"error", "cancelled"}:
            try:
                future.result()
            except Exception as exc:
                return {"state": "ERROR", "message": str(exc)}
        return {"state": "RUNNING", "message": "PA-LAW agents are processing the concern."}

    def close(self) -> None:
        self.client.close()
        self.cluster.close()
