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


class PaLawEngine:
    """Application service that owns the LangGraph workflow lifecycle."""
    def __init__(self, repository: TicketRepository | None = None,
                 initialize_knowledge: bool = True) -> None:
        self.repository = repository or TicketRepository()
        legacy.initialize_database()
        if initialize_knowledge:
            # Existing RAG layer falls back to keyword retrieval if embeddings fail.
            legacy.load_handbook_and_index()
        self.graph = legacy.build_agentic_graph()

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
