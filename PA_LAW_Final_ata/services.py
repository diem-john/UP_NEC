"""Integration adapter between the Dash UI and the PA-LAW agentic workflow."""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from threading import RLock
from typing import Any

import Palaw_agentic as legacy


class TicketRepository:
    def __init__(self, database_path: str | Path | None = None):
        self.database_path = Path(database_path or legacy.DB_NAME)

    def list_priority_sorted(self) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.database_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT ticket_id, date_created, classification, priority,
                       complexity, primary_department, status, assigned_to
                FROM tickets
                ORDER BY CASE priority
                    WHEN 'CRITICAL' THEN 1 WHEN 'URGENT' THEN 2
                    WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 4
                    WHEN 'LOW' THEN 5 ELSE 6 END, date_created ASC
            """).fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, ticket_id: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.database_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.strip(),)
            ).fetchone()
        return dict(row) if row else None


class PaLawEngine:
    """Owns the target multi-agent graph, database, and chat checkpoint."""
    def __init__(self, repository: TicketRepository | None = None,
                 initialize_knowledge: bool = True):
        self.repository = repository or TicketRepository()
        legacy.initialize_database()
        if initialize_knowledge:
            legacy.load_handbook_and_index()

        self.graph = legacy.build_agentic_graph()

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "Missing LangGraph SQLite checkpoint support. "
                "Install requirements_gui.txt."
            ) from exc

        checkpoint_path = Path(str(legacy.DB_NAME) + ".checkpoints.sqlite")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_context = SqliteSaver.from_conn_string(str(checkpoint_path))
        self.checkpointer = self._checkpointer_context.__enter__()
        self.checkpointer.setup()

        try:
            from conversations import ConversationRepository, build_chat_graph
        except ImportError:
            from .conversations import ConversationRepository, build_chat_graph
        self.conversation_repository = ConversationRepository(legacy.DB_NAME)
        self.chat_graph = build_chat_graph(self.checkpointer)

    @staticmethod
    def validate_submission(nature_choice: str, description: str):
        choice = str(nature_choice or "").strip()
        text = (description or "").strip()
        if choice not in legacy.NATURE_OPTIONS:
            raise ValueError("Choose a valid nature of concern.")
        if len(text) < 10:
            raise ValueError("Please provide at least 10 characters.")
        if len(text) > 10000:
            raise ValueError("Concern descriptions must not exceed 10,000 characters.")
        return choice, text

    def process_concern(self, nature_choice: str, description: str,
                        conversation_id: str | None = None) -> dict[str, Any]:
        choice, text = self.validate_submission(nature_choice, description)
        initial_state = {
            "nature_choice": choice,
            "nature_of_concern": legacy.NATURE_OPTIONS[choice],
            "description": text,
        }
        # The agentic graph is the exact graph defined by the supplied target file.
        # Each submission receives its own thread identifier.
        thread_id = conversation_id or str(uuid.uuid4())
        return dict(self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        ))

    def close(self):
        context = getattr(self, "_checkpointer_context", None)
        if context:
            context.__exit__(None, None, None)
            self._checkpointer_context = None


_SHARED_ENGINE: PaLawEngine | None = None


def _run_engine_task(nature_choice: str, description: str,
                     conversation_id: str) -> dict[str, Any]:
    if _SHARED_ENGINE is None:
        raise RuntimeError("PA-LAW engine has not been initialized.")
    return _SHARED_ENGINE.process_concern(nature_choice, description, conversation_id)


class DaskTaskManager:
    """Run long Ollama/LangGraph submissions asynchronously."""
    def __init__(self, engine: PaLawEngine, workers: int = 2):
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as exc:
            raise RuntimeError(
                "Dask Distributed is missing. Install requirements_gui.txt."
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
            _run_engine_task, choice, text, job_id,
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
                result = future.result()
                with self._lock:
                    self._futures.pop(job_id, None)
                return {"state": "FINISHED", "result": result}
            except Exception as exc:
                with self._lock:
                    self._futures.pop(job_id, None)
                return {"state": "ERROR", "message": str(exc)}
        if future.status in {"error", "cancelled"}:
            try:
                future.result()
            except Exception as exc:
                with self._lock:
                    self._futures.pop(job_id, None)
                return {"state": "ERROR", "message": str(exc)}
        return {"state": "RUNNING", "message": "PA-LAW agents are processing the concern."}

    def close(self):
        self.client.close()
        self.cluster.close()
