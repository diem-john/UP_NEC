# PA-LAW OOP + Dash GUI

This GUI is a new presentation and application layer for the existing PA-LAW implementation in `Palaw_agentic.py`. The original LangGraph graph, agent prompts, routing rules, policy retrieval, escalation behavior, and SQLite ticket format remain the processing source of truth.

## Architecture

- `gui/services.py` — object-oriented application services:
  - `PaLawAgentSuite` exposes the seven agent nodes as class methods and assembles the LangGraph workflow.
  - `PaLawEngine` owns the workflow lifecycle and validates submissions.
  - `TicketRepository` encapsulates ticket queries.
  - `DaskTaskManager` schedules workflow execution asynchronously.
- `gui/app.py` — Dash web interface, persistent chat, submission status polling, ticket dashboard, and ticket lookup.
- `gui/conversations.py` — session-owned SQLite conversation history and the checkpointed conversational LangGraph.
- `gui/assets/style.css` — responsive styling.
- `Palaw_agentic.py` — existing agentic engine; intentionally retained so this GUI does not fork or silently rewrite the established workflow.

Dask uses a local, in-process distributed cluster with worker threads. This keeps the compiled LangGraph engine and Ollama client in the application process instead of attempting to serialize them to separate worker processes. The Dash UI polls Dask futures and remains responsive while a concern is being processed.

## Persistent conversations and isolation

- The **Chat with PA-LAW** tab provides a conversational interface.
- A UUID conversation ID is kept in browser local storage, so a refresh restores the same conversation.
- A signed, persistent Flask session cookie supplies a separate owner ID. Every conversation lookup and message write checks that owner ID; a conversation ID from another browser session is rejected and replaced with a new conversation.
- User and assistant messages are stored in the main SQLite database (`palaw_conversations` and `palaw_messages` tables).
- The Flask signing secret is generated once and persisted in `palaw_app_settings`, so session cookies remain verifiable across server restarts.
- LangGraph's `SqliteSaver` stores graph checkpoints in a separate SQLite file alongside the ticket database. The conversation UUID is passed as LangGraph's `thread_id`, keeping checkpoint state separate per conversation.
- The chat graph restores its prior message state from the checkpoint and also loads the visible transcript from SQLite.
- **New conversation** creates a new ID and therefore a clean context.

This is browser-session isolation for a locally hosted, single-application deployment; it is not a substitute for organization-wide authentication or role-based access control. For a multi-user production deployment, put the app behind authenticated identity (for example, institutional SSO), use a production secret from environment-managed secrets, and deploy a database designed for concurrent multi-user access. Anyone with access to the same browser profile and its local storage/cookies is using the same browser session.

## Setup

Run from the repository root:

```powershell
pip install -r requirements.txt
pip install -r draft/requirements_gui.txt
ollama serve
```

Ensure the configured LLM and embedding models are installed in Ollama. The defaults are defined in `draft/Palaw_agentic.py` and can be overridden with `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_BASE_URL`.

## Tests

Run the service-layer unit tests from the repository root:

```powershell
pytest -q draft/gui/tests
```

These tests cover submission validation and SQLite repository behavior; they do not call Ollama or measure LLM quality.

## Launch

```powershell
python draft/gui/app.py
```

Open http://127.0.0.1:8050.

## Notes

- The ticket dashboard reads the same SQLite database used by the existing workflow.
- Conversation history is stored in the existing SQLite database; LangGraph checkpoints are stored in a sibling `.checkpoints.sqlite` file. Back up both files to preserve tickets, transcripts, and workflow state.
- Complaints and complex/urgent matters retain the existing human-review and escalation paths.
- The supplied company handbook is explicitly a fictional demonstration document. It must not be treated as an approved real-world policy source.
- The GUI does not send emails or create external referrals. Existing legal notifications are simulation-only.
- Dask is used for asynchronous local task scheduling; it does not change model inference or make Ollama itself distributed.
