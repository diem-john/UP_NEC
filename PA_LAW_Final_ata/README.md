# PA-LAW UI + Agentic Workflow

## Folder contents
- `app.py` — Dash web UI
- `services.py` — adapter that connects the UI to `Palaw_agentic.py`
- `Palaw_agentic.py` — your supplied DOLE/PA-LAW multi-agent LangGraph system
- `conversations.py` — persistent chat and handbook-grounded chat graph
- `assets/style.css` — supplied UI styling
- `requirements_gui.txt` — Python dependencies

## Important handbook file
Place `PA_LAW_Dummy_Company_Policy_Handbook.md` in this same folder.
The agent code expects this exact filename. Without it, the concern workflow can
still start but handbook retrieval will have no policy content.

## Setup (Windows PowerShell)
1. Open PowerShell in this folder.
2. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements_gui.txt
   ```
4. Ensure Ollama is running and pull the configured models:
   ```powershell
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
5. Start the UI:
   ```powershell
   python app.py
   ```
6. Open `http://127.0.0.1:8050`.

## Model configuration
Defaults are read by `Palaw_agentic.py`:
- `OLLAMA_BASE_URL` defaults to `http://localhost:11434`
- `OLLAMA_MODEL` defaults to `llama3.2:3b`
- `OLLAMA_EMBEDDING_MODEL` defaults to `nomic-embed-text`

Override in PowerShell before launching if needed:
```powershell
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
python app.py
```

## Notes
- The UI uses the same SQLite ticket database (`concern_tickets.db`) as the agent file.
- Chat history and LangGraph chat checkpoints are stored in SQLite.
- The agentic workflow remains the source of truth; the adapter invokes
  `build_agentic_graph()` directly rather than recreating its nodes.
- Legal alerts in the supplied agent are simulations only; no email is sent.
- AI outputs are preliminary. Complaints, complex, urgent, and critical matters
  remain subject to human review.
