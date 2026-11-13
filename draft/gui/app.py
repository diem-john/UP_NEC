"""Dash GUI for PA-LAW. Run: python draft/gui/app.py"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from flask import session
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update

GUI_DIR = Path(__file__).resolve().parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))
from services import DaskTaskManager, PaLawEngine, TicketRepository  # noqa: E402
from services import legacy  # noqa: E402
from conversations import ConversationService  # noqa: E402

engine = PaLawEngine()
repository: TicketRepository = engine.repository
task_manager = DaskTaskManager(engine)
conversation_service = ConversationService(engine.conversation_repository, engine.chat_graph)

NATURE_OPTIONS = [{"label": value, "value": key} for key, value in legacy.NATURE_OPTIONS.items()]
app = Dash(__name__, title="PA-LAW | Policy Adviser", suppress_callback_exceptions=True)
server = app.server
server.secret_key = engine.conversation_repository.get_or_create_secret()
server.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=365),
)


def current_owner_id() -> str:
    session.permanent = True
    if "palaw_owner_id" not in session:
        import uuid
        session["palaw_owner_id"] = uuid.uuid4().hex
    return session["palaw_owner_id"]


def render_chat_messages(messages):
    if not messages:
        return [html.Div([
            html.H3("Welcome to PA-LAW"),
            html.P("Ask a question about workplace concerns, labor policies, or the next steps for a concern."),
            html.P("Your conversation is saved locally and restored when you return.", className="muted"),
        ], className="chat-empty")]
    return [
        html.Div([
            html.Div("You" if item["role"] == "user" else "PA-LAW", className="chat-role"),
            html.Div(item["content"], className="chat-bubble-content"),
            html.Div(item["created_at"], className="chat-timestamp"),
        ], className=f"chat-message {item['role']}")
        for item in messages
    ]

app.layout = html.Div(className="app-shell", children=[
    dcc.Store(id="active-job", storage_type="memory"),
    dcc.Store(id="conversation-id", storage_type="local"),
    dcc.Interval(id="job-poll", interval=1200, n_intervals=0, disabled=True),
    html.Aside(className="sidebar", children=[
        html.Div([html.Div("⚖", className="brand-mark"), html.H2("PA-LAW"),
                  html.P("Policy Assistant for Labor and Welfare")], className="brand"),
        html.Div("WORKSPACE", className="nav-label"),
        dcc.Tabs(id="main-tabs", value="submit", vertical=True, className="nav-tabs", children=[
            dcc.Tab(label="Chat with PA-LAW", value="chat"),
            dcc.Tab(label="Submit a concern", value="submit"),
            dcc.Tab(label="Ticket dashboard", value="dashboard"),
            dcc.Tab(label="Find a ticket", value="search"),
            dcc.Tab(label="About PA-LAW", value="about"),
        ]),
        html.Div("LOCAL AI • HUMAN REVIEW", className="sidebar-foot"),
    ]),
    html.Main(className="main-area", children=[
        html.Div([html.Div("EMPLOYEE SUPPORT", className="eyebrow"),
                  html.H1("Concern Management Portal"),
                  html.P("Submit a workplace concern and let PA-LAW coordinate the review workflow.")], className="page-header"),
        html.Div(id="page-content"),
    ]),
])


def card(title: str, body, subtitle: str | None = None):
    children = [html.H3(title)]
    if subtitle:
        children.append(html.P(subtitle, className="muted"))
    children.append(body)
    return html.Section(children, className="panel")


def page_layout(tab: str):
    if tab == "chat":
        return html.Div([
            dcc.Interval(id="chat-init", interval=150, max_intervals=1, n_intervals=0),
            card("Conversation with PA-LAW", html.Div([
                html.Div(id="chat-history", className="chat-history"),
                html.Div([
                    dcc.Textarea(id="chat-input", className="chat-input",
                                 placeholder="Type your message…", maxLength=10000),
                    html.Div([
                        html.Button("Send message", id="send-chat", n_clicks=0, className="primary-button"),
                        html.Button("New conversation", id="new-chat", n_clicks=0, className="secondary-button"),
                    ], className="chat-actions"),
                    html.Div(id="chat-feedback", className="feedback"),
                ], className="chat-composer"),
            ]), "Your conversation is stored in the local SQLite database and isolated by a signed browser session.")
        ])
    if tab == "submit":
        return html.Div([
            card("Submit a new concern", html.Div([
                html.Label("Nature of concern", htmlFor="nature-choice"),
                dcc.Dropdown(id="nature-choice", options=NATURE_OPTIONS, placeholder="Choose the closest category", clearable=False),
                html.Label("Describe what happened", htmlFor="concern-description"),
                dcc.Textarea(id="concern-description", className="concern-input", placeholder="Include relevant details, dates, and what assistance you need…", maxLength=10000),
                html.Div("Minimum 10 characters. Do not enter passwords, bank details, or one-time codes.", className="helper-text"),
                html.Button("Submit concern", id="submit-concern", n_clicks=0, className="primary-button"),
                html.Div(id="submit-feedback", className="feedback"),
            ]), "Your submission is processed locally through the configured Ollama and LangGraph workflow."),
            html.Div(id="processing-result"),
        ])
    if tab == "dashboard":
        return html.Div([
            html.Div([html.Button("Refresh tickets", id="refresh-tickets", className="secondary-button")], className="toolbar"),
            html.Div(id="ticket-summary", className="metric-grid"),
            card("Priority-sorted tickets", html.Div(id="ticket-table")),
        ])
    if tab == "search":
        return html.Div([card("Find a ticket", html.Div([
            html.Label("Ticket ID", htmlFor="ticket-id"),
            dcc.Input(id="ticket-id", type="text", placeholder="PA-LAW-YYYYMMDD-XXXXXX", className="text-input"),
            html.Button("Search", id="search-ticket", n_clicks=0, className="primary-button"),
            html.Div(id="ticket-detail", className="ticket-detail"),
        ]))])
    return html.Div([card("About PA-LAW", html.Div([
        html.P("PA-LAW is a local, agentic policy-assistance proof of concept. It uses a LangGraph workflow with specialized agents for intake, classification, priority assessment, department routing, complexity assessment, policy retrieval, and escalation."),
        html.Ul([html.Li("Dash provides the browser-based user interface."), html.Li("Dask Distributed runs workflows asynchronously and the interface polls their status."), html.Li("Ollama runs the configured local language and embedding models."), html.Li("SQLite stores concern tickets.")]),
        html.P("AI-generated drafts are preliminary. Complaints and complex or urgent concerns require authorized human review. The demo handbook is fictional; verify all policies before real deployment.", className="notice"),
    ]))])


@app.callback(Output("page-content", "children"), Input("main-tabs", "value"))
def render_page(tab):
    return page_layout(tab)



@app.callback(
    Output("conversation-id", "data"), Output("chat-history", "children"),
    Input("chat-init", "n_intervals"), State("conversation-id", "data"),
    prevent_initial_call=True,
)
def restore_conversation(_ticks, stored_id):
    owner_id = current_owner_id()
    conversation_id, messages = conversation_service.restore(stored_id, owner_id)
    return conversation_id, render_chat_messages(messages)


@app.callback(
    Output("conversation-id", "data", allow_duplicate=True),
    Output("chat-history", "children", allow_duplicate=True),
    Output("chat-input", "value"),
    Output("chat-feedback", "children"),
    Input("send-chat", "n_clicks"), Input("new-chat", "n_clicks"),
    State("chat-input", "value"), State("conversation-id", "data"),
    prevent_initial_call=True,
)
def handle_chat_actions(_send_clicks, _new_clicks, message, stored_id):
    owner_id = current_owner_id()
    triggered = callback_context.triggered_id
    if triggered == "new-chat":
        conversation_id, messages = conversation_service.restore(None, owner_id)
        return conversation_id, render_chat_messages(messages), "", html.Div("Started a new private conversation.", className="status-message")
    if triggered == "send-chat":
        try:
            conversation_id, _ = conversation_service.restore(stored_id, owner_id)
            messages = conversation_service.send(conversation_id, owner_id, message or "")
            return conversation_id, render_chat_messages(messages), "", html.Div("Response saved.", className="success-message")
        except Exception as exc:
            return no_update, no_update, no_update, html.Div(str(exc), className="error-message")
    return no_update, no_update, no_update, no_update


@app.callback(
    Output("active-job", "data"), Output("job-poll", "disabled"),
    Output("submit-feedback", "children"), Output("processing-result", "children"),
    Input("submit-concern", "n_clicks"),
    State("nature-choice", "value"), State("concern-description", "value"),
    prevent_initial_call=True,
)
def submit_concern(n_clicks, nature_choice, description):
    if not n_clicks:
        return no_update, True, no_update, no_update
    try:
        job_id = task_manager.submit(nature_choice or "", description or "")
    except Exception as exc:
        return None, True, html.Div(str(exc), className="error-message"), no_update
    return job_id, False, html.Div("Concern accepted. The agent workflow is running…", className="status-message"), html.Div([
        html.Div(className="spinner"), html.P("Processing your concern. This may take a little while.")
    ], className="processing-panel")


@app.callback(
    Output("job-poll", "disabled", allow_duplicate=True),
    Output("active-job", "data", allow_duplicate=True),
    Output("submit-feedback", "children", allow_duplicate=True),
    Output("processing-result", "children", allow_duplicate=True),
    Input("job-poll", "n_intervals"), State("active-job", "data"),
    prevent_initial_call=True,
)
def poll_concern(_ticks, job_id):
    if not job_id:
        return True, None, no_update, no_update
    status = task_manager.status(job_id)
    if status["state"] == "RUNNING":
        return False, job_id, html.Div(status["message"], className="status-message"), no_update
    if status["state"] == "ERROR":
        return True, None, html.Div(status["message"], className="error-message"), no_update
    if status["state"] != "FINISHED":
        return True, None, html.Div(status["message"], className="error-message"), no_update
    result = status["result"]
    details = [
        ("Ticket ID", result.get("ticket_id")), ("Request type", result.get("request_type")),
        ("Nature", result.get("nature_of_concern")), ("Priority", result.get("priority")),
        ("Complexity", result.get("complexity")), ("Primary department", result.get("primary_department")),
        ("Additional departments", ", ".join(result.get("additional_departments", []))),
        ("Status", result.get("status")), ("Assigned to", result.get("assigned_to")),
        ("Ticket saved", "Yes" if result.get("saved") else "No"),
    ]
    blocks = [html.Div([html.Span(label), html.Strong(value if value not in (None, "") else "—")], className="result-item") for label, value in details]
    if result.get("draft_answer"):
        blocks.append(html.Div([html.H4("Preliminary draft answer"), html.Pre(result["draft_answer"])], className="draft-answer"))
    if result.get("handbook_sources"):
        blocks.append(html.Div([html.H4("Retrieved handbook sources"), html.Ul([html.Li(x) for x in result["handbook_sources"]])], className="source-list"))
    if result.get("handbook_excerpt"):
        blocks.append(html.Details([html.Summary("View retrieved policy evidence"), html.Pre(result["handbook_excerpt"])], className="evidence"))
    if result.get("result_message"):
        blocks.append(html.P(result["result_message"], className="notice"))
    blocks.append(html.P("AI-generated output is preliminary. It is not legal advice or an approved company decision.", className="disclaimer"))
    return True, None, html.Div("Workflow completed.", className="success-message"), card("Concern processing result", html.Div(blocks, className="result-grid"))


@app.callback(
    Output("ticket-summary", "children"), Output("ticket-table", "children"),
    Input("main-tabs", "value"), Input("refresh-tickets", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_dashboard(tab, _clicks):
    if tab != "dashboard":
        return no_update, no_update
    rows = repository.list_priority_sorted()
    priorities = ["CRITICAL", "URGENT", "HIGH", "MEDIUM", "LOW"]
    metrics = [html.Div([html.Span(label), html.Strong(str(value))], className="metric-card")
               for label, value in [("Total tickets", len(rows)), *[(p, sum(r.get("priority") == p for r in rows)) for p in priorities]]]
    table = dash_table.DataTable(
        data=rows, columns=[{"name": key.replace("_", " ").title(), "id": key} for key in
            ["ticket_id", "date_created", "classification", "priority", "complexity", "primary_department", "status", "assigned_to"]],
        page_size=10, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Inter, sans-serif", "fontSize": "13px", "padding": "12px", "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "700", "backgroundColor": "#F1F5F9"},
    ) if rows else html.P("No tickets have been submitted yet.", className="muted")
    return metrics, table


@app.callback(Output("ticket-detail", "children"), Input("search-ticket", "n_clicks"), State("ticket-id", "value"), prevent_initial_call=True)
def search_ticket(_clicks, ticket_id):
    if not ticket_id or not ticket_id.strip():
        return html.P("Enter a ticket ID.", className="error-message")
    ticket = repository.get_by_id(ticket_id)
    if not ticket:
        return html.P("No ticket found with that ID.", className="muted")
    blocks = [html.Div([html.Span(key.replace("_", " ").title()), html.Strong(str(value) if value is not None else "—")], className="result-item") for key, value in ticket.items()]
    if ticket.get("draft_answer"):
        blocks.append(html.Div([html.H4("Draft answer"), html.Pre(ticket["draft_answer"])], className="draft-answer"))
    return html.Div(blocks, className="result-grid")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
