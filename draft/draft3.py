import sqlite3
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. SYSTEM CONFIGURATION
# ============================================================

DB_NAME = "concern_tickets.db"

# Place the handbook in the same folder as this script.
# Supported: .txt, .md; .pdf and .docx if optional libraries are installed.
HANDBOOK_PATH = "Dummy_Company_Policy_Handbook.md"

LEGAL_CONSULTANTS = [
    {"name": "Legal Consultant 1", "email": "legal1@company.com"},
    {"name": "Legal Consultant 2", "email": "legal2@company.com"},
    {"name": "Legal Consultant 3", "email": "legal3@company.com"},
]

PRIORITY_ORDER = {
    "CRITICAL": 1,
    "URGENT": 2,
    "HIGH": 3,
    "MEDIUM": 4,
    "LOW": 5,
}


# ============================================================
# 2. DEPARTMENT DIRECTORY
# ============================================================

DEPARTMENTS = {
    "HR": {
        "name": "Human Resources",
        "keywords": [
            "employee", "staff", "supervisor", "attendance", "leave",
            "benefits", "harassment", "workplace", "recruitment",
            "termination", "disciplinary action", "employee relations",
            "workplace conflict",
        ],
    },
    "FINANCE": {
        "name": "Finance",
        "keywords": [
            "payroll", "salary discrepancy", "reimbursement", "budget",
            "funds", "financial", "invoice", "accounting", "deduction",
            "unpaid salary", "payment", "allowance", "tax",
        ],
    },
    "IT": {
        "name": "Information Technology",
        "keywords": [
            "computer", "network", "internet", "software", "hardware",
            "password", "system access", "email account", "server",
            "cybersecurity", "data breach", "database", "application",
            "technical issue",
        ],
    },
    "LEGAL": {
        "name": "Legal Department",
        "keywords": [
            "lawsuit", "legal action", "contract dispute", "legal violation",
            "litigation", "regulatory", "legal advice", "legal dispute",
            "court", "legal complaint",
        ],
    },
    "OPERATIONS": {
        "name": "Operations",
        "keywords": [
            "workflow", "daily operations", "production", "service delivery",
            "operational issue", "process", "work procedure",
            "business operations", "service disruption",
        ],
    },
    "ADMIN": {
        "name": "Administration",
        "keywords": [
            "office supplies", "facilities", "maintenance", "office equipment",
            "workspace", "building", "office space", "administrative services",
        ],
    },
    "PROCUREMENT": {
        "name": "Procurement",
        "keywords": [
            "purchase order", "supplier", "vendor", "purchasing", "quotation",
            "procurement", "bidding", "delivery of supplies",
        ],
    },
    "COMPLIANCE": {
        "name": "Compliance",
        "keywords": [
            "policy violation", "fraud", "corruption", "misconduct", "ethics",
            "conflict of interest", "internal investigation", "noncompliance",
            "regulatory compliance",
        ],
    },
    "SAFETY": {
        "name": "Safety and Security",
        "keywords": [
            "accident", "injury", "fire", "emergency", "physical threat",
            "violence", "unsafe", "security", "life-threatening",
            "workplace hazard", "physical assault",
        ],
    },
    "MANAGEMENT": {
        "name": "Management",
        "keywords": [
            "executive decision", "management decision", "organizational policy",
            "strategic decision", "leadership decision", "executive management",
        ],
    },
}


# ============================================================
# 3. LANGGRAPH SHARED STATE
# ============================================================

class ConcernState(TypedDict, total=False):
    # User input
    nature_of_concern: str
    nature_choice: str
    description: str

    # Agent outputs
    classification: str
    priority: str
    complexity: str
    primary_department: str
    additional_departments: str
    routing_status: str

    # Workflow and ticket data
    ticket_id: str
    date_created: str
    status: str
    assigned_to: str
    next_action: str
    draft_answer: Optional[str]
    handbook_excerpt: str
    handbook_found: bool
    legal_recipients: List[Dict[str, str]]
    result_message: str
    saved: bool


# ============================================================
# 4. DATABASE INITIALIZATION AND STORAGE
# ============================================================

def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            date_created TEXT,
            nature_of_concern TEXT,
            classification TEXT,
            description TEXT,
            priority TEXT,
            complexity TEXT,
            status TEXT,
            assigned_to TEXT,
            next_action TEXT,
            draft_answer TEXT
        )
        """
    )

    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(tickets)").fetchall()
    }

    if "primary_department" not in existing_columns:
        conn.execute("ALTER TABLE tickets ADD COLUMN primary_department TEXT")

    if "additional_departments" not in existing_columns:
        conn.execute("ALTER TABLE tickets ADD COLUMN additional_departments TEXT")

    conn.commit()
    conn.close()
    print("[SYSTEM] Database initialized successfully.")


def save_ticket(state: ConcernState) -> bool:
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(
            """
            INSERT INTO tickets (
                ticket_id, date_created, nature_of_concern, classification,
                description, priority, complexity, status, assigned_to,
                next_action, draft_answer, primary_department,
                additional_departments
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["ticket_id"],
                state["date_created"],
                state["nature_of_concern"],
                state["classification"],
                state["description"],
                state["priority"],
                state["complexity"],
                state["status"],
                state["assigned_to"],
                state["next_action"],
                state.get("draft_answer"),
                state["primary_department"],
                state.get("additional_departments", ""),
            ),
        )
        conn.commit()
        print("\n[SYSTEM] Ticket successfully saved.")
        return True
    except sqlite3.Error as error:
        conn.rollback()
        print(f"\n[DATABASE ERROR] {error}")
        return False
    finally:
        conn.close()


# ============================================================
# 5. CONSOLE INPUT
# ============================================================

def get_nature_of_concern():
    options = {
        "1": "Concern About Frontline Staff",
        "2": "Concern About Supervisor",
        "3": "Concern About Executive and Management",
        "4": "Violation of Company Policy",
        "5": "Life-Threatening Concern",
    }

    print("\n============================================")
    print("           NATURE OF CONCERN")
    print("============================================")
    for key, value in options.items():
        print(f"{key}. {value}")

    while True:
        choice = input("\nSelect Nature of Concern (1-5): ").strip()
        if choice in options:
            return options[choice], choice
        print("[ERROR] Invalid selection. Please choose 1-5.")


def get_concern_description():
    print("\n============================================")
    print("           CONCERN DESCRIPTION")
    print("============================================")
    while True:
        description = input("\nDescribe your concern: ").strip()
        if len(description) >= 10:
            return description
        print("[ERROR] Please provide at least 10 characters of detail.")


# ============================================================
# 6. SPECIALIZED AGENT NODES
# ============================================================

def intake_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 1: validates and normalizes the submitted concern."""
    print("\n[AGENT 1 - INTAKE] Concern received.")
    return {
        "description": state["description"].strip(),
        "date_created": datetime.now().isoformat(timespec="seconds"),
        "ticket_id": (
            f"DOLE-{datetime.now().strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        ),
    }


def classification_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 2: classifies the concern based on the selected concern type."""
    mapping = {
        "1": "FRONTLINE_STAFF",
        "2": "SUPERVISOR",
        "3": "MANAGEMENT",
        "4": "VIOLATION",
        "5": "VIOLATION",
    }
    classification = mapping[state["nature_choice"]]
    print(f"[AGENT 2 - CLASSIFICATION] {classification}")
    return {"classification": classification}


def priority_assessment_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 3: assigns priority using the original concern-type rules."""
    mapping = {
        "1": "LOW",
        "2": "MEDIUM",
        "3": "HIGH",
        "4": "URGENT",
        "5": "CRITICAL",
    }
    priority = mapping[state["nature_choice"]]
    print(f"[AGENT 3 - PRIORITY] Assigned priority: {priority}")
    return {"priority": priority}


def department_routing_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 4: routes the concern using base routing and keyword scoring."""
    text = state["description"].lower()

    base_department = {
        "1": "HR",
        "2": "HR",
        "3": "MANAGEMENT",
        "4": "COMPLIANCE",
        "5": "SAFETY",
    }

    scores: Dict[str, int] = {base_department[state["nature_choice"]]: 2}

    for department, details in DEPARTMENTS.items():
        for keyword in details["keywords"]:
            if keyword in text:
                scores[department] = scores.get(department, 0) + 1

    if state["nature_choice"] == "5":
        scores["SAFETY"] = max(scores.get("SAFETY", 0), 3)
        scores["LEGAL"] = max(scores.get("LEGAL", 0), 2)

    sorted_departments = sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )

    primary_code = sorted_departments[0][0]
    additional_codes = [
        code for code, score in sorted_departments[1:] if score > 0
    ]

    primary = DEPARTMENTS[primary_code]["name"]
    additional = [DEPARTMENTS[code]["name"] for code in additional_codes]

    print(f"[AGENT 4 - DEPARTMENT ROUTING] Primary: {primary}")
    print(f"[AGENT 4 - DEPARTMENT ROUTING] Additional: {additional}")

    return {
        "primary_department": primary,
        "additional_departments": ", ".join(additional),
        "routing_status": "AUTO_ROUTED",
    }


def complexity_assessment_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 5: detects concerns requiring qualified human assessment."""
    text = state["description"].lower()

    if state["nature_choice"] == "5":
        complexity = "COMPLEX"
    else:
        complex_keywords = [
            "harassment", "sexual harassment", "discrimination",
            "retaliation", "threat", "violence", "assault", "fraud",
            "corruption", "illegal dismissal", "abuse",
            "conflict of interest", "data breach", "repeated violation",
            "multiple violations", "legal action", "lawsuit",
            "physical injury", "coercion", "blackmail",
        ]

        if any(keyword in text for keyword in complex_keywords):
            complexity = "COMPLEX"
        else:
            issue_indicators = [
                "also", "additionally", "another issue",
                "at the same time", "multiple incidents", "repeatedly",
            ]
            indicator_count = sum(text.count(word) for word in issue_indicators)
            complexity = "COMPLEX" if indicator_count >= 2 else "NOT COMPLEX"

    print(f"[AGENT 5 - COMPLEXITY] {complexity}")
    return {"complexity": complexity}


# ============================================================
# 7. POLICY KNOWLEDGE AGENT / LOCAL HANDBOOK RETRIEVAL
# ============================================================

def load_handbook_text() -> str:
    path = Path(HANDBOOK_PATH)

    if not path.exists():
        return ""

    suffix = path.suffix.lower()

    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                print("[HANDBOOK] Install pypdf to read PDF handbooks.")
                return ""
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if suffix == ".docx":
            try:
                from docx import Document
            except ImportError:
                print("[HANDBOOK] Install python-docx to read DOCX handbooks.")
                return ""
            document = Document(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)

        print(f"[HANDBOOK] Unsupported file type: {suffix}")
        return ""

    except Exception as error:
        print(f"[HANDBOOK ERROR] Could not read handbook: {error}")
        return ""


def retrieve_handbook_excerpt(query: str, handbook_text: str) -> str:
    """
    Lightweight keyword-based retrieval for a dummy handbook.
    This is not embedding-based semantic RAG.
    """
    if not handbook_text.strip():
        return ""

    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", handbook_text)
        if chunk.strip()
    ]

    query_terms = {
        term for term in re.findall(r"[a-zA-Z]{3,}", query.lower())
        if term not in {
            "the", "and", "for", "with", "that", "this", "from",
            "have", "has", "was", "were", "are", "about", "employee",
            "concern", "please",
        }
    }

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for term in query_terms if term in chunk_lower)
        if score:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored[:3]]
    return "\n\n---\n\n".join(selected)


def policy_knowledge_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 6: retrieves relevant handbook text and prepares a draft."""
    print("[AGENT 6 - POLICY KNOWLEDGE] Searching the company handbook...")

    handbook_text = load_handbook_text()
    excerpt = retrieve_handbook_excerpt(
        state["description"], handbook_text
    )

    handbook_found = bool(excerpt)

    if handbook_found:
        policy_section = (
            "Relevant company handbook reference:\n"
            f"{excerpt}\n\n"
            "The reference above was retrieved by keyword matching. "
            "A qualified HR/legal reviewer should verify its applicability."
        )
    elif handbook_text:
        policy_section = (
            "The handbook was loaded, but no sufficiently matching section "
            "was found. This draft does not make a policy determination."
        )
    else:
        policy_section = (
            "The company handbook file was not found or could not be read. "
            "This is an acknowledgment only, not a handbook-based answer. "
            f"Place the handbook beside this script and name it: {HANDBOOK_PATH}"
        )

    draft = f"""
SUBJECT: Acknowledgment of Concern

Dear Employee,

Thank you for bringing this concern to our attention.

We acknowledge receipt of your concern regarding:

Nature of Concern:
{state['nature_of_concern']}

Concern Details:
{state['description']}

Priority Classification:
{state['priority']}

Responsible Department:
{state['primary_department']}

{policy_section}

Your concern will be reviewed in accordance with applicable company policies
and procedures. This preliminary draft is not a final legal determination
or resolution.

Regards,
{state['primary_department']}
""".strip()

    print(
        "[AGENT 6 - POLICY KNOWLEDGE] "
        + ("Relevant handbook text retrieved." if handbook_found
           else "No matching handbook section; acknowledgment draft prepared.")
    )

    return {
        "handbook_excerpt": excerpt,
        "handbook_found": handbook_found,
        "draft_answer": draft,
        "status": "DRAFT_GENERATED",
        "assigned_to": state["primary_department"],
        "next_action": (
            "Review the preliminary draft and verify the cited handbook "
            "section before sending it to the employee."
        ),
    }


# ============================================================
# 8. LEGAL ESCALATION AND HUMAN REVIEW AGENT
# ============================================================

def legal_escalation_agent(state: ConcernState) -> Dict[str, Any]:
    """Agent 7: handles urgent/critical alerts and complex human review."""
    priority = state["priority"]

    if priority == "CRITICAL":
        recipients = LEGAL_CONSULTANTS
        assigned_to = ", ".join(item["name"] for item in recipients)
        status = "LEGAL_ALERTED"
        next_action = (
            "CRITICAL: Notify all legal consultants. Immediate safety and "
            "legal intervention required. Await human/legal assessment."
        )
        result_message = (
            "CRITICAL: All legal consultants must be alerted. "
            "No automated draft answer was generated."
        )

    elif priority == "URGENT":
        recipients = [LEGAL_CONSULTANTS[0]]
        assigned_to = recipients[0]["name"]
        status = "LEGAL_ALERTED"
        next_action = (
            "URGENT: Notify one legal consultant. Await legal assessment "
            "and intervention."
        )
        result_message = (
            "URGENT: One legal consultant must be alerted. "
            "No automated draft answer was generated."
        )

    else:
        recipients = []
        assigned_to = state["primary_department"]
        status = "FOR_HUMAN_RESOLUTION"
        next_action = (
            "Concern is complex. Refer to the responsible department's "
            "qualified human reviewer. No automated draft answer generated."
        )
        result_message = (
            "This concern is complex and requires qualified human review. "
            "No automated draft answer was generated."
        )

    if recipients:
        print("\n============================================")
        print("           LEGAL NOTIFICATION")
        print("============================================")
        for consultant in recipients:
            print("\n*** SIMULATED LEGAL ALERT ***")
            print(f"Recipient: {consultant['name']}")
            print(f"Email: {consultant['email']}")
            print(f"Ticket ID: {state['ticket_id']}")
            print(f"Priority: {priority}")
            print(f"Nature: {state['nature_of_concern']}")
            print("--------------------------------------------")
        print("[NOTICE] Simulation only. No email was sent.")

    return {
        "legal_recipients": recipients,
        "assigned_to": assigned_to,
        "status": status,
        "next_action": next_action,
        "draft_answer": None,
        "result_message": result_message,
    }


def human_review_agent(state: ConcernState) -> Dict[str, Any]:
    """Human-review path for complex but non-urgent concerns."""
    return legal_escalation_agent(state)


# ============================================================
# 9. LANGGRAPH ROUTING AND PERSISTENCE NODES
# ============================================================

def route_after_complexity(state: ConcernState) -> str:
    if state["priority"] in {"URGENT", "CRITICAL"}:
        return "legal_escalation"
    if state["complexity"] == "COMPLEX":
        return "human_review"
    return "policy_knowledge"


def ticket_persistence_node(state: ConcernState) -> Dict[str, Any]:
    """Tool node: persists the completed ticket in SQLite."""
    saved = save_ticket(state)
    if not saved:
        return {
            "saved": False,
            "status": "DATABASE_ERROR",
            "next_action": "Ticket could not be saved. Check the database error.",
        }
    return {"saved": True}


def build_concern_graph():
    workflow = StateGraph(ConcernState)

    workflow.add_node("intake", intake_agent)
    workflow.add_node("classification", classification_agent)
    workflow.add_node("priority", priority_assessment_agent)
    workflow.add_node("department_routing", department_routing_agent)
    workflow.add_node("complexity", complexity_assessment_agent)
    workflow.add_node("policy_knowledge", policy_knowledge_agent)
    workflow.add_node("legal_escalation", legal_escalation_agent)
    workflow.add_node("human_review", human_review_agent)
    workflow.add_node("save_ticket", ticket_persistence_node)

    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "classification")
    workflow.add_edge("classification", "priority")
    workflow.add_edge("priority", "department_routing")
    workflow.add_edge("department_routing", "complexity")

    workflow.add_conditional_edges(
        "complexity",
        route_after_complexity,
        {
            "legal_escalation": "legal_escalation",
            "human_review": "human_review",
            "policy_knowledge": "policy_knowledge",
        },
    )

    workflow.add_edge("policy_knowledge", "save_ticket")
    workflow.add_edge("legal_escalation", "save_ticket")
    workflow.add_edge("human_review", "save_ticket")
    workflow.add_edge("save_ticket", END)

    return workflow.compile()


# ============================================================
# 10. TICKET DISPLAY AND DATABASE VIEWING
# ============================================================

def display_ticket_result(ticket: ConcernState):
    print("\n============================================")
    print("         CONCERN PROCESSING RESULT")
    print("============================================")
    print(f"Ticket ID: {ticket.get('ticket_id')}")
    print(f"Date Created: {ticket.get('date_created')}")
    print(f"Nature: {ticket.get('nature_of_concern')}")
    print(f"Classification: {ticket.get('classification')}")
    print(f"Description: {ticket.get('description')}")
    print(f"Priority: {ticket.get('priority')}")
    print(f"Complexity: {ticket.get('complexity')}")
    print(f"Primary Department: {ticket.get('primary_department')}")
    print(f"Additional Departments: {ticket.get('additional_departments')}")
    print(f"Status: {ticket.get('status')}")
    print(f"Assigned To: {ticket.get('assigned_to')}")
    print(f"Next Action: {ticket.get('next_action')}")

    if not ticket.get("saved"):
        print("\n[ERROR] Ticket was not saved. Please check the database.")
        return

    if ticket.get("priority") == "CRITICAL":
        print("\n============================================")
        print("          CRITICAL LEGAL ALERT")
        print("============================================")
        print("All legal consultants have been alerted in this simulation.")
        print("No automated draft answer was generated.")
        print("Immediate safety and legal intervention is required.")

    elif ticket.get("priority") == "URGENT":
        print("\n============================================")
        print("           URGENT LEGAL ALERT")
        print("============================================")
        print("One legal consultant has been alerted in this simulation.")
        print("No automated draft answer was generated.")

    elif ticket.get("complexity") == "COMPLEX":
        print("\n============================================")
        print("          HUMAN RESOLUTION REQUIRED")
        print("============================================")
        print("The ticket has been saved for qualified human review.")
        print("No automated draft answer was generated.")

    else:
        print("\n============================================")
        print("          GENERATED DRAFT ANSWER")
        print("============================================")
        print(ticket.get("draft_answer") or "No draft answer available.")

    print("\n============================================")
    print("TICKET SUBMITTED SUCCESSFULLY")
    print("The ticket details above have been saved.")
    print("Returning to the main menu...\n")


def view_sorted_tickets():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    tickets = conn.execute(
        """
        SELECT * FROM tickets
        ORDER BY
            CASE priority
                WHEN 'CRITICAL' THEN 1
                WHEN 'URGENT' THEN 2
                WHEN 'HIGH' THEN 3
                WHEN 'MEDIUM' THEN 4
                WHEN 'LOW' THEN 5
                ELSE 6
            END ASC,
            date_created ASC
        """
    ).fetchall()
    conn.close()

    print("\n============================================")
    print("          PRIORITY-SORTED TICKETS")
    print("============================================")

    if not tickets:
        print("No tickets found.")
        return

    for ticket in tickets:
        print(f"\nTicket ID: {ticket['ticket_id']}")
        print(f"Date: {ticket['date_created']}")
        print(f"Nature: {ticket['nature_of_concern']}")
        print(f"Priority: {ticket['priority']}")
        print(f"Complexity: {ticket['complexity']}")
        print(f"Primary Department: {ticket['primary_department']}")
        print(f"Additional Departments: {ticket['additional_departments']}")
        print(f"Status: {ticket['status']}")
        print(f"Assigned To: {ticket['assigned_to']}")
        print("--------------------------------------------")


def view_ticket():
    ticket_id = input("\nEnter Ticket ID: ").strip()

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    ticket = conn.execute(
        "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    conn.close()

    if not ticket:
        print("[ERROR] Ticket not found.")
        return

    print("\n============================================")
    print("             TICKET DETAILS")
    print("============================================")
    for key in ticket.keys():
        print(f"{key}: {ticket[key]}")

    if ticket["draft_answer"]:
        print("\n============================================")
        print("             DRAFT ANSWER")
        print("============================================")
        print(ticket["draft_answer"])
    else:
        print("\nNo automated draft answer available.")
        print("This ticket requires human/legal intervention.")


# ============================================================
# 11. MAIN CONCERN SUBMISSION
# ============================================================

def concern_management_tool(graph):
    nature, choice = get_nature_of_concern()
    description = get_concern_description()

    initial_state: ConcernState = {
        "nature_of_concern": nature,
        "nature_choice": choice,
        "description": description,
    }

    print("\n============================================")
    print("       LANGGRAPH WORKFLOW EXECUTION")
    print("============================================")

    try:
        result = graph.invoke(initial_state)
        display_ticket_result(result)
        return result
    except Exception as error:
        print(f"\n[WORKFLOW ERROR] {error}")
        print(
            "If the error mentions LangGraph, install dependencies using "
            "'pip install -r requirements.txt'."
        )
        return None


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():
    initialize_database()
    concern_graph = build_concern_graph()

    while True:
        print("\n============================================")
        print("          DOLE POLICY ADVISER")
        print("       LANGGRAPH CONCERN MANAGEMENT")
        print("============================================")
        print("1. Submit New Concern")
        print("2. View Priority-Sorted Tickets")
        print("3. View Individual Ticket")
        print("4. Exit")

        action = input("\nSelect Action: ").strip()

        if action == "1":
            concern_management_tool(concern_graph)
        elif action == "2":
            view_sorted_tickets()
        elif action == "3":
            view_ticket()
        elif action == "4":
            print("\nExiting DOLE Policy Adviser.")
            break
        else:
            print("\n[ERROR] Invalid selection. Please choose 1-4.")


if __name__ == "__main__":
    main()
