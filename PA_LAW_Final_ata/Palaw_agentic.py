"""
DOLE POLICY ADVISER - AGENTIC AI MULTI-AGENT SYSTEM
===================================================
Architecture:
  Orchestrator: LangGraph StateGraph
  Local LLM: Ollama / Llama 3.2
  Specialized LangGraph nodes:
    Intake, Classification, Priority, Department Routing, Complexity,
    Task Planner, Policy Knowledge/RAG, Follow-up, Legal Escalation,
    Complaint Review, and Ticket Persistence

Console-only application. SQLite ticket storage.
Legal alerts are simulated; no email is sent.

IMPORTANT:
- The supplied policy handbook is fictional and for demonstration only.
- RAG answers are drafts, not legal advice or approved company decisions.
- Complaints, complex matters, and urgent/critical matters require human review.
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = BASE_DIR / "concern_tickets.db"
HANDBOOK_PATH = BASE_DIR / "PA_LAW_Dummy_Company_Policy_Handbook.md"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

LEGAL_CONSULTANTS = [
    {"name": "Legal Consultant 1", "email": "legal1@company.com"},
    {"name": "Legal Consultant 2", "email": "legal2@company.com"},
    {"name": "Legal Consultant 3", "email": "legal3@company.com"},
]

NATURE_OPTIONS = {
    "1": "Concern About Frontline Staff",
    "2": "Concern About Supervisor",
    "3": "Concern About Executive and Management",
    "4": "Violation of Company Policy",
    "5": "Life-Threatening Concern",
}

# Preserve the priority mapping from the original implementation.
PRIORITY_BY_NATURE = {
    "1": "LOW",
    "2": "MEDIUM",
    "3": "HIGH",
    "4": "URGENT",
    "5": "CRITICAL",
}

DEPARTMENTS = {
    "HR": {
        "name": "Human Resources",
        "keywords": [
            "employee", "staff", "supervisor", "attendance", "leave",
            "benefits", "harassment", "workplace", "recruitment",
            "termination", "disciplinary action", "employee relations",
            "workplace conflict", "unfair treatment",
        ],
    },
    "FINANCE": {
        "name": "Finance",
        "keywords": [
            "payroll", "salary", "reimbursement", "budget", "funds",
            "financial", "invoice", "accounting", "deduction",
            "unpaid salary", "payment", "allowance", "tax", "payslip",
        ],
    },
    "IT": {
        "name": "Information Technology",
        "keywords": [
            "computer", "network", "internet", "software", "hardware",
            "password", "system access", "email account", "server",
            "cybersecurity", "data breach", "database", "application",
            "technical issue", "phishing",
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
            "business operations", "service disruption", "work assignment",
        ],
    },
    "ADMIN": {
        "name": "Administration",
        "keywords": [
            "office supplies", "facilities", "maintenance", "office equipment",
            "workspace", "building", "office space", "air conditioner",
        ],
    },
    "PROCUREMENT": {
        "name": "Procurement",
        "keywords": [
            "purchase order", "supplier", "vendor", "purchasing",
            "quotation", "procurement", "bidding", "delivery of supplies",
        ],
    },
    "COMPLIANCE": {
        "name": "Compliance",
        "keywords": [
            "policy violation", "fraud", "corruption", "misconduct",
            "ethics", "conflict of interest", "internal investigation",
            "noncompliance", "regulatory compliance",
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
            "executive decision", "management decision",
            "organizational policy", "strategic decision",
            "leadership decision", "executive management",
        ],
    },
}


# ============================================================
# 2. STRUCTURED OUTPUT SCHEMAS
# ============================================================

class IntakeResult(BaseModel):
    summary: str = Field(description="Short neutral summary of the concern")
    key_facts: List[str] = Field(description="Facts explicitly stated by the employee")
    missing_information: List[str] = Field(
        description="Important information that may need human follow-up"
    )


class ClassificationResult(BaseModel):
    request_type: Literal["INQUIRY", "COMPLAINT"] = Field(
        description="INQUIRY asks for information or policy guidance. COMPLAINT reports dissatisfaction, wrongdoing, harm, or an alleged incident."
    )
    reason: str


class DepartmentResult(BaseModel):
    primary_department: str = Field(
        description="One department code from the supplied department list"
    )
    additional_departments: List[str] = Field(
        description="Other relevant department codes, if any"
    )
    reason: str


class ComplexityResult(BaseModel):
    complexity: Literal["NOT COMPLEX", "COMPLEX"]
    reason: str
    requires_human_review: bool


class PlannerResult(BaseModel):
    action: Literal["POLICY_LOOKUP", "REQUEST_INFORMATION", "HUMAN_REVIEW", "LEGAL_ESCALATION"]
    selected_tools: List[Literal["handbook_search", "create_followup", "department_referral", "legal_alert"]]
    reason: str


class EvidenceReviewResult(BaseModel):
    sufficient: bool
    missing_topics: List[str]
    refined_query: str
    reason: str


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class ConcernState(TypedDict, total=False):
    nature_choice: str
    nature_of_concern: str
    description: str

    intake_summary: str
    key_facts: List[str]
    missing_information: List[str]

    request_type: str
    classification_reason: str

    priority: str
    primary_department: str
    additional_departments: List[str]
    routing_reason: str

    complexity: str
    complexity_reason: str

    handbook_excerpt: str
    handbook_sources: List[str]
    rag_used: bool
    draft_answer: Optional[str]

    ticket_id: str
    date_created: str
    status: str
    assigned_to: str
    next_action: str
    legal_recipients: List[Dict[str, str]]
    saved: bool
    result_message: str
    error: str
    planned_action: str
    planner_reason: str
    selected_tools: List[str]
    rag_attempts: int
    evidence_sufficient: bool
    rag_feedback: str
    follow_up_question: str
    follow_up_id: str
    approval_status: str
    approved_by: str
    approved_at: str
    review_notes: str


# ============================================================
# 4. LLM AND EMBEDDING CLIENTS
# ============================================================

def build_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=8192,
    )


def build_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


llm = build_llm()
embeddings = build_embeddings()


def invoke_structured(schema, system_prompt: str, user_prompt: str):
    """Invoke Ollama and return a validated Pydantic object."""
    structured_llm = llm.with_structured_output(schema, method="json_schema")
    return structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])


# ============================================================
# 5. DATABASE
# ============================================================

def initialize_database():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
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
                draft_answer TEXT,
                primary_department TEXT,
                additional_departments TEXT,
                intake_summary TEXT,
                handbook_sources TEXT,
                approval_status TEXT DEFAULT 'NOT_REQUIRED',
                approved_by TEXT,
                approved_at TEXT,
                review_notes TEXT,
                rag_attempts INTEGER DEFAULT 0,
                selected_tools TEXT,
                follow_up_question TEXT
            )
        """)

        existing = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(tickets)"
            ).fetchall()
        }

        migrations = {
            "primary_department": "TEXT",
            "additional_departments": "TEXT",
            "intake_summary": "TEXT",
            "handbook_sources": "TEXT",
            "approval_status": "TEXT DEFAULT 'NOT_REQUIRED'",
            "approved_by": "TEXT",
            "approved_at": "TEXT",
            "review_notes": "TEXT",
            "rag_attempts": "INTEGER DEFAULT 0",
            "selected_tools": "TEXT",
            "follow_up_question": "TEXT",
        }

        for column, column_type in migrations.items():
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE tickets ADD COLUMN {column} {column_type}"
                )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS followups (
                follow_up_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                response TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """)

    print("[DATABASE] SQLite database initialized.")


def save_ticket(state: ConcernState) -> bool:
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("""
                INSERT INTO tickets (
                    ticket_id, date_created, nature_of_concern,
                    classification, description, priority, complexity,
                    status, assigned_to, next_action, draft_answer,
                    primary_department, additional_departments,
                    intake_summary, handbook_sources, approval_status,
                    approved_by, approved_at, review_notes, rag_attempts,
                    selected_tools, follow_up_question
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state["ticket_id"],
                state["date_created"],
                state["nature_of_concern"],
                state["request_type"],
                state["description"],
                state["priority"],
                state["complexity"],
                state["status"],
                state["assigned_to"],
                state["next_action"],
                state.get("draft_answer"),
                state["primary_department"],
                ", ".join(state.get("additional_departments", [])),
                state.get("intake_summary", ""),
                json.dumps(state.get("handbook_sources", [])),
                state.get("approval_status", "NOT_REQUIRED"),
                state.get("approved_by"),
                state.get("approved_at"),
                state.get("review_notes", ""),
                int(state.get("rag_attempts", 0)),
                json.dumps(state.get("selected_tools", [])),
                state.get("follow_up_question", ""),
            ))
        print("[DATABASE] Ticket saved successfully.")
        return True
    except sqlite3.Error as exc:
        print(f"[DATABASE ERROR] {exc}")
        return False


# ============================================================
# 6. HANDBOOK SEMANTIC RAG
# ============================================================

HANDBOOK_CHUNKS: List[Dict[str, str]] = []
HANDBOOK_VECTORS: Optional[np.ndarray] = None


def split_handbook_into_chunks(text: str, max_chars: int = 1100):
    """Split markdown while retaining section headings as source labels."""
    chunks = []
    current_heading = "General"
    buffer = []

    for line in text.splitlines():
        if line.strip().startswith("#"):
            if buffer:
                content = "\n".join(buffer).strip()
                if content:
                    chunks.append((current_heading, content))
                buffer = []
            current_heading = line.strip("# ").strip()
        else:
            buffer.append(line)

    if buffer:
        content = "\n".join(buffer).strip()
        if content:
            chunks.append((current_heading, content))

    final_chunks = []
    for heading, content in chunks:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            candidate = (current + "\n\n" + paragraph).strip()
            if len(candidate) > max_chars and current:
                final_chunks.append({
                    "source": f"PA-LAW Dummy Company Policy Handbook — {heading}",
                    "text": current,
                })
                current = paragraph
            else:
                current = candidate
        if current:
            final_chunks.append({
                "source": f"PA-LAW Dummy Company Policy Handbook — {heading}",
                "text": current,
            })
    return final_chunks


def load_handbook_and_index():
    global HANDBOOK_CHUNKS, HANDBOOK_VECTORS

    if not HANDBOOK_PATH.exists():
        print(f"[RAG WARNING] Handbook not found: {HANDBOOK_PATH.name}")
        print("[RAG WARNING] Place the supplied handbook beside this script.")
        return

    text = HANDBOOK_PATH.read_text(encoding="utf-8")
    HANDBOOK_CHUNKS = split_handbook_into_chunks(text)

    if not HANDBOOK_CHUNKS:
        print("[RAG WARNING] No handbook chunks were extracted.")
        return

    print(f"[RAG] Loaded {len(HANDBOOK_CHUNKS)} handbook chunks.")
    print(f"[RAG] Creating embeddings with {EMBEDDING_MODEL}...")

    try:
        texts = [item["text"] for item in HANDBOOK_CHUNKS]
        vectors = embeddings.embed_documents(texts)
        HANDBOOK_VECTORS = np.asarray(vectors, dtype=np.float32)
        print("[RAG] Semantic handbook index is ready.")
    except Exception as exc:
        HANDBOOK_VECTORS = None
        print(f"[RAG WARNING] Embedding initialization failed: {exc}")
        print("[RAG WARNING] Semantic RAG is unavailable; keyword fallback will be used.")


def retrieve_handbook(query: str, top_k: int = 4):
    if not HANDBOOK_CHUNKS:
        return []

    if HANDBOOK_VECTORS is not None:
        try:
            query_vector = np.asarray(
                embeddings.embed_query(query), dtype=np.float32
            )
            denom = (
                np.linalg.norm(HANDBOOK_VECTORS, axis=1)
                * np.linalg.norm(query_vector)
            )
            denom = np.where(denom == 0, 1e-10, denom)
            scores = (HANDBOOK_VECTORS @ query_vector) / denom
            indices = np.argsort(scores)[::-1][:top_k]
            return [
                {
                    **HANDBOOK_CHUNKS[int(i)],
                    "score": float(scores[int(i)]),
                }
                for i in indices
                if float(scores[int(i)]) > 0.15
            ]
        except Exception as exc:
            print(f"[RAG WARNING] Semantic retrieval failed: {exc}")

    # Transparent fallback, used only if local embedding retrieval is unavailable.
    terms = {
        t for t in re.findall(r"[a-zA-Z]{3,}", query.lower())
        if t not in {"the", "and", "for", "with", "that", "this", "from", "about"}
    }
    ranked = []
    for chunk in HANDBOOK_CHUNKS:
        body = chunk["text"].lower()
        score = sum(body.count(term) for term in terms)
        if score:
            ranked.append((score, chunk))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [
        {**chunk, "score": float(score)}
        for score, chunk in ranked[:top_k]
    ]


# ============================================================
# APPROVED TOOL REGISTRY — AGENTS MAY SELECT ONLY THESE TOOLS
# ============================================================

def tool_handbook_search(query: str):
    return retrieve_handbook(query, top_k=5)


def tool_create_followup(ticket_id: str, question: str):
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.execute(
            "INSERT INTO followups (ticket_id, question, status, created_at) VALUES (?, ?, 'PENDING', ?)",
            (ticket_id, question, datetime.now().isoformat(timespec="seconds")),
        )
        return int(cur.lastrowid)


def tool_department_referral(state: ConcernState):
    return f"Refer to {state.get('primary_department', 'the responsible department')} for human assessment."


def tool_legal_alert(state: ConcernState):
    # Simulation only; this does not send email.
    return legal_escalation_agent(state)


def execute_legal_alert_tool(state: ConcernState):
    print("[TOOL SELECTION] Executing approved tool: legal_alert")
    return APPROVED_TOOLS["legal_alert"](state)


APPROVED_TOOLS = {
    "handbook_search": tool_handbook_search,
    "create_followup": tool_create_followup,
    "department_referral": tool_department_referral,
    "legal_alert": tool_legal_alert,
}


def review_retrieved_evidence(question: str, evidence: str) -> EvidenceReviewResult:
    prompt = f"""Employee question:
{question}

Retrieved handbook evidence:
{evidence}

Assess whether the evidence directly and sufficiently answers the question. If not, identify missing topics and provide a better retrieval query. Do not answer the employee."""
    return invoke_structured(
        EvidenceReviewResult,
        "You are an evidence-checking agent. Be conservative. Evidence is sufficient only when the supplied handbook passages directly support an answer.",
        prompt,
    )




# ============================================================
# 7. AGENT 1 — INTAKE AGENT
# ============================================================

def intake_agent(state: ConcernState) -> Dict[str, Any]:
    print("\n[AGENT 1 - INTAKE] Understanding the submitted concern...")

    prompt = f"""
Nature selected by employee: {state['nature_of_concern']}
Employee description:
{state['description']}

Summarize only facts present in the message. Do not invent facts.
Identify missing information that a human reviewer may need.
"""
    try:
        result = invoke_structured(
            IntakeResult,
            "You are the Intake Agent of a company concern management system. "
            "Be neutral, concise, and do not make legal conclusions.",
            prompt,
        )
        return {
            "intake_summary": result.summary,
            "key_facts": result.key_facts,
            "missing_information": result.missing_information,
        }
    except Exception as exc:
        print(f"[AGENT 1 WARNING] LLM unavailable; using safe fallback: {exc}")
        return {
            "intake_summary": state["description"][:500],
            "key_facts": [],
            "missing_information": [],
            "error": f"Intake LLM fallback: {exc}",
        }


# ============================================================
# 8. AGENT 2 — CLASSIFICATION AGENT
# ============================================================

def classification_agent(state: ConcernState) -> Dict[str, Any]:
    print("[AGENT 2 - CLASSIFICATION] Classifying inquiry or complaint...")

    prompt = f"""
Employee concern:
{state['description']}

Classify as:
- INQUIRY: asks a question, requests information, or seeks policy guidance.
- COMPLAINT: reports dissatisfaction, wrongdoing, unfair treatment, harm,
  an incident, alleged violation, or asks the organization to investigate.

When uncertain, choose COMPLAINT so a human can review it.
"""
    try:
        result = invoke_structured(
            ClassificationResult,
            "You are a careful request classification agent. "
            "Return exactly one supported classification.",
            prompt,
        )
        return {
            "request_type": result.request_type,
            "classification_reason": result.reason,
        }
    except Exception as exc:
        text = state["description"].lower()
        complaint_terms = [
            "complain", "report", "harass", "unfair", "abuse", "threat",
            "discriminat", "retaliat", "violation", "mistreat", "wrongdoing",
            "stole", "fraud", "assault", "shout", "humiliat",
        ]
        request_type = (
            "COMPLAINT"
            if any(term in text for term in complaint_terms)
            else "INQUIRY"
        )
        print(f"[AGENT 2 WARNING] Keyword fallback used: {exc}")
        return {
            "request_type": request_type,
            "classification_reason": "Fallback classification; verify manually.",
        }


# ============================================================
# 9. AGENT 3 — PRIORITY ASSESSMENT AGENT
# ============================================================

def priority_assessment_agent(state: ConcernState) -> Dict[str, Any]:
    print("[AGENT 3 - PRIORITY] Applying the approved nature-to-priority rules...")

    priority = PRIORITY_BY_NATURE[state["nature_choice"]]
    print(f"[AGENT 3 - PRIORITY] Assigned priority: {priority}")

    return {"priority": priority}


# ============================================================
# 10. AGENT 4 — DEPARTMENT ROUTING AGENT
# ============================================================

def department_routing_agent(state: ConcernState) -> Dict[str, Any]:
    print("[AGENT 4 - ROUTING] Selecting responsible department...")

    department_list = {
        code: details["name"] for code, details in DEPARTMENTS.items()
    }

    prompt = f"""
Concern category: {state['nature_of_concern']}
Concern description: {state['description']}
Available department codes and names:
{json.dumps(department_list, indent=2)}

Choose one primary department code and any relevant additional department
codes. Use only the supplied codes. If the concern is ambiguous, use HR
or MANAGEMENT as the primary department and explain why.
"""
    try:
        result = invoke_structured(
            DepartmentResult,
            "You are the Department Routing Agent. Route concerns to the "
            "most appropriate organizational department. Do not invent departments.",
            prompt,
        )
        primary = result.primary_department.upper().strip()
        additional = [x.upper().strip() for x in result.additional_departments]

        if primary not in DEPARTMENTS:
            raise ValueError(f"Invalid primary department: {primary}")

        additional = [
            x for x in additional
            if x in DEPARTMENTS and x != primary
        ]

        return {
            "primary_department": DEPARTMENTS[primary]["name"],
            "additional_departments": [
                DEPARTMENTS[x]["name"] for x in additional
            ],
            "routing_reason": result.reason,
        }

    except Exception as exc:
        print(f"[AGENT 4 WARNING] Rule-based routing fallback: {exc}")
        text = state["description"].lower()
        scores = {}
        for code, details in DEPARTMENTS.items():
            score = sum(
                1 for keyword in details["keywords"] if keyword in text
            )
            if score:
                scores[code] = score

        # Nature-based fallback when no keywords match.
        if not scores:
            fallback = {
                "1": "HR",
                "2": "HR",
                "3": "MANAGEMENT",
                "4": "COMPLIANCE",
                "5": "SAFETY",
            }
            scores[fallback[state["nature_choice"]]] = 1

        ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        primary = ordered[0][0]
        additional = [code for code, _ in ordered[1:]]

        return {
            "primary_department": DEPARTMENTS[primary]["name"],
            "additional_departments": [
                DEPARTMENTS[x]["name"] for x in additional
            ],
            "routing_reason": "Keyword and nature-based fallback routing.",
        }


# ============================================================
# 11. AGENT 5 — COMPLEXITY ASSESSMENT AGENT
# ============================================================

def complexity_assessment_agent(state: ConcernState) -> Dict[str, Any]:
    print("[AGENT 5 - COMPLEXITY] Assessing whether human review is required...")

    # Life-threatening concerns always require human intervention.
    if state["nature_choice"] == "5":
        return {
            "complexity": "COMPLEX",
            "complexity_reason": "Life-threatening concerns always require human intervention.",
        }

    prompt = f"""
Concern:
{state['description']}

Classify as COMPLEX if it involves serious allegations, harassment,
discrimination, retaliation, threats, violence, assault, fraud, corruption,
illegal dismissal, abuse, blackmail, coercion, legal action, data breach,
multiple incidents, or facts requiring investigation.

Use NOT COMPLEX only for straightforward requests that can safely be
answered from a company handbook. Do not decide whether an allegation is true.
"""
    try:
        result = invoke_structured(
            ComplexityResult,
            "You are a risk and complexity assessment agent. "
            "Err on the side of human review. Never make legal findings.",
            prompt,
        )
        complexity = result.complexity
        if result.requires_human_review:
            complexity = "COMPLEX"

        return {
            "complexity": complexity,
            "complexity_reason": result.reason,
        }

    except Exception as exc:
        print(f"[AGENT 5 WARNING] Conservative keyword fallback: {exc}")
        complex_terms = [
            "harassment", "sexual harassment", "discrimination",
            "retaliation", "threat", "violence", "assault", "fraud",
            "corruption", "illegal dismissal", "abuse", "data breach",
            "lawsuit", "blackmail", "coercion", "repeated violation",
        ]
        text = state["description"].lower()
        complex_case = any(term in text for term in complex_terms)
        return {
            "complexity": "COMPLEX" if complex_case else "NOT COMPLEX",
            "complexity_reason": "Fallback assessment; human verification advised.",
        }


# ============================================================
# AGENT 6 — TASK PLANNER / DYNAMIC DELEGATION
# ============================================================

def task_planner_agent(state: ConcernState) -> Dict[str, Any]:
    print("[PLANNER AGENT] Selecting the next specialist and approved tools...")
    # Mandatory guardrails override any model-selected action.
    if state.get("priority") in {"URGENT", "CRITICAL"}:
        action, tools = "LEGAL_ESCALATION", ["legal_alert", "department_referral"]
        reason = "Priority policy requires legal escalation."
    elif state.get("request_type") == "COMPLAINT" or state.get("complexity") == "COMPLEX":
        action, tools = "HUMAN_REVIEW", ["department_referral"]
        reason = "Complaints and complex concerns require authorized human review."
    else:
        prompt = f"""Concern: {state['description']}
Intake summary: {state.get('intake_summary','')}
Missing information: {state.get('missing_information', [])}

Choose the next action. Use REQUEST_INFORMATION only when essential facts are missing and cannot be responsibly answered from the handbook. Otherwise choose POLICY_LOOKUP. Select only tools from the supplied allowlist. Do not bypass complaint, complexity, or priority safeguards. Allowed tools: handbook_search, create_followup, department_referral, legal_alert."""
        try:
            result = invoke_structured(
                PlannerResult,
                "You are the task-planning and delegation agent. Select the most suitable next specialist and only approved tools. Never authorize a final legal decision.",
                prompt,
            )
            action = result.action
            tools = list(dict.fromkeys(result.selected_tools))
            reason = result.reason
            if action == "LEGAL_ESCALATION":
                action, tools = "HUMAN_REVIEW", ["department_referral"]
            if action == "POLICY_LOOKUP" and "handbook_search" not in tools:
                tools.append("handbook_search")
            if action == "REQUEST_INFORMATION" and "create_followup" not in tools:
                tools.append("create_followup")
        except Exception as exc:
            action, tools = "POLICY_LOOKUP", ["handbook_search"]
            reason = f"Planner fallback to bounded handbook lookup: {exc}"
    print(f"[PLANNER AGENT] Delegated action: {action}; tools: {', '.join(tools)}")
    return {"planned_action": action, "selected_tools": tools, "planner_reason": reason}


def followup_agent(state: ConcernState) -> Dict[str, Any]:
    missing = state.get("missing_information", [])
    question = "Please provide the following information so the responsible department can review your concern: " + "; ".join(missing[:5])
    if not missing:
        question = "Please provide any relevant dates, people or departments involved, and supporting details needed to review your concern."
    return {
        "follow_up_question": question,
        "status": "WAITING_FOR_EMPLOYEE_INFORMATION",
        "assigned_to": state.get("primary_department", "HR"),
        "next_action": "A follow-up request has been recorded. Await the employee's response before proceeding.",
        "approval_status": "NOT_REQUIRED",
        "result_message": "The Planner delegated this concern to the Follow-up Agent because key information is missing.",
    }


# ============================================================
# 12. AGENT 6 — POLICY KNOWLEDGE AGENT (RAG)
# ============================================================

def policy_knowledge_agent(state: ConcernState) -> Dict[str, Any]:
    print("[AGENT 7 - POLICY KNOWLEDGE] Executing selected handbook-search tool...")
    query = state.get("intake_summary", "") + "\n" + state["description"]
    retrieved = []
    feedback = ""
    evidence_ok = False
    attempts = 0

    # Bounded feedback loop: at most 2 retrieval/evaluation attempts.
    for attempts in range(1, 3):
        retrieved = tool_handbook_search(query)
        if not retrieved:
            feedback = "No handbook passages matched the query."
            continue
        evidence = "\n\n".join(
            f"SOURCE: {item['source']}\n{item['text']}" for item in retrieved
        )
        try:
            review = review_retrieved_evidence(state["description"], evidence)
            evidence_ok = bool(review.sufficient)
            feedback = review.reason
            if evidence_ok:
                break
            if review.refined_query.strip():
                query = review.refined_query.strip()
        except Exception as exc:
            # If the evaluator cannot run, fail closed instead of assuming evidence is enough.
            feedback = f"Evidence evaluator unavailable: {exc}"
            evidence_ok = False
            break

    sources = list(dict.fromkeys(item["source"] for item in retrieved))
    excerpt = "\n\n".join(
        f"[{item['source']}]\n{item['text']}" for item in retrieved
    )

    if not retrieved or not evidence_ok:
        return {
            "handbook_excerpt": excerpt,
            "handbook_sources": sources,
            "rag_used": bool(retrieved),
            "rag_attempts": attempts,
            "evidence_sufficient": False,
            "rag_feedback": feedback,
            "draft_answer": None,
            "status": "FOR_DEPARTMENT_REVIEW",
            "assigned_to": state["primary_department"],
            "next_action": "The bounded RAG feedback loop could not verify sufficient handbook evidence. Refer to the responsible department.",
            "approval_status": "NOT_REQUIRED",
            "result_message": "Evidence remained insufficient after retrieval attempts; no answer was generated.",
        }

    prompt = f"""Employee inquiry:
{state['description']}

Responsible department: {state['primary_department']}

Evidence reviewed as sufficient:
{excerpt}

Draft a concise answer using ONLY these passages. Cite the relevant handbook section names. If a detail is not supported, say it is not specified. Do not state fictional policy as Philippine law. End by saying this is a preliminary draft requiring authorized approval."""
    try:
        response = llm.invoke([
            SystemMessage(content="You are the Policy Knowledge Agent. Use only the supplied evidence. Do not fabricate or make legal determinations."),
            HumanMessage(content=prompt),
        ])
        draft = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        draft = None
        feedback = f"Draft generation failed: {exc}"

    return {
        "handbook_excerpt": excerpt,
        "handbook_sources": sources,
        "rag_used": True,
        "rag_attempts": attempts,
        "evidence_sufficient": True,
        "rag_feedback": feedback,
        "draft_answer": draft,
        "status": "DRAFT_PENDING_APPROVAL" if draft else "FOR_DEPARTMENT_REVIEW",
        "assigned_to": state["primary_department"],
        "approval_status": "PENDING" if draft else "NOT_REQUIRED",
        "next_action": "An authorized department reviewer must approve the draft before it can be released." if draft else "Refer to the responsible department because draft generation failed.",
        "result_message": "The evidence-checking agent validated the retrieved passages. The draft is awaiting human approval." if draft else "No draft was generated; human review is required.",
    }


# ============================================================
# 13. AGENT 7 — LEGAL ESCALATION / HUMAN REVIEW AGENT
# ============================================================

def legal_escalation_agent(state: ConcernState) -> Dict[str, Any]:
    print("[AGENT 7 - LEGAL ESCALATION] Evaluating escalation path...")

    priority = state["priority"]
    recipients = []

    if priority == "CRITICAL":
        recipients = LEGAL_CONSULTANTS
        status = "LEGAL_ALERTED"
        assigned_to = ", ".join(x["name"] for x in recipients)
        next_action = (
            "CRITICAL: Alert all legal consultants. Immediate safety and "
            "legal intervention required. Await human assessment."
        )
        message = "All three legal consultants must be alerted."
    elif priority == "URGENT":
        recipients = [LEGAL_CONSULTANTS[0]]
        status = "LEGAL_ALERTED"
        assigned_to = recipients[0]["name"]
        next_action = (
            "URGENT: Alert one legal consultant. Await legal assessment."
        )
        message = "One legal consultant must be alerted."
    else:
        status = "FOR_HUMAN_RESOLUTION"
        assigned_to = state["primary_department"]
        next_action = (
            "Human review required. The responsible department must assess "
            "the concern and prepare an approved response."
        )
        message = "This matter requires human review."

    if recipients:
        print("\n---------------- LEGAL ALERT (SIMULATION) ----------------")
        for person in recipients:
            print(f"Recipient : {person['name']}")
            print(f"Email     : {person['email']}")
            print(f"Ticket ID : {state.get('ticket_id', 'Pending ID')}")
            print(f"Priority  : {priority}")
            print("----------------------------------------------------------")
        print("[NOTICE] Simulation only. No email was sent.")

    return {
        "legal_recipients": recipients,
        "status": status,
        "assigned_to": assigned_to,
        "next_action": next_action,
        "draft_answer": None,
        "result_message": message,
    }


def complaint_human_review_agent(state: ConcernState) -> Dict[str, Any]:
    """Complaints are always routed to a human reviewer."""
    referral = tool_department_referral(state)
    print("[TOOL SELECTION] Executing approved tool: department_referral")
    return {
        "status": "WAITING_FOR_HUMAN_REVIEW",
        "assigned_to": state["primary_department"],
        "next_action": (
            f"{referral} An authorized human reviewer must assess the allegation "
            "and approve any response."
        ),
        "draft_answer": None,
        "result_message": (
            "Complaint automatically classified and routed for human review. "
            "No automated finding or final response was generated."
        ),
    }


# ============================================================
# 14. ORCHESTRATOR AND CONDITIONAL ROUTING
# ============================================================

def create_ticket_id_node(state: ConcernState) -> Dict[str, Any]:
    return {
        "ticket_id": (
            f"PA-LAW-{datetime.now().strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        ),
        "date_created": datetime.now().isoformat(timespec="seconds"),
    }


def route_after_planner(state: ConcernState) -> str:
    # Enforce policy guardrails again at the routing boundary.
    if state.get("priority") in {"URGENT", "CRITICAL"}:
        return "legal_escalation"
    if state.get("request_type") == "COMPLAINT" or state.get("complexity") == "COMPLEX":
        return "complaint_review"
    if state.get("planned_action") == "REQUEST_INFORMATION":
        return "followup_agent"
    if state.get("planned_action") == "HUMAN_REVIEW":
        return "complaint_review"
    return "policy_knowledge"


def persist_ticket_node(state: ConcernState) -> Dict[str, Any]:
    saved = save_ticket(state)
    follow_up_id = None
    if saved and state.get("follow_up_question"):
        follow_up_id = tool_create_followup(state["ticket_id"], state["follow_up_question"])
    return {
        "follow_up_id": str(follow_up_id) if follow_up_id else "",
        "saved": saved,
        "status": state["status"] if saved else "DATABASE_ERROR",
        "next_action": (
            state["next_action"] if saved
            else "Ticket could not be saved. Check the SQLite database."
        ),
    }


def build_agentic_graph():
    graph = StateGraph(ConcernState)

    graph.add_node("intake_agent", intake_agent)
    graph.add_node("classification_agent", classification_agent)
    graph.add_node("priority_agent", priority_assessment_agent)
    graph.add_node("routing_agent", department_routing_agent)
    graph.add_node("complexity_agent", complexity_assessment_agent)
    graph.add_node("create_ticket_id", create_ticket_id_node)
    graph.add_node("task_planner_agent", task_planner_agent)
    graph.add_node("followup_agent", followup_agent)
    graph.add_node("policy_knowledge_agent", policy_knowledge_agent)
    graph.add_node("legal_escalation_agent", execute_legal_alert_tool)
    graph.add_node("complaint_review_agent", complaint_human_review_agent)
    graph.add_node("persist_ticket", persist_ticket_node)

    graph.add_edge(START, "intake_agent")
    graph.add_edge("intake_agent", "classification_agent")
    graph.add_edge("classification_agent", "priority_agent")
    graph.add_edge("priority_agent", "routing_agent")
    graph.add_edge("routing_agent", "complexity_agent")
    graph.add_edge("complexity_agent", "task_planner_agent")
    graph.add_edge("task_planner_agent", "create_ticket_id")

    graph.add_conditional_edges(
        "create_ticket_id",
        route_after_planner,
        {
            "legal_escalation": "legal_escalation_agent",
            "complaint_review": "complaint_review_agent",
            "policy_knowledge": "policy_knowledge_agent",
            "followup_agent": "followup_agent",
        },
    )

    graph.add_edge("legal_escalation_agent", "persist_ticket")
    graph.add_edge("complaint_review_agent", "persist_ticket")
    graph.add_edge("policy_knowledge_agent", "persist_ticket")
    graph.add_edge("followup_agent", "persist_ticket")
    graph.add_edge("persist_ticket", END)

    return graph.compile()


# ============================================================
# 15. CONSOLE INPUT AND DISPLAY
# ============================================================

def get_nature_of_concern():
    print("\n================ NATURE OF CONCERN ================")
    for key, value in NATURE_OPTIONS.items():
        print(f"{key}. {value}")

    while True:
        choice = input("\nSelect Nature of Concern (1-5): ").strip()
        if choice in NATURE_OPTIONS:
            return NATURE_OPTIONS[choice], choice
        print("[ERROR] Invalid selection. Choose 1-5.")


def get_concern_description():
    print("\n================ CONCERN DESCRIPTION ================")
    while True:
        description = input("Describe your concern: ").strip()
        if len(description) >= 10:
            return description
        print("[ERROR] Please provide at least 10 characters.")


def display_result(state: ConcernState):
    print("\n================ PROCESSING RESULT ================")
    fields = [
        ("Ticket ID", "ticket_id"),
        ("Request Type", "request_type"),
        ("Nature", "nature_of_concern"),
        ("Priority", "priority"),
        ("Complexity", "complexity"),
        ("Primary Department", "primary_department"),
        ("Additional Departments", "additional_departments"),
        ("Status", "status"),
        ("Assigned To", "assigned_to"),
        ("Next Action", "next_action"),
        ("Ticket Saved", "saved"),
        ("Planner Action", "planned_action"),
        ("Selected Tools", "selected_tools"),
        ("RAG Attempts", "rag_attempts"),
        ("Evidence Sufficient", "evidence_sufficient"),
        ("Approval Status", "approval_status"),
        ("Follow-up ID", "follow_up_id"),
    ]

    for label, key in fields:
        value = state.get(key, "")
        if isinstance(value, list):
            value = ", ".join(value)
        print(f"{label}: {value}")

    print("\n---------------- AGENT ANALYSIS ----------------")
    print("Intake Summary:", state.get("intake_summary", ""))
    print("Classification Reason:", state.get("classification_reason", ""))
    print("Routing Reason:", state.get("routing_reason", ""))
    print("Complexity Reason:", state.get("complexity_reason", ""))

    if state.get("handbook_sources"):
        print("\nHandbook Sources:")
        for source in state["handbook_sources"]:
            print(f"- {source}")

    if state.get("planner_reason"):
        print("Planner Reason:", state.get("planner_reason"))
    if state.get("rag_feedback"):
        print("Evidence Feedback:", state.get("rag_feedback"))
    if state.get("follow_up_question"):
        print("\nFOLLOW-UP REQUEST:\n", state.get("follow_up_question"))
    if state.get("draft_answer"):
        print("\n================ DRAFT ANSWER (NOT YET APPROVED) ================")
        print(state["draft_answer"])

    if state.get("handbook_excerpt"):
        print("\n================ RETRIEVED POLICY EVIDENCE ================")
        print(state["handbook_excerpt"])

    if state.get("result_message"):
        print("\nSystem Message:", state["result_message"])

    if state.get("missing_information"):
        print("\nInformation a reviewer may need:")
        for item in state["missing_information"]:
            print(f"- {item}")

    print("\n[DISCLAIMER] AI-generated output is preliminary. "
          "It is not legal advice or an approved company decision.")


def submit_concern(graph):
    nature, choice = get_nature_of_concern()
    description = get_concern_description()

    initial_state: ConcernState = {
        "nature_choice": choice,
        "nature_of_concern": nature,
        "description": description,
    }

    print("\n================ LANGGRAPH AGENT EXECUTION ================")
    try:
        result = graph.invoke(initial_state)
        display_result(result)
        return result
    except Exception as exc:
        print(f"\n[WORKFLOW ERROR] {exc}")
        print("Check that Ollama is running and required models are installed.")
        return None


def view_sorted_tickets():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT ticket_id, date_created, classification, priority,
                   complexity, primary_department, status, assigned_to
            FROM tickets
            ORDER BY
                CASE priority
                    WHEN 'CRITICAL' THEN 1
                    WHEN 'URGENT' THEN 2
                    WHEN 'HIGH' THEN 3
                    WHEN 'MEDIUM' THEN 4
                    WHEN 'LOW' THEN 5
                    ELSE 6
                END,
                date_created ASC
        """).fetchall()

    print("\n================ PRIORITY-SORTED TICKETS ================")
    if not rows:
        print("No tickets found.")
        return

    for row in rows:
        print("-" * 55)
        for key in row.keys():
            print(f"{key}: {row[key]}")


def view_ticket():
    ticket_id = input("\nEnter Ticket ID: ").strip()

    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()

    if not row:
        print("[ERROR] Ticket not found.")
        return

    print("\n================ TICKET DETAILS ================")
    for key in row.keys():
        print(f"{key}: {row[key]}")

    if row["draft_answer"]:
        print("\n================ DRAFT ANSWER ================")
        print(row["draft_answer"])


# ============================================================
# HUMAN APPROVAL AND FOLLOW-UP CONSOLE TOOLS
# ============================================================

def review_pending_approvals():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM tickets WHERE approval_status='PENDING' ORDER BY date_created").fetchall()
    if not rows:
        print("\nNo drafts are waiting for approval.")
        return
    for row in rows:
        print("\n" + "=" * 65)
        print(f"Ticket: {row['ticket_id']} | Department: {row['primary_department']} | Priority: {row['priority']}")
        print("Concern:", row['description'])
        print("Draft:\n", row['draft_answer'])
        print("Sources:", row['handbook_sources'])
        print("\n1. Approve draft for release")
        print("2. Reject / return for revision")
        print("3. Skip")
        choice = input("Decision: ").strip()
        if choice == "1":
            reviewer = input("Authorized reviewer name: ").strip()
            if not reviewer:
                print("[APPROVAL BLOCKED] Reviewer name is required.")
                continue
            confirm = input("Confirm approval? Type APPROVE: ").strip()
            if confirm != "APPROVE":
                print("Approval cancelled.")
                continue
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("UPDATE tickets SET approval_status='APPROVED', approved_by=?, approved_at=?, status='APPROVED_FOR_RELEASE', next_action='Draft approved by authorized reviewer; release through the approved communication channel.' WHERE ticket_id=?",
                             (reviewer, datetime.now().isoformat(timespec="seconds"), row['ticket_id']))
            print("[APPROVED] Draft marked as approved. This console does not send it.")
        elif choice == "2":
            reviewer = input("Reviewer name: ").strip()
            notes = input("Revision notes: ").strip()
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("UPDATE tickets SET approval_status='REJECTED', approved_by=?, review_notes=?, status='RETURNED_FOR_REVISION', next_action='Revise the draft according to reviewer notes.' WHERE ticket_id=?",
                             (reviewer, notes, row['ticket_id']))
            print("[RETURNED] Draft returned for revision.")


def manage_followups():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM followups WHERE status='PENDING' ORDER BY created_at").fetchall()
    if not rows:
        print("\nNo pending follow-up requests.")
        return
    for row in rows:
        print("\n" + "-" * 60)
        print(f"Follow-up ID: {row['follow_up_id']} | Ticket: {row['ticket_id']}")
        print("Question:", row['question'])
        response = input("Record employee response (blank to skip): ").strip()
        if response:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("UPDATE followups SET status='RESPONDED', response=?, updated_at=? WHERE follow_up_id=?",
                             (response, datetime.now().isoformat(timespec="seconds"), row['follow_up_id']))
                conn.execute("UPDATE tickets SET status='FOLLOW_UP_RECEIVED', next_action='Employee response received. Responsible department should review the response.' WHERE ticket_id=?",
                             (row['ticket_id'],))
            print("[FOLLOW-UP] Response recorded. The ticket is ready for department review.")


def view_followup_history():
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT follow_up_id, ticket_id, question, status, response, created_at FROM followups ORDER BY created_at DESC").fetchall()
    if not rows:
        print("No follow-up records.")
        return
    print("\n================ FOLLOW-UP HISTORY ================")
    for row in rows:
        print(f"ID: {row[0]} | Ticket: {row[1]} | Status: {row[3]}")
        print("Question:", row[2])
        print("Response:", row[4] or "(no response yet)")


# ============================================================
# 16. STARTUP CHECKS AND MAIN MENU
# ============================================================

def check_ollama():
    print("\n[OLLAMA] Configuration")
    print(f"Base URL: {OLLAMA_BASE_URL}")
    print(f"LLM Model: {OLLAMA_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("Ensure Ollama is running before submitting concerns.")


def main():
    print("\n====================================================")
    print("          PA-LAW AGENTIC AI POLICY ADVISER")
    print("        LANGGRAPH MULTI-AGENT CONSOLE SYSTEM")
    print("====================================================")

    initialize_database()
    check_ollama()
    load_handbook_and_index()

    print("\n[ORCHESTRATOR] Building agent workflow with dynamic delegation...")
    graph = build_agentic_graph()
    print("[ORCHESTRATOR] Workflow ready.")

    while True:
        print("\n====================================================")
        print("                 MAIN MENU")
        print("====================================================")
        print("1. Submit New Concern")
        print("2. View Priority-Sorted Tickets")
        print("3. View Individual Ticket")
        print("4. Review Pending Draft Approvals")
        print("5. Manage Pending Follow-ups")
        print("6. View Follow-up History")
        print("7. Exit")

        action = input("\nSelect Action: ").strip()

        if action == "1":
            submit_concern(graph)
        elif action == "2":
            view_sorted_tickets()
        elif action == "3":
            view_ticket()
        elif action == "4":
            review_pending_approvals()
        elif action == "5":
            manage_followups()
        elif action == "6":
            view_followup_history()
        elif action == "7":
            print("\nExiting PA-LAW Agentic AI Policy Adviser.")
            break
        else:
            print("[ERROR] Invalid selection. Choose 1-7.")


if __name__ == "__main__":
    main()
