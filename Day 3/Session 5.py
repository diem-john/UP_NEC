#ollama pull nomic-embed-text
#%% EXERCISE 1: ANATOMY OF A DV HELPDESK AGENT

import os
import sqlite3
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

DB_FILE = "dv_helpdesk.db"
PDF_FILE = "PreAudit Checklist for Common Transactions.pdf"
MODEL = "qwen3.5:4b"


# ---------------------------------------------------------
# 1. CHECK SYSTEM COMPONENTS
# ---------------------------------------------------------

def check_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    return os.path.abspath(filename)


print("SQLite Database :", check_file(DB_FILE))
print("Knowledge Source :", check_file(PDF_FILE))


# ---------------------------------------------------------
# 2. INSPECT SAMPLE DV RECORDS
# ---------------------------------------------------------

def get_sample_dvs():
    conn = sqlite3.connect(DB_FILE)

    rows = conn.execute("""
        SELECT dv_number, dv_type, status, remarks
        FROM disbursement_vouchers
        ORDER BY dv_number
        LIMIT 5
    """).fetchall()

    conn.close()

    return rows


print("\nSAMPLE DV RECORDS")

for dv in get_sample_dvs():
    print(f"\nDV Number : {dv[0]}")
    print(f"DV Type   : {dv[1]}")
    print(f"Status    : {dv[2]}")
    print(f"Remarks   : {dv[3]}")


# ---------------------------------------------------------
# 3. INITIAL AGENT STATE
# ---------------------------------------------------------

state = {
    "user_question": "Why was DV-2026-00123 returned?",
    "dv_number": None,
    "dv_type": None,
    "dv_status": None,
    "remarks": None,
    "response": None,
    "status": "Started"
}

print("\nINITIAL STATE")
print(state)


# ---------------------------------------------------------
# 4. LOCAL LLM
# ---------------------------------------------------------

llm = ChatOllama(
    model=MODEL,
    reasoning=False,
    temperature=0
)


# ---------------------------------------------------------
# 5. EXPLAIN THE FUTURE AGENT WORKFLOW
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are helping design a DV Helpdesk Agent.

The system has:
- an SQLite database containing DV status and remarks,
- a PDF containing supporting-document requirements,
- a local LLM,
- state for tracking the current inquiry.

The database and PDF are not yet available as tools.
Explain what the completed agent should eventually do.
"""
    ),
    (
        "user",
        "User question: {question}"
    )
])


response = (prompt | llm).invoke({
    "question": state["user_question"]
})


print("\nPROPOSED AGENT PROCESS")
print(response.content)



#%% EXERCISE 2: TOOLS AND FUNCTION CALLING
# Required files in the working directory:
#   dv_helpdesk.db
#   rag_knowledge.db
#   dv_rag.py
#
# One-time Ollama setup:
#   ollama pull qwen3.5:4b
#   ollama pull nomic-embed-text

import sqlite3
import json
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from dv_rag import retrieve_dv_requirements

DB_FILE = "dv_helpdesk.db"
MODEL = "qwen3.5:4b"


# ---------------------------------------------------------
# 1. REUSABLE APPLICATION FUNCTIONS
# ---------------------------------------------------------

def query_database(sql, params=()):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return dict(row) if row else None


def get_dv(dv_number):
    row = query_database("""
        SELECT dv_number, dv_type, payee, amount, status, remarks,
               accounting_analyst, last_updated, payment_date
        FROM disbursement_vouchers
        WHERE UPPER(dv_number) = UPPER(?)
    """, (dv_number.strip(),))

    return row if row else {"error": "DV not found", "dv_number": dv_number}


# ---------------------------------------------------------
# 2. EXPOSE THE FUNCTIONS AS LANGCHAIN TOOLS
# ---------------------------------------------------------

@tool
def query_dv(dv_number: str) -> dict:
    """Retrieve the current record for one Disbursement Voucher."""
    return get_dv(dv_number)


@tool
def search_dv_requirements(dv_type: str, question: str, analyst_remarks: str = "") -> list:
    """Search the pre-audit checklist for attachment and supporting-document requirements."""
    return retrieve_dv_requirements(
        dv_type=dv_type,
        question=question,
        analyst_remarks=analyst_remarks,
        k=3
    )


# ---------------------------------------------------------
# 3. CREATE THE SINGLE DV HELPDESK AGENT
# ---------------------------------------------------------

llm = ChatOllama(model=MODEL, reasoning=False, temperature=0)

agent = create_agent(
    model=llm,
    tools=[query_dv, search_dv_requirements],
    system_prompt="""
You are a DV Helpdesk Agent.

Follow this process:
1. If the user refers to a specific DV, call query_dv first.
2. Use the returned SQLite record as the authoritative source for the DV type,
   current status, Accounting Analyst remarks, and payment information.
3. If the user asks why a DV was returned, what attachment is missing, what
   document is required, or another compliance question, call
   search_dv_requirements after query_dv.
4. Pass the DV type and Accounting Analyst remarks returned by query_dv into
   search_dv_requirements. Do not invent these values.
5. Use the RAG result only as checklist evidence. Clearly distinguish the
   specific analyst remark from the general checklist requirement.
6. Never invent a DV status, reason, attachment requirement, or payment date.
7. If the DV is not found or the retrieved evidence does not support an answer,
   say so.
8. Keep the final response concise and helpful.
"""
)


# ---------------------------------------------------------
# 4. RUN AND INSPECT THE AGENT
# ---------------------------------------------------------

def run_agent(question):
    return agent.invoke({"messages": [{"role": "user", "content": question}]})


def show_trace(result):
    print("\n" + "=" * 70)
    print("AGENT EXECUTION TRACE")
    print("=" * 70)

    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []) or []:
            print(f"\nTOOL: {call['name']}")
            print("ARGUMENTS:", json.dumps(call["args"], indent=2))

        if type(message).__name__ == "ToolMessage":
            print("\nTOOL RESULT:")
            print(message.content)

    print("\n" + "=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)
    print(result["messages"][-1].content)


question = "Why was DV-2026-00123 returned, and what document do I need to submit?"
result = run_agent(question)
show_trace(result)

#%% EXERCISE 3: STATE AND WORKFLOW CONTROL
#
# Demonstrates two workflow directions:
#
# Example A:
# Returned / compliance question
# → DV lookup
# → RAG required
# → Retrieve checklist evidence
#
# Example B:
# Simple status question
# → DV lookup
# → RAG not required
# → Ready for response

import sqlite3
import json
import re
from datetime import datetime
import dv_rag

STATE_DB = "agent_state.db"

def initialize_state_db():
    conn=sqlite3.connect(STATE_DB)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS inquiry_state(
            workflow_id TEXT PRIMARY KEY,
            user_question TEXT,
            dv_number TEXT,
            dv_type TEXT,
            dv_status TEXT,
            remarks TEXT,
            needs_rag INTEGER DEFAULT 0,
            rag_evidence TEXT,
            response TEXT,
            workflow_status TEXT,
            updated_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history(
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

initialize_state_db()
# ---------------------------------------------------------
# 1. STATE DATABASE FUNCTIONS
# ---------------------------------------------------------

def start_workflow(workflow_id, question):

    conn = sqlite3.connect(STATE_DB)

    conn.execute("""
        INSERT OR REPLACE INTO inquiry_state
        (
            workflow_id,
            user_question,
            dv_number,
            dv_type,
            dv_status,
            remarks,
            needs_rag,
            rag_evidence,
            response,
            workflow_status,
            updated_at
        )
        VALUES (?, ?, NULL, NULL, NULL, NULL, 0, NULL, NULL, ?, ?)
    """, (
        workflow_id,
        question,
        "Started",
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def get_state(workflow_id):

    conn = sqlite3.connect(STATE_DB)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT *
        FROM inquiry_state
        WHERE workflow_id = ?
        """,
        (workflow_id,)
    ).fetchone()

    conn.close()

    if not row:
        return None

    state = dict(row)

    if state["rag_evidence"]:

        state["rag_evidence"] = json.loads(
            state["rag_evidence"]
        )

    state["needs_rag"] = bool(
        state["needs_rag"]
    )

    return state


def update_state(workflow_id, **changes):

    allowed = {
        "user_question",
        "dv_number",
        "dv_type",
        "dv_status",
        "remarks",
        "needs_rag",
        "rag_evidence",
        "response",
        "workflow_status"
    }

    changes = {
        key: value
        for key, value in changes.items()
        if key in allowed
    }

    if not changes:
        return

    changes["updated_at"] = (
        datetime.now().isoformat()
    )

    columns = ", ".join(
        f"{key} = ?"
        for key in changes
    )

    values = (
        list(changes.values())
        + [workflow_id]
    )

    conn = sqlite3.connect(STATE_DB)

    conn.execute(
        f"""
        UPDATE inquiry_state
        SET {columns}
        WHERE workflow_id = ?
        """,
        values
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# 2. DISPLAY HELPERS
# ---------------------------------------------------------

def clean_text(text):

    if text is None:
        return None

    text = str(text)

    # Replace escaped / actual newlines
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")

    # Replace common Unicode bullets
    text = text.replace("\\u00b7", "-")
    text = text.replace("\u00b7", "-")
    text = text.replace("•", "-")

    # Remove repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_rag_evidence(evidence):

    if not evidence:
        return evidence

    cleaned = []

    for item in evidence:

        new_item = dict(item)

        if "text" in new_item:
            new_item["text"] = clean_text(
                new_item["text"]
            )

        cleaned.append(
            new_item
        )

    return cleaned


def print_state(workflow_id):

    state = get_state(
        workflow_id
    )

    if not state:
        print("State not found.")
        return

    print("\n" + "=" * 70)
    print(
        "CURRENT STATE:",
        workflow_id
    )
    print("=" * 70)

    print(
        "Question        :",
        clean_text(
            state["user_question"]
        )
    )

    print(
        "DV Number       :",
        state["dv_number"]
    )

    print(
        "DV Type         :",
        state["dv_type"]
    )

    print(
        "DV Status       :",
        state["dv_status"]
    )

    print(
        "Remarks         :",
        clean_text(
            state["remarks"]
        )
    )

    print(
        "Needs RAG       :",
        state["needs_rag"]
    )

    print(
        "Workflow Status :",
        state["workflow_status"]
    )

    if state["rag_evidence"]:

        print("\nRAG EVIDENCE")

        evidence = clean_rag_evidence(
            state["rag_evidence"]
        )

        for item in evidence:

            print(
                f"\nRank   : {item.get('rank')}"
            )

            print(
                f"Page   : {item.get('page')}"
            )

            print(
                f"Source : {item.get('source')}"
            )

            print(
                "Text   :",
                item.get("text")
            )


# ---------------------------------------------------------
# 3. WORKFLOW FUNCTIONS
# ---------------------------------------------------------

def extract_dv_number(text):

    match = re.search(
        r"DV-\d{4}-\d+",
        text.upper()
    )

    return (
        match.group()
        if match
        else None
    )


def needs_rag(state):

    text = " ".join([
        state["user_question"] or "",
        state["dv_status"] or "",
        state["remarks"] or ""
    ]).lower()

    keywords = [
        "returned",
        "compliance",
        "attach",
        "attachment",
        "missing",
        "document",
        "requirement",
        "submit"
    ]

    return any(
        word in text
        for word in keywords
    )


def determine_next_step(state):

    if state["workflow_status"] == "Started":
        return "identify_dv"

    if state["workflow_status"] == "DV Identified":
        return "query_dv"

    if state["workflow_status"] == "DV Retrieved":
        return "check_rag"

    if state["workflow_status"] == "RAG Required":
        return "retrieve_rag"

    return "stop"


# ---------------------------------------------------------
# 4. WORKFLOW ORCHESTRATOR
# ---------------------------------------------------------

def run_workflow(workflow_id):

    while True:

        state = get_state(
            workflow_id
        )

        action = determine_next_step(
            state
        )

        print("\n" + "-" * 70)

        print(
            f"{state['workflow_status']}"
            f"  →  {action}"
        )

        print("-" * 70)


        # -------------------------------------------------
        # IDENTIFY DV
        # -------------------------------------------------

        if action == "identify_dv":

            dv_number = extract_dv_number(
                state["user_question"]
            )

            if dv_number:

                update_state(
                    workflow_id,
                    dv_number=dv_number,
                    workflow_status="DV Identified"
                )

            else:

                update_state(
                    workflow_id,
                    workflow_status="Awaiting DV Number"
                )

            print_state(
                workflow_id
            )


        # -------------------------------------------------
        # QUERY DV DATABASE
        # -------------------------------------------------

        elif action == "query_dv":

            dv = get_dv(
                state["dv_number"]
            )

            if "error" in dv:

                update_state(
                    workflow_id,
                    workflow_status="DV Not Found"
                )

            else:

                update_state(
                    workflow_id,
                    dv_type=dv["dv_type"],
                    dv_status=dv["status"],
                    remarks=dv["remarks"],
                    workflow_status="DV Retrieved"
                )

            print_state(
                workflow_id
            )


        # -------------------------------------------------
        # CHECK WHETHER RAG IS NEEDED
        # -------------------------------------------------

        elif action == "check_rag":

            use_rag = needs_rag(
                state
            )

            update_state(
                workflow_id,
                needs_rag=int(use_rag),
                workflow_status=(
                    "RAG Required"
                    if use_rag
                    else "Ready for Response"
                )
            )

            print_state(
                workflow_id
            )


        # -------------------------------------------------
        # RETRIEVE RAG EVIDENCE
        # -------------------------------------------------

        elif action == "retrieve_rag":

            evidence = (
                dv_rag.retrieve_dv_requirements(
                    dv_type=state["dv_type"],
                    question=state["user_question"],
                    analyst_remarks=state["remarks"],
                    k=3
                )
            )

            evidence = clean_rag_evidence(
                evidence
            )

            update_state(
                workflow_id,
                rag_evidence=json.dumps(
                    evidence,
                    ensure_ascii=False
                ),
                workflow_status="Ready for Response"
            )

            print_state(
                workflow_id
            )


        else:

            print(
                "\nWorkflow complete."
            )

            break


# ---------------------------------------------------------
# 5. EXAMPLE A:
#    RETURNED FOR COMPLIANCE
#    → RAG REQUIRED
# ---------------------------------------------------------

print("\n" + "#" * 70)
print("EXAMPLE A: RETURNED / COMPLIANCE INQUIRY")
print("#" * 70)

WORKFLOW_ID_1 = "INQ-001"

question_1 = (
    "Why was DV-2026-00123 returned, "
    "and what document do I need to submit?"
)

start_workflow(
    WORKFLOW_ID_1,
    question_1
)

print_state(
    WORKFLOW_ID_1
)

run_workflow(
    WORKFLOW_ID_1
)


# ---------------------------------------------------------
# 6. EXAMPLE B:
#    SIMPLE STATUS INQUIRY
#    → NO RAG REQUIRED
# ---------------------------------------------------------

print("\n\n" + "#" * 70)
print("EXAMPLE B: SIMPLE STATUS INQUIRY")
print("#" * 70)

WORKFLOW_ID_2 = "INQ-002"

question_2 = (
    "What is the status of DV-2026-00125?"
)

start_workflow(
    WORKFLOW_ID_2,
    question_2
)

print_state(
    WORKFLOW_ID_2
)

run_workflow(
    WORKFLOW_ID_2
)


# ---------------------------------------------------------
# 7. COMPARE FINAL STATES
# ---------------------------------------------------------

print("\n\n" + "#" * 70)
print("FINAL WORKFLOW COMPARISON")
print("#" * 70)

for workflow_id in [
    WORKFLOW_ID_1,
    WORKFLOW_ID_2
]:

    state = get_state(
        workflow_id
    )

    print(
        f"\n{workflow_id}"
    )

    print(
        "Question        :",
        clean_text(
            state["user_question"]
        )
    )

    print(
        "DV Status       :",
        state["dv_status"]
    )

    print(
        "Needs RAG       :",
        state["needs_rag"]
    )

    print(
        "Workflow Status :",
        state["workflow_status"]
    )
#%% EXERCISE 4: MEMORY AND CONTEXT

import sqlite3
from datetime import datetime

STATE_DB = "agent_state.db"


# ---------------------------------------------------------
# 1. SAVE A MESSAGE TO CONVERSATION MEMORY
# ---------------------------------------------------------

def save_message(workflow_id, role, message):
    conn = sqlite3.connect(STATE_DB)

    last = conn.execute("""
        SELECT role, message
        FROM conversation_history
        WHERE workflow_id = ?
        ORDER BY message_id DESC
        LIMIT 1
    """, (workflow_id,)).fetchone()

    if last != (role, message):
        conn.execute("""
            INSERT INTO conversation_history
            (workflow_id, role, message, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            workflow_id,
            role,
            message,
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# 2. RETRIEVE RECENT CONVERSATION MEMORY
# ---------------------------------------------------------

def get_recent_messages(workflow_id, limit=6):
    conn = sqlite3.connect(STATE_DB)

    rows = conn.execute("""
        SELECT role, message
        FROM conversation_history
        WHERE workflow_id = ?
        ORDER BY message_id DESC
        LIMIT ?
    """, (workflow_id, limit)).fetchall()

    conn.close()

    rows.reverse()

    return [
        {
            "role": row[0],
            "message": row[1]
        }
        for row in rows
    ]


# ---------------------------------------------------------
# 3. BUILD THE CONTEXT FOR THE LLM
# ---------------------------------------------------------

def build_context(workflow_id):

    state = get_state(workflow_id)

    memory = get_recent_messages(
        workflow_id,
        limit=6
    )

    evidence = state["rag_evidence"] or []

    if evidence:
        evidence_text = dv_rag.format_evidence(
            evidence
        )
    else:
        evidence_text = "No RAG evidence is currently stored."

    memory_text = "\n".join(
        f"{item['role'].upper()}: {item['message']}"
        for item in memory
    )

    context = f"""
CURRENT DV STATE

DV Number:
{state['dv_number']}

DV Type:
{state['dv_type']}

DV Status:
{state['dv_status']}

Accounting Analyst Remarks:
{state['remarks']}


RETRIEVED CHECKLIST EVIDENCE

{evidence_text}


RECENT CONVERSATION

{memory_text}
"""

    return context


# ---------------------------------------------------------
# 4. GENERATE A GROUNDED HELPDESK RESPONSE
# ---------------------------------------------------------

def generate_helpdesk_response(
    workflow_id,
    question
):

    save_message(
        workflow_id,
        "user",
        question
    )

    context = build_context(
        workflow_id
    )

    messages = [

        {
            "role": "system",
            "content": """
You are a DV Helpdesk Agent.

Answer the user's question using only the supplied context.

Rules:

1. The DV state is authoritative for the specific DV's
   current status and Accounting Analyst remarks.

2. The retrieved checklist evidence is the source for
   supporting-document requirements.

3. Use conversation memory only to understand references
   such as "my DV", "it", "that document", or follow-up questions.

4. Do not invent a DV status, missing attachment,
   requirement, reason, or payment date.

5. Clearly distinguish the Accounting Analyst's specific
   remark from general checklist requirements.

6. If the available context does not support the answer,
   say that the information is insufficient.

7. Keep the response concise and helpful.
"""
        },

        {
            "role": "user",
            "content": f"""
CONTEXT:

{context}


CURRENT QUESTION:

{question}
"""
        }

    ]

    response = llm.invoke(
        messages
    )

    answer = response.content

    save_message(
        workflow_id,
        "assistant",
        answer
    )

    update_state(
        workflow_id,
        response=answer,
        workflow_status="Response Generated"
    )

    return answer

# ---------------------------------------------------------
# RESET CONVERSATION MEMORY FOR ONE WORKFLOW
# ---------------------------------------------------------

def clear_memory(workflow_id):
    conn = sqlite3.connect(STATE_DB)

    conn.execute("""
        DELETE FROM conversation_history
        WHERE workflow_id = ?
    """, (workflow_id,))

    conn.commit()
    conn.close()
# ---------------------------------------------------------
# 5. FIRST CONVERSATION TURN
# ---------------------------------------------------------

WORKFLOW_ID = "INQ-001"

# Remove conversation messages from previous runs
clear_memory(WORKFLOW_ID)

question1 = (
    "Why was DV-2026-00123 returned? "
   
)

answer1 = generate_helpdesk_response(
    WORKFLOW_ID,
    question1
)

print("\n" + "=" * 70)
print("TURN 1")
print("=" * 70)

print("\nUSER:")
print(question1)

print("\nAGENT:")
print(answer1)


# ---------------------------------------------------------
# 6. FOLLOW-UP QUESTION
# ---------------------------------------------------------

question2 = "What exactly should I submit?"

answer2 = generate_helpdesk_response(
    WORKFLOW_ID,
    question2
)

print("\n" + "=" * 70)
print("TURN 2")
print("=" * 70)

print("\nUSER:")
print(question2)

print("\nAGENT:")
print(answer2)


# ---------------------------------------------------------
# 7. VIEW STORED MEMORY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONVERSATION MEMORY")
print("=" * 70)

memory = get_recent_messages(
    WORKFLOW_ID,
    limit=10
)

for item in memory:
    print(
        f"\n{item['role'].upper()}: "
        f"{item['message']}"
    )


# ---------------------------------------------------------
# 8. VIEW THE CONTEXT SENT TO THE LLM
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT LLM CONTEXT")
print("=" * 70)

print(
    build_context(
        WORKFLOW_ID
    )
)


#%% EXERCISE 5: CONTROLS AND HUMAN APPROVAL
import json
# ---------------------------------------------------------
# 1. DISPLAY CURRENT STATE AND MEMORY
# ---------------------------------------------------------

def show_control_progress(workflow_id, stage):

    state = get_state(workflow_id)

    memory = get_recent_messages(
        workflow_id,
        limit=10
    )

    print("\n" + "=" * 70)
    print(stage)
    print("=" * 70)

    print("\nCURRENT STATE")

    print(
        "Workflow Status :",
        state["workflow_status"]
    )

    print(
        "DV Number       :",
        state["dv_number"]
    )

    print(
        "DV Type         :",
        state["dv_type"]
    )

    print(
        "DV Status       :",
        state["dv_status"]
    )

    print(
        "Remarks         :",
        clean_text(
            state["remarks"]
        )
    )

    print(
        "Needs RAG       :",
        state["needs_rag"]
    )

    if state["rag_evidence"]:

        print(
            "RAG Evidence    :",
            len(state["rag_evidence"]),
            "retrieved chunks"
        )

    else:

        print(
            "RAG Evidence    : None"
        )


    print("\nCONVERSATION MEMORY")

    if not memory:

        print(
            "No messages stored."
        )

    else:

        for item in memory:

            print(
                f"{item['role'].upper()}: "
                f"{clean_text(item['message'])}"
            )


# ---------------------------------------------------------
# 2. GENERATE A DRAFT WITHOUT RELEASING IT
# ---------------------------------------------------------

def generate_helpdesk_draft(
    workflow_id,
    question
):

    # Save the user's message
    save_message(
        workflow_id,
        "user",
        question
    )

    context = build_context(
        workflow_id
    )

    messages = [

        {
            "role": "system",
            "content": """
You are a DV Helpdesk Agent.

Prepare a draft response using only the supplied context.

Rules:

1. The DV state is authoritative for the specific DV's
   current status and Accounting Analyst remarks.

2. The retrieved checklist evidence is the source for
   supporting-document requirements.

3. Use conversation memory only to understand references
   and follow-up questions.

4. Do not invent a DV status, missing attachment,
   requirement, reason, or payment date.

5. Clearly distinguish the Accounting Analyst's specific
   remark from general checklist requirements.

6. If the available context does not support the answer,
   say that the information is insufficient.

7. Keep the response concise and helpful.

The response is only a draft and may still require
human approval before release.
"""
        },

        {
            "role": "user",
            "content": f"""
CONTEXT:

{context}

CURRENT QUESTION:

{question}
"""
        }

    ]

    response = llm.invoke(
        messages
    )

    return response.content


# ---------------------------------------------------------
# 3. VALIDATE THE GENERATED RESPONSE
# ---------------------------------------------------------

def validate_response(
    workflow_id,
    response
):

    state = get_state(
        workflow_id
    )

    issues = []


    if not state:

        issues.append(
            "Workflow state not found."
        )

        return {
            "valid": False,
            "issues": issues
        }


    if not state["dv_number"]:

        issues.append(
            "DV number is missing."
        )


    if not state["dv_status"]:

        issues.append(
            "DV status is missing."
        )


    if (
        not response
        or not response.strip()
    ):

        issues.append(
            "Generated response is empty."
        )


    if (
        state["needs_rag"]
        and not state["rag_evidence"]
    ):

        issues.append(
            "RAG evidence is required "
            "but was not retrieved."
        )


    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


# ---------------------------------------------------------
# 4. DETERMINE WHETHER HUMAN APPROVAL IS REQUIRED
# ---------------------------------------------------------

def requires_human_approval(
    workflow_id
):

    state = get_state(
        workflow_id
    )

    reasons = []


    if (
        state["dv_status"]
        == "Returned for Compliance"
    ):

        reasons.append(
            "DV was returned for compliance."
        )


    if state["needs_rag"]:

        reasons.append(
            "Response uses supporting-document "
            "requirements retrieved through RAG."
        )


    return {
        "required": len(reasons) > 0,
        "reasons": reasons
    }


# ---------------------------------------------------------
# 5. REQUEST HUMAN APPROVAL
# ---------------------------------------------------------

def request_human_approval(
    workflow_id,
    response
):

    state = get_state(
        workflow_id
    )

    decision = requires_human_approval(
        workflow_id
    )


    print("\n" + "=" * 70)
    print("HUMAN REVIEW REQUIRED")
    print("=" * 70)


    print("\nDV NUMBER:")
    print(
        state["dv_number"]
    )


    print("\nDV STATUS:")
    print(
        state["dv_status"]
    )


    print("\nACCOUNTING ANALYST REMARKS:")
    print(
        clean_text(
            state["remarks"]
        )
    )


    print("\nWHY APPROVAL IS REQUIRED:")

    for reason in decision["reasons"]:

        print(
            "-",
            reason
        )


    print("\nPROPOSED RESPONSE:")

    print(
        response
    )


    approval = input(
        "\nApprove this response? (yes/no): "
    ).strip().lower()


    if approval == "yes":

        update_state(
            workflow_id,
            workflow_status="Approved"
        )

        show_control_progress(
            workflow_id,
            "HUMAN DECISION: APPROVED"
        )

        return True


    update_state(
        workflow_id,
        workflow_status="Approval Rejected"
    )

    show_control_progress(
        workflow_id,
        "HUMAN DECISION: REJECTED"
    )

    return False


# ---------------------------------------------------------
# 6. RELEASE THE RESPONSE
# ---------------------------------------------------------

def release_response(
    workflow_id,
    response
):

    update_state(
        workflow_id,
        response=response,
        workflow_status="Response Released"
    )


    # Only released responses become assistant memory
    save_message(
        workflow_id,
        "assistant",
        response
    )


    show_control_progress(
        workflow_id,
        "RESPONSE RELEASED"
    )


    print("\nFINAL RESPONSE TO USER")

    print(
        response
    )


    return response


# ---------------------------------------------------------
# 7. CONTROL THE PROPOSED RESPONSE
# ---------------------------------------------------------

def control_response(
    workflow_id,
    response
):

    # Validate first
    validation = validate_response(
        workflow_id,
        response
    )


    print("\nVALIDATION RESULT")

    print(
        "Valid:",
        validation["valid"]
    )


    if validation["issues"]:

        for issue in validation["issues"]:

            print(
                "-",
                issue
            )


    if not validation["valid"]:

        update_state(
            workflow_id,
            workflow_status="Response Blocked"
        )

        show_control_progress(
            workflow_id,
            "CONTROL RESULT: RESPONSE BLOCKED"
        )

        return None


    # Determine approval requirement
    decision = requires_human_approval(
        workflow_id
    )


    print("\nAPPROVAL CHECK")

    print(
        "Human approval required:",
        decision["required"]
    )


    if decision["required"]:

        for reason in decision["reasons"]:

            print(
                "-",
                reason
            )


        update_state(
            workflow_id,
            workflow_status="Awaiting Approval"
        )


        show_control_progress(
            workflow_id,
            "AWAITING HUMAN APPROVAL"
        )


        approved = request_human_approval(
            workflow_id,
            response
        )


        if not approved:

            print(
                "\nResponse was not released."
            )

            return None


    return release_response(
        workflow_id,
        response
    )


# ---------------------------------------------------------
# 8. COMPLETE CONTROLLED DV WORKFLOW
# ---------------------------------------------------------

def run_controlled_dv_agent(
    workflow_id,
    question
):


    # -----------------------------------------------------
    # START WORKFLOW
    # -----------------------------------------------------

    start_workflow(
        workflow_id,
        question
    )


    show_control_progress(
        workflow_id,
        "1. WORKFLOW STARTED"
    )


    # -----------------------------------------------------
    # IDENTIFY DV
    # -----------------------------------------------------

    dv_number = extract_dv_number(
        question
    )


    if not dv_number:

        update_state(
            workflow_id,
            workflow_status="Awaiting DV Number"
        )

        show_control_progress(
            workflow_id,
            "2. DV NUMBER NOT FOUND"
        )

        print(
            "\nPlease provide the DV number."
        )

        return None


    update_state(
        workflow_id,
        dv_number=dv_number,
        workflow_status="DV Identified"
    )


    show_control_progress(
        workflow_id,
        "2. DV IDENTIFIED"
    )


    # -----------------------------------------------------
    # RETRIEVE DV
    # -----------------------------------------------------

    dv = get_dv(
        dv_number
    )


    if "error" in dv:

        update_state(
            workflow_id,
            workflow_status="DV Not Found"
        )


        show_control_progress(
            workflow_id,
            "3. DV NOT FOUND"
        )


        print(
            "\nDV not found. "
            "Please verify the DV number."
        )

        return None


    update_state(
        workflow_id,
        dv_type=dv["dv_type"],
        dv_status=dv["status"],
        remarks=dv["remarks"],
        workflow_status="DV Retrieved"
    )


    show_control_progress(
        workflow_id,
        "3. DV RECORD RETRIEVED"
    )


    # -----------------------------------------------------
    # DETERMINE WHETHER RAG IS REQUIRED
    # -----------------------------------------------------

    state = get_state(
        workflow_id
    )


    use_rag = needs_rag(
        state
    )


    if use_rag:

        update_state(
            workflow_id,
            needs_rag=1,
            workflow_status="RAG Required"
        )


        show_control_progress(
            workflow_id,
            "4. RAG REQUIRED"
        )


        evidence = (
            dv_rag.retrieve_dv_requirements(
                dv_type=state["dv_type"],
                question=question,
                analyst_remarks=state["remarks"],
                k=3
            )
        )


        # Reuse Exercise 3 cleaning helper
        evidence = clean_rag_evidence(
            evidence
        )


        update_state(
            workflow_id,
            rag_evidence=json.dumps(
                evidence,
                ensure_ascii=False
            ),
            workflow_status="Evidence Retrieved"
        )


        show_control_progress(
            workflow_id,
            "5. RAG EVIDENCE RETRIEVED"
        )


    else:

        update_state(
            workflow_id,
            needs_rag=0,
            workflow_status="Ready for Response"
        )


        show_control_progress(
            workflow_id,
            "4. RAG NOT REQUIRED"
        )


    # -----------------------------------------------------
    # GENERATE DRAFT
    # -----------------------------------------------------

    response = generate_helpdesk_draft(
        workflow_id,
        question
    )


    update_state(
        workflow_id,
        response=response,
        workflow_status="Draft Generated"
    )


    show_control_progress(
        workflow_id,
        "6. DRAFT GENERATED"
    )


    print("\nDRAFT RESPONSE")

    print(
        response
    )


    # -----------------------------------------------------
    # APPLY CONTROLS
    # -----------------------------------------------------

    return control_response(
        workflow_id,
        response
    )


# ---------------------------------------------------------
# 9. TEST CONTROLLED WORKFLOW
# ---------------------------------------------------------

WORKFLOW_ID = "INQ-006"


question = (
    "Why was DV-2026-00123 returned, "
    "and what document do I need to submit?"
)


final_response = run_controlled_dv_agent(
    WORKFLOW_ID,
    question
)


# ---------------------------------------------------------
# 10. FINAL STATE AND MEMORY
# ---------------------------------------------------------

show_control_progress(
    WORKFLOW_ID,
    "FINAL STATE AND MEMORY"
)