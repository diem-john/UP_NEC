# CASE STUDY 5: DV HELPDESK AGENT | Run #%% cells in order (Spyder)
# Required: dv_helpdesk.db, agent_state.db, rag_knowledge.db, dv_rag.py, PreAudit Checklist PDF
# One-time: ollama pull qwen3.5:4b && ollama pull nomic-embed-text
#%%  ANATOMY OF A DV HELPDESK AGENT
import json,re,sqlite3
from pathlib import Path
from contextlib import closing
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
import dv_rag
MODEL="qwen3.5:4b"; DV_DB="dv_helpdesk.db"; STATE_DB="agent_state.db"
FILES=[DV_DB,STATE_DB,"rag_knowledge.db","dv_rag.py","PreAudit Checklist for Common Transactions.pdf"]
missing=[f for f in FILES if not Path(f).is_file()]
if missing: raise FileNotFoundError("Missing workshop files: "+", ".join(missing))
def read_db(sql,args=(),file=DV_DB,all_rows=False):
    with closing(sqlite3.connect(file)) as c:
        c.row_factory=sqlite3.Row; c.execute("PRAGMA query_only=ON")
        rows=c.execute(sql,args).fetchall() if all_rows else c.execute(sql,args).fetchone()
        return [dict(r) for r in rows] if all_rows else (dict(rows) if rows else None)
def write_state(sql,args):
    with closing(sqlite3.connect(STATE_DB)) as c: c.execute(sql,args); c.commit()
def get_dv(number):
    row=read_db("SELECT dv_number,dv_type,payee,amount,status,remarks,accounting_analyst,last_updated,payment_date FROM disbursement_vouchers WHERE UPPER(dv_number)=UPPER(?)",(number.strip(),))
    return row or {"error":"DV not found","dv_number":number}
llm=ChatOllama(model=MODEL,reasoning=False,temperature=0)
print("MODEL:",MODEL,"| FILES:",len(FILES),"found")
for r in read_db("SELECT dv_number,dv_type,status,remarks FROM disbursement_vouchers ORDER BY dv_number LIMIT 3",all_rows=True): print(r)
initial={"question":"Why was DV-2026-00123 returned?","dv_number":None,"status":"Started"}
print("INITIAL STATE:",initial)
print("AGENT DESIGN:",llm.invoke([("system","Describe the future roles of SQLite, RAG, tools, LLM, and workflow state in a DV helpdesk. Do not claim any tool was used."),("human",initial["question"])]).content)
#%%  TOOLS AND FUNCTION CALLING
@tool
def query_dv(dv_number:str)->dict:
    """Read the authoritative status, type, and remarks of a single DV."""
    return get_dv(dv_number)
@tool
def search_dv_requirements(dv_type:str,question:str,analyst_remarks:str="")->list:
    """Retrieve checklist evidence about required DV attachments and compliance."""
    return dv_rag.retrieve_dv_requirements(dv_type=dv_type,question=question,analyst_remarks=analyst_remarks,k=3)
agent=create_agent(model=llm,tools=[query_dv,search_dv_requirements],system_prompt="""You are a read-only DV Helpdesk Agent. For a specific DV, query_dv first. For missing attachment, returned-for-compliance, or policy questions, then search_dv_requirements using the DV type and actual analyst remarks returned by query_dv. Use the database as authority for DV status and remarks, and checklist retrieval as evidence for general requirements. Never invent facts, change records, or approve a DV. State clearly when evidence is insufficient.""")
def run_agent(question): return agent.invoke({"messages":[{"role":"user","content":question}]})
def show_trace(result):
    for m in result["messages"]:
        for t in getattr(m,"tool_calls",[]) or []: print("TOOL:",t["name"],"ARGS:",t["args"])
    print("AGENT ANSWER:",result["messages"][-1].content)
show_trace(run_agent("Why was DV-2026-00123 returned, and what document should I submit?"))
#%%  PERSISTENT STATE AND WORKFLOW CONTROL
FIELDS={"dv_number","dv_type","dv_status","remarks","needs_rag","rag_evidence","response","workflow_status"}
def start_workflow(w,q):
    write_state("INSERT OR REPLACE INTO inquiry_state(workflow_id,user_question,dv_number,dv_type,dv_status,remarks,needs_rag,rag_evidence,response,workflow_status,updated_at) VALUES(?,?,NULL,NULL,NULL,NULL,0,NULL,NULL,?,?)",(w,q,"Started",datetime.now().isoformat()))
def get_state(w):
    s=read_db("SELECT * FROM inquiry_state WHERE workflow_id=?",(w,),STATE_DB)
    if s:
        s["needs_rag"]=bool(s["needs_rag"])
        if isinstance(s["rag_evidence"],str): s["rag_evidence"]=json.loads(s["rag_evidence"])
    return s
def update_state(w,**changes):
    if not changes or set(changes)-FIELDS: raise ValueError("Invalid state fields")
    changes["updated_at"]=datetime.now().isoformat()
    write_state("UPDATE inquiry_state SET "+", ".join(f"{k}=?" for k in changes)+" WHERE workflow_id=?",[*changes.values(),w])
def clean(text): return re.sub(r"\s+"," ",str(text or "")).strip()
def show_state(w,stage):
    s=get_state(w)
    print(f"{stage} | {w} | {s['workflow_status']} | DV:{s['dv_number']} | Status:{s['dv_status']} | RAG:{s['needs_rag']} | Evidence:{len(s['rag_evidence'] or [])}")
def extract_dv(question):
    m=re.search(r"DV-\d{4}-\d+",question,re.I)
    return m.group(0).upper() if m else None
def needs_rag(question):
    return any(w in question.lower() for w in ("why","return","compliance","attach","missing","document","requirement","submit"))
def run_workflow(w,q):
    start_workflow(w,q); show_state(w,"START")
    number=extract_dv(q)
    if not number:
        update_state(w,workflow_status="Awaiting DV Number"); show_state(w,"STOP"); return
    update_state(w,dv_number=number,workflow_status="DV Identified"); show_state(w,"IDENTIFIED")
    dv=get_dv(number)
    if "error" in dv:
        update_state(w,workflow_status="DV Not Found"); show_state(w,"STOP"); return
    update_state(w,dv_type=dv["dv_type"],dv_status=dv["status"],remarks=dv["remarks"],workflow_status="DV Retrieved"); show_state(w,"LOOKUP")
    if needs_rag(q):
        update_state(w,needs_rag=1,workflow_status="RAG Required"); show_state(w,"ROUTE")
        evidence=dv_rag.retrieve_dv_requirements(dv_type=dv["dv_type"],question=q,analyst_remarks=dv["remarks"] or "",k=3)
        update_state(w,rag_evidence=json.dumps(evidence,ensure_ascii=False),workflow_status="Ready for Response"); show_state(w,"RAG DONE")
    else:
        update_state(w,needs_rag=0,workflow_status="Ready for Response"); show_state(w,"NO RAG")
RUN=datetime.now().strftime("%H%M%S%f")
WA=f"INQ-5A-{RUN}"; WB=f"INQ-5B-{RUN}"
run_workflow(WA,"Why was DV-2026-00123 returned, and what document should I submit?")
run_workflow(WB,"What is the status of DV-2026-00125?")
#%%  CONVERSATION MEMORY AND GROUNDED RESPONSE
def save_message(w,role,message):
    last=read_db("SELECT role,message FROM conversation_history WHERE workflow_id=? ORDER BY message_id DESC LIMIT 1",(w,),STATE_DB)
    if not last or (last["role"],last["message"])!=(role,message):
        write_state("INSERT INTO conversation_history(workflow_id,role,message,created_at) VALUES(?,?,?,?)",(w,role,message,datetime.now().isoformat()))
def recent(w,limit=6):
    rows=read_db("SELECT role,message FROM conversation_history WHERE workflow_id=? ORDER BY message_id DESC LIMIT ?",(w,limit),STATE_DB,True)
    return rows[::-1]
def build_context(w):
    s=get_state(w); chunks=s["rag_evidence"] or []
    evidence="\n".join(f"Page {x.get('page')}: {clean(x.get('text'))[:350]}" for x in chunks[:3]) or "No checklist evidence retrieved."
    history="\n".join(f"{x['role']}: {clean(x['message'])}" for x in recent(w))
    return f"DV: {s['dv_number']} | Type: {s['dv_type']} | Status: {s['dv_status']}\nAnalyst remarks: {s['remarks']}\nCHECKLIST: {evidence}\nRECENT CONVERSATION:\n{history}"
RESPONSE_RULES="""You are a read-only DV Helpdesk Agent. Use DV state for the actual status and analyst remarks and retrieved checklist snippets for general documentary requirements. Never invent details, missing attachments, payment dates, or policies. Use conversation history only for follow-up references. State when evidence is insufficient. Write a concise answer, as a draft until the application authorizes its release."""
def draft_answer(w,question):
    save_message(w,"user",question)
    text=""; print("DRAFT: ",end="",flush=True)
    for chunk in llm.stream([("system",RESPONSE_RULES),("human",f"CONTEXT:\n{build_context(w)}\nQUESTION:\n{question}")]):
        token=chunk.content if isinstance(chunk.content,str) else ""
        print(token,end="",flush=True); text+=token
    print(); return text.strip()
q1="What is the status of DV-2026-00125?"
a1=draft_answer(WB,q1); save_message(WB,"assistant",a1)
q2="Please repeat the DV number and status from my previous question."
a2=draft_answer(WB,q2); save_message(WB,"assistant",a2)
print("MEMORY:",[(m["role"],clean(m["message"])[:85]) for m in recent(WB,10)])
#%%  VALIDATION AND HUMAN APPROVAL BEFORE RELEASE
def validate_response(w,text):
    s=get_state(w); issues=[]
    if not s: return ["Workflow not found"]
    if not s["dv_number"]: issues.append("DV number missing")
    if not s["dv_status"]: issues.append("DV status missing")
    if not text.strip(): issues.append("Empty response")
    if s["needs_rag"] and not s["rag_evidence"]: issues.append("Missing required RAG evidence")
    return issues
def approval_reasons(w):
    s=get_state(w); reasons=[]
    if s["dv_status"]=="Returned for Compliance": reasons.append("Returned for compliance")
    if s["needs_rag"]: reasons.append("Response includes checklist-based requirements")
    return reasons
def release_response(w,text):
    update_state(w,response=text,workflow_status="Response Released")
    save_message(w,"assistant",text); show_state(w,"RELEASED"); print("FINAL RESPONSE:",text)
    return text
def control_response(w,text):
    issues=validate_response(w,text)
    if issues:
        update_state(w,workflow_status="Response Blocked"); print("BLOCKED:",issues); return None
    reasons=approval_reasons(w)
    if reasons:
        update_state(w,workflow_status="Awaiting Approval"); show_state(w,"APPROVAL REQUIRED")
        print("REASONS:","; ".join(reasons))
        if input("Approve draft for release? (yes/no): ").strip().lower()!="yes":
            update_state(w,workflow_status="Approval Rejected"); show_state(w,"NOT RELEASED"); return None
        update_state(w,workflow_status="Approved"); show_state(w,"APPROVED")
    return release_response(w,text)
def run_controlled_agent(w,question):
    run_workflow(w,question)
    if get_state(w)["workflow_status"]!="Ready for Response": return None
    draft=draft_answer(w,question)
    update_state(w,response=draft,workflow_status="Draft Generated"); show_state(w,"DRAFT SAVED")
    return control_response(w,draft)
WC=f"INQ-5C-{RUN}"
final_response=run_controlled_agent(WC,"Why was DV-2026-00123 returned, and what document should I submit?")
print("FINAL MEMORY:",[(m["role"],clean(m["message"])[:85]) for m in recent(WC,10)])
