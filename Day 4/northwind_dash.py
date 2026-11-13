#%% NORTHWIND AGENT CHATBOT — FLASK + DASH + LANGGRAPH + OLLAMA
# Requires northwind_agent.py and northwind.db in the same folder
import re,uuid
from pathlib import Path
from flask import Flask,jsonify
from dash import Dash,html,dcc,Input,Output,State,ctx,no_update
from langgraph.types import Command
import northwind_agent as nw

if not Path(nw.DB).is_file():
    raise FileNotFoundError(f"Place northwind.db beside northwind_agent.py: {nw.DB}")
server=Flask(__name__)
app=Dash(__name__,server=server)
app.title="Northwind Order Agent"
MAX_HISTORY=20
PAGE={"fontFamily":"Arial","maxWidth":"920px","margin":"24px auto","padding":"18px","backgroundColor":"#f5f6f8"}
PANEL={"backgroundColor":"white","padding":"16px","borderRadius":"12px","marginBottom":"12px"}
USER={"backgroundColor":"#dbeafe","margin":"8px 0 8px auto","padding":"12px","borderRadius":"12px","maxWidth":"80%"}
BOT={"backgroundColor":"#f0f0f0","margin":"8px auto 8px 0","padding":"12px","borderRadius":"12px","maxWidth":"80%"}
BUTTON={"padding":"11px 16px","cursor":"pointer","fontSize":"15px"}

@server.get("/health")
def health():
    return jsonify({"status":"running","app":"Northwind Agent"})

def summary(s,pending=False):
    text=f"**Order {s.get('order_id','?')}**\n\n**Assessment:** {s.get('assessment','Unavailable')}\n\n**Recommendation:** {s.get('recommendation','Unavailable')}\n\n**Reason:** {s.get('recommendation_reason','Unavailable')}"
    text+=f"\n\n**Status:** {'AWAITING HUMAN APPROVAL' if pending else s.get('status','Unknown')}"
    if s.get("trace"):
        text+="\n\n**Agent trace:** "+" → ".join(s["trace"])
    return text

app.layout=html.Div(style=PAGE,children=[
    html.H2("Northwind Order Investigation Agent",style={"textAlign":"center"}),
    html.P("LangGraph specialist agents + SQLite + Ollama + human approval",style={"textAlign":"center"}),
    dcc.Store(id="session",storage_type="session",data={"history":[],"order_id":None,"pending":None}),
    html.Div(style=PANEL,children=[
        html.Div(id="chat-window",style={"height":"440px","overflowY":"auto","padding":"12px","border":"1px solid #ddd","borderRadius":"8px"})
    ]),
    html.Div(style=PANEL,children=[
        dcc.Textarea(id="user-input",placeholder="Example: Investigate order 11077 and recommend an action.",style={"width":"100%","height":"80px","boxSizing":"border-box","padding":"10px","fontSize":"16px"}),
        html.Div(style={"display":"flex","gap":"8px","flexWrap":"wrap","marginTop":"10px"},children=[
            html.Button("Send",id="send",style=BUTTON),
            html.Button("Approve",id="approve",disabled=True,style=BUTTON),
            html.Button("Reject",id="reject",disabled=True,style=BUTTON),
            html.Button("Clear",id="clear",style=BUTTON)
        ])
    ]),
    html.P("This workshop app simulates order actions. Approval does not modify Northwind database records.",style={"fontSize":"13px","color":"#555"})
])

@app.callback(Output("session","data"),Output("user-input","value"),Input("send","n_clicks"),Input("approve","n_clicks"),Input("reject","n_clicks"),Input("clear","n_clicks"),State("user-input","value"),State("session","data"),prevent_initial_call=True)
def chat(send_clicks,approve_clicks,reject_clicks,clear_clicks,message,data):
    event=ctx.triggered_id
    if event=="clear":
        return {"history":[],"order_id":None,"pending":None},""
    data=dict(data or {})
    history=list(data.get("history") or [])
    pending=data.get("pending")
    if event in ("approve","reject"):
        if not pending:
            return no_update,no_update
        decision="approve" if event=="approve" else "reject"
        history.append({"role":"user","content":decision.title()+" recommendation"})
        try:
            result=nw.graph.invoke(Command(resume=decision),config={"configurable":{"thread_id":pending}})
            answer=summary(result)
            data["pending"]=None
        except Exception as e:
            answer=f"Approval was not completed: {e}"
        history.append({"role":"assistant","content":answer})
    elif event=="send":
        question=(message or "").strip()
        if not question:
            return no_update,no_update
        history.append({"role":"user","content":question})
        if pending:
            answer="Please approve or reject the pending recommendation before starting another investigation."
        else:
            match=re.search(r"\border\s*#?\s*(\d+)\b",question,re.I)
            order_id=int(match.group(1)) if match else data.get("order_id")
            if not order_id:
                answer="Please include a Northwind order number, for example: Investigate order 11077."
            else:
                try:
                    if not nw.get_order(order_id):
                        answer=f"Order {order_id} was not found in Northwind."
                    else:
                        thread_id=uuid.uuid4().hex
                        config={"configurable":{"thread_id":thread_id}}
                        prior="\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-5:-1])
                        request=f"Recent conversation:\n{prior}\nCurrent request: {question}" if prior else question
                        result=nw.graph.invoke({"request":request,"order_id":order_id,"status":"RECEIVED","status_history":[],"memory":[],"trace":[]},config=config)
                        waiting=bool(result.get("__interrupt__"))
                        state=nw.graph.get_state(config).values if waiting else result
                        answer=summary(state,waiting)
                        data["order_id"]=order_id
                        data["pending"]=thread_id if waiting else None
                except Exception as e:
                    answer=f"The Northwind workflow failed: {e}"
        history.append({"role":"assistant","content":answer})
    else:
        return no_update,no_update
    data["history"]=history[-MAX_HISTORY:]
    return data,""

@app.callback(Output("chat-window","children"),Output("approve","disabled"),Output("reject","disabled"),Input("session","data"))
def render(data):
    data=data or {}
    history=data.get("history") or []
    bubbles=[html.Div([html.B("You" if m["role"]=="user" else "Northwind Agent"),dcc.Markdown(m["content"])],style=USER if m["role"]=="user" else BOT) for m in history]
    pending=bool(data.get("pending"))
    return bubbles or html.P("Try: Investigate order 11077 and recommend an action."),not pending,not pending

if __name__=="__main__":
    app.run(host="127.0.0.1",port=8050,debug=False,use_reloader=False)
