#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
from dash import Dash,dcc,html
import plotly.graph_objects as go

app=Dash(__name__)

fig=go.Figure([
    go.Bar(x=[1,2,3],y=[4,1,2],name="SF"),
    go.Bar(x=[1,2,3],y=[2,4,5],name="Montréal")
])

fig.update_layout(title="Dash Data Visualization",barmode="group")

app.layout=html.Div([
    html.H1("Hello Dash Rex Jalao"),
    html.P("Dash: A web application framework for Python."),
    dcc.Graph(id="example-graph",figure=fig)
])

if __name__ == "__main__":   
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )
#%%
from dash import jupyter_dash

server = jupyter_dash._servers.get(("127.0.0.1", 8050))

if server:
    server.shutdown()
    del jupyter_dash._servers[("127.0.0.1", 8050)]
    print("Dash server stopped.")
else:
    print("No Dash server is running on port 8050.")    
#%%
from dash import Dash,dcc,html

app=Dash(__name__)

cities=[
    {"label":"New York City","value":"NYC"},
    {"label":"Montréal","value":"MTL"},
    {"label":"San Francisco","value":"SF"}
]

app.layout=html.Div([
    html.Label("Dropdown"),
    dcc.Dropdown(options=cities,value="MTL"),

    html.Label("Multi-Select Dropdown"),
    dcc.Dropdown(options=cities,value=["MTL","SF"],multi=True),

    html.Label("Slider"),
    dcc.Slider(
        min=-5,max=10,step=0.5,
        marks={i:str(i) for i in range(-5,11)},
        value=-3
    ),

    html.Label("Radio Items"),
    dcc.RadioItems(options=cities,value="MTL")
],style={"width":"50%"})

if __name__=="__main__":
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )

#%% EXERCISE 1: CREATE A DASH CHATBOT LAYOUT
from flask import Flask
from dash import Dash,html,dcc

# 1. CREATE FLASK AND DASH APPLICATION
server=Flask(__name__)
app=Dash(__name__,server=server)
app.title="Northwind Chatbot"

# 2. DEFINE BASIC STYLES
page={"fontFamily":"Arial","maxWidth":"850px","margin":"30px auto","padding":"20px"}
panel={"backgroundColor":"#f3f4f6","padding":"15px","borderRadius":"10px"}

# 3. CREATE APPLICATION LAYOUT
app.layout=html.Div([
    html.H2("Northwind Order Investigation Agent"),
    html.P("Agentic AI powered by LangGraph and Ollama"),
    html.H4("Conversation"),
    html.Div([
        html.Div("Northwind Agent: Welcome! How can I help you?",
                 style={"backgroundColor":"white","padding":"12px","marginBottom":"10px","borderRadius":"10px"}),
        html.Div("You: Investigate order 11077.",
                 style={"backgroundColor":"#dbeafe","padding":"12px","borderRadius":"10px","textAlign":"right"})
    ],id="chat-window",style={**panel,"height":"300px","overflowY":"auto"}),
    html.H4("Enter Your Question"),
    dcc.Textarea(
        id="question",
        placeholder="Type a question about a Northwind order...",
        style={"width":"100%","height":"80px","marginBottom":"10px","boxSizing":"border-box"}
    ),
    html.Div([
        html.Button("Send",id="send",style={"padding":"10px 25px","marginRight":"10px"}),
        html.Button("Clear",id="clear",style={"padding":"10px 25px"})
    ]),
    html.P("Agent Status: Not Connected",id="status",style={"color":"gray"})
],style=page)

# 4. LAUNCH APPLICATION
if __name__=="__main__":
    app.run(debug=False,use_reloader=False,port=8050,jupyter_mode="external",)
#%%

from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)


app.layout = html.Div([
    html.H3("Change the value in the text box to see callbacks in action!"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='initial value', type='text')]),
    html.Br(),
    html.Div(id='my-output'),

])


@app.callback(
    [Output(component_id='my-output', component_property='children')],
    [Input(component_id='my-input', component_property='value')]
)
def update_output_div(input_value):
    return ['Output: {}'.format(input_value)]


if __name__ == '__main__':
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )
    
#%%


from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        step=None
    )
])


@app.callback(
    [Output('graph-with-slider', 'figure')],
    [Input('year-slider', 'value')])
def update_figure(selected_year):
    filtered_df = df[df.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return [fig]


if __name__ == '__main__':
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )
#%% EXERCISE 2: IMPLEMENT INTERACTIVE CALLBACKS
import re
from dash import Input,Output,State,no_update
from flask import Flask
from dash import Dash,html,dcc

# 1. CREATE FLASK AND DASH APPLICATION
server=Flask(__name__)
app=Dash(__name__,server=server)
app.title="Northwind Chatbot"

# 2. DEFINE BASIC STYLES
page={"fontFamily":"Arial","maxWidth":"850px","margin":"30px auto","padding":"20px"}
panel={"backgroundColor":"#f3f4f6","padding":"15px","borderRadius":"10px"}

# 3. CREATE APPLICATION LAYOUT
app.layout=html.Div([
    html.H2("Northwind Order Investigation Agent"),
    html.P("Agentic AI powered by LangGraph and Ollama"),
    html.H4("Conversation"),
    html.Div([
        html.Div("Northwind Agent: Welcome! How can I help you?",
                 style={"backgroundColor":"white","padding":"12px","marginBottom":"10px","borderRadius":"10px"}),
        html.Div("You: Investigate order 11077.",
                 style={"backgroundColor":"#dbeafe","padding":"12px","borderRadius":"10px","textAlign":"right"})
    ],id="chat-window",style={**panel,"height":"300px","overflowY":"auto"}),
    html.H4("Enter Your Question"),
    dcc.Textarea(
        id="question",
        placeholder="Type a question about a Northwind order...",
        style={"width":"100%","height":"80px","marginBottom":"10px","boxSizing":"border-box"}
    ),
    html.Div([
        html.Button("Send",id="send",style={"padding":"10px 25px","marginRight":"10px"}),
        html.Button("Clear",id="clear",style={"padding":"10px 25px"})
    ]),
    html.P("Agent Status: Not Connected",id="status",style={"color":"gray"})
],style=page)

# 1. CALLBACK: CONNECT SEND BUTTON TO PYTHON
@app.callback(
    Output("chat-window","children"),
    Output("question","value"),
    Input("send","n_clicks"),
    State("question","value"),
    prevent_initial_call=True
)
def send_message(n,question):
    question=(question or "").strip()
    if not question:
        return no_update,no_update

    # 2. SIMULATE NORTHWIND AGENT RESPONSE
    match=re.search(r"order\s*#?\s*(\d+)",question,re.I)
    if match:
        reply=f"Order {match.group(1)} received. The investigation agent is not yet connected."
    else:
        reply="Please include an order number, such as order 11077."

    # 3. DISPLAY USER AND ASSISTANT MESSAGES
    messages=[
        html.Div(f"You: {question}",
                 style={"backgroundColor":"#dbeafe","padding":"12px","marginBottom":"10px","borderRadius":"10px","textAlign":"right"}),
        html.Div(f"Northwind Agent: {reply}",
                 style={"backgroundColor":"white","padding":"12px","borderRadius":"10px"})
    ]
    return messages,""

# 4. LAUNCH APPLICATION
if __name__=="__main__":
    app.run(debug=False,use_reloader=False,port=8050)


#%%

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

available_indicators = df['Indicator Name'].unique()

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic'),

    dcc.Slider(
        id='year--slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()},
        step=None
    )
])

@app.callback(
    [Output('indicator-graphic', 'figure')],
   [ Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value'),
    Input('year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                     y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                     hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    return [fig]

if __name__ == '__main__':
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )
#%%

from dash import Dash, dcc, html, Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

all_options = {
    'America': ['New York City', 'San Francisco', 'Cincinnati'],
    'Canada': [u'Montréal', 'Toronto', 'Ottawa']
}
app.layout = html.Div([
    dcc.RadioItems(
        id='countries-radio',
        options=[{'label': k, 'value': k} for k in all_options.keys()],
        value='America'
    ),

    html.Hr(),

    dcc.RadioItems(id='cities-radio'),

    html.Hr(),

    html.Div(id='display-selected-values')
])


@app.callback(
    [Output('cities-radio', 'options')],
   [ Input('countries-radio', 'value')])
def set_cities_options(selected_country):
    list = [{'label': i, 'value': i} for i in all_options[selected_country]]
    return [list]


@app.callback(
    [Output('cities-radio', 'value')],
    [Input('cities-radio', 'options')])
def set_cities_value(available_options):
    return [available_options[0]['value']]


@app.callback(
    [Output('display-selected-values', 'children')],
    [Input('countries-radio', 'value'),
    Input('cities-radio', 'value')])
def set_display_children(selected_country, selected_city):
    return [u'{} is a city in {}'.format(
        selected_city, selected_country,
    )]


if __name__ == '__main__':
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )
#%%
from dash import Dash, dcc, html, Input, Output, State, ctx

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div('Convert Temperature'),
    'Celsius',
    dcc.Input(
        id="celsius",
        value=0.0,
        type="number"
    ),
    ' = Fahrenheit',
    dcc.Input(
        id="fahrenheit",
        value=32.0,
        type="number",
    ),
])

@app.callback(
    Output("celsius", "value"),
    Output("fahrenheit", "value"),
    Input("celsius", "value"),
    Input("fahrenheit", "value"),
)
def sync_input(celsius, fahrenheit):

    input_id = ctx.triggered_id
    if input_id == "celsius":
        fahrenheit= None if celsius is None else (float(celsius) * 9/5) + 32
    else:
        celsius = None if fahrenheit is None else (float(fahrenheit) - 32) * 5/9
    return celsius, fahrenheit

if __name__ == "__main__":
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False
        
    )

#%% EXERCISE 3: CONVERSATION HISTORY AND MULTIPLE CONTROLS
  
 
import re
from dash import Input,Output,State,no_update
from flask import Flask
from dash import Dash,html,dcc

# 1. CREATE FLASK AND DASH APPLICATION
server=Flask(__name__)
app=Dash(__name__,server=server)
app.title="Northwind Chatbot"

# 2. DEFINE BASIC STYLES
page={"fontFamily":"Arial","maxWidth":"850px","margin":"30px auto","padding":"20px"}
panel={"backgroundColor":"#f3f4f6","padding":"15px","borderRadius":"10px"}

# 3. CREATE APPLICATION LAYOUT
app.layout=html.Div([
    dcc.Store(id="chat-history",data=[],storage_type="session"),
    html.H2("Northwind Order Investigation Agent"),
    html.P("Agentic AI powered by LangGraph and Ollama"),
    html.H4("Conversation"),
    html.Div([
        html.Div("Northwind Agent: Welcome! How can I help you?",
                 style={"backgroundColor":"white","padding":"12px","marginBottom":"10px","borderRadius":"10px"}),
        html.Div("You: Investigate order 11077.",
                 style={"backgroundColor":"#dbeafe","padding":"12px","borderRadius":"10px","textAlign":"right"})
    ],id="chat-window",style={**panel,"height":"300px","overflowY":"auto"}),
    html.H4("Enter Your Question"),
    dcc.Textarea(
        id="question",
        placeholder="Type a question about a Northwind order...",
        style={"width":"100%","height":"80px","marginBottom":"10px","boxSizing":"border-box"}
    ),
    html.Div([
        html.Button("Send",id="send",style={"padding":"10px 25px","marginRight":"10px"}),
        html.Button("Clear",id="clear",style={"padding":"10px 25px"})
    ]),
    html.P("Agent Status: Not Connected",id="status",style={"color":"gray"})
],style=page)

# 1. CALLBACK: CONNECT SEND BUTTON TO PYTHON
@app.callback(
    Output("chat-window","children"),
    Output("question","value"),
    Input("send","n_clicks"),
    State("question","value"),
    prevent_initial_call=True
)
def send_message(n,question):
    question=(question or "").strip()
    if not question:
        return no_update,no_update

    # 2. SIMULATE NORTHWIND AGENT RESPONSE
    match=re.search(r"order\s*#?\s*(\d+)",question,re.I)
    if match:
        reply=f"Order {match.group(1)} received. The investigation agent is not yet connected."
    else:
        reply="Please include an order number, such as order 11077."

    # 3. DISPLAY USER AND ASSISTANT MESSAGES
    messages=[
        html.Div(f"You: {question}",
                 style={"backgroundColor":"#dbeafe","padding":"12px","marginBottom":"10px","borderRadius":"10px","textAlign":"right"}),
        html.Div(f"Northwind Agent: {reply}",
                 style={"backgroundColor":"white","padding":"12px","borderRadius":"10px"})
    ]
    return messages,""


# 2. UPDATE HISTORY USING SEND AND CLEAR BUTTONS
@app.callback(
    Output("chat-history","data"),
    Output("question","value"),
    Input("send","n_clicks"),
    Input("clear","n_clicks"),
    State("question","value"),
    State("chat-history","data"),
    prevent_initial_call=True
)
def update_chat(send,clear,question,history):
    if ctx.triggered_id=="clear":
        return [],""
    if ctx.triggered_id!="send" or not (question or "").strip():
        return no_update,no_update

    question=question.strip()
    match=re.search(r"order\s*#?\s*(\d+)",question,re.I)
    if match:
        reply=f"Order {match.group(1)} received. The Northwind agent is not yet connected."
    else:
        reply="Please include an order number, such as order 11077."

    history=list(history or [])
    history.append({"role":"user","content":question})
    history.append({"role":"assistant","content":reply})
    return history[-20:],""

# 3. CHAINED CALLBACK: DISPLAY CONVERSATION HISTORY
@app.callback(
    Output("chat-window","children"),
    Input("chat-history","data")
)
def display_chat(history):
    if not history:
        return html.P("Start a conversation with the Northwind Agent.")

    messages=[]
    for item in history:
        user=item["role"]=="user"
        style={
            "backgroundColor":"#dbeafe" if user else "white",
            "padding":"12px",
            "marginBottom":"10px",
            "borderRadius":"10px",
            "textAlign":"right" if user else "left"
        }
        name="You" if user else "Northwind Agent"
        messages.append(
            html.Div([
                html.B(name),
                html.P(item["content"])
            ],style=style)
        )
    return messages

# 4. RUN APPLICATION
if __name__=="__main__":
    app.run(debug=False,use_reloader=False,port=8050)
#%% EXERCISE 4: CONNECT DASH TO THE NORTHWIND LANGGRAPH AGENT
# Requires northwind_agent.py and northwind.db in the same folder
# pip install flask dash langgraph ollama
# ollama pull qwen3.5:4b
import re
import uuid
from pathlib import Path
from flask import Flask,jsonify
from dash import Dash,html,dcc,Input,Output,State,ctx,no_update
from langgraph.types import Command
import northwind_agent as nw

if not Path(nw.DB).is_file():
    raise FileNotFoundError(f"Missing Northwind database: {nw.DB}")
server=Flask(__name__)
app=Dash(__name__,server=server)
app.title="Northwind Agent Chatbot"
PAGE={"fontFamily":"Arial","maxWidth":"900px","margin":"25px auto","padding":"20px","backgroundColor":"#f5f6f8"}
PANEL={"backgroundColor":"white","padding":"16px","borderRadius":"12px","marginBottom":"12px"}
USER={"backgroundColor":"#dbeafe","margin":"8px 0 8px auto","padding":"12px","borderRadius":"10px","maxWidth":"80%"}
BOT={"backgroundColor":"#f0f0f0","margin":"8px auto 8px 0","padding":"12px","borderRadius":"10px","maxWidth":"80%"}
BUTTON={"padding":"10px 18px","cursor":"pointer","marginRight":"8px"}

# 1. FLASK HEALTH ENDPOINT AND RESPONSE FORMAT
@server.get("/health")
def health():
    return jsonify({"status":"running","service":"Northwind Dash Agent"})

def show_result(s,pending=False):
    text=f"**Order:** {s.get('order_id','?')}\n\n**Assessment:** {s.get('assessment','Not available')}"
    text+=f"\n\n**Recommendation:** {s.get('recommendation','Not available')}"
    text+=f"\n\n**Reason:** {s.get('recommendation_reason','Not available')}"
    text+=f"\n\n**Workflow status:** {'AWAITING HUMAN APPROVAL' if pending else s.get('status','Unknown')}"
    if s.get("trace"):
        text+="\n\n**Agent trace:** "+" → ".join(s["trace"])
    return text

# 2. DASH CHATBOT LAYOUT
app.layout=html.Div(style=PAGE,children=[
    html.H2("Northwind Order Investigation Agent",style={"textAlign":"center"}),
    html.P("Flask + Dash + LangGraph + Ollama",style={"textAlign":"center"}),
    dcc.Store(id="session",storage_type="local",data={"history":[],"order_id":None,"pending":None}),
    html.Div(style=PANEL,children=[
        html.Div(id="chat-window",style={"height":"420px","overflowY":"auto","backgroundColor":"#fafafa","padding":"12px","border":"1px solid #ddd","borderRadius":"8px"})
    ]),
    html.Div(style=PANEL,children=[
        dcc.Textarea(id="question",placeholder="Investigate order 11077 and recommend an action.",style={"width":"100%","height":"80px","padding":"10px","boxSizing":"border-box","fontSize":"16px"}),
        html.Div(style={"marginTop":"10px"},children=[
            html.Button("Send",id="send",style=BUTTON),
            html.Button("Approve",id="approve",disabled=True,style=BUTTON),
            html.Button("Reject",id="reject",disabled=True,style=BUTTON),
            html.Button("Clear",id="clear",style=BUTTON)
        ])
    ]),
    html.P("Workshop simulation: approving a recommendation does not modify database records.",style={"fontSize":"13px","color":"#666"})
])

# 3. CALLBACK: EXECUTE OR RESUME LANGGRAPH
@app.callback(
    Output("session","data"),
    Output("question","value"),
    Input("send","n_clicks"),
    Input("approve","n_clicks"),
    Input("reject","n_clicks"),
    Input("clear","n_clicks"),
    State("question","value"),
    State("session","data"),
    prevent_initial_call=True
)
def update_chat(send,approve,reject,clear,question,data):
    action=ctx.triggered_id
    if action=="clear":
        return {"history":[],"order_id":None,"pending":None},""
    data=dict(data or {})
    history=list(data.get("history") or [])
    pending=data.get("pending")
    if action in ("approve","reject"):
        if not pending:
            return no_update,no_update
        decision="approve" if action=="approve" else "reject"
        history.append({"role":"user","content":decision.title()+" recommendation"})
        try:
            result=nw.graph.invoke(Command(resume=decision),config={"configurable":{"thread_id":pending}})
            answer=show_result(result)
            data["pending"]=None
        except Exception as error:
            answer=f"Could not resume workflow: {error}"
    elif action=="send":
        question=(question or "").strip()
        if not question:
            return no_update,no_update
        history.append({"role":"user","content":question})
        if pending:
            answer="Please approve or reject the pending investigation first."
        else:
            match=re.search(r"\border\s*#?\s*(\d+)\b",question,re.I)
            order_id=int(match.group(1)) if match else data.get("order_id")
            if not order_id:
                answer="Please enter an order number, such as: Investigate order 11077."
            else:
                try:
                    if not nw.get_order(order_id):
                        answer=f"Order {order_id} was not found."
                    else:
                        thread=uuid.uuid4().hex
                        config={"configurable":{"thread_id":thread}}
                        state={"request":question,"order_id":order_id,"status":"RECEIVED","status_history":[],"memory":[],"trace":[]}
                        result=nw.graph.invoke(state,config=config)
                        waiting=bool(result.get("__interrupt__"))
                        values=nw.graph.get_state(config).values if waiting else result
                        answer=show_result(values,waiting)
                        data["order_id"]=order_id
                        data["pending"]=thread if waiting else None
                except Exception as error:
                    answer=f"Agent execution failed: {error}"
    else:
        return no_update,no_update
    history.append({"role":"assistant","content":answer})
    data["history"]=history[-20:]
    return data,""

# 4. CHAINED CALLBACK: SHOW MESSAGES AND APPROVAL BUTTONS
@app.callback(
    Output("chat-window","children"),
    Output("approve","disabled"),
    Output("reject","disabled"),
    Input("session","data")
)
def display_chat(data):
    data=data or {}
    messages=[]
    for item in data.get("history",[]):
        is_user=item["role"]=="user"
        messages.append(html.Div([
            html.B("You" if is_user else "Northwind Agent"),
            dcc.Markdown(item["content"])
        ],style=USER if is_user else BOT))
    pending=bool(data.get("pending"))
    return messages or html.P("Start with: Investigate order 11077."),not pending,not pending

# 5. START FLASK/DASH SERVER
if __name__=="__main__":
    app.run(host="127.0.0.1",port=8050,debug=False,use_reloader=False)
