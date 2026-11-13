# -*- coding: utf-8 -*-
#%pip install langgraph
#%pip install grandalf
# -*- coding: utf-8 -*-
#%pip install langgraph grandalf

#%% EXERCISE 1: INTRODUCTION TO MULTI-AGENT SYSTEMS
# NORTHWIND ORDER INVESTIGATION

import json
import sqlite3
import ollama
from typing import TypedDict
from langgraph.graph import StateGraph,START,END

DB="northwind.db"
MODEL="qwen3.5:4b"

# ==========================================================
# COMMON HELPERS
# ==========================================================

def ask_json(system_prompt,user_prompt):
    r=ollama.chat(
        model=MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        format="json",
        think=False
    )
    return json.loads(r.message.content)

def add_trace(state,message):
    return state.get("trace",[])+[message]

def show_graph(graph):
    print("\nLANGGRAPH WORKFLOW")
    print(graph.get_graph().draw_ascii())

def llm_summary(role,instruction,evidence,key):
    r=ask_json(
        f"""You are the Northwind {role}.
            {instruction}
            Use only the database evidence provided.
            Do not invent information.
            Return JSON:
            {{"{key}":"..."}}""",
                    json.dumps(evidence,indent=2,default=str)
                )
    return r.get(key,"")

# ==========================================================
# DATABASE HELPERS
# ==========================================================

def db_query(sql,params=()):
    with sqlite3.connect(DB) as conn:
        conn.row_factory=sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return [dict(x) for x in conn.execute(sql,params).fetchall()]

def get_order(order_id):
    rows=db_query(
        """SELECT o.OrderID,o.CustomerID,o.OrderDate,o.RequiredDate,
        o.ShippedDate,o.Freight,o.ShipCountry,s.CompanyName AS Shipper
        FROM Orders o
        LEFT JOIN Shippers s ON o.ShipVia=s.ShipperID
        WHERE o.OrderID=?""",
        (order_id,)
    )
    return rows[0] if rows else None

def get_customer(customer_id):
    rows=db_query(
        """SELECT c.CustomerID,c.CompanyName,c.ContactName,c.City,c.Country,
        COUNT(o.OrderID) AS TotalOrders
        FROM Customers c
        LEFT JOIN Orders o ON c.CustomerID=o.CustomerID
        WHERE c.CustomerID=?
        GROUP BY c.CustomerID,c.CompanyName,c.ContactName,c.City,c.Country""",
        (customer_id,)
    )
    return rows[0] if rows else None

def get_order_items(order_id):
    return db_query(
        """SELECT p.ProductID,p.ProductName,od.Quantity,od.UnitPrice,
        od.Discount,p.UnitsInStock,p.UnitsOnOrder,p.ReorderLevel,
        CASE
            WHEN p.UnitsInStock<od.Quantity THEN 'Potential shortage'
            WHEN p.UnitsInStock<=p.ReorderLevel THEN 'Low stock'
            ELSE 'Adequate'
        END AS InventoryStatus
        FROM "Order Details" od
        JOIN Products p ON od.ProductID=p.ProductID
        WHERE od.OrderID=?""",
        (order_id,)
    )

# ==========================================================
# SHARED STATE
# ==========================================================

AgentState=TypedDict(
    "AgentState",
    {
        "request":str,
        "order_id":int,
        "order":dict,
        "order_summary":str,
        "customer":dict,
        "customer_summary":str,
        "items":list[dict],
        "inventory_summary":str,
        "response":str,
        "trace":list[str]
    },
    total=False
)

# ==========================================================
# LLM-POWERED SPECIALIST AGENTS
# ==========================================================

def order_agent(state):
    order=get_order(state["order_id"])
    if not order:
        raise ValueError(f'Order {state["order_id"]} not found.')
    summary=llm_summary(
        "Order Agent",
        "Summarize the order, shipping dates, freight, and fulfillment status.",
        order,
        "order_summary"
    )
    return {
        "order":order,
        "order_summary":summary,
        "trace":add_trace(state,"Order Agent [LLM]")
    }

def customer_agent(state):
    customer=get_customer(state["order"]["CustomerID"])
    summary=llm_summary(
        "Customer Agent",
        "Summarize the customer and previous order activity.",
        customer,
        "customer_summary"
    )
    return {
        "customer":customer,
        "customer_summary":summary,
        "trace":add_trace(state,"Customer Agent [LLM]")
    }

def inventory_agent(state):
    items=get_order_items(state["order_id"])
    summary=llm_summary(
        "Inventory Agent",
        "Identify low-stock and potential-shortage concerns in the order.",
        items,
        "inventory_summary"
    )
    return {
        "items":items,
        "inventory_summary":summary,
        "trace":add_trace(state,"Inventory Agent [LLM]")
    }

def response_agent(state):
    evidence={
        "order":state["order_summary"],
        "customer":state["customer_summary"],
        "inventory":state["inventory_summary"]
    }
    response=llm_summary(
        "Response Agent",
        "Combine the specialist findings into a concise operational assessment.",
        evidence,
        "response"
    )
    return {
        "response":response,
        "trace":add_trace(state,"Response Agent [LLM]")
    }

# ==========================================================
# LANGGRAPH WORKFLOW
# ==========================================================

builder=StateGraph(AgentState)
builder.add_node("order_agent",order_agent)
builder.add_node("customer_agent",customer_agent)
builder.add_node("inventory_agent",inventory_agent)
builder.add_node("response_agent",response_agent)

builder.add_edge(START,"order_agent")
builder.add_edge("order_agent","customer_agent")
builder.add_edge("customer_agent","inventory_agent")
builder.add_edge("inventory_agent","response_agent")
builder.add_edge("response_agent",END)

graph=builder.compile()

# ==========================================================
# RUN
# ==========================================================

ORDER_ID=11077

result=graph.invoke({
    "request":f"Investigate Order {ORDER_ID}.",
    "order_id":ORDER_ID,
    "trace":[]
})

# ==========================================================
# OUTPUT
# ==========================================================

print("\nORDER AGENT")
print(result["order_summary"])

print("\nCUSTOMER AGENT")
print(result["customer_summary"])

print("\nINVENTORY AGENT")
print(result["inventory_summary"])

print("\nFINAL RESPONSE")
print(result["response"])

print("\nEXECUTION TRACE")
print(" -> ".join(result["trace"]))

show_graph(graph)
#%% EXERCISE 2: DESIGNING AGENT ROLES AND RESPONSIBILITIES
# ==========================================================
# ROLE DEFINITIONS
# ==========================================================

role_specs={
    "order_agent":{
        "goal":"Investigate order and shipping information.",
        "inputs":["order_id"],
        "outputs":["order","order_summary"],
        "database":["Orders","Shippers"]
    },
    "customer_agent":{
        "goal":"Assess the customer and previous order activity.",
        "inputs":["order"],
        "outputs":["customer","customer_summary"],
        "database":["Customers","Orders"]
    },
    "inventory_agent":{
        "goal":"Assess ordered products for inventory risk.",
        "inputs":["order_id"],
        "outputs":["items","inventory_summary"],
        "database":["Order Details","Products"]
    }
}

# ==========================================================
# NEW REUSABLE ROLE HELPERS
# ==========================================================

def show_roles(specs):
    print("\nAGENT ROLE CONTRACTS")
    for name,spec in specs.items():
        print(f"\n{name}")
        print(" Goal:",spec["goal"])
        print(" Inputs:",", ".join(spec["inputs"]))
        print(" Outputs:",", ".join(spec["outputs"]))
        print(" Database:",", ".join(spec["database"]))

def role_summary(name,evidence,instruction,key):
    spec=role_specs[name]
    rules=(
        f'{spec["goal"]}\n'
        f'Use only these inputs: {", ".join(spec["inputs"])}.\n'
        f'Database scope: {", ".join(spec["database"])}.\n'
        f'{instruction}'
    )
    return llm_summary(
        name.replace("_"," ").title(),
        rules,
        evidence,
        key
    )

# ==========================================================
# SPECIALIST AGENTS
# ==========================================================

def order_agent(state):
    order=get_order(state["order_id"])
    if not order:
        raise ValueError(f'Order {state["order_id"]} not found.')
    return {
        "order":order,
        "order_summary":role_summary(
            "order_agent",
            order,
            "Summarize order and shipping status only.",
            "order_summary"
        ),
        "trace":add_trace(state,"Order Agent")
    }

def customer_agent(state):
    customer=get_customer(state["order"]["CustomerID"])
    return {
        "customer":customer,
        "customer_summary":role_summary(
            "customer_agent",
            customer,
            "Summarize customer information and order history only.",
            "customer_summary"
        ),
        "trace":add_trace(state,"Customer Agent")
    }

def inventory_agent(state):
    items=get_order_items(state["order_id"])
    return {
        "items":items,
        "inventory_summary":role_summary(
            "inventory_agent",
            items,
            "Identify low-stock and potential-shortage concerns only.",
            "inventory_summary"
        ),
        "trace":add_trace(state,"Inventory Agent")
    }

# ==========================================================
# BUILD WORKFLOW
# ==========================================================

builder=StateGraph(AgentState)
builder.add_node("order_agent",order_agent)
builder.add_node("customer_agent",customer_agent)
builder.add_node("inventory_agent",inventory_agent)

builder.add_edge(START,"order_agent")
builder.add_edge("order_agent","customer_agent")
builder.add_edge("customer_agent","inventory_agent")
builder.add_edge("inventory_agent",END)

graph=builder.compile()

# ==========================================================
# RUN
# ==========================================================

result=graph.invoke({
    "request":f"Investigate Order {ORDER_ID}.",
    "order_id":ORDER_ID,
    "trace":[]
})

# ==========================================================
# OUTPUT
# ==========================================================

show_roles(role_specs)

print("\nORDER AGENT")
print(result["order_summary"])

print("\nCUSTOMER AGENT")
print(result["customer_summary"])

print("\nINVENTORY AGENT")
print(result["inventory_summary"])

print("\nEXECUTION TRACE")
print(" -> ".join(result["trace"]))

show_graph(graph)
#%% EXERCISE 3: ROUTER-SPECIALIST ARCHITECTURE



# ==========================================================
# ADD SUPPLIER ROLE AND DATABASE HELPER
# ==========================================================

role_specs["supplier_agent"]={
    "goal":"Identify suppliers of products in the order.",
    "inputs":["order_id"],
    "outputs":["response"],
    "database":["Order Details","Products","Suppliers"]
}

def get_order_suppliers(order_id):
    return db_query(
        """SELECT DISTINCT s.SupplierID,s.CompanyName,s.ContactName,
        s.Country,p.ProductName
        FROM "Order Details" od
        JOIN Products p ON od.ProductID=p.ProductID
        JOIN Suppliers s ON p.SupplierID=s.SupplierID
        WHERE od.OrderID=?""",
        (order_id,)
    )

def validate_choice(value,allowed,default):
    choices={str(x).strip().lower():x for x in allowed}
    return choices.get(str(value).strip().lower(),default)

# ==========================================================
# NEW REUSABLE SPECIALIST HELPER
# ==========================================================

def specialist_reply(state,name,evidence,instruction):
    response=role_summary(
        name,
        evidence,
        instruction,
        "response"
    )
    label=name.replace("_"," ").title()
    return {
        "response":response,
        "agent_used":label,
        "trace":add_trace(state,label)
    }

def show_route_result(name,result):
    print("\n"+"="*60)
    print(name)
    print("="*60)
    print("Request:",result["request"])
    print("Category:",result["category"])
    print("Agent:",result["agent_used"])
    print("Response:",result["response"])
    print("Trace:"," -> ".join(result["trace"]))

# ==========================================================
# SHARED STATE
# ==========================================================

RouterState=TypedDict(
    "RouterState",
    {
        "request":str,
        "order_id":int,
        "category":str,
        "agent_used":str,
        "response":str,
        "trace":list[str]
    },
    total=False
)

# ==========================================================
# ROUTER AGENT
# ==========================================================

def router_agent(state):
    result=ask_json(
        """You are the Northwind Operations Router.

Classify the request into exactly one category:

order = order status, dates, shipping or freight
customer = customer identity or order history
inventory = products, quantities or stock concerns
supplier = suppliers of products in the order

Return JSON:
{"category":"order|customer|inventory|supplier"}""",
        state["request"]
    )
    category=validate_choice(
        result.get("category",""),
        ["order","customer","inventory","supplier"],
        "order"
    )
    return {
        "category":category,
        "trace":add_trace(state,"Router -> "+category)
    }

# ==========================================================
# SPECIALIST AGENTS
# ==========================================================

def order_specialist(state):
    return specialist_reply(
        state,
        "order_agent",
        get_order(state["order_id"]),
        "Answer only the order and shipping question."
    )

def customer_specialist(state):
    order=get_order(state["order_id"])
    return specialist_reply(
        state,
        "customer_agent",
        get_customer(order["CustomerID"]),
        "Answer only the customer question."
    )

def inventory_specialist(state):
    return specialist_reply(
        state,
        "inventory_agent",
        get_order_items(state["order_id"]),
        "Answer only the inventory question and identify stock concerns."
    )

def supplier_specialist(state):
    return specialist_reply(
        state,
        "supplier_agent",
        get_order_suppliers(state["order_id"]),
        "Answer only the supplier question."
    )

def choose_specialist(state):
    return state["category"]

# ==========================================================
# BUILD ROUTER-SPECIALIST WORKFLOW
# ==========================================================

builder=StateGraph(RouterState)
builder.add_node("router",router_agent)
builder.add_node("order",order_specialist)
builder.add_node("customer",customer_specialist)
builder.add_node("inventory",inventory_specialist)
builder.add_node("supplier",supplier_specialist)

builder.add_edge(START,"router")
builder.add_conditional_edges(
    "router",
    choose_specialist,
    {
        "order":"order",
        "customer":"customer",
        "inventory":"inventory",
        "supplier":"supplier"
    }
)

for node in ["order","customer","inventory","supplier"]:
    builder.add_edge(node,END)

graph=builder.compile()

# ==========================================================
# RUN 1 - INVENTORY QUESTION
# ==========================================================

result1=graph.invoke({
    "request":f"Are there inventory problems in Order {ORDER_ID}?",
    "order_id":ORDER_ID,
    "trace":[]
})

show_route_result(
    "RUN 1 - INVENTORY QUESTION",
    result1
)

# ==========================================================
# RUN 2 - SUPPLIER QUESTION
# ==========================================================

result2=graph.invoke({
    "request":f"Who supplies the products in Order {ORDER_ID}?",
    "order_id":ORDER_ID,
    "trace":[]
})

show_route_result(
    "RUN 2 - SUPPLIER QUESTION",
    result2
)

# ==========================================================
# WORKFLOW VISUALIZATION
# ==========================================================

show_graph(graph)
#%% EXERCISE 4: AGENT COMMUNICATION, HANDOFFS, AND SHARED STATE
# ==========================================================
# NEW REUSABLE HELPERS
# ==========================================================

def set_status(state,status,agent):
    old=state.get("status","START")
    return {
        "status":status,
        "status_history":state.get("status_history",[])+[
            f"{old} -> {status} ({agent})"
        ]
    }

def add_memory(state,message):
    return state.get("memory",[])+[message]

def make_handoff(from_agent,to_agent,task,evidence):
    return {
        "from":from_agent,
        "to":to_agent,
        "task":task,
        "evidence":evidence
    }

def agent_update(state,agent,status,memory,**values):
    return {
        **values,
        "memory":add_memory(state,memory),
        "trace":add_trace(state,agent),
        **set_status(state,status,agent)
    }

def get_product_suppliers(product_ids):
    if not product_ids:
        return []
    marks=",".join(["?"]*len(product_ids))
    return db_query(
        f"""SELECT p.ProductID,p.ProductName,
        s.CompanyName,s.Country
        FROM Products p
        JOIN Suppliers s ON p.SupplierID=s.SupplierID
        WHERE p.ProductID IN ({marks})
        ORDER BY p.ProductName""",
        tuple(product_ids)
    )

# ==========================================================
# SHARED STATE
# ==========================================================

HandoffState=TypedDict(
    "HandoffState",
    {
        "request":str,
        "order_id":int,
        "order":dict,
        "items":list[dict],
        "risk_items":list[dict],
        "suppliers":list[dict],
        "handoff":dict,
        "supplier_review_required":bool,
        "response":str,
        "status":str,
        "status_history":list[str],
        "memory":list[str],
        "trace":list[str]
    },
    total=False
)

# ==========================================================
# ORDER AGENT
# ==========================================================

def order_agent(state):
    order=get_order(state["order_id"])

    if not order:
        raise ValueError(
            f'Order {state["order_id"]} not found.'
        )

    handoff=make_handoff(
        "Order Agent",
        "Inventory Agent",
        state["request"],
        {
            "order_id":state["order_id"],
            "order":order
        }
    )

    return agent_update(
        state,
        "Order Agent",
        "HANDED TO INVENTORY",
        f'Order {state["order_id"]} retrieved and handed to Inventory Agent.',
        order=order,
        handoff=handoff
    )

# ==========================================================
# INVENTORY AGENT
# ==========================================================

def inventory_agent(state):
    items=get_order_items(state["order_id"])

    risks=[
        x for x in items
        if x["InventoryStatus"]!="Adequate"
    ]

    needs_supplier=bool(risks)

    if needs_supplier:
        handoff=make_handoff(
            "Inventory Agent",
            "Supplier Agent",
            "Review suppliers for inventory-risk products.",
            {
                "order_id":state["order_id"],
                "risk_products":[
                    {
                        "ProductID":x["ProductID"],
                        "ProductName":x["ProductName"],
                        "InventoryStatus":x["InventoryStatus"]
                    }
                    for x in risks
                ]
            }
        )
        status="SUPPLIER REVIEW REQUIRED"
        memory=(
            f"Inventory Agent found {len(risks)} risk product(s) "
            "and handed the case to Supplier Agent."
        )
        response=""
    else:
        handoff=state["handoff"]
        status="COMPLETED"
        memory="Inventory Agent found no inventory risk."
        response="No supplier follow-up is required."

    return agent_update(
        state,
        "Inventory Agent",
        status,
        memory,
        items=items,
        risk_items=risks,
        supplier_review_required=needs_supplier,
        handoff=handoff,
        response=response
    )

# ==========================================================
# SUPPLIER AGENT
# ==========================================================

def supplier_agent(state):
    ids=[
        x["ProductID"]
        for x in state["risk_items"]
    ]

    suppliers=get_product_suppliers(ids)

    response=llm_summary(
        "Supplier Agent",
        """Review the inventory-risk products and their suppliers.
Recommend a concise operational follow-up.""",
        {
            "request":state["request"],
            "risk_items":state["risk_items"],
            "suppliers":suppliers
        },
        "response"
    )

    return agent_update(
        state,
        "Supplier Agent",
        "COMPLETED",
        "Supplier Agent reviewed suppliers for the inventory-risk products.",
        suppliers=suppliers,
        response=response
    )

# ==========================================================
# CONDITIONAL ROUTING
# ==========================================================

def route_inventory(state):
    return (
        "supplier"
        if state["supplier_review_required"]
        else "END"
    )

# ==========================================================
# BUILD LANGGRAPH WORKFLOW
# ==========================================================

builder=StateGraph(HandoffState)

builder.add_node("order_agent",order_agent)
builder.add_node("inventory_agent",inventory_agent)
builder.add_node("supplier_agent",supplier_agent)

builder.add_edge(START,"order_agent")
builder.add_edge("order_agent","inventory_agent")

builder.add_conditional_edges(
    "inventory_agent",
    route_inventory,
    {
        "supplier":"supplier_agent",
        "END":END
    }
)

builder.add_edge("supplier_agent",END)

graph=builder.compile()

# ==========================================================
# RUN
# ==========================================================

ORDER_ID=11077

result=graph.invoke({
    "request":(
        f"Investigate Order {ORDER_ID} for fulfillment risk. "
        "Check for inventory problems and determine whether "
        "supplier follow-up is required."
    ),
    "order_id":ORDER_ID,
    "status":"RECEIVED",
    "status_history":[],
    "memory":[],
    "trace":[]
})

# ==========================================================
# OUTPUT
# ==========================================================

print("\nREQUEST")
print(result["request"])

print("\nFINAL HANDOFF")
for key,value in result["handoff"].items():
    print(key+":",value)

print("\nRISK PRODUCTS")
for item in result["risk_items"]:
    print(
        item["ProductName"],
        "| Qty:",item["Quantity"],
        "| Stock:",item["UnitsInStock"],
        "|",item["InventoryStatus"]
    )

print("\nFINAL RESPONSE")
print(result["response"])

print("\nWORKFLOW MEMORY")
for item in result["memory"]:
    print("-",item)

print("\nSTATUS HISTORY")
for item in result["status_history"]:
    print("-",item)

print("\nEXECUTION TRACE")
print(" -> ".join(result["trace"]))

show_graph(graph)
#%% EXERCISE 5: STATEFUL FAILURE AND RECOVERY


from langgraph.checkpoint.memory import InMemorySaver

# ==========================================================
# SIMULATE SUPPLIER LLM AVAILABILITY
# ==========================================================

SUPPLIER_LLM_AVAILABLE=False

# ==========================================================
# SUPPLIER AGENT WITH SIMULATED LLM FAILURE
# ==========================================================

def supplier_agent_recovery(state):
    print("Supplier Agent started.")

    if not SUPPLIER_LLM_AVAILABLE:
        print("Supplier LLM unreachable.")
        raise ConnectionError("Supplier Agent LLM is unavailable.")

    ids=[x["ProductID"] for x in state["risk_items"]]
    suppliers=get_product_suppliers(ids)

    response=llm_summary(
        "Supplier Agent",
        "Review the inventory-risk products and recommend supplier follow-up.",
        {
            "risk_items":state["risk_items"],
            "suppliers":suppliers
        },
        "response"
    )

    return agent_update(
        state,
        "Supplier Agent",
        "SUPPLIERS REVIEWED",
        "Supplier Agent completed supplier review.",
        suppliers=suppliers,
        response=response
    )

# ==========================================================
# RESPONSE AGENT
# ==========================================================

def response_agent(state):
    response=llm_summary(
        "Response Agent",
        "Prepare a concise final operational assessment and next action.",
        {
            "order":state["order"],
            "risk_items":state["risk_items"],
            "supplier_finding":state["response"]
        },
        "response"
    )

    return agent_update(
        state,
        "Response Agent",
        "COMPLETED",
        "Response Agent completed the investigation.",
        response=response
    )

# ==========================================================
# BUILD CHECKPOINTED WORKFLOW
# ==========================================================

builder=StateGraph(HandoffState)

builder.add_node("order_agent",order_agent)
builder.add_node("inventory_agent",inventory_agent)
builder.add_node("supplier_agent",supplier_agent_recovery)
builder.add_node("response_agent",response_agent)

builder.add_edge(START,"order_agent")
builder.add_edge("order_agent","inventory_agent")
builder.add_edge("inventory_agent","supplier_agent")
builder.add_edge("supplier_agent","response_agent")
builder.add_edge("response_agent",END)

graph=builder.compile(checkpointer=InMemorySaver())

config={
    "configurable":{
        "thread_id":f"northwind-{ORDER_ID}"
    }
}

# ==========================================================
# PHASE 1: LLM UNAVAILABLE
# ==========================================================

print("\n"+"="*60)
print("PHASE 1 - SUPPLIER LLM UNAVAILABLE")
print("="*60)

try:
    graph.invoke(
        {
            "request":(
                f"Investigate Order {ORDER_ID} for fulfillment risk "
                "and recommend supplier follow-up."
            ),
            "order_id":ORDER_ID,
            "status":"RECEIVED",
            "status_history":[],
            "memory":[],
            "trace":[]
        },
        config=config
    )
except ConnectionError as e:
    print("\nWORKFLOW FAILED")
    print(e)

state=graph.get_state(config)

print("\nSAVED STATE")
print("Status:",state.values["status"])
print("Next node:",state.next)
print("Trace:"," -> ".join(state.values["trace"]))

# ==========================================================
# PHASE 2: LLM RESTORED AND SAME TRANSACTION RESUMED
# ==========================================================

print("\n"+"="*60)
print("PHASE 2 - SUPPLIER LLM RESTORED")
print("="*60)

SUPPLIER_LLM_AVAILABLE=True

result=graph.invoke(
    None,
    config=config
)

print("\nFINAL RESPONSE")
print(result["response"])

print("\nFINAL STATUS")
print(result["status"])

print("\nFINAL TRACE")
print(" -> ".join(result["trace"]))

show_graph(graph)



#%% EXERCISE 6: SUPERVISOR, ROUTING, AND WORKFLOW CONTROL
# Run Exercises 1-5 first

# ==========================================================
# EXTEND PREVIOUS STATE
# ==========================================================

SupervisorState=TypedDict(
    "SupervisorState",
    {
        **HandoffState.__annotations__,
        "customer":dict,
        "customer_summary":str,
        "allowed_actions":list[str],
        "next_agent":str
    },
    total=False
)

# ==========================================================
# SUPERVISOR CONTROL
# ==========================================================

def get_allowed_actions(state):
    if "order" not in state:
        return ["order_agent"]

    allowed=[]

    if "customer" not in state:
        allowed.append("customer_agent")

    if "items" not in state:
        allowed.append("inventory_agent")

    if (
        "items" in state
        and state.get("risk_items")
        and "suppliers" not in state
    ):
        allowed.append("supplier_agent")

    return allowed or ["response_agent"]

def choose_next_agent(state,allowed):
    if len(allowed)==1:
        return allowed[0]

    result=ask_json(
        f"""You are the Northwind Workflow Supervisor.

Choose the most useful next agent from:
{allowed}

customer_agent = review customer information
inventory_agent = assess inventory risk
supplier_agent = review suppliers for risk products

Choose only from the allowed agents.

Return JSON:
{{"next_agent":"..."}}""",
        state["request"]
    )

    return validate_choice(
        result.get("next_agent",""),
        allowed,
        allowed[0]
    )

def supervisor(state):
    allowed=get_allowed_actions(state)
    next_agent=choose_next_agent(state,allowed)

    return {
        "allowed_actions":allowed,
        "next_agent":next_agent,
        "trace":add_trace(
            state,
            f"Supervisor -> {next_agent} "
            f"[allowed: {', '.join(allowed)}]"
        )
    }

# ==========================================================
# CUSTOMER AGENT
# ==========================================================

def customer_agent(state):
    customer=get_customer(
        state["order"]["CustomerID"]
    )

    summary=role_summary(
        "customer_agent",
        customer,
        "Summarize the customer and previous order activity.",
        "customer_summary"
    )

    return agent_update(
        state,
        "Customer Agent",
        "CUSTOMER REVIEWED",
        "Customer Agent completed customer review.",
        customer=customer,
        customer_summary=summary
    )

# ==========================================================
# RESPONSE AGENT
# ==========================================================

def response_agent(state):
    response=llm_summary(
        "Response Agent",
        """Prepare a concise final operational assessment.
Include the order, customer, inventory risks, supplier findings
if applicable, and recommended action.""",
        {
            "order":state["order"],
            "customer":state["customer_summary"],
            "risk_items":state["risk_items"],
            "supplier_finding":(
                state["response"]
                if state.get("suppliers")
                else "Supplier review was not required."
            )
        },
        "response"
    )

    return agent_update(
        state,
        "Response Agent",
        "COMPLETED",
        "Response Agent completed the investigation.",
        response=response
    )

# ==========================================================
# BUILD SUPERVISOR WORKFLOW
# ==========================================================

builder=StateGraph(SupervisorState)

builder.add_node("supervisor",supervisor)
builder.add_node("order_agent",order_agent)
builder.add_node("customer_agent",customer_agent)
builder.add_node("inventory_agent",inventory_agent)
builder.add_node("supplier_agent",supplier_agent)
builder.add_node("response_agent",response_agent)

builder.add_edge(START,"supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state:state["next_agent"],
    {
        "order_agent":"order_agent",
        "customer_agent":"customer_agent",
        "inventory_agent":"inventory_agent",
        "supplier_agent":"supplier_agent",
        "response_agent":"response_agent"
    }
)

for node in [
    "order_agent",
    "customer_agent",
    "inventory_agent",
    "supplier_agent"
]:
    builder.add_edge(node,"supervisor")

builder.add_edge("response_agent",END)

graph=builder.compile()

# ==========================================================
# RUN
# ==========================================================

result=graph.invoke({
    "request":(
        f"Investigate Order {ORDER_ID}. Review the customer, "
        "inventory risks, affected suppliers, and recommend "
        "an operational action."
    ),
    "order_id":ORDER_ID,
    "status":"RECEIVED",
    "status_history":[],
    "memory":[],
    "trace":[]
})

# ==========================================================
# OUTPUT
# ==========================================================

print("\nFINAL RESPONSE")
print(result["response"])

print("\nWORKFLOW MEMORY")
for x in result["memory"]:
    print("-",x)

print("\nSUPERVISOR TRACE")
for x in result["trace"]:
    print("-",x)

show_graph(graph)


#%% EXERCISE 7: HUMAN-IN-THE-LOOP APPROVAL
 

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command

# ==========================================================
# STATE + SMALL NEW HELPERS
# ==========================================================

HITLState=TypedDict(
    "HITLState",
    {
        **HandoffState.__annotations__,
        "customer":dict,
        "risk_level":str,
        "recommendation":str,
        "recommendation_reason":str,
        "approval_status":str
    },
    total=False
)

def show_approval(data):
    print("\nHUMAN APPROVAL REQUIRED")
    print("Order:",data["order_id"])
    print("Customer:",data["customer"])
    print("Risk:",data["risk_level"])
    print("Risk items:",data["risk_count"])
    print("AI recommendation:",data["recommendation"])
    print("Reason:",data["reason"])

def get_decision():
    while True:
        x=input("\nApprove recommendation? [approve/reject]: ").strip().lower()
        if x in ["approve","reject"]:
            return x
        print("Enter approve or reject.")

# ==========================================================
# INVESTIGATION AGENT
# ==========================================================

def investigation_agent(state):
    order=get_order(state["order_id"])
    if not order:
        raise ValueError(f'Order {state["order_id"]} not found.')

    customer=get_customer(order["CustomerID"])
    items=get_order_items(state["order_id"])
    risks=[x for x in items if x["InventoryStatus"]!="Adequate"]

    risk=(
        "HIGH"
        if any(x["InventoryStatus"]=="Potential shortage" for x in risks)
        else "MEDIUM"
        if risks
        else "LOW"
    )

    return agent_update(
        state,
        "Investigation Agent",
        "INVESTIGATION COMPLETED",
        f"Order {state['order_id']} investigated; {len(risks)} risk item(s) found.",
        order=order,
        customer=customer,
        items=items,
        risk_items=risks,
        risk_level=risk
    )

# ==========================================================
# RECOMMENDATION AGENT
# ==========================================================

def recommendation_agent(state):
    result=ask_json(
        """You are the Northwind Operations Recommendation Agent.

Recommend exactly one action:
PRIORITIZE = priority handling is required.
HOLD = inventory problems require review before fulfillment.
STANDARD = no significant intervention is required.

This is only a recommendation. A human makes the final decision.

Return JSON:
{"recommendation":"...","recommendation_reason":"..."}""",
        json.dumps({
            "order":state["order"],
            "customer":state["customer"],
            "risk_level":state["risk_level"],
            "risk_items":state["risk_items"]
        },default=str)
    )

    recommendation=validate_choice(
        result.get("recommendation",""),
        ["PRIORITIZE","HOLD","STANDARD"],
        "HOLD"
    )

    return agent_update(
        state,
        "Recommendation Agent",
        "AWAITING APPROVAL",
        f"AI recommended {recommendation}.",
        recommendation=recommendation,
        recommendation_reason=result.get(
            "recommendation_reason",
            "Human review required."
        )
    )

# ==========================================================
# HUMAN APPROVAL
# ==========================================================

def human_approval(state):
    decision=interrupt({
        "order_id":state["order_id"],
        "customer":state["customer"]["CompanyName"],
        "risk_level":state["risk_level"],
        "risk_count":len(state["risk_items"]),
        "recommendation":state["recommendation"],
        "reason":state["recommendation_reason"]
    })

    decision=validate_choice(
        decision,
        ["approve","reject"],
        "reject"
    )

    return agent_update(
        state,
        "Human Reviewer",
        "APPROVED" if decision=="approve" else "REJECTED",
        f"Human reviewer selected {decision.upper()}.",
        approval_status=decision
    )

# ==========================================================
# FINAL ACTIONS
# ==========================================================

def execute_action(state):
    return agent_update(
        state,
        "Action Agent",
        "COMPLETED",
        f"Simulated action executed: {state['recommendation']}."
    )

def reject_action(state):
    return agent_update(
        state,
        "Action Agent",
        "REJECTED",
        "Recommendation rejected; no action executed."
    )

# ==========================================================
# BUILD HITL WORKFLOW
# ==========================================================

builder=StateGraph(HITLState)
builder.add_node("investigation_agent",investigation_agent)
builder.add_node("recommendation_agent",recommendation_agent)
builder.add_node("human_approval",human_approval)
builder.add_node("execute",execute_action)
builder.add_node("reject",reject_action)

builder.add_edge(START,"investigation_agent")
builder.add_edge("investigation_agent","recommendation_agent")
builder.add_edge("recommendation_agent","human_approval")

builder.add_conditional_edges(
    "human_approval",
    lambda s:"execute" if s["approval_status"]=="approve" else "reject",
    {
        "execute":"execute",
        "reject":"reject"
    }
)

builder.add_edge("execute",END)
builder.add_edge("reject",END)

graph=builder.compile(
    checkpointer=InMemorySaver()
)

# ==========================================================
# PHASE 1: RUN UNTIL HUMAN APPROVAL
# ==========================================================

config={
    "configurable":{
        "thread_id":f"northwind-hitl-{ORDER_ID}"
    }
}

first=graph.invoke(
    {
        "request":(
            f"Investigate Order {ORDER_ID}, assess fulfillment risk "
            "and recommend an operational action."
        ),
        "order_id":ORDER_ID,
        "status":"RECEIVED",
        "status_history":[],
        "memory":[],
        "trace":[]
    },
    config=config
)

interrupts=first.get("__interrupt__",())
if not interrupts:
    raise RuntimeError("Workflow did not pause for human approval.")

approval=interrupts[0].value
show_approval(approval)

# ==========================================================
# PHASE 2: HUMAN DECISION + RESUME
# ==========================================================

decision=get_decision()

result=graph.invoke(
    Command(resume=decision),
    config=config
)

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\nFINAL DECISION")
print("AI Recommendation:",result["recommendation"])
print("Human Decision:",result["approval_status"].upper())
print("Final Status:",result["status"])

print("\nWORKFLOW MEMORY")
for x in result["memory"]:
    print("-",x)

print("\nTRACE")
print(" -> ".join(result["trace"]))

show_graph(graph)