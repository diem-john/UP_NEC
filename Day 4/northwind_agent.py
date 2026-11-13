#%% NORTHWIND MULTI-AGENT CORE — IMPORT-SAFE
# Based on the existing Northwind agent. No console demo or input() on import.
# Requires northwind.db in this folder, Ollama qwen3.5:4b, LangGraph.
import sqlite3, json, ollama
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
DB = str(Path(__file__).with_name('northwind.db'))
MODEL = 'qwen3.5:4b'
SUPPLIER_LLM_AVAILABLE = True

def ask_json(system, user):
    r = ollama.chat(model=MODEL, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], format='json', think=False)
    return json.loads(r.message.content)

def validate_choice(value, allowed, default):
    m = {str(x).lower(): x for x in allowed}
    return m.get(str(value).strip().lower(), default)

def db_query(sql, params=()):
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA query_only=ON')
        return [dict(x) for x in c.execute(sql, params).fetchall()]

def get_order(order_id):
    r = db_query('SELECT o.OrderID,o.CustomerID,o.OrderDate,o.RequiredDate,o.ShippedDate,o.Freight,o.ShipCountry,s.CompanyName AS Shipper\n    FROM Orders o LEFT JOIN Shippers s ON o.ShipVia=s.ShipperID WHERE o.OrderID=?', (order_id,))
    return r[0] if r else None

def get_customer(customer_id):
    r = db_query('SELECT c.CustomerID,c.CompanyName,c.ContactName,c.City,c.Country,COUNT(o.OrderID) AS TotalOrders\n    FROM Customers c LEFT JOIN Orders o ON c.CustomerID=o.CustomerID WHERE c.CustomerID=?\n    GROUP BY c.CustomerID,c.CompanyName,c.ContactName,c.City,c.Country', (customer_id,))
    return r[0] if r else None

def get_order_items(order_id):
    return db_query('SELECT p.ProductID,p.ProductName,od.Quantity,od.UnitPrice,od.Discount,p.UnitsInStock,p.ReorderLevel,\n    CASE WHEN p.UnitsInStock<od.Quantity THEN \'Potential shortage\'\n    WHEN p.UnitsInStock<=p.ReorderLevel THEN \'Low stock\' ELSE \'Adequate\' END AS InventoryStatus\n    FROM "Order Details" od JOIN Products p ON od.ProductID=p.ProductID WHERE od.OrderID=?', (order_id,))

def get_product_suppliers(ids):
    if not ids:
        return []
    q = ','.join(['?'] * len(ids))
    return db_query(f'SELECT p.ProductID,p.ProductName,s.SupplierID,s.CompanyName,s.Country\n    FROM Products p JOIN Suppliers s ON p.SupplierID=s.SupplierID WHERE p.ProductID IN ({q}) ORDER BY p.ProductName', tuple(ids))

def add_trace(state, msg):
    return state.get('trace', []) + [msg]

def add_memory(state, msg):
    return state.get('memory', []) + [msg]

def update_state(state, agent, status, memory, **values):
    old = state.get('status', 'START')
    return {**values, 'status': status, 'status_history': state.get('status_history', []) + [f'{old} -> {status} ({agent})'], 'memory': add_memory(state, memory), 'trace': add_trace(state, agent)}
ROLES = {'order_agent': ('Investigate order and shipping facts.', 'Orders, Shippers'), 'customer_agent': ('Assess the customer and previous order activity.', 'Customers, Orders'), 'inventory_agent': ('Assess ordered products for inventory risk.', 'Order Details, Products'), 'supplier_agent': ('Assess suppliers for inventory-risk products.', 'Products, Suppliers')}

def role_summary(name, evidence, instruction, key):
    goal, scope = ROLES[name]
    return ask_json(f'''You are the Northwind {name.replace('_', ' ').title()}.\nGoal: {goal}\nDatabase scope: {scope}\n{instruction}\nUse only supplied evidence. Do not invent information.\nReturn JSON: {{"{key}":"..."}}''', json.dumps(evidence, default=str)).get(key, '')

def handoff(state, to_agent):
    return {'from': 'Supervisor', 'to': to_agent, 'task': state['request'], 'order_id': state['order_id']}
AppState = TypedDict('AppState', {'request': str, 'order_id': int, 'order': dict, 'order_summary': str, 'customer': dict, 'customer_summary': str, 'items': list[dict], 'risk_items': list[dict], 'inventory_summary': str, 'suppliers': list[dict], 'supplier_summary': str, 'handoff': dict, 'allowed_actions': list[str], 'next_agent': str, 'assessment': str, 'recommendation': str, 'recommendation_reason': str, 'approval_status': str, 'status': str, 'status_history': list[str], 'memory': list[str], 'trace': list[str]}, total=False)

def order_agent(state):
    order = get_order(state['order_id'])
    if not order:
        raise ValueError(f"Order {state['order_id']} not found.")
    summary = role_summary('order_agent', order, 'Summarize fulfillment and shipping status.', 'order_summary')
    return update_state(state, 'Order Agent', 'ORDER REVIEWED', 'Order Agent completed order review.', order=order, order_summary=summary)

def customer_agent(state):
    customer = get_customer(state['order']['CustomerID'])
    summary = role_summary('customer_agent', customer, 'Summarize the customer and previous order activity.', 'customer_summary')
    return update_state(state, 'Customer Agent', 'CUSTOMER REVIEWED', 'Customer Agent completed customer review.', customer=customer, customer_summary=summary)

def inventory_agent(state):
    items = get_order_items(state['order_id'])
    risks = [x for x in items if x['InventoryStatus'] != 'Adequate']
    summary = role_summary('inventory_agent', {'items': items, 'risk_items': risks}, 'Summarize inventory risks.', 'inventory_summary')
    return update_state(state, 'Inventory Agent', 'INVENTORY REVIEWED', f'Inventory Agent found {len(risks)} risk item(s).', items=items, risk_items=risks, inventory_summary=summary)

def supplier_agent(state):
    global SUPPLIER_LLM_AVAILABLE
    suppliers = get_product_suppliers([x['ProductID'] for x in state['risk_items']])
    print('Supplier Agent: contacting LLM...')
    if not SUPPLIER_LLM_AVAILABLE:
        raise ConnectionError('Supplier Agent LLM is unreachable.')
    summary = role_summary('supplier_agent', {'risk_items': state['risk_items'], 'suppliers': suppliers}, 'Recommend supplier follow-up.', 'supplier_summary')
    return update_state(state, 'Supplier Agent', 'SUPPLIER REVIEWED', 'Supplier Agent completed supplier review.', suppliers=suppliers, supplier_summary=summary)

def allowed_actions(state):
    if 'order' not in state:
        return ['order_agent']
    a = []
    if 'customer' not in state:
        a.append('customer_agent')
    if 'items' not in state:
        a.append('inventory_agent')
    if 'items' in state and state.get('risk_items') and ('suppliers' not in state):
        a.append('supplier_agent')
    return a or ['response_agent']

def supervisor(state):
    allowed = allowed_actions(state)
    if len(allowed) == 1:
        next_agent = allowed[0]
    else:
        r = ask_json(f'You are the Northwind Workflow Supervisor.\nChoose exactly one next agent from {allowed}.\nChoose the most useful valid next step.\nReturn JSON: {{"next_agent":"..."}}', json.dumps({'request': state['request'], 'completed': state.get('trace', [])}))
        next_agent = validate_choice(r.get('next_agent', ''), allowed, allowed[0])
    return {'allowed_actions': allowed, 'next_agent': next_agent, 'handoff': handoff(state, next_agent), 'trace': add_trace(state, f'Supervisor -> {next_agent}')}

def response_agent(state):
    r = ask_json('You are the Northwind Operations Response Agent.\nPrepare a concise assessment and recommend exactly one action:\nPRIORITIZE, HOLD, or STANDARD.\nReturn JSON:\n{"assessment":"...","recommendation":"...","recommendation_reason":"..."}', json.dumps({'order': state['order_summary'], 'customer': state['customer_summary'], 'inventory': state['inventory_summary'], 'supplier': state.get('supplier_summary', 'Supplier review not required.')}, default=str))
    rec = validate_choice(r.get('recommendation', ''), ['PRIORITIZE', 'HOLD', 'STANDARD'], 'HOLD')
    return update_state(state, 'Response Agent', 'AWAITING APPROVAL', f'AI recommended {rec}.', assessment=r.get('assessment', ''), recommendation=rec, recommendation_reason=r.get('recommendation_reason', 'Human review required.'))

def human_approval(state):
    decision = interrupt({'order_id': state['order_id'], 'customer': state['customer']['CompanyName'], 'assessment': state['assessment'], 'recommendation': state['recommendation'], 'reason': state['recommendation_reason']})
    decision = validate_choice(decision, ['approve', 'reject'], 'reject')
    return update_state(state, 'Human Reviewer', 'APPROVED' if decision == 'approve' else 'REJECTED', f'Human reviewer selected {decision.upper()}.', approval_status=decision)

def execute_action(state):
    return update_state(state, 'Action Agent', 'COMPLETED', f"Simulated action executed: {state['recommendation']}.")

def reject_action(state):
    return update_state(state, 'Action Agent', 'REJECTED', 'Recommendation rejected. No action executed.')
builder = StateGraph(AppState)
for name, node in {'supervisor': supervisor, 'order_agent': order_agent, 'customer_agent': customer_agent, 'inventory_agent': inventory_agent, 'supplier_agent': supplier_agent, 'response_agent': response_agent, 'human_approval': human_approval, 'execute': execute_action, 'reject': reject_action}.items():
    builder.add_node(name, node)
builder.add_edge(START, 'supervisor')
builder.add_conditional_edges('supervisor', lambda s: s['next_agent'], {'order_agent': 'order_agent', 'customer_agent': 'customer_agent', 'inventory_agent': 'inventory_agent', 'supplier_agent': 'supplier_agent', 'response_agent': 'response_agent'})
for n in ['order_agent', 'customer_agent', 'inventory_agent', 'supplier_agent']:
    builder.add_edge(n, 'supervisor')
builder.add_edge('response_agent', 'human_approval')
builder.add_conditional_edges('human_approval', lambda s: 'execute' if s['approval_status'] == 'approve' else 'reject', {'execute': 'execute', 'reject': 'reject'})
builder.add_edge('execute', END)
builder.add_edge('reject', END)
graph = builder.compile(checkpointer=InMemorySaver())
