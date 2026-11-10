
#%% CASE STUDY 3: SETUP
import json, ollama
MODEL = "qwen3.5:4b"

purchase_requests = [
    {"request_id":"PR-2026-001","department":"Engineering Laboratory",
     "item":"20 laptop computers","amount":1200000.0,
     "purpose":"Replace obsolete laboratory computers.",
     "supplier":"TechSource Systems"},
    {"request_id":"PR-2026-002","department":"Engineering Laboratory",
     "item":"Laboratory printer","amount":850000.0,
     "purpose":"Replace damaged laboratory equipment.",
     "supplier":None},
    {"request_id":"PR-2026-003","department":"Information Technology Office",
     "item":"Network switches","amount":450000.0,
     "purpose":"Replace aging network equipment.",
     "supplier":"Network Systems Corporation"}
]

def request_text(r):
    return json.dumps(r,indent=2)

def ask_llm(prompt,system="",schema=None,stream=False):
    args={"model":MODEL,"messages":[
        {"role":"system","content":system},
        {"role":"user","content":prompt}],
        "think":False,"options":{"temperature":0},"stream":stream}
    if schema is not None: args["format"]=schema
    response=ollama.chat(**args)
    if not stream: return response["message"]["content"].strip()
    text=""
    for chunk in response:
        token=chunk["message"]["content"] or ""
        print(token,end="",flush=True)
        text+=token
    print()
    return text.strip()

print("MODEL:",MODEL)
for r in purchase_requests:
    print(r["request_id"],"|",r["item"],"|",r["amount"],"|",r["supplier"])


#%% FROM LLM RESPONSE TO WORKFLOW DECISION
request=purchase_requests[0]

print("\nGENERAL LLM RESPONSE")
general_response=ask_llm(
    "Review this purchase request and explain what should be done:\n"
    +request_text(request),stream=True)

workflow_prompt="""You are a Purchase Request Review Assistant.
If amount > PHP 1,000,000, return HUMAN_REVIEW.
Otherwise, return STANDARD_REVIEW.
Return only one of these two labels."""

print("\nBOUNDED WORKFLOW DECISION")
workflow_decision=ask_llm(
    request_text(request),workflow_prompt,stream=True)


#%%  LOCAL LLM ARCHITECTURE
print("\nLOCAL LLM ARCHITECTURE")
print("Python Application → ollama.chat() → Ollama Runtime")
print("→",MODEL,"→ Model Response → Python Application")
print("Model:",MODEL,"| Proposed Decision:",workflow_decision)
print("Business workflow remains controlled by Python.")


#%% PROMPT DESIGN
request=purchase_requests[0]

controlled_prompt="""ROLE: Purchase Request Review Assistant.
TASK: Determine the appropriate review workflow.
BUSINESS RULES:
Amount > PHP 1,000,000: HUMAN_REVIEW.
Otherwise: STANDARD_REVIEW.
CONSTRAINTS:
Use only supplied information. Do not invent details.
Do not approve transactions. Keep the reason concise.
OUTPUT FORMAT (exactly four lines):
DECISION:
REASON:
POLICY_APPLIED:
NEXT_ACTION:"""

print("\nBASIC PROMPT")
basic=ask_llm("Review this request:\n"+request_text(request),
              stream=True)

print("\nCONTROLLED PROMPT")
controlled=ask_llm(request_text(request),controlled_prompt,
                   stream=True)


#%%  STRUCTURED OUTPUTS
schema={
    "type":"object",
    "properties":{
        "request_id":{"type":"string"},
        "item":{"type":"string"},
        "amount":{"type":"number"},
        "supplier":{"type":["string","null"]},
        "decision":{"type":"string",
                    "enum":["HUMAN_REVIEW","STANDARD_REVIEW"]},
        "reason":{"type":"string"}},
    "required":["request_id","item","amount","supplier",
                "decision","reason"],
    "additionalProperties":False
}

structured_prompt="""You are a Purchase Request Review Assistant.
Extract request_id, item, amount, and supplier from the source.
Do not change values or invent missing information.
If supplier is missing, return null.
Return the amount as a number.
Amount > PHP 1,000,000: HUMAN_REVIEW.
Otherwise: STANDARD_REVIEW.
Return only JSON matching the supplied schema."""

def generate_review(r):
    return ask_llm(request_text(r),structured_prompt,schema)

raw_output=generate_review(purchase_requests[0])
print("\nRAW STRUCTURED OUTPUT:\n",raw_output)

try:
    structured_review=json.loads(raw_output)
    print("JSON PARSING: PASSED")
    print("PYTHON DICTIONARY:",structured_review)
except json.JSONDecodeError:
    structured_review={}
    print("JSON PARSING: FAILED")


#%% FAILURE MODES AND GUARDRAILS
def control_workflow(source,raw):
    try:
        review=json.loads(raw)
        json_ok=isinstance(review,dict)
    except (json.JSONDecodeError,TypeError):
        review={}
        json_ok=False

    required=set(schema["required"])
    fields_ok=(json_ok and set(review)==required and
        all(isinstance(review.get(k),str)
            for k in ["request_id","item","decision","reason"]) and
        bool(review.get("reason","").strip()) and
        type(review.get("amount")) in (int,float) and
        (review.get("supplier") is None or
         isinstance(review.get("supplier"),str)))

    allowed_ok=fields_ok and review["decision"] in (
        "HUMAN_REVIEW","STANDARD_REVIEW")

    source_ok=fields_ok and all(
        review[k]==source[k]
        for k in ["request_id","item","amount","supplier"])

    expected=("HUMAN_REVIEW" if source["amount"]>1000000
              else "STANDARD_REVIEW")
    rule_ok=allowed_ok and review["decision"]==expected

    hallucination_ok=fields_ok and (
        source["supplier"] is not None or review["supplier"] is None)

    supplier_ok=bool(source["supplier"])

    checks={
        "1.JSON":json_ok,
        "2.Fields/Types":fields_ok,
        "3.Allowed Decision":allowed_ok,
        "4.Source Values":source_ok,
        "5.Business Rule":rule_ok,
        "6.No Hallucination":hallucination_ok,
        "7.Supplier Provided":supplier_ok
    }

    if not all(list(checks.values())[:6]):
        status,action="STOP","Failed validation; manual review required"
    elif not supplier_ok:
        status,action="HOLD","Request supplier information"
    elif expected=="HUMAN_REVIEW":
        status,action="CONTINUE","Route to human review"
    else:
        status,action="CONTINUE","Proceed to standard review"

    return review,checks,status,action

expected_results={
    "PR-2026-001":("CONTINUE","Route to human review"),
    "PR-2026-002":("HOLD","Request supplier information"),
    "PR-2026-003":("CONTINUE","Proceed to standard review")
}

results={}
for source in purchase_requests:
    raw=(raw_output if source["request_id"]=="PR-2026-001"
         else generate_review(source))
    review,checks,status,action=control_workflow(source,raw)
    results[source["request_id"]]=(review,checks,status,action)
    expected=expected_results[source["request_id"]]

    print("\n",source["request_id"])
    print("LLM Decision:",review.get("decision","INVALID"))
    for name,passed in checks.items():
        print(name,":","PASSED" if passed else "FAILED")
    print("FINAL:",status,"→",action)
    print("EXPECTED RESULT:","PASS" if (status,action)==expected
          else "FAIL")


#%% ADDITIONAL FAILURE TESTS: SIMULATED HALLUCINATIONS
# Modify LLM proposals to demonstrate deterministic safeguards.

source=purchase_requests[0]
bad=dict(results[source["request_id"]][0])
bad["amount"]=800000
bad["decision"]="STANDARD_REVIEW"
_,checks,status,action=control_workflow(source,json.dumps(bad))
print("\nALTERED AMOUNT:",status,"|",action)
print("Source values:",checks["4.Source Values"])
print("Business rule:",checks["5.Business Rule"])

source=purchase_requests[1]
bad=dict(results[source["request_id"]][0])
bad["supplier"]="Invented Supplier Inc."
_,checks,status,action=control_workflow(source,json.dumps(bad))
print("\nINVENTED SUPPLIER:",status,"|",action)
print("No hallucination:",checks["6.No Hallucination"])
