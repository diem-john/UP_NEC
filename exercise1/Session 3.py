
#%% EXERCISE 1: UNSTRUCTURED VS STRUCTURED PROMPTING

import ollama

model_name = "gemma3:4b"


# ----------------------------------------------------------
# 1. PURCHASE REQUEST DATA
# ----------------------------------------------------------

request_id = "PR-2026-001"
amount = 1200000

purchase_request = f"""
Request ID: {request_id}
Item: 20 laptop computers
Amount: PHP {amount:,}
Purpose: Replace obsolete laboratory computers.
"""


# ----------------------------------------------------------
# 2. SHOW LOCAL LLM ARCHITECTURE
# ----------------------------------------------------------

print("========================================")
print("LOCAL LLM ARCHITECTURE")
print("========================================")

print("Python Application")
print("       ↓")
print("  ollama.chat()")
print("       ↓")
print(" Ollama Runtime")
print("       ↓")
print(" " + model_name)


# ==========================================================
# PART A: UNSTRUCTURED PROMPT
# ==========================================================

unstructured_prompt = f"""
Review this purchase request and tell me what should happen:

{purchase_request}
"""

print("\n========================================")
print("PART A: UNSTRUCTURED PROMPT")
print("========================================")

print("\nPROMPT:")
print(unstructured_prompt.strip())

print("\nMODEL RESPONSE:")

stream = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": unstructured_prompt
        }
    ],
    stream=True,
    think=False,
    options={
        "temperature": 0
    }
)

unstructured_result = ""

for chunk in stream:

    text = chunk["message"]["content"]

    print(
        text,
        end="",
        flush=True
    )

    unstructured_result += text

unstructured_result = unstructured_result.strip()

print()


# ==========================================================
# PART B: STRUCTURED PROMPT
# ==========================================================

system_message = """
You are a Purchase Request Review Assistant.
Your task is to determine the correct review path
for a purchase request.
"""

context_data = f"""
PURCHASE REQUEST:

Request ID: {request_id}
Item: 20 laptop computers
Amount: PHP {amount:,}
Purpose: Replace obsolete laboratory computers.
"""

constraints = """
BUSINESS RULE:

- If the purchase amount is greater than PHP 1,000,000,
  select HUMAN_REVIEW.
- Otherwise, select STANDARD_REVIEW.

Do not invent information.
Use only the purchase request provided.
"""

output_format = """
Return exactly one of these values:

HUMAN_REVIEW
STANDARD_REVIEW

Do not include explanations.
"""

user_message = f"""
Review the following purchase request.

{context_data}

{constraints}

{output_format}
"""


print("\n========================================")
print("PART B: STRUCTURED PROMPT")
print("========================================")

print("\nSYSTEM MESSAGE:")
print(system_message.strip())

print("\nUSER MESSAGE:")
print(user_message.strip())

print("\nMODEL RESPONSE:")

stream = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ],
    stream=True,
    think=False,
    options={
        "temperature": 0
    }
)

structured_result = ""

for chunk in stream:

    text = chunk["message"]["content"]

    print(
        text,
        end="",
        flush=True
    )

    structured_result += text

structured_result = structured_result.strip()

print()


# ----------------------------------------------------------
# 3. VALIDATE STRUCTURED RESULT
# ----------------------------------------------------------

expected_decision = (
    "HUMAN_REVIEW"
    if amount > 1000000
    else "STANDARD_REVIEW"
)

decision_valid = (
    structured_result == expected_decision
)


# ----------------------------------------------------------
# 4. DISPLAY COMPARISON
# ----------------------------------------------------------

print("\n========================================")
print("COMPARISON")
print("========================================")

print("Unstructured Response:")
print(unstructured_result)

print("\nStructured Response:")
print(structured_result)

print("\nExpected Decision:")
print(expected_decision)

print(
    "\nStructured Validation:",
    "PASSED" if decision_valid else "FAILED"
)


# ----------------------------------------------------------
# 5. NEXT WORKFLOW ACTION
# ----------------------------------------------------------

print("\n========================================")
print("NEXT ACTION")
print("========================================")

if not decision_valid:

    print(
        "STOP: Structured model output failed validation."
    )

elif structured_result == "HUMAN_REVIEW":

    print(
        f"Route {request_id} to a human approver."
    )

else:

    print(
        f"Continue {request_id} to standard review."
    )
#%% EXERCISE 2: ZERO-SHOT, ONE-SHOT, AND FEW-SHOT PROMPTING

import ollama

model_name = "gemma3:4b"

# ----------------------------------------------------------
# 1. PURCHASE REQUEST TO CLASSIFY
# ----------------------------------------------------------

purchase_request = """
Request ID: PR-2026-002
Item: 10 high-performance desktop workstations
Purpose: Upgrade the university AI and data science laboratory.
"""

categories = """
ICT Equipment
Office Supplies
Laboratory Equipment
Professional Services
"""

# ----------------------------------------------------------
# 2. ZERO-SHOT PROMPT
# ----------------------------------------------------------

zero_shot_prompt = f"""
Classify the purchase request into exactly one of these categories:

{categories}

PURCHASE REQUEST:
{purchase_request}

Return only the category.
"""

zero_response = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": zero_shot_prompt
        }
    ],
    think=False,
    options={
        "temperature": 0
    }
)

zero_result = (
    zero_response["message"]["content"].strip()
)

# ----------------------------------------------------------
# 3. ONE-SHOT PROMPT
# ----------------------------------------------------------

one_shot_prompt = f"""
Classify the purchase request into exactly one of these categories:

ICT Equipment
Office Supplies
Laboratory Equipment
Professional Services

CLASSIFICATION GUIDELINES:

- ICT Equipment:
  Computers, laptops, desktop workstations, servers, networking equipment,
  storage devices, and other general-purpose computing hardware.

- Office Supplies:
  Consumable office materials such as paper, pens, folders, printer ink,
  and similar supplies.

- Laboratory Equipment:
  Specialized scientific instruments primarily used for experimentation,
  measurement, testing, or laboratory analysis, such as microscopes,
  spectrometers, and laboratory analyzers.

- Professional Services:
  Work performed by external specialists, consultants, auditors,
  trainers, or other professional service providers.

IMPORTANT RULE:
Classify the item according to its PRIMARY FUNCTION, not according to
the office, laboratory, department, or location where it will be used.

EXAMPLE:

Purchase Request:
Item: 25 desktop computers
Purpose: Install computers in an engineering laboratory for CAD,
simulation, and student computing activities.

Classification:
ICT Equipment

NOW CLASSIFY THIS PURCHASE REQUEST:

{purchase_request}

Return exactly one category and nothing else.
"""

one_response = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": one_shot_prompt
        }
    ],
    think=False,
    options={
        "temperature": 0
    }
)

one_result = (
    one_response["message"]["content"].strip()
)

# ----------------------------------------------------------
# 4. FEW-SHOT PROMPT
# ----------------------------------------------------------

few_shot_prompt = f"""
Classify purchase requests into exactly one of these categories:

{categories}

EXAMPLE 1:

Purchase Request:
Item: 40 laptop computers
Purpose: Replace obsolete administrative computers.

Classification:
ICT Equipment


EXAMPLE 2:

Purchase Request:
Item: Bond paper, folders, pens, and printer ink
Purpose: Replenish office consumable supplies.

Classification:
Office Supplies


EXAMPLE 3:

Purchase Request:
Item: Digital microscope
Purpose: Equip the biology teaching laboratory.

Classification:
Laboratory Equipment


EXAMPLE 4:

Purchase Request:
Item: External cybersecurity assessment
Purpose: Conduct an independent security audit.

Classification:
Professional Services


NOW CLASSIFY THIS REQUEST:

{purchase_request}

Return only the category.
"""

few_response = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": few_shot_prompt
        }
    ],
    think=False,
    options={
        "temperature": 0
    }
)

few_result = (
    few_response["message"]["content"].strip()
)

# ----------------------------------------------------------
# 5. DISPLAY RESULTS
# ----------------------------------------------------------

print("========================================")
print("PURCHASE REQUEST")
print("========================================")
print(purchase_request.strip())

print("\n========================================")
print("ZERO-SHOT RESULT")
print("========================================")
print(zero_result)

print("\n========================================")
print("ONE-SHOT RESULT")
print("========================================")
print(one_result)

print("\n========================================")
print("FEW-SHOT RESULT")
print("========================================")
print(few_result)


#%% EXERCISE 3: COMMON PROMPTING PATTERNS
import ollama

model_name = "gemma3:4b"


# ----------------------------------------------------------
# STREAMING HELPER
# ----------------------------------------------------------

def ask_model(prompt):

    stream = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True,
        think=False,
        options={
            "temperature": 0
        }
    )

    full_response = ""

    for chunk in stream:

        text = chunk["message"]["content"]

        print(
            text,
            end="",
            flush=True
        )

        full_response += text

    print()

    return full_response.strip()


# ==========================================================
# PATTERN 1: CLASSIFICATION
# ==========================================================

classification_prompt = """
Classify the following document into exactly one category:

PURCHASE_REQUEST
PURCHASE_ORDER
INVOICE
DELIVERY_RECEIPT

Document:

"The supplier submitted a request for payment of
PHP 85,000 for computer equipment that has already
been delivered."

Return only the category.
"""

print("\n" + "=" * 60)
print("PATTERN 1: CLASSIFICATION")
print("=" * 60)

print("\nPURPOSE:")
print("Assign an input to one predefined category.")

print("\nRESULT:")

classification_result = ask_model(
    classification_prompt
)


# ==========================================================
# PATTERN 2: INFORMATION EXTRACTION
# ==========================================================

extraction_prompt = """
Extract the following fields:

- supplier
- document_date
- item
- quantity
- amount

Return the result as JSON.

Use null if a value is not provided.
Do not invent missing information.

Text:

"ABC Technologies submitted an invoice dated
September 10, 2026 for 10 laptop computers
worth PHP 850,000."
"""

print("\n" + "=" * 60)
print("PATTERN 2: INFORMATION EXTRACTION")
print("=" * 60)

print("\nPURPOSE:")
print(
    "Convert unstructured text into structured data."
)

print("\nRESULT:")

extraction_result = ask_model(
    extraction_prompt
)


# ==========================================================
# PATTERN 3: SUMMARIZATION
# ==========================================================

summarization_prompt = """
Summarize the following purchase justification
in no more than three sentences.

Preserve:

- the requested item,
- the reason for the purchase,
- the amount, and
- the expected benefit.

Purchase justification:

"The department is requesting the purchase of
10 new laptop computers costing PHP 850,000.

The existing computers are more than six years old
and cannot efficiently run current engineering,
data-analysis, and simulation software.

The new laptops will support faculty research,
administrative work, and student projects and are
expected to improve staff productivity."
"""

print("\n" + "=" * 60)
print("PATTERN 3: SUMMARIZATION")
print("=" * 60)

print("\nPURPOSE:")
print(
    "Reduce information while preserving important meaning."
)

print("\nRESULT:")

summarization_result = ask_model(
    summarization_prompt
)


# ==========================================================
# PATTERN 4: GROUNDED QUESTION ANSWERING
# ==========================================================

grounded_qa_prompt = """
Answer the question using ONLY the evidence below.

If the evidence does not contain enough information,
reply:

INSUFFICIENT INFORMATION

EVIDENCE:

"Complete reimbursement requests are normally
processed within 10 working days after all required
supporting documents have been received."

QUESTION:

How long does reimbursement processing normally take?
"""

print("\n" + "=" * 60)
print("PATTERN 4: GROUNDED QUESTION ANSWERING")
print("=" * 60)

print("\nPURPOSE:")
print(
    "Answer a question using only supplied evidence."
)

print("\nRESULT:")

grounded_qa_result = ask_model(
    grounded_qa_prompt
)


# ==========================================================
# PATTERN 5: ROUTING
# ==========================================================

routing_prompt = """
Determine which workflow should process the
following user request.

Choose exactly one route:

DOCUMENT_REVIEW
PAYMENT_STATUS
PROCUREMENT
HR_REQUEST
GENERAL_INQUIRY

User request:

"Can you check whether my reimbursement has
already been processed?"

Return the selected route only.
"""

print("\n" + "=" * 60)
print("PATTERN 5: ROUTING")
print("=" * 60)

print("\nPURPOSE:")
print(
    "Determine which workflow should handle the request next."
)

print("\nRESULT:")

routing_result = ask_model(
    routing_prompt
)


# ==========================================================
# PATTERN 6: TOOL SELECTION
# ==========================================================

tool_selection_prompt = """
You may use only these approved functions:

get_payment_status(dv_number)
search_purchase_order(po_number)
get_supplier_information(supplier_name)
send_notification(recipient, message)

User request:

"Has payment for DV-2026-1055 already been released?"

Select the most appropriate function.

Return JSON containing:

- tool
- arguments

Do not execute the function.
"""

print("\n" + "=" * 60)
print("PATTERN 6: TOOL SELECTION")
print("=" * 60)

print("\nPURPOSE:")
print(
    "Propose which approved tool should be used."
)

print("\nRESULT:")

tool_selection_result = ask_model(
    tool_selection_prompt
)


# ==========================================================
# PATTERN 7: CONTENT GENERATION
# ==========================================================

generation_prompt = """
Write a short and professional notification
using the following validated information.

DV Number: DV-2026-1055
Status: Returned
Reason: Missing supplier quotation

Tell the requester:

- why the DV was returned, and
- what action is required before resubmission.

Use no more than three sentences.

Do not add information that is not provided.
"""

print("\n" + "=" * 60)
print("PATTERN 7: CONTENT GENERATION")
print("=" * 60)

print("\nPURPOSE:")
print(
    "Transform validated information into human-readable text."
)

print("\nRESULT:")

generation_result = ask_model(
    generation_prompt
)


#%% EXERCISE 4: STRUCTURED AND SCHEMA-BASED OUTPUT
import ollama

from pydantic import TypeAdapter, ValidationError

model_name = "gemma3:4b"


# ----------------------------------------------------------
# 1. DEFINE JSON SCHEMA
# ----------------------------------------------------------

purchase_review_schema = {
    "type": "object",

    "properties": {

        "request_id": {
            "type": "string"
        },

        "decision": {
            "type": "string",
            "enum": [
                "HUMAN_REVIEW",
                "STANDARD_REVIEW"
            ]
        },

        "reason": {
            "type": "string"
        },

        "policy_applied": {
            "type": "string",
            "enum": [
                "AMOUNT_GT_1M",
                "AMOUNT_LE_1M"
            ]
        },

        "next_action": {
            "type": "string",
            "enum": [
                "ROUTE_TO_HUMAN_APPROVER",
                "CONTINUE_STANDARD_REVIEW"
            ]
        }
    },

    "required": [
        "request_id",
        "decision",
        "reason",
        "policy_applied",
        "next_action"
    ],

    "additionalProperties": False
}


# ----------------------------------------------------------
# 2. PURCHASE REQUEST
# ----------------------------------------------------------

purchase_request = """
Request ID: PR-2026-001
Department: Engineering Laboratory
Item: 20 laptop computers
Amount: PHP 1,200,000
Purpose: Replace obsolete laboratory computers.
Requested by: Laboratory Manager
"""


# ----------------------------------------------------------
# 3. SYSTEM PROMPT
# ----------------------------------------------------------

system_prompt = """
You are a Purchase Request Review Assistant.

BUSINESS RULE:

If the purchase amount is greater than PHP 1,000,000:

decision = HUMAN_REVIEW
policy_applied = AMOUNT_GT_1M
next_action = ROUTE_TO_HUMAN_APPROVER

Otherwise:

decision = STANDARD_REVIEW
policy_applied = AMOUNT_LE_1M
next_action = CONTINUE_STANDARD_REVIEW

Use only information supplied in the purchase request.
Do not invent information.

The reason should briefly explain why the selected
business rule applies.

Return only data conforming to the supplied JSON schema.
"""


# ----------------------------------------------------------
# 4. STREAM STRUCTURED OUTPUT
# ----------------------------------------------------------

print("========================================")
print("STREAMING STRUCTURED OUTPUT")
print("========================================")

stream = ollama.chat(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": purchase_request
        }
    ],
    format=purchase_review_schema,
    stream=True,
    think=False,
    options={
        "temperature": 0
    }
)

raw_output = ""

for chunk in stream:

    text = chunk["message"]["content"]

    print(
        text,
        end="",
        flush=True
    )

    raw_output += text

print()


# ----------------------------------------------------------
# 5. PYDANTIC VALIDATION
# ----------------------------------------------------------

review_adapter = TypeAdapter(
    dict[str, str]
)

try:

    review = review_adapter.validate_json(
        raw_output
    )

    pydantic_valid = True

except ValidationError as error:

    review = None
    pydantic_valid = False
    validation_error = error


# ----------------------------------------------------------
# 6. REQUIRED FIELDS
# ----------------------------------------------------------

required_fields = [
    "request_id",
    "decision",
    "reason",
    "policy_applied",
    "next_action"
]

if pydantic_valid:

    fields_present = all(
        field in review
        for field in required_fields
    )

else:

    fields_present = False


# ----------------------------------------------------------
# 7. ALLOWED VALUES
# ----------------------------------------------------------

allowed_decisions = [
    "HUMAN_REVIEW",
    "STANDARD_REVIEW"
]

allowed_policies = [
    "AMOUNT_GT_1M",
    "AMOUNT_LE_1M"
]

allowed_actions = [
    "ROUTE_TO_HUMAN_APPROVER",
    "CONTINUE_STANDARD_REVIEW"
]


if pydantic_valid and fields_present:

    values_valid = (
        review["decision"] in allowed_decisions
        and
        review["policy_applied"] in allowed_policies
        and
        review["next_action"] in allowed_actions
    )

else:

    values_valid = False


# ----------------------------------------------------------
# 8. DISPLAY VALIDATION
# ----------------------------------------------------------

print("\n========================================")
print("PYDANTIC AND SCHEMA VALIDATION")
print("========================================")

if pydantic_valid and fields_present and values_valid:

    print("VALIDATION: PASSED")

    print("\nVALIDATED PURCHASE REVIEW")

    print("Request ID     :", review["request_id"])
    print("Decision       :", review["decision"])
    print("Reason         :", review["reason"])
    print("Policy Applied :", review["policy_applied"])
    print("Next Action    :", review["next_action"])

else:

    print("VALIDATION: FAILED")

    if not pydantic_valid:

        print(
            "Reason: Pydantic could not validate "
            "the completed JSON output."
        )

        print(validation_error)

    elif not fields_present:

        print(
            "Reason: Required fields are missing."
        )

    elif not values_valid:

        print(
            "Reason: Unsupported values were returned."
        )


#%% EXERCISE 5: FAILURE MODES AND GUARDRAILS
import re
import ollama

from pydantic import TypeAdapter, ValidationError

model_name = "gemma3:4b"


# ----------------------------------------------------------
# 1. EXPECTED JSON SCHEMA
# ----------------------------------------------------------

purchase_review_schema = {
    "type": "object",

    "properties": {

        "request_id": {
            "type": "string"
        },

        "decision": {
            "type": "string",
            "enum": [
                "HUMAN_REVIEW",
                "STANDARD_REVIEW"
            ]
        },

        "reason": {
            "type": "string"
        },

        "policy_applied": {
            "type": "string",
            "enum": [
                "AMOUNT_GT_1M",
                "AMOUNT_LE_1M",
                "INSUFFICIENT_INFORMATION"
            ]
        },

        "next_action": {
            "type": "string",
            "enum": [
                "ROUTE_TO_HUMAN_APPROVER",
                "CONTINUE_STANDARD_REVIEW"
            ]
        }
    },

    "required": [
        "request_id",
        "decision",
        "reason",
        "policy_applied",
        "next_action"
    ],

    "additionalProperties": False
}


# ----------------------------------------------------------
# 2. TEST PURCHASE REQUESTS
# ----------------------------------------------------------

purchase_requests = [

    """
    Request ID: PR-2026-001
    Item: 20 laptop computers
    Amount: PHP 1,200,000
    Purpose: Replace obsolete laboratory computers.
    """,

    """
    Request ID: PR-2026-002
    Item: Office supplies
    Amount: PHP 250,000
    Purpose: Replenish office supplies.
    """,

    """
    Request ID: PR-2026-003
    Item: Laboratory equipment
    Amount: Not provided
    Purpose: Upgrade laboratory facilities.
    """
]


# ----------------------------------------------------------
# 3. SYSTEM PROMPT
# ----------------------------------------------------------

system_prompt = """
You are a Purchase Request Review Assistant.

BUSINESS RULES:

If amount > PHP 1,000,000:

decision = HUMAN_REVIEW
policy_applied = AMOUNT_GT_1M
next_action = ROUTE_TO_HUMAN_APPROVER


If amount <= PHP 1,000,000:

decision = STANDARD_REVIEW
policy_applied = AMOUNT_LE_1M
next_action = CONTINUE_STANDARD_REVIEW


If the amount is missing or cannot be determined:

decision = HUMAN_REVIEW
policy_applied = INSUFFICIENT_INFORMATION
next_action = ROUTE_TO_HUMAN_APPROVER


GUARDRAILS:

1. Use only information contained in the purchase request.
2. Never invent a missing amount.
3. Never assume missing information.
4. If information is insufficient, route to human review.
5. Return only data conforming to the supplied schema.
"""


# ----------------------------------------------------------
# 4. PYDANTIC VALIDATOR
# ----------------------------------------------------------

review_adapter = TypeAdapter(
    dict[str, str]
)


# ----------------------------------------------------------
# 5. PROCESS EACH REQUEST
# ----------------------------------------------------------

for purchase_request in purchase_requests:

    print("\n========================================")
    print("PURCHASE REQUEST")
    print("========================================")

    print(
        purchase_request.strip()
    )


    # ------------------------------------------------------
    # INPUT GUARDRAIL
    # ------------------------------------------------------

    amount_match = re.search(
        r"Amount:\s*PHP\s*([\d,]+)",
        purchase_request
    )

    if amount_match:

        amount = int(
            amount_match.group(1).replace(",", "")
        )

        input_complete = True

    else:

        amount = None
        input_complete = False


    # ------------------------------------------------------
    # STREAM LLM OUTPUT
    # ------------------------------------------------------

    print("\nLLM OUTPUT:")

    stream = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": purchase_request
            }
        ],
        format=purchase_review_schema,
        stream=True,
        think=False,
        options={
            "temperature": 0
        }
    )

    raw_output = ""

    for chunk in stream:

        text = chunk["message"]["content"]

        print(
            text,
            end="",
            flush=True
        )

        raw_output += text

    print()


    # ------------------------------------------------------
    # PYDANTIC VALIDATION
    # ------------------------------------------------------

    try:

        review = review_adapter.validate_json(
            raw_output
        )

        pydantic_valid = True

    except ValidationError:

        review = None
        pydantic_valid = False


    # ------------------------------------------------------
    # REQUIRED FIELDS
    # ------------------------------------------------------

    required_fields = [
        "request_id",
        "decision",
        "reason",
        "policy_applied",
        "next_action"
    ]

    if pydantic_valid:

        fields_present = all(
            field in review
            for field in required_fields
        )

    else:

        fields_present = False


    # ------------------------------------------------------
    # INDEPENDENT BUSINESS RULE
    # ------------------------------------------------------

    if not input_complete:

        expected_decision = "HUMAN_REVIEW"

        expected_policy = (
            "INSUFFICIENT_INFORMATION"
        )

        expected_action = (
            "ROUTE_TO_HUMAN_APPROVER"
        )

    elif amount > 1_000_000:

        expected_decision = "HUMAN_REVIEW"
        expected_policy = "AMOUNT_GT_1M"

        expected_action = (
            "ROUTE_TO_HUMAN_APPROVER"
        )

    else:

        expected_decision = "STANDARD_REVIEW"
        expected_policy = "AMOUNT_LE_1M"

        expected_action = (
            "CONTINUE_STANDARD_REVIEW"
        )


    # ------------------------------------------------------
    # BUSINESS RULE CHECK
    # ------------------------------------------------------

    if pydantic_valid and fields_present:

        business_rule_valid = (
            review["decision"]
            == expected_decision
            and
            review["policy_applied"]
            == expected_policy
            and
            review["next_action"]
            == expected_action
        )

    else:

        business_rule_valid = False


    # ------------------------------------------------------
    # FINAL GUARDRAIL
    # ------------------------------------------------------

    guardrails_passed = (
        pydantic_valid
        and
        fields_present
        and
        business_rule_valid
    )


    # ------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------

    print("\n========================================")
    print("GUARDRAIL RESULTS")
    print("========================================")

    print(
        "Input Complete       :",
        "YES"
        if input_complete
        else "NO"
    )

    print(
        "Pydantic Validation  :",
        "PASSED"
        if pydantic_valid
        else "FAILED"
    )

    print(
        "Required Fields      :",
        "PASSED"
        if fields_present
        else "FAILED"
    )

    print(
        "Business Rule Check  :",
        "PASSED"
        if business_rule_valid
        else "FAILED"
    )


    # ------------------------------------------------------
    # FINAL WORKFLOW ACTION
    # ------------------------------------------------------

    print("\n========================================")
    print("FINAL WORKFLOW ACTION")
    print("========================================")

    if not guardrails_passed:

        print(
            "STOP: Guardrail failure. "
            "Route to human review."
        )

    elif expected_policy == (
        "INSUFFICIENT_INFORMATION"
    ):

        print(
            "HUMAN REVIEW REQUIRED: "
            "Purchase amount is missing."
        )

    elif expected_decision == "HUMAN_REVIEW":

        print(
            "Route purchase request "
            "to human approver."
        )

    else:

        print(
            "Continue to standard review."
        )

