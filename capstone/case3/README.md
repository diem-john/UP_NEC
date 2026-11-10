# Pa-Law POC — Complaint/Inquiry Classification and Department Routing

## Goal and scope
This POC performs two sequential classification tasks on an employee's message:

1. **Message classification:** `complaint` or `inquiry`.
2. **Department routing:** identify the single most relevant receiving office department.

It classifies and suggests a destination only. It does not submit a complaint, contact a department, retrieve DOLE policies, interpret law, decide legal merits, or provide legal advice.

## Department taxonomy
The initial POC uses a configurable starter list:

| Output code | Display name | Typical concerns |
|---|---|---|
| `HR` | Human Resources | Leave, attendance, benefits, workplace conduct, harassment, employee relations |
| `FINANCE_PAYROLL` | Finance / Payroll | Salary, 13th-month pay, payslips, deductions, withholding, reimbursements |
| `LEGAL_COMPLIANCE` | Legal / Compliance | Formal legal/compliance review, alleged regulatory violations, formal DOLE/legal complaint escalation |
| `ADMIN_OPERATIONS` | Administration / Operations | Facilities, supplies, office access, administrative services |
| `IT` | IT / Technical Support | Devices, accounts, software, network, access systems |
| `GENERAL_UNCLEAR` | General / Needs Manual Triage | Insufficient detail or no suitable destination |

The taxonomy is a POC assumption, not a verified organization chart. Confirm the actual departments and routing ownership with the participating office before deployment.

## Two-stage workflow
1. Validate the employee message.
2. Stage 1 classifies complaint vs inquiry. Existing deterministic complaint patterns remain active; otherwise Qwen 3.5:4B classifies the message.
3. Stage 2 sends the original message and Stage 1 label to Qwen 3.5:4B to select one receiving department.
4. Python validates the department against the fixed enum and validates confidence/reason fields.
5. Low-confidence department predictions are mapped to `GENERAL_UNCLEAR` for manual triage.
6. The Streamlit UI shows both classifications and the routing rationale. It does not automatically send or disclose the message to any department.

## Model and tools
Python, Ollama, `qwen3.5:4b`, JSON Schema structured output, deterministic temperature 0, Python-side validation, Streamlit, and pytest. The two stages use separate model calls so each output has a narrow task and independently validated schema.

## Setup
```powershell
ollama pull qwen3.5:4b
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r capstone/case3/requirements.txt
```

## Run
```bash
streamlit run capstone/case3/palaw_app.py
```

## Tests
```bash
pytest -q capstone/case3/test_palaw_classifier.py
```

Unit tests mock the Ollama calls; they validate component logic but do not measure live model accuracy.

## Guardrails and limitations
- Only the two allowed message categories and six allowed department codes can pass validation.
- Empty, overlong, malformed, or unsupported outputs are rejected.
- User content is treated as untrusted input.
- Low-confidence department routing goes to manual triage rather than a guessed department.
- Complaint classification does not establish that a policy or law was violated.
- The department list and routing rules must be adapted to the actual organization's structure.
- Build a human-labelled Tagalog/English/Taglish evaluation set and report complaint recall, department macro-F1, and confusion matrices before operational use.
