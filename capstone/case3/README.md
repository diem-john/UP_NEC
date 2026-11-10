# Pa-Law POC — Unified Classification and Routing

## Goal and scope
The POC accepts one employee message and returns a single structured decision with exactly five fields:

- `category`: `INQUIRY` or `COMPLAINT`
- `confidence`: model-reported category confidence from `0.0` to `1.0` (displayed as a percentage)
- `routed_team`: `HR`, `LEGAL`, `ACCOUNTING`, or `IT_SUPPORT`
- `reason`: concise explanation for the classification and routing
- `next_action`: `ROUTE_TO_LEGAL_AI_ASSISTANT` or `ROUTE_TO_OTHER_TEAM`

The model performs category classification and team routing in one Ollama call using the JSON schema. The confidence value is a model-reported estimate, not a statistically calibrated probability; it should be evaluated against a labelled dataset before being used for automated decisions.

## Routing rules
| Routed team | Typical concerns | Next action |
|---|---|---|
| HR | Leave, attendance, benefits, workplace conduct, harassment, employee relations | `ROUTE_TO_OTHER_TEAM` |
| LEGAL | Formal legal review, regulatory concerns, formal DOLE/legal complaint matters | `ROUTE_TO_LEGAL_AI_ASSISTANT` |
| ACCOUNTING | Salary, 13th-month pay, payslips, payroll deductions, tax withholding, reimbursements | `ROUTE_TO_OTHER_TEAM` |
| IT_SUPPORT | Company accounts, devices, software, network, access systems | `ROUTE_TO_OTHER_TEAM` |

The next-action mapping is also validated deterministically in Python: LEGAL must use `ROUTE_TO_LEGAL_AI_ASSISTANT`; all other teams must use `ROUTE_TO_OTHER_TEAM`.

## Workflow
1. Validate the employee message.
2. Send it to Qwen 3.5:4B with the combined system prompt and JSON schema.
3. Parse the JSON response.
4. Validate exact keys, enum values, confidence range, non-empty reason, and the team/next-action relationship.
5. Display the structured decision in Streamlit.

## Model and tools
Python, Ollama, `qwen3.5:4b`, JSON Schema structured output, deterministic temperature 0, 2,048-token context window, Python-side validation, Streamlit, and pytest.

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

Unit tests mock the Ollama call. They validate schema and routing logic but do not measure live model accuracy.

## Limitations
- The team taxonomy is a POC assumption and must be confirmed against the actual office structure.
- The output is a routing recommendation; no email, ticket, or referral is sent.
- A complaint label does not establish that a law or policy was violated.
- Build a human-labelled Tagalog/English/Taglish evaluation set and measure category and team-routing performance before operational use.
