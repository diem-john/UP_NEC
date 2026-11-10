# Pa-Law POC — Complaint vs Inquiry Classifier

## Goal and scope
Implement the first AI-supported activity in the Pa-Law swimlane: classify an employee's message into exactly one of two labels.

- **complaint** — the message reports a workplace problem, grievance, unfair treatment, or alleged violation.
- **inquiry** — the message asks for information, explanation, eligibility, procedure, or clarification without reporting a specific grievance.

This POC classifies intent only. It does not retrieve DOLE policies, interpret law, decide whether a complaint is legally valid, or provide legal advice.

## Alignment with the existing UP_NEC examples
Uses the same stack and calling pattern as \`exercise1/Case 3.py\`: Python, Ollama, \`qwen3.5:4b\`, deterministic temperature, JSON-schema output, and Python-side validation. The user's PALAW slide deck identifies DOLE policies as the POC input and uses employee examples such as emergency leave, missing 13th-month pay, deductions, and workplace complaints.

## Setup
Install and start Ollama, then pull the model:

\`\`\`bash
ollama pull qwen3.5:4b
\`\`\`

Create and activate a virtual environment, then install the root project requirements (they already include \`ollama\` and \`pytest\`):

\`\`\`bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
\`\`\`

## Run
From the repository root:

\`\`\`bash
python capstone/case3/palaw_classifier.py
\`\`\`

Or import the function:

\`\`\`python
from capstone.case3.palaw_classifier import classify_message
print(classify_message("Bakit wala akong 13th month pay?"))
\`\`\`

## Tests
\`\`\`bash
pytest -q capstone/case3/test_palaw_classifier.py
\`\`\`

Unit tests mock the Ollama call, so they do not require the model to be running.

## Guardrails
1. Reject empty, overlong, non-string, and unsupported-control-character input.
2. Constrain model output to a JSON schema with only the two allowed labels.
3. Validate JSON keys, category, confidence range, and reason length in Python.
4. Apply a deterministic Python complaint rule to several high-signal grievance patterns.
5. Use temperature 0 and a conservative inquiry fallback when model confidence is below 0.50.
6. Treat user input as untrusted and instruct the model to ignore prompt-injection instructions.

## Evaluation checklist
Test normal inquiry, normal complaint, incomplete input, overlong input, adversarial prompt text, malformed model output, and low-confidence output. For a more meaningful POC evaluation, build a human-labelled Tagalog/English/Taglish sample set and report complaint recall separately from inquiry precision. Current unit tests validate component behavior; they are not a measured model-accuracy benchmark.

## Limitation
The deterministic patterns are a starter safeguard, not a complete language understanding system. Review and expand them using consented, anonymized, human-labelled examples. Ambiguous cases should be clarified or escalated in a later workflow rather than treated as a legal determination.
