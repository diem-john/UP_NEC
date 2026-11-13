"""Streamlit interface for Pa-Law's unified decision-schema POC."""
from __future__ import annotations

import streamlit as st

from palaw_classifier import ClassificationError, classify_and_route

TEAM_LABELS = {
    "HR": "Human Resources (HR)",
    "LEGAL": "Legal",
    "ACCOUNTING": "Accounting",
    "IT_SUPPORT": "IT Support",
}
ACTION_LABELS = {
    "ROUTE_TO_LEGAL_AI_ASSISTANT": "Route to Legal AI Assistant",
    "ROUTE_TO_OTHER_TEAM": "Refer to the selected office team",
}

st.set_page_config(page_title="Pa-Law | Labor Policy Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ Pa-Law")
st.caption("Policy Assistant for Labor and Welfare")
st.write("Describe your workplace concern. Pa-Law will classify the message and recommend the appropriate receiving team.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

message = st.chat_input("Type your labor-related question or concern...")

if message:
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        try:
            with st.status("Pa-Law is analyzing your message...", expanded=True) as status:
                st.write("✓ Message received")
                st.write("⏳ Classifying message and selecting the receiving team...")
                result = classify_and_route(message)
                st.write("✓ Decision schema validated")
                status.update(label="Classification and routing complete", state="complete", expanded=False)

            st.caption(f"Category confidence (model estimate): {result['confidence']:.0%}")

            if result["category"] == "COMPLAINT":
                st.error("**COMPLAINT**")
            else:
                st.info("**INQUIRY**")

            st.success(f"**Routed team:** {TEAM_LABELS[result['routed_team']]}")
            if result["next_action"] == "ROUTE_TO_LEGAL_AI_ASSISTANT":
                st.info("**Next action:** Route to Legal AI Assistant")
            else:
                st.info("**Next action:** Refer to the selected office team")

            st.caption(f"Reason: {result['reason']}")
            st.caption("This is a routing recommendation only. No message has been automatically referred.")
            st.code(result, language="json")

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"**Category:** {result['category']} ({result['confidence']:.0%} model-estimated confidence)\n\n"
                    f"**Routed team:** {TEAM_LABELS[result['routed_team']]}\n\n"
                    f"**Reason:** {result['reason']}\n\n"
                    f"**Next action:** {ACTION_LABELS[result['next_action']]}\n\n"
                    "_This is a routing recommendation only. No message has been automatically referred._"
                ),
            })
        except ClassificationError as exc:
            st.warning(f"Please review the message: {exc}")
        except Exception as exc:
            st.error("Pa-Law could not complete the decision. Check that Ollama is running and the configured model is available.")
            st.caption(f"Technical detail: {exc}")

if st.session_state.messages and st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
