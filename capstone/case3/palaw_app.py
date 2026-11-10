"""Streamlit interface for the Pa-Law two-stage classification and routing POC."""
from __future__ import annotations

import streamlit as st

from palaw_classifier import ClassificationError, classify_message, classify_department

DEPARTMENT_LABELS = {
    "HR": "Human Resources (HR)",
    "FINANCE_PAYROLL": "Finance / Payroll",
    "LEGAL_COMPLIANCE": "Legal / Compliance",
    "ADMIN_OPERATIONS": "Administration / Operations",
    "IT": "IT / Technical Support",
    "GENERAL_UNCLEAR": "General / Needs Manual Triage",
}

st.set_page_config(
    page_title="Pa-Law | Labor Policy Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Pa-Law")
st.caption("Policy Assistant for Labor and Welfare")
st.write(
    "Tell us about your workplace concern. Pa-Law will first classify it as a "
    "complaint or inquiry, then identify the most relevant office department."
)

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
            with st.status("Pa-Law is processing your message...", expanded=True) as status:
                st.write("✓ Message received")
                st.write("⏳ Stage 1: Classifying complaint or inquiry...")
                first = classify_message(message)

                st.write(f"✓ Initial classification: **{first['category'].upper()}**")
                st.write("⏳ Stage 2: Identifying the receiving department...")
                routing = classify_department(message, first["category"])

                st.write(f"✓ Department identified: **{DEPARTMENT_LABELS[routing['department']]}**")
                st.write("✓ Both outputs validated")
                status.update(
                    label="Classification and department routing complete",
                    state="complete",
                    expanded=False,
                )

            if first["category"] == "complaint":
                st.error("**COMPLAINT** — Your message describes a workplace problem or grievance.")
            else:
                st.info("**INQUIRY** — Your message is primarily asking for information or clarification.")

            st.success(f"**Suggested receiving department:** {DEPARTMENT_LABELS[routing['department']]}")
            st.caption(f"Classification confidence: {first['confidence']:.0%} · Source: {first['source']}")
            st.caption(f"Department-routing confidence: {routing['confidence']:.0%} · Source: {routing['source']}")
            st.caption(f"Classification: {first['reason']}")
            st.caption(f"Routing rationale: {routing['reason']}")
            st.caption("This is an initial routing suggestion, not a legal assessment or an automatic referral.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"**Type:** {first['category'].upper()}\n\n"
                    f"**Suggested department:** {DEPARTMENT_LABELS[routing['department']]}\n\n"
                    f"**Classification confidence:** {first['confidence']:.0%}\n\n"
                    f"**Routing confidence:** {routing['confidence']:.0%}\n\n"
                    f"**Why:** {routing['reason']}\n\n"
                    "_This is a routing suggestion only. No automatic referral has been made._"
                ),
            })
        except ClassificationError as exc:
            st.warning(f"Please review your message: {exc}")
        except Exception as exc:
            st.error(
                "Pa-Law could not complete the classification and routing. "
                "Check that Ollama is running and the configured model is available."
            )
            st.caption(f"Technical detail: {exc}")

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
