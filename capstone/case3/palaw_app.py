"""Streamlit interface for the Pa-Law complaint/inquiry POC."""
from __future__ import annotations

import streamlit as st

from palaw_classifier import ClassificationError, classify_message

st.set_page_config(
    page_title="Pa-Law | Labor Policy Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Pa-Law")
st.caption("Policy Assistant for Labor and Welfare")
st.write("Tell us about your workplace concern. Pa-Law will classify your message as a complaint or an inquiry.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("category"):
            st.caption(f"Classification: {item['category'].upper()}")

message = st.chat_input("Type your labor-related question or concern...")

if message:
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        try:
            with st.status("Pa-Law is processing your message...", expanded=True) as status:
                st.write("✓ Message received")
                st.write("⏳ Sending message to the classification model...")
                result = classify_message(message)
                st.write("✓ Classification received")
                st.write("✓ Output validated")
                status.update(
                    label="Classification complete",
                    state="complete",
                    expanded=False,
                )

            if result["category"] == "complaint":
                st.error("**COMPLAINT** — Your message describes a workplace problem or grievance.")
            else:
                st.info("**INQUIRY** — Your message is primarily asking for information or clarification.")

            st.caption(result["reason"])
            st.caption(f"Confidence: {result['confidence']:.0%} · Source: {result['source']}")
            st.caption("This is an initial message classification only, not a legal assessment.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"**{result['category'].upper()}** — {result['reason']}\n\n"
                    f"Confidence: {result['confidence']:.0%} · Source: {result['source']}\n\n"
                    "_This is an initial message classification only, not a legal assessment._"
                ),
                "category": result["category"],
            })
        except ClassificationError as exc:
            st.warning(f"Please review your message: {exc}")
        except Exception as exc:
            st.error("Pa-Law could not complete the classification. Check that Ollama is running and the configured model is available.")
            st.caption(f"Technical detail: {exc}")

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
