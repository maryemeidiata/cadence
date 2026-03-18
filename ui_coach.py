"""
UI for the Sync tab — AI strategy coach with clickable prompts.
"""

import streamlit as st
from llm_coach import chat_with_coach


def render_coach() -> None:

    st.markdown("## Sync — AI Strategy Coach")
    st.caption(
        "Your workload advisor — ask about risks, run what-if scenarios, "
        "and get strategic recommendations powered by your live data."
    )

    if "coach_history" not in st.session_state:
        st.session_state.coach_history = []
    if "coach_display" not in st.session_state:
        st.session_state.coach_display = []

    # ── Clickable suggested prompts ──
    if not st.session_state.coach_display:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What's my biggest risk?", key="sug_1", use_container_width=True):
                st.session_state["queued_prompt"] = "What's my biggest risk right now?"
                st.rerun()
            if st.button("What if I drop a task?", key="sug_3", use_container_width=True):
                st.session_state["queued_prompt"] = "What if I drop my highest risk task?"
                st.rerun()
        with col2:
            if st.button("What should I focus on today?", key="sug_2", use_container_width=True):
                st.session_state["queued_prompt"] = "What should I focus on today?"
                st.rerun()
            if st.button("Compare two options", key="sug_4", use_container_width=True):
                st.session_state["queued_prompt"] = "Compare dropping my highest risk task vs extending its deadline by 5 days"
                st.rerun()

    # ── Display conversation history ──
    for role, text in st.session_state.coach_display:
        with st.chat_message(role):
            st.markdown(text)

    # ── Check for queued prompt ──
    prompt = st.session_state.pop("queued_prompt", None)

    # ── Chat input ──
    typed = st.chat_input("Ask Sync anything about your workload...")
    if typed:
        prompt = typed

    if prompt:
        api_key = ""
        try:
            api_key = st.secrets["CO_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
        if not api_key:
            import os
            api_key = os.environ.get("CO_API_KEY", "")
        if not api_key:
            st.error("Cohere API key not configured. Set CO_API_KEY in your environment or Streamlit secrets.")
            return

        if not st.session_state.get("tasks"):
            st.session_state.coach_display.append(("user", prompt))
            st.session_state.coach_display.append(("assistant",
                "You don't have any tasks yet. Open the sidebar to add tasks first, "
                "then I can help you analyse your workload."))
            st.rerun()
            return

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.coach_display.append(("user", prompt))

        with st.chat_message("assistant"):
            with st.spinner("Sync is thinking..."):
                try:
                    response_text, updated_history = chat_with_coach(
                        prompt, st.session_state.coach_history)
                    st.markdown(response_text)
                    st.session_state.coach_history = updated_history
                    st.session_state.coach_display.append(("assistant", response_text))
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.coach_display.append(("assistant", error_msg))

        st.rerun()

    if st.session_state.coach_display:
        if st.button("Clear conversation", key="clear_coach"):
            st.session_state.coach_history = []
            st.session_state.coach_display = []
            st.rerun()