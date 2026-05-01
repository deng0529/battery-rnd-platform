import streamlit as st

from agents.router_agent import route_user_request


def render_copilot_sidebar(current_module: str):
    prompt = st.sidebar.text_area(
        "Ask Copilot",
        placeholder="e.g. show voltage curve for cycle 5",
        height=120,
    )

    if st.sidebar.button("Run", use_container_width=True):
        if prompt.strip():
            st.session_state["copilot_prompt"] = prompt
            st.session_state["copilot_result"] = route_user_request(
                user_input=prompt,
                current_module=current_module,
            )
            st.rerun()