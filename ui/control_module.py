import streamlit as st
from core.control_service import run_demo_control_optimisation


def render_control_module():

    st.title("🎛️ AI Control & Optimisation")
    st.caption("Optimisation objective, control policy and decision support.")


    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Setup")
        objective = st.selectbox("Objective", ["Extend RUL", "Reduce degradation", "Improve fast charging safety"])
        method = st.selectbox("Method", ["Rule-based", "Optimisation", "Reinforcement Learning", "BRB-ER Decision"])
        run_button = st.button("Run Demo Control", use_container_width=True)

    with col2:
        st.subheader("Control Recommendation")
        if run_button:
            result = run_demo_control_optimisation(objective=objective, method=method)
            st.json(result)
        else:
            st.info("Choose an objective and method, then run the demo control module.")
