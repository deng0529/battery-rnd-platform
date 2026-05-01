import streamlit as st
from core.model_service import run_demo_prediction


def render_modelling_module():

    st.title("🧠 Modelling & Prediction")
    st.caption("Model selection, train/test configuration and prediction evaluation.")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Configuration")
        model_type = st.selectbox("Model type", ["Demo Linear Model", "LSTM", "Transformer", "BRB-ER Hybrid"])
        train_ratio = st.slider("Training ratio", 0.5, 0.9, 0.7)
        run_button = st.button("Run Demo Prediction", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")
        if run_button:
            result = run_demo_prediction(model_type=model_type, train_ratio=train_ratio)
            st.json(result)
        else:
            st.info("Configure a model and click Run Demo Prediction.")
