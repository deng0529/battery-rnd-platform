import streamlit as st
import pandas as pd

from core.model_service import run_demo_prediction
from core.data_service import get_aging_data, get_cycles


def render_modelling_module():

    st.title("🧠 Modelling & Prediction")
    st.caption(
        "Choose physical and AI models, then run prediction evaluation using the globally selected battery data."
    )

    # -----------------------------
    # Read global sidebar selections
    # -----------------------------
    dataset_id = st.session_state.get("global_dataset_id")
    unit_id = st.session_state.get("global_unit_id")
    cycle_type = st.session_state.get("global_cycle_type", "all")
    cycle_index = st.session_state.get("global_cycle_index")
    signal_name = st.session_state.get("global_signal_name")
    train_ratio = st.session_state.get("global_train_ratio", 0.7)

    if dataset_id is None or unit_id is None:
        st.warning("Please select dataset and battery unit from the left sidebar.")
        return

    # -----------------------------
    # Main layout
    # -----------------------------
    model_col, result_col = st.columns([1.2, 2.4])

    # =============================
    # Left/Main: Model Configuration
    # =============================
    with model_col:
        st.subheader("⚙ Model Configuration")

        physical_model = st.selectbox(
            "Physical model",
            [
                "Capacity degradation curve",
                "Exponential degradation model",
                "Semi-empirical ageing model",
                "None",
            ],
        )

        ai_model = st.selectbox(
            "AI model",
            [
                "GRU",
                "LSTM",
                "Transformer",
                "MLP",
                "BRB-ER Hybrid",
            ],
        )

        st.markdown("### Current Data")

        st.write("**Dataset:**", dataset_id)
        st.write("**Battery Unit:**", unit_id)
        st.write("**Cycle Type:**", cycle_type)
        st.write("**Selected Cycle:**", cycle_index)
        st.write("**Signal:**", signal_name)
        st.write("**Training Ratio:**", train_ratio)

        run_button = st.button(
            "Run Prediction",
            use_container_width=True,
        )

    # =============================
    # Right: Data Preview + Results
    # =============================
    with result_col:
        st.subheader("📊 Selected Data")

        try:
            if cycle_type == "all":
                preview_df = get_cycles(dataset_id, unit_id)
            else:
                preview_df = get_cycles(dataset_id, unit_id)
                preview_df = preview_df[preview_df["cycle_type"] == cycle_type]

            if preview_df.empty:
                st.warning("No selected cycle data found.")
            else:
                st.dataframe(
                    preview_df.head(200),
                    use_container_width=True,
                    height=220,
                )

        except Exception as e:
            st.error(f"Failed to load selected data: {e}")
            preview_df = pd.DataFrame()

        st.divider()

        st.subheader("📈 Prediction Results")

        if run_button:
            result = run_demo_prediction(
                dataset=dataset_id,
                battery_unit=unit_id,
                cycle_type=cycle_type,
                physical_model=physical_model,
                ai_model=ai_model,
                train_ratio=train_ratio,
            )

            st.json(result)

            try:
                aging_df = get_aging_data(dataset_id, unit_id)

                if not aging_df.empty:
                    st.markdown("### Aging Data Used for Prediction")
                    st.dataframe(
                        aging_df.head(300),
                        use_container_width=True,
                        height=220,
                    )

            except Exception as e:
                st.warning(f"Prediction completed, but aging data preview failed: {e}")

        else:
            st.info("Select models and click Run Prediction.")