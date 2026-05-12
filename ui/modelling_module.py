import streamlit as st
import pandas as pd

from core.model_service import run_demo_prediction
# 后续可以替换成真实数据读取函数
# from core.data_service import load_battery_data


def render_modelling_module():

    st.title("🧠 Modelling & Prediction")
    st.caption("Select battery data, choose physical and AI models, then run prediction evaluation.")

    left_col, middle_col, right_col = st.columns([1.1, 1.3, 2.4])

    # =========================
    # Left: Data Selection
    # =========================
    with left_col:
        st.subheader("Data Selection")

        dataset = st.selectbox(
            "Select dataset",
            ["NASA"],
        )

        battery_unit = st.selectbox(
            "Select battery cell",
            ["B0005", "B0006", "B0007", "B0018"],
        )

        cycle_type = st.selectbox(
            "Cycle type",
            ["Discharge", "Charge", "Impedance"],
        )

        train_ratio = st.slider(
            "Training ratio",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
        )

    # =========================
    # Middle: Model Selection
    # =========================
    with middle_col:
        st.subheader("Model Configuration")

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

        run_button = st.button(
            "Run Prediction",
            use_container_width=True,
        )

    # =========================
    # Right: Data Preview + Results
    # =========================
    with right_col:
        st.subheader("Selected Data")

        # 这里先用 demo data，后续替换成 Supabase 数据
        demo_df = pd.DataFrame({
            "cycle_index": [1, 2, 3, 4, 5],
            "capacity": [1.86, 1.84, 1.82, 1.81, 1.79],
            "soh": [1.00, 0.99, 0.98, 0.97, 0.96],
            "temperature_mean": [24.5, 24.8, 25.1, 25.0, 25.3],
        })

        st.dataframe(demo_df, use_container_width=True, height=180)

        st.divider()

        st.subheader("Prediction Results")

        if run_button:
            result = run_demo_prediction(
                dataset=dataset,
                battery_unit=battery_unit,
                cycle_type=cycle_type,
                physical_model=physical_model,
                ai_model=ai_model,
                train_ratio=train_ratio,
            )

            st.json(result)

        else:
            st.info("Select data and models, then click Run Prediction.")