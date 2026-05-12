import streamlit as st
import pandas as pd
import plotly.express as px

from core.data_service import get_aging_data

from core.modelling.rul_soh_test import (
    RULSOHExperimentConfig,
    RULSOHExperimentRunner,
)

from core.modelling.model_register import (
    PHYSICAL_MODEL_REGISTRY,
    AI_MODEL_REGISTRY,
    HYBRID_MODEL_CLASS,
)


def render_modelling_module():

    st.title("🧠 Modelling & Prediction")
    st.caption(
        "SOH/RUL prediction using a selected physical model and GRU neural network."
    )

    dataset_id = st.session_state.get("global_dataset_id")
    unit_id = st.session_state.get("global_unit_id")
    train_ratio = st.session_state.get("global_train_ratio", 0.7)

    if dataset_id is None or unit_id is None:
        st.warning("Please select dataset and battery unit from the left sidebar.")
        return

    config_col, result_col = st.columns([1.15, 2.85])

    with config_col:
        st.subheader("⚙ Model Configuration")

        physical_model_key = st.selectbox(
            "Physical model",
            options=list(PHYSICAL_MODEL_REGISTRY.keys()),
            format_func=lambda key: PHYSICAL_MODEL_REGISTRY[key]["display_name"],
        )

        ai_model_key = st.selectbox(
            "AI model",
            options=["gru"],
            format_func=lambda key: AI_MODEL_REGISTRY[key]["display_name"],
        )

        st.divider()

        st.markdown("### Training Setup")

        eol_threshold = st.slider(
            "EOL SOH threshold",
            min_value=0.6,
            max_value=0.9,
            value=0.7,
            step=0.05,
        )

        window_size = st.slider(
            "GRU sequence window",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
        )

        epochs = st.slider(
            "Training epochs",
            min_value=50,
            max_value=500,
            value=250,
            step=50,
        )

        st.markdown("### Current Data")
        st.write("**Dataset:**", dataset_id)
        st.write("**Battery Unit:**", unit_id)
        st.write("**Training Ratio:**", train_ratio)

        run_button = st.button(
            "▶ Run SOH/RUL Prediction",
            use_container_width=True,
        )

    with result_col:

        try:
            aging_df = get_aging_data(dataset_id, unit_id)
        except Exception as e:
            st.error(f"Failed to load aging data: {e}")
            return

        if aging_df.empty:
            st.warning("No aging data found for the selected battery.")
            return

        config = RULSOHExperimentConfig(
            eol_threshold=eol_threshold,
            train_ratio=train_ratio,
            window_size=window_size,
            epochs=epochs,
        )

        runner = RULSOHExperimentRunner(config)

        try:
            prepared_df = runner.prepare_feature_preview(
                aging_df=aging_df,
                dataset_id=dataset_id,
                unit_id=unit_id,
            )
        except Exception as e:
            st.error(f"Failed to prepare feature data: {e}")
            return

        st.subheader("📊 Feature Data Used for Modelling")

        feature_display_cols = [
            "dataset_id",
            "unit_id",
            "cycle_index",
            "cycle_type",
            "capacity",
            "soh",
            "rul",
            "capacity_fade",
            "delta_soh",
            "fade_rate",
            "voltage_mean",
            "voltage_min",
            "voltage_max",
            "temperature_mean",
            "temperature_max",
            "time_duration",
        ]

        existing_cols = [
            col for col in feature_display_cols if col in prepared_df.columns
        ]

        st.dataframe(
            prepared_df[existing_cols].head(300),
            use_container_width=True,
            height=260,
        )

        st.divider()

        st.subheader("📈 SOH Prediction Comparison")

        if not run_button:
            st.info("Click Run SOH/RUL Prediction to compare physical, GRU and hybrid models.")
            return

        with st.spinner("Training physical model, GRU model and hybrid model..."):

            try:
                output = runner.run(
                    aging_df=aging_df,
                    dataset_id=dataset_id,
                    unit_id=unit_id,
                    physical_model_class=PHYSICAL_MODEL_REGISTRY[physical_model_key]["class"],
                    ai_model_class=AI_MODEL_REGISTRY[ai_model_key]["class"],
                    hybrid_model_class=HYBRID_MODEL_CLASS,
                )
            except Exception as e:
                st.error(f"Model experiment failed: {e}")
                return

        prediction_df = output["predictions"]
        metrics_df = output["metrics"]
        health_summary = output["health_summary"]

        fig_df = prediction_df.copy()

        true_df = fig_df[["cycle_index", "true_soh"]].drop_duplicates()
        true_df = true_df.rename(columns={"true_soh": "SOH"})
        true_df["Model"] = "True SOH"

        pred_df = fig_df.rename(
            columns={
                "pred_soh": "SOH",
                "model": "Model",
            }
        )
        pred_df = pred_df[["cycle_index", "SOH", "Model"]]

        plot_df = pd.concat([true_df, pred_df], ignore_index=True)

        fig = px.line(
            plot_df,
            x="cycle_index",
            y="SOH",
            color="Model",
            title="SOH Prediction: Physical Model vs GRU vs Hybrid",
        )

        fig.update_layout(
            height=500,
            xaxis_title="Cycle Index",
            yaxis_title="SOH",
            legend_title="Model",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 RUL Prediction Comparison")

        rul_true_df = fig_df[["cycle_index", "true_rul"]].drop_duplicates()
        rul_true_df = rul_true_df.rename(columns={"true_rul": "RUL"})
        rul_true_df["Model"] = "True RUL"

        rul_pred_df = fig_df.rename(
            columns={
                "pred_rul": "RUL",
                "model": "Model",
            }
        )
        rul_pred_df = rul_pred_df[["cycle_index", "RUL", "Model"]]

        rul_plot_df = pd.concat([rul_true_df, rul_pred_df], ignore_index=True)

        rul_fig = px.line(
            rul_plot_df,
            x="cycle_index",
            y="RUL",
            color="Model",
            title="RUL Prediction: Physical Model vs GRU vs Hybrid",
        )

        rul_fig.update_layout(
            height=415,
            xaxis_title="Cycle Index",
            yaxis_title="RUL",
            legend_title="Model",
        )

        st.plotly_chart(rul_fig, use_container_width=True)

        st.subheader("📏 Model Evaluation Metrics")

        metric_cols = [
            "Model",
            "SOH RMSE",
            "SOH MAE",
            "RUL RMSE",
            "RUL MAE",
            "RUL MAPE (%)",
        ]

        existing_metric_cols = [
            col for col in metric_cols if col in metrics_df.columns
        ]

        st.dataframe(
            metrics_df[existing_metric_cols],
            use_container_width=True,
        )

        st.subheader("🔋 Battery Health Indicators")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Latest SOH", health_summary.get("Latest SOH"))

        with c2:
            st.metric("Capacity Loss", f"{health_summary.get('Capacity Loss (%)')}%")

        with c3:
            st.metric("EOL Status", health_summary.get("EOL Status"))

        with c4:
            st.metric("Recent Fade Rate", health_summary.get("Recent Avg Fade Rate"))

        st.markdown("### Detailed Battery Performance Summary")

        summary_df = pd.DataFrame(
            [{"Indicator": k, "Value": v} for k, v in health_summary.items()]
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
        )