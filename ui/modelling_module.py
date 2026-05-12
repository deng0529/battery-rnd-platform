import streamlit as st
import pandas as pd
import plotly.express as px

from core.data_service import get_aging_data

from core.modelling.rul_soh_test import (
    RULSOHExperimentConfig,
    RULSOHExperimentRunner,
)

from core.modelling.model_register import MODEL_REGISTRY


def render_modelling_module():

    st.title("🧠 Modelling & Prediction")
    st.caption(
        "SOH/RUL prediction using physical degradation model, GRU model, and hybrid model."
    )

    # -------------------------------------------------
    # Read global sidebar selections
    # -------------------------------------------------
    dataset_id = st.session_state.get("global_dataset_id")
    unit_id = st.session_state.get("global_unit_id")
    train_ratio = st.session_state.get("global_train_ratio", 0.7)

    if dataset_id is None or unit_id is None:
        st.warning("Please select dataset and battery unit from the left sidebar.")
        return

    if unit_id != "B0005":
        st.info(
            "Current SOH/RUL demo is configured for B0005 first. "
            "Other cells can be enabled after validating the workflow."
        )

    config_col, result_col = st.columns([1.15, 2.85])

    # -------------------------------------------------
    # Left: Model Configuration
    # -------------------------------------------------
    with config_col:
        st.subheader("⚙ Model Configuration")

        model_options = {
            key: value["display_name"]
            for key, value in MODEL_REGISTRY.items()
        }

        default_models = [
            key for key in [
                "physical_capacity",
                "gru",
                "hybrid_physical_gru",
            ]
            if key in MODEL_REGISTRY
        ]

        selected_model_keys = st.multiselect(
            "Select models to compare",
            options=list(model_options.keys()),
            default=default_models,
            format_func=lambda key: model_options[key],
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

    # -------------------------------------------------
    # Right: Data + Prediction Results
    # -------------------------------------------------
    with result_col:

        st.subheader("📊 Selected Aging Data")

        try:
            aging_df = get_aging_data(dataset_id, unit_id)
        except Exception as e:
            st.error(f"Failed to load aging data: {e}")
            return

        if aging_df.empty:
            st.warning("No aging data found for the selected battery.")
            return

        st.dataframe(
            aging_df.head(200),
            use_container_width=True,
            height=220,
        )

        st.divider()

        st.subheader("📈 SOH Prediction Comparison")

        if not run_button:
            st.info("Click Run SOH/RUL Prediction to compare selected models.")
            return

        if not selected_model_keys:
            st.warning("Please select at least one model.")
            return

        with st.spinner("Training and evaluating selected models..."):

            config = RULSOHExperimentConfig(
                eol_threshold=eol_threshold,
                train_ratio=train_ratio,
                window_size=window_size,
                epochs=epochs,
            )

            runner = RULSOHExperimentRunner(config)

            try:
                output = runner.run(
                    aging_df=aging_df,
                    selected_model_keys=selected_model_keys,
                    model_registry=MODEL_REGISTRY,
                )
            except Exception as e:
                st.error(f"Model experiment failed: {e}")
                return

        prediction_df = output["predictions"]
        metrics_df = output["metrics"]
        health_summary = output["health_summary"]

        # -------------------------------------------------
        # SOH plot
        # -------------------------------------------------
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
            markers=False,
            title="SOH Prediction Comparison",
        )

        fig.update_layout(
            height=520,
            xaxis_title="Cycle Index",
            yaxis_title="SOH",
            legend_title="Model",
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------
        # RUL plot
        # -------------------------------------------------
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
            markers=False,
            title="RUL Prediction Comparison",
        )

        rul_fig.update_layout(
            height=420,
            xaxis_title="Cycle Index",
            yaxis_title="RUL",
            legend_title="Model",
        )

        st.plotly_chart(rul_fig, use_container_width=True)

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Battery expert health indicators
        # -------------------------------------------------
        st.subheader("🔋 Battery Health Indicators")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Latest SOH",
                health_summary.get("Latest SOH"),
            )

        with c2:
            st.metric(
                "Capacity Loss",
                f"{health_summary.get('Capacity Loss (%)')}%",
            )

        with c3:
            st.metric(
                "EOL Status",
                health_summary.get("EOL Status"),
            )

        with c4:
            st.metric(
                "Recent Fade Rate",
                health_summary.get("Recent Avg Fade Rate"),
            )

        st.markdown("### Detailed Battery Performance Summary")

        summary_df = pd.DataFrame(
            [
                {"Indicator": key, "Value": value}
                for key, value in health_summary.items()
            ]
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
        )