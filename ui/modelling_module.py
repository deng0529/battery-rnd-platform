import streamlit as st
import pandas as pd
import plotly.express as px

from core.data_service import get_model_features

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
        "SOH/RUL prediction using discharge model features from Supabase."
    )

    dataset_id = st.session_state.get("global_dataset_id")
    unit_id = st.session_state.get("global_unit_id")
    train_ratio = st.session_state.get("global_train_ratio", 0.7)

    if dataset_id is None or unit_id is None:
        st.warning("Please select dataset and battery unit from the left sidebar.")
        return

    config_col, result_col = st.columns([1.05, 2.95])

    with config_col:
        st.subheader("⚙ Model Selection")

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

        run_button = st.button(
            "▶ Run SOH/RUL Prediction",
            use_container_width=True,
        )

    with result_col:

        try:
            feature_df = get_model_features(dataset_id, unit_id)
        except Exception as e:
            st.error(f"Failed to load battery_model_features: {e}")
            return

        if feature_df.empty:
            st.warning("No model feature data found in Supabase.")
            return

        config = RULSOHExperimentConfig(
            train_ratio=train_ratio,
            eol_threshold=0.7,
            window_size=10,
            epochs=250,
        )

        runner = RULSOHExperimentRunner(config)

        try:
            prepared_df = runner.prepare_feature_preview(feature_df)
            train_df, test_df = runner.train_test_split_by_time(prepared_df)
        except Exception as e:
            st.error(f"Failed to prepare model features: {e}")
            return

        st.subheader("📊 Model Feature Table from Supabase")

        input_features = [
            "cycle_index",
            "voltage_mean",
            "voltage_std",
            "voltage_min",
            "voltage_max",
            "voltage_drop",
            "current_mean",
            "current_std",
            "temperature_mean",
            "temperature_max",
            "temperature_rise",
            "time_duration",
        ]

        output_labels = ["soh", "rul"]

        display_cols = [
            "dataset_id",
            "unit_id",
            "cycle_index",
            "cycle_type",
        ] + input_features + output_labels

        existing_cols = [c for c in display_cols if c in prepared_df.columns]

        st.dataframe(
            prepared_df[existing_cols].head(300),
            use_container_width=True,
            height=260,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Input Features")
            st.dataframe(
                pd.DataFrame({"Input Feature": input_features}),
                use_container_width=True,
                height=260,
            )

        with c2:
            st.markdown("### Output Labels")
            st.dataframe(
                pd.DataFrame({"Output Label": output_labels}),
                use_container_width=True,
                height=120,
            )

            st.markdown("### Train / Test Split")
            split_info = pd.DataFrame(
                [
                    {
                        "Part": "Training",
                        "Rows": len(train_df),
                        "Cycle Range": f"{int(train_df['cycle_index'].min())} - {int(train_df['cycle_index'].max())}",
                    },
                    {
                        "Part": "Testing",
                        "Rows": len(test_df),
                        "Cycle Range": f"{int(test_df['cycle_index'].min())} - {int(test_df['cycle_index'].max())}",
                    },
                ]
            )

            st.dataframe(split_info, use_container_width=True)

        st.divider()

        st.subheader("📈 Prediction Results")

        if not run_button:
            st.info("Click Run SOH/RUL Prediction to compare physical, GRU and hybrid models.")
            return

        with st.spinner("Training physical model, GRU model and hybrid model..."):

            try:
                output = runner.run(
                    feature_df=feature_df,
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

        pred_df = fig_df.rename(columns={"pred_soh": "SOH", "model": "Model"})
        pred_df = pred_df[["cycle_index", "SOH", "Model"]]

        plot_df = pd.concat([true_df, pred_df], ignore_index=True)

        fig = px.line(
            plot_df,
            x="cycle_index",
            y="SOH",
            color="Model",
            title="SOH Prediction Comparison",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 RUL Prediction Comparison")

        rul_true_df = fig_df[["cycle_index", "true_rul"]].drop_duplicates()
        rul_true_df = rul_true_df.rename(columns={"true_rul": "RUL"})
        rul_true_df["Model"] = "True RUL"

        rul_pred_df = fig_df.rename(columns={"pred_rul": "RUL", "model": "Model"})
        rul_pred_df = rul_pred_df[["cycle_index", "RUL", "Model"]]

        rul_plot_df = pd.concat([rul_true_df, rul_pred_df], ignore_index=True)

        rul_fig = px.line(
            rul_plot_df,
            x="cycle_index",
            y="RUL",
            color="Model",
            title="RUL Prediction Comparison",
        )

        st.plotly_chart(rul_fig, use_container_width=True)

        st.subheader("📏 Model Evaluation Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("🔋 Battery Health Indicators")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Latest SOH", health_summary.get("Latest SOH"))

        with c2:
            st.metric("EOL Status", health_summary.get("EOL Status"))

        with c3:
            st.metric("EOL Cycle", health_summary.get("EOL Cycle"))

        summary_df = pd.DataFrame(
            [{"Indicator": k, "Value": v} for k, v in health_summary.items()]
        )

        st.dataframe(summary_df, use_container_width=True)