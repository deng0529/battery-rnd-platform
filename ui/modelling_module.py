import streamlit as st
import pandas as pd
import plotly.express as px

from core.data_service import get_model_features

from core.modelling.rul_soh_test import (
    RULSOHExperimentConfig,
    RULSOHExperimentRunner,
    INPUT_FEATURES,
    OUTPUT_LABELS,
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

    # =====================================================
    # Left: model selection only
    # =====================================================
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

    # =====================================================
    # Right: variables + results
    # =====================================================
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

        # -------------------------------------------------
        # One clean variable table only
        # -------------------------------------------------
        st.subheader("📊 Model Variables")

        variables_df = pd.DataFrame(
            [{"Variable Type": "Input Feature", "Variable Name": col} for col in INPUT_FEATURES]
            +
            [{"Variable Type": "Output Label", "Variable Name": col} for col in OUTPUT_LABELS]
        )

        st.dataframe(
            variables_df,
            use_container_width=True,
            height=360,
        )

        st.divider()

        # -------------------------------------------------
        # Results area
        # -------------------------------------------------
        st.subheader("📈 Prediction Results")

        st.caption(
            f"Training/testing split is cycle-based: "
            f"{len(train_df)} training cycles "
            f"({int(train_df['cycle_index'].min())}–{int(train_df['cycle_index'].max())}) "
            f"and {len(test_df)} testing cycles "
            f"({int(test_df['cycle_index'].min())}–{int(test_df['cycle_index'].max())}), "
            f"using train_ratio={train_ratio}."
        )

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

        # -------------------------------------------------
        # SOH plot
        # -------------------------------------------------
        fig_df = prediction_df.copy()

        true_soh_df = fig_df[["cycle_index", "true_soh"]].drop_duplicates()
        true_soh_df = true_soh_df.rename(columns={"true_soh": "SOH"})
        true_soh_df["Model"] = "True SOH"

        pred_soh_df = fig_df.rename(
            columns={
                "pred_soh": "SOH",
                "model": "Model",
            }
        )
        pred_soh_df = pred_soh_df[["cycle_index", "SOH", "Model"]]

        soh_plot_df = pd.concat([true_soh_df, pred_soh_df], ignore_index=True)

        soh_fig = px.line(
            soh_plot_df,
            x="cycle_index",
            y="SOH",
            color="Model",
            title="SOH Prediction Comparison",
        )

        soh_fig.update_layout(
            height=480,
            xaxis_title="Cycle Index",
            yaxis_title="SOH",
            legend_title="Model",
        )

        st.plotly_chart(soh_fig, use_container_width=True)

        # -------------------------------------------------
        # RUL plot
        # -------------------------------------------------
        st.subheader("📉 RUL Prediction Comparison")

        true_rul_df = fig_df[["cycle_index", "true_rul"]].drop_duplicates()
        true_rul_df = true_rul_df.rename(columns={"true_rul": "RUL"})
        true_rul_df["Model"] = "True RUL"

        pred_rul_df = fig_df.rename(
            columns={
                "pred_rul": "RUL",
                "model": "Model",
            }
        )
        pred_rul_df = pred_rul_df[["cycle_index", "RUL", "Model"]]

        rul_plot_df = pd.concat([true_rul_df, pred_rul_df], ignore_index=True)

        rul_fig = px.line(
            rul_plot_df,
            x="cycle_index",
            y="RUL",
            color="Model",
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

        st.dataframe(
            metrics_df,
            use_container_width=True,
        )

        # -------------------------------------------------
        # Battery health indicators
        # -------------------------------------------------
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

        st.dataframe(
            summary_df,
            use_container_width=True,
        )