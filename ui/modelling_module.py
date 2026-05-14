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
    st.caption("SOH/RUL prediction using discharge model features from Supabase.")

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

        ai_model_keys = st.multiselect(
            "AI model(s)",
            options=list(AI_MODEL_REGISTRY.keys()),
            default=["mlp", "residual_mlp"],
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

        st.subheader("📊 Model Variables and Example Data")

        example_row = prepared_df.iloc[0]

        max_len = max(len(INPUT_FEATURES), len(OUTPUT_LABELS))
        rows = []

        for i in range(max_len):
            input_name = INPUT_FEATURES[i] if i < len(INPUT_FEATURES) else ""
            output_name = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else ""

            rows.append({
                "输入特征": input_name,
                "输入示例值": example_row.get(input_name, ""),
                "输出": output_name,
                "输出示例值": example_row.get(output_name, ""),
            })

        variable_df = pd.DataFrame(rows)

        st.dataframe(
            variable_df,
            use_container_width=True,
            height=420,
        )

        st.divider()

        st.subheader("📈 Prediction Results")

        st.caption(
            f"Cycle-based split: {len(train_df)} training cycles "
            f"({int(train_df['cycle_index'].min())}–{int(train_df['cycle_index'].max())}) "
            f"and {len(test_df)} testing cycles "
            f"({int(test_df['cycle_index'].min())}–{int(test_df['cycle_index'].max())}), "
            f"train_ratio={train_ratio}."
        )

        if not run_button:
            st.info("Select models and click Run SOH/RUL Prediction.")
            return

        if not ai_model_keys:
            st.warning("Please select at least one AI model.")
            return

        ai_model_classes = [
            AI_MODEL_REGISTRY[key]["class"]
            for key in ai_model_keys
        ]

        with st.spinner("Training selected models..."):
            try:
                output = runner.run(
                    feature_df=feature_df,
                    physical_model_class=PHYSICAL_MODEL_REGISTRY[physical_model_key]["class"],
                    ai_model_classes=ai_model_classes,
                    hybrid_model_class=HYBRID_MODEL_CLASS,
                )
            except Exception as e:
                st.error(f"Model experiment failed: {e}")
                return

        prediction_df = output["predictions"]
        metrics_df = output["metrics"]
        health_summary = output["health_summary"]

        fig_df = prediction_df.copy()

        true_soh_df = fig_df[["cycle_index", "true_soh"]].drop_duplicates()
        true_soh_df = true_soh_df.rename(columns={"true_soh": "SOH"})
        true_soh_df["Model"] = "True SOH"

        pred_soh_df = fig_df.rename(columns={"pred_soh": "SOH", "model": "Model"})
        pred_soh_df = pred_soh_df[["cycle_index", "SOH", "Model"]]

        soh_plot_df = pd.concat([true_soh_df, pred_soh_df], ignore_index=True)

        soh_fig = px.line(
            soh_plot_df,
            x="cycle_index",
            y="SOH",
            color="Model",
            title="SOH Prediction Comparison",
        )

        st.plotly_chart(soh_fig, use_container_width=True)

        st.subheader("📉 RUL Prediction Comparison")

        true_rul_df = fig_df[["cycle_index", "true_rul"]].drop_duplicates()
        true_rul_df = true_rul_df.rename(columns={"true_rul": "RUL"})
        true_rul_df["Model"] = "True RUL"

        pred_rul_df = fig_df.rename(columns={"pred_rul": "RUL", "model": "Model"})
        pred_rul_df = pred_rul_df[["cycle_index", "RUL", "Model"]]

        rul_plot_df = pd.concat([true_rul_df, pred_rul_df], ignore_index=True)

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