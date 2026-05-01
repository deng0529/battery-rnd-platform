import streamlit as st
import pandas as pd
import plotly.express as px

from agents.data_agent import run_data_copilot

from core.data_service import (
    get_datasets,
    get_units,
    get_cycles,
    get_available_signals,
    get_measurements,
    get_aging_data,
    get_impedance_data,
)


def render_data_module():

    st.title("📊 Data System")
    st.caption("Battery data browsing, preview and diagnostic analysis")

    datasets = get_datasets()

    if datasets.empty:
        st.warning("No dataset found in database.")
        return

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.header("Data Selection")

    dataset_id = st.sidebar.selectbox(
        "Dataset",
        datasets["dataset_id"].tolist(),
    )

    units = get_units(dataset_id)

    if units.empty:
        st.warning("No battery units found for this dataset.")
        return

    unit_id = st.sidebar.selectbox(
        "Battery Unit / Cell",
        units["unit_id"].tolist(),
    )

    cycles = get_cycles(dataset_id, unit_id)

    if cycles.empty:
        st.warning("No cycles found for this unit.")
        return

    cycle_types = ["all"] + sorted(cycles["cycle_type"].dropna().unique().tolist())
    selected_cycle_type = st.sidebar.selectbox("Cycle Type", cycle_types)

    filtered_cycles = cycles.copy()
    if selected_cycle_type != "all":
        filtered_cycles = filtered_cycles[filtered_cycles["cycle_type"] == selected_cycle_type]

    cycle_index = st.sidebar.selectbox(
        "Cycle",
        filtered_cycles["cycle_index"].tolist(),
    )

    signals = get_available_signals(dataset_id, unit_id, cycle_index)

    if not signals:
        st.sidebar.warning("No measurement signals found for this cycle.")
        signal_name = None
    else:
        signal_name = st.sidebar.selectbox("Signal", signals)

    # -----------------------------
    # Dataset info
    # -----------------------------
    dataset_row = datasets[datasets["dataset_id"] == dataset_id].iloc[0]
    unit_row = units[units["unit_id"] == unit_id].iloc[0]

    # -----------------------------
    # Copilot Result Panel（核心）
    # -----------------------------
    if "copilot_result" in st.session_state:
        copilot_result = st.session_state["copilot_result"]

        # ⚠️ 关键修复：这里必须是 "data"
        if copilot_result.get("target_module") == "data":

            context = {
                "dataset_id": dataset_id,
                "unit_id": unit_id,
                "cycle_index": cycle_index,
                "signal_name": signal_name,
            }

            result = run_data_copilot(
                copilot_result["user_input"],
                context,
            )

            with st.container(border=True):
                col_title, col_clear = st.columns([5, 1])

                with col_title:
                    st.subheader(result.get("title", "Copilot Result"))

                with col_clear:
                    if st.button("Clear", key="clear_data_copilot_result", use_container_width=True):
                        if "copilot_result" in st.session_state:
                            del st.session_state["copilot_result"]
                        if "copilot_prompt" in st.session_state:
                            del st.session_state["copilot_prompt"]
                        st.rerun()

                st.write(result.get("message", ""))

                # ---------- Cycle curve ----------
                if result["type"] == "cycle_curve":
                    df = result["data"]

                    if df.empty:
                        st.warning("No data found.")
                    else:
                        fig = px.line(
                            df,
                            x="time_seconds",
                            y="signal_value",
                            title=f"{unit_id} | Cycle {result['cycle_index']} | {result['signal_name']}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df.head(200), use_container_width=True)

                # ---------- Aging ----------
                elif result["type"] == "aging":
                    df = result["data"]

                    if df.empty:
                        st.warning("No aging data found.")
                    else:
                        if "capacity" in df.columns:
                            fig = px.line(df, x="cycle_index", y="capacity", title="Capacity")
                            st.plotly_chart(fig, use_container_width=True)

                        if "soh" in df.columns:
                            fig = px.line(df, x="cycle_index", y="soh", title="SOH")
                            st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(df, use_container_width=True)

                # ---------- Impedance ----------
                elif result["type"] == "impedance":
                    df = result["data"]

                    if df.empty:
                        st.warning("No impedance data found.")
                    else:
                        st.dataframe(df, use_container_width=True)

                # ---------- Summary ----------
                elif result["type"] == "summary":
                    st.dataframe(result["table"], use_container_width=True)

                # ---------- Message ----------
                elif result["type"] == "message":
                    st.info(result["message"])

    # -----------------------------
    # Tabs
    # -----------------------------
    tab_overview, tab_curve, tab_aging, tab_impedance, tab_raw = st.tabs(
        [
            "Overview",
            "Cycle Curve",
            "Aging Analysis",
            "Impedance",
            "Raw Data",
        ]
    )

    # -----------------------------
    # Overview
    # -----------------------------
    with tab_overview:
        st.subheader("Dataset Information")

        st.write("**Dataset Name:**", dataset_row.get("dataset_name"))
        st.write("**Source:**", dataset_row.get("source_name"))
        st.write("**Type:**", dataset_row.get("source_type"))
        st.write("**Raw Format:**", dataset_row.get("raw_format"))
        st.write("**Description:**", dataset_row.get("description"))

        with st.expander("Dataset Metadata"):
            st.json(dataset_row.get("metadata"))

        st.subheader("Battery Unit Information")

        unit_info = pd.DataFrame([unit_row.to_dict()])
        st.dataframe(unit_info, use_container_width=True)

        st.subheader("Cycle Summary")
        st.dataframe(cycles, use_container_width=True)

    # -----------------------------
    # Cycle Curve
    # -----------------------------
    with tab_curve:
        st.subheader("Selected Cycle Signal Curve")

        if signal_name is None:
            st.warning("No signal available for the selected cycle.")
        else:
            df = get_measurements(
                dataset_id=dataset_id,
                unit_id=unit_id,
                cycle_index=cycle_index,
                signal_name=signal_name,
            )

            if df.empty:
                st.warning("No measurement data found.")
            else:
                fig = px.line(
                    df,
                    x="time_seconds",
                    y="signal_value",
                    title=f"{unit_id} | Cycle {cycle_index} | {signal_name}",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.head(200), use_container_width=True)

    # -----------------------------
    # Aging
    # -----------------------------
    with tab_aging:
        st.subheader("Capacity / SOH / RUL Trend")

        aging_df = get_aging_data(dataset_id, unit_id)

        if aging_df.empty:
            st.warning("No discharge aging data found.")
        else:
            if "capacity" in aging_df.columns:
                fig = px.line(aging_df, x="cycle_index", y="capacity")
                st.plotly_chart(fig, use_container_width=True)

            if "soh" in aging_df.columns:
                fig = px.line(aging_df, x="cycle_index", y="soh")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(aging_df, use_container_width=True)

    # -----------------------------
    # Impedance
    # -----------------------------
    with tab_impedance:
        st.subheader("Impedance Analysis")

        imp_df = get_impedance_data(dataset_id, unit_id)

        if imp_df.empty:
            st.info("No impedance data found.")
        else:
            st.dataframe(imp_df, use_container_width=True)

    # -----------------------------
    # Raw Data
    # -----------------------------
    with tab_raw:
        st.subheader("Raw Database Preview")

        raw_option = st.selectbox(
            "Select Table Preview",
            [
                "datasets",
                "battery_units",
                "battery_cycles",
                "battery_measurements",
                "battery_impedance",
            ],
        )

        if raw_option == "datasets":
            st.dataframe(datasets, use_container_width=True)

        elif raw_option == "battery_units":
            st.dataframe(units, use_container_width=True)

        elif raw_option == "battery_cycles":
            st.dataframe(cycles, use_container_width=True)

        elif raw_option == "battery_measurements":
            if signal_name is not None:
                preview_df = get_measurements(
                    dataset_id=dataset_id,
                    unit_id=unit_id,
                    cycle_index=cycle_index,
                    signal_name=signal_name,
                )
                st.dataframe(preview_df.head(500), use_container_width=True)
            else:
                st.warning("No signal selected.")

        elif raw_option == "battery_impedance":
            st.dataframe(get_impedance_data(dataset_id, unit_id), use_container_width=True)