import streamlit as st
import pandas as pd
import plotly.express as px

from agents.data_agent import run_data_copilot

from core.data_service import (
    get_datasets,
    get_units,
    get_cycles,
    get_measurements,
    get_aging_data,
    get_impedance_data,
)


def render_data_module():

    st.title("📊 Data System")
    st.caption("Battery data browsing, preview and diagnostic analysis")

    # -----------------------------
    # Load basic database tables
    # -----------------------------
    datasets = get_datasets()

    if datasets.empty:
        st.warning("No dataset found in database.")
        return

    # -----------------------------
    # Read global sidebar selections
    # These are created in app.py via render_sidebar_data_selection()
    # -----------------------------
    dataset_id = st.session_state.get("global_dataset_id")
    unit_id = st.session_state.get("global_unit_id")
    selected_cycle_type = st.session_state.get("global_cycle_type", "all")
    cycle_index = st.session_state.get("global_cycle_index")
    signal_name = st.session_state.get("global_signal_name")

    if dataset_id is None or unit_id is None or cycle_index is None:
        st.warning("Please select dataset, battery unit and cycle from the left sidebar.")
        return

    # -----------------------------
    # Load units and cycles based on global selection
    # -----------------------------
    units = get_units(dataset_id)

    if units.empty:
        st.warning("No battery units found for this dataset.")
        return

    cycles = get_cycles(dataset_id, unit_id)

    if cycles.empty:
        st.warning("No cycles found for this unit.")
        return

    filtered_cycles = cycles.copy()

    if selected_cycle_type != "all":
        filtered_cycles = filtered_cycles[
            filtered_cycles["cycle_type"] == selected_cycle_type
        ]

    if filtered_cycles.empty:
        st.warning("No cycles found for the selected cycle type.")
        return

    # -----------------------------
    # Dataset and unit info
    # -----------------------------
    dataset_row = datasets[datasets["dataset_id"] == dataset_id].iloc[0]
    unit_row = units[units["unit_id"] == unit_id].iloc[0]

    # -----------------------------
    # Copilot Result Panel
    # -----------------------------
    if "copilot_result" in st.session_state:
        copilot_result = st.session_state["copilot_result"]

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
                    if st.button(
                        "Clear",
                        key="clear_data_copilot_result",
                        use_container_width=True,
                    ):
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
                            fig = px.line(
                                df,
                                x="cycle_index",
                                y="capacity",
                                title="Capacity",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        if "soh" in df.columns:
                            fig = px.line(
                                df,
                                x="cycle_index",
                                y="soh",
                                title="SOH",
                            )
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

        st.write("**Dataset ID:**", dataset_id)
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

        if selected_cycle_type == "all":
            st.dataframe(cycles, use_container_width=True)
        else:
            st.dataframe(filtered_cycles, use_container_width=True)

    # -----------------------------
    # Cycle Curve
    # -----------------------------
    with tab_curve:
        st.subheader("Selected Cycle Signal Curve")

        st.write("**Dataset:**", dataset_id)
        st.write("**Battery Unit:**", unit_id)
        st.write("**Cycle:**", cycle_index)
        st.write("**Signal:**", signal_name)

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
                fig = px.line(
                    aging_df,
                    x="cycle_index",
                    y="capacity",
                    title=f"{unit_id} Capacity Trend",
                )
                st.plotly_chart(fig, use_container_width=True)

            if "soh" in aging_df.columns:
                fig = px.line(
                    aging_df,
                    x="cycle_index",
                    y="soh",
                    title=f"{unit_id} SOH Trend",
                )
                st.plotly_chart(fig, use_container_width=True)

            if "rul" in aging_df.columns:
                fig = px.line(
                    aging_df,
                    x="cycle_index",
                    y="rul",
                    title=f"{unit_id} RUL Trend",
                )
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

            numeric_cols = imp_df.select_dtypes(include="number").columns.tolist()

            if "cycle_index" in numeric_cols:
                plot_cols = [c for c in numeric_cols if c != "cycle_index"]

                if plot_cols:
                    selected_imp_signal = st.selectbox(
                        "Select impedance signal to plot",
                        plot_cols,
                    )

                    fig = px.line(
                        imp_df,
                        x="cycle_index",
                        y=selected_imp_signal,
                        title=f"{unit_id} | {selected_imp_signal}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

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
            if selected_cycle_type == "all":
                st.dataframe(cycles, use_container_width=True)
            else:
                st.dataframe(filtered_cycles, use_container_width=True)

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
            st.dataframe(
                get_impedance_data(dataset_id, unit_id),
                use_container_width=True,
            )