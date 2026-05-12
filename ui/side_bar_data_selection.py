import streamlit as st

from core.data_service import (
    get_datasets,
    get_units,
    get_cycles,
    get_available_signals,
)


def render_sidebar_data_selection():

    st.sidebar.divider()
    st.sidebar.markdown("### 📂 Data Selection")

    datasets = get_datasets()

    if datasets.empty:
        st.sidebar.warning("No dataset found.")
        return None

    dataset_id = st.sidebar.selectbox(
        "Dataset",
        datasets["dataset_id"].tolist(),
        key="global_dataset_id",
    )

    units = get_units(dataset_id)

    if units.empty:
        st.sidebar.warning("No battery units found.")
        return None

    unit_id = st.sidebar.selectbox(
        "Battery Unit / Cell",
        units["unit_id"].tolist(),
        key="global_unit_id",
    )

    cycles = get_cycles(dataset_id, unit_id)

    if cycles.empty:
        st.sidebar.warning("No cycles found.")
        return None

    cycle_types = ["all"] + sorted(cycles["cycle_type"].dropna().unique().tolist())

    selected_cycle_type = st.sidebar.selectbox(
        "Cycle Type",
        cycle_types,
        key="global_cycle_type",
    )

    filtered_cycles = cycles.copy()
    if selected_cycle_type != "all":
        filtered_cycles = filtered_cycles[
            filtered_cycles["cycle_type"] == selected_cycle_type
        ]

    cycle_index = st.sidebar.selectbox(
        "Cycle",
        filtered_cycles["cycle_index"].tolist(),
        key="global_cycle_index",
    )

    signals = get_available_signals(dataset_id, unit_id, cycle_index)

    if not signals:
        signal_name = None
        st.sidebar.warning("No signals found for this cycle.")
    else:
        signal_name = st.sidebar.selectbox(
            "Signal",
            signals,
            key="global_signal_name",
        )

    train_ratio = st.sidebar.slider(
        "Training Ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        key="global_train_ratio",
    )

    return {
        "dataset_id": dataset_id,
        "unit_id": unit_id,
        "cycle_type": selected_cycle_type,
        "cycle_index": cycle_index,
        "signal_name": signal_name,
        "train_ratio": train_ratio,
    }