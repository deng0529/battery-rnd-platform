import pandas as pd
from core.supabase_client import get_supabase_client


PAGE_SIZE = 1000


def _fetch_all(query, page_size: int = PAGE_SIZE) -> list[dict]:
    """
    Fetch all rows from Supabase with pagination.
    Supabase often returns limited rows by default, so this avoids missing signals/data.
    """
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1
        res = query.range(start, end).execute()
        rows = res.data or []

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        start += page_size

    return all_rows


def get_datasets() -> pd.DataFrame:
    supabase = get_supabase_client()
    query = supabase.table("datasets").select("*")
    return pd.DataFrame(_fetch_all(query))


def get_units(dataset_id: str) -> pd.DataFrame:
    supabase = get_supabase_client()
    query = (
        supabase.table("battery_units")
        .select("*")
        .eq("dataset_id", dataset_id)
        .order("unit_id")
    )
    return pd.DataFrame(_fetch_all(query))


def get_cycles(dataset_id: str, unit_id: str) -> pd.DataFrame:
    supabase = get_supabase_client()
    query = (
        supabase.table("battery_cycles")
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .order("cycle_index")
    )

    df = pd.DataFrame(_fetch_all(query))

    if not df.empty and "cycle_index" in df.columns:
        df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
        df = df.dropna(subset=["cycle_index"])
        df["cycle_index"] = df["cycle_index"].astype(int)
        df = df.sort_values("cycle_index")

    return df


def get_available_signals(
    dataset_id: str,
    unit_id: str,
    cycle_index: int,
) -> list[str]:
    supabase = get_supabase_client()

    query = (
        supabase.table("battery_measurements")
        .select("signal_name")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .eq("cycle_index", int(cycle_index))
    )

    df = pd.DataFrame(_fetch_all(query))

    if df.empty or "signal_name" not in df.columns:
        return []

    signals = (
        df["signal_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(signals)


def get_measurements(
    dataset_id: str,
    unit_id: str,
    cycle_index: int,
    signal_name: str,
) -> pd.DataFrame:
    supabase = get_supabase_client()

    query = (
        supabase.table("battery_measurements")
        .select(
            "dataset_id, unit_id, cycle_index, time_seconds, "
            "signal_name, signal_value, signal_unit, source_signal_name"
        )
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .eq("cycle_index", int(cycle_index))
        .eq("signal_name", signal_name)
        .order("time_seconds")
    )

    df = pd.DataFrame(_fetch_all(query))

    if not df.empty:
        if "time_seconds" in df.columns:
            df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")

        if "signal_value" in df.columns:
            df["signal_value"] = pd.to_numeric(df["signal_value"], errors="coerce")

        df = df.dropna(subset=["time_seconds", "signal_value"])
        df = df.sort_values("time_seconds")

    return df


def get_aging_data(dataset_id: str, unit_id: str) -> pd.DataFrame:
    cycles = get_cycles(dataset_id, unit_id)

    if cycles.empty:
        return cycles

    if "cycle_type" not in cycles.columns:
        return pd.DataFrame()

    df = cycles[
        cycles["cycle_type"].astype(str).str.lower() == "discharge"
    ].copy()

    return df.sort_values("cycle_index")


def get_impedance_data(dataset_id: str, unit_id: str) -> pd.DataFrame:
    supabase = get_supabase_client()

    query = (
        supabase.table("battery_impedance")
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .order("cycle_index")
    )

    df = pd.DataFrame(_fetch_all(query))

    if not df.empty and "cycle_index" in df.columns:
        df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
        df = df.dropna(subset=["cycle_index"])
        df["cycle_index"] = df["cycle_index"].astype(int)
        df = df.sort_values("cycle_index")

    return df


def get_capacity_summary(dataset_id: str, unit_id: str) -> pd.DataFrame:
    aging_df = get_aging_data(dataset_id, unit_id)

    if aging_df.empty:
        return aging_df

    cols = [
        "dataset_id",
        "unit_id",
        "cycle_index",
        "cycle_type",
        "capacity",
    ]

    existing_cols = [c for c in cols if c in aging_df.columns]

    return aging_df[existing_cols].copy()