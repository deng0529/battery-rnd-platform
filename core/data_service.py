import pandas as pd
from core.supabase_client import get_supabase_client


def get_datasets() -> pd.DataFrame:
    supabase = get_supabase_client()
    res = supabase.table("datasets").select("*").execute()
    return pd.DataFrame(res.data)


def get_units(dataset_id: str) -> pd.DataFrame:
    supabase = get_supabase_client()
    res = (
        supabase.table("battery_units")
        .select("*")
        .eq("dataset_id", dataset_id)
        .order("unit_id")
        .execute()
    )
    return pd.DataFrame(res.data)


def get_cycles(dataset_id: str, unit_id: str) -> pd.DataFrame:
    supabase = get_supabase_client()
    res = (
        supabase.table("battery_cycles")
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .order("cycle_index")
        .execute()
    )
    return pd.DataFrame(res.data)


def get_available_signals(dataset_id: str, unit_id: str, cycle_index: int) -> list[str]:
    supabase = get_supabase_client()
    res = (
        supabase.table("battery_measurements")
        .select("signal_name")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .eq("cycle_index", cycle_index)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return []
    return sorted(df["signal_name"].dropna().unique().tolist())


def get_measurements(
    dataset_id: str,
    unit_id: str,
    cycle_index: int,
    signal_name: str,
) -> pd.DataFrame:
    supabase = get_supabase_client()
    res = (
        supabase.table("battery_measurements")
        .select("time_seconds, signal_value, signal_unit, source_signal_name")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .eq("cycle_index", cycle_index)
        .eq("signal_name", signal_name)
        .order("time_seconds")
        .execute()
    )
    return pd.DataFrame(res.data)


def get_aging_data(dataset_id: str, unit_id: str) -> pd.DataFrame:
    cycles = get_cycles(dataset_id, unit_id)
    if cycles.empty:
        return cycles
    return cycles[cycles["cycle_type"] == "discharge"].copy()


def get_impedance_data(dataset_id: str, unit_id: str) -> pd.DataFrame:
    supabase = get_supabase_client()
    res = (
        supabase.table("battery_impedance")
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("unit_id", unit_id)
        .order("cycle_index")
        .execute()
    )
    return pd.DataFrame(res.data)

def get_capacity_summary(dataset_id: str, unit_id: str):
    return None