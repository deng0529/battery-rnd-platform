from pathlib import Path
import json
import toml
import psycopg2
import pandas as pd
import numpy as np

from data_adapters.nasa_adapter import DATASET_INFO


BASE_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

DATASET_ID = DATASET_INFO["dataset_id"]
OUTPUT_CSV = PROCESSED_DATA_DIR / "battery_discharge_model_features.csv"


def load_database_url() -> str:
    secrets = toml.load(SECRETS_PATH)
    return secrets["SUPABASE_DATABASE_URL"]


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_csv_files():
    cycles_path = PROCESSED_DATA_DIR / "battery_cycles.csv"
    timeseries_path = PROCESSED_DATA_DIR / "battery_timeseries.csv"

    if not cycles_path.exists():
        raise FileNotFoundError(f"Missing file: {cycles_path}")

    if not timeseries_path.exists():
        raise FileNotFoundError(f"Missing file: {timeseries_path}")

    cycles = pd.read_csv(cycles_path)
    timeseries = pd.read_csv(timeseries_path)

    print("battery_cycles:", cycles.shape)
    print("battery_timeseries:", timeseries.shape)

    return cycles, timeseries


def build_discharge_cycle_base(cycles: pd.DataFrame) -> pd.DataFrame:
    required = ["cell_id", "cycle_index", "cycle_type", "capacity"]

    missing = [c for c in required if c not in cycles.columns]
    if missing:
        raise ValueError(f"battery_cycles.csv missing columns: {missing}")

    df = cycles.copy()

    df["cycle_type"] = df["cycle_type"].astype(str).str.lower()
    df = df[df["cycle_type"] == "discharge"].copy()

    df["dataset_id"] = DATASET_ID
    df["unit_id"] = df["cell_id"]

    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    df = df.dropna(subset=["cycle_index", "capacity"])
    df["cycle_index"] = df["cycle_index"].astype(int)

    keep_cols = [
        "dataset_id",
        "unit_id",
        "cell_id",
        "cycle_index",
        "cycle_type",
        "capacity",
    ]

    return df[keep_cols].copy()


def build_discharge_timeseries_features(timeseries: pd.DataFrame) -> pd.DataFrame:
    required = [
        "cell_id",
        "cycle_index",
        "cycle_type",
        "time_seconds",
        "voltage_measured",
        "current_measured",
        "temperature_measured",
    ]

    missing = [c for c in required if c not in timeseries.columns]
    if missing:
        raise ValueError(f"battery_timeseries.csv missing columns: {missing}")

    df = timeseries.copy()

    df["cycle_type"] = df["cycle_type"].astype(str).str.lower()
    df = df[df["cycle_type"] == "discharge"].copy()

    df["dataset_id"] = DATASET_ID
    df["unit_id"] = df["cell_id"]

    numeric_cols = [
        "cycle_index",
        "time_seconds",
        "voltage_measured",
        "current_measured",
        "temperature_measured",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "cycle_index",
            "time_seconds",
            "voltage_measured",
            "current_measured",
            "temperature_measured",
        ]
    )

    df["cycle_index"] = df["cycle_index"].astype(int)

    group_cols = [
        "dataset_id",
        "unit_id",
        "cell_id",
        "cycle_index",
        "cycle_type",
    ]

    features = (
        df.groupby(group_cols)
        .agg(
            voltage_mean=("voltage_measured", "mean"),
            voltage_std=("voltage_measured", "std"),
            voltage_min=("voltage_measured", "min"),
            voltage_max=("voltage_measured", "max"),
            voltage_start=("voltage_measured", "first"),
            voltage_end=("voltage_measured", "last"),

            current_mean=("current_measured", "mean"),
            current_std=("current_measured", "std"),

            temperature_mean=("temperature_measured", "mean"),
            temperature_max=("temperature_measured", "max"),
            temperature_start=("temperature_measured", "first"),
            temperature_end=("temperature_measured", "last"),

            time_start=("time_seconds", "min"),
            time_end=("time_seconds", "max"),
            n_points=("time_seconds", "count"),
        )
        .reset_index()
    )

    features["voltage_drop"] = (
        features["voltage_start"] - features["voltage_end"]
    )

    features["temperature_rise"] = (
        features["temperature_end"] - features["temperature_start"]
    )

    features["time_duration"] = (
        features["time_end"] - features["time_start"]
    )

    final_cols = [
        "dataset_id",
        "unit_id",
        "cell_id",
        "cycle_index",
        "cycle_type",

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
        "n_points",
    ]

    return features[final_cols].copy()


def add_soh_rul(df: pd.DataFrame, eol_threshold: float = 0.7) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["dataset_id", "unit_id", "cycle_index"])

    df["discharge_index"] = np.nan
    df["soh"] = np.nan
    df["rul"] = np.nan
    df["eol_cycle_index"] = np.nan

    for (dataset_id, unit_id), group in df.groupby(["dataset_id", "unit_id"]):
        group = group.sort_values("cycle_index")

        q0 = group["capacity"].iloc[0]
        soh = group["capacity"] / q0

        eol_rows = group[soh <= eol_threshold]

        if eol_rows.empty:
            eol_cycle = int(group["cycle_index"].iloc[-1])
        else:
            eol_cycle = int(eol_rows["cycle_index"].iloc[0])

        rul = eol_cycle - group["cycle_index"]
        rul = rul.clip(lower=0)

        df.loc[group.index, "discharge_index"] = range(1, len(group) + 1)
        df.loc[group.index, "soh"] = soh
        df.loc[group.index, "rul"] = rul
        df.loc[group.index, "eol_cycle_index"] = eol_cycle

    df["capacity_fade"] = 1 - df["soh"]
    df["delta_soh"] = df.groupby(["dataset_id", "unit_id"])["soh"].diff()
    df["fade_rate"] = -df["delta_soh"]

    return df


def build_model_features(eol_threshold: float = 0.7) -> pd.DataFrame:
    cycles, timeseries = load_csv_files()

    base = build_discharge_cycle_base(cycles)
    ts_features = build_discharge_timeseries_features(timeseries)

    merge_cols = [
        "dataset_id",
        "unit_id",
        "cell_id",
        "cycle_index",
        "cycle_type",
    ]

    features = base.merge(
        ts_features,
        on=merge_cols,
        how="inner",
    )

    features = add_soh_rul(features, eol_threshold=eol_threshold)

    final_cols = [
        "dataset_id",
        "unit_id",
        "cell_id",
        "cycle_index",
        "discharge_index",
        "cycle_type",

        # labels
        "capacity",
        "soh",
        "rul",
        "eol_cycle_index",

        # optional degradation descriptors
        "capacity_fade",
        "delta_soh",
        "fade_rate",

        # GRU input features
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

        # quality check
        "n_points",
    ]

    features = features[final_cols].copy()

    features = features.sort_values(
        ["dataset_id", "unit_id", "cycle_index"]
    ).reset_index(drop=True)

    return features


def save_features_csv(features: pd.DataFrame):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {len(features.columns)}")


def create_supabase_table(conn):
    sql = """
    create table if not exists battery_model_features (
        dataset_id text not null,
        unit_id text not null,
        cycle_index integer not null,

        cell_id text,
        discharge_index integer,
        cycle_type text,

        capacity double precision,
        soh double precision,
        rul double precision,
        eol_cycle_index integer,

        capacity_fade double precision,
        delta_soh double precision,
        fade_rate double precision,

        voltage_mean double precision,
        voltage_std double precision,
        voltage_min double precision,
        voltage_max double precision,
        voltage_drop double precision,

        current_mean double precision,
        current_std double precision,

        temperature_mean double precision,
        temperature_max double precision,
        temperature_rise double precision,

        time_duration double precision,
        n_points integer,

        metadata jsonb,

        primary key (dataset_id, unit_id, cycle_index)
    );
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    print("Ensured table: battery_model_features")


def upload_to_supabase(features: pd.DataFrame):
    database_url = load_database_url()

    conn = psycopg2.connect(database_url)
    conn.autocommit = False

    create_supabase_table(conn)

    sql = """
    insert into battery_model_features (
        dataset_id,
        unit_id,
        cycle_index,

        cell_id,
        discharge_index,
        cycle_type,

        capacity,
        soh,
        rul,
        eol_cycle_index,

        capacity_fade,
        delta_soh,
        fade_rate,

        voltage_mean,
        voltage_std,
        voltage_min,
        voltage_max,
        voltage_drop,

        current_mean,
        current_std,

        temperature_mean,
        temperature_max,
        temperature_rise,

        time_duration,
        n_points,

        metadata
    )
    values (
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s,
        %s::jsonb
    )
    on conflict (dataset_id, unit_id, cycle_index)
    do update set
        cell_id = excluded.cell_id,
        discharge_index = excluded.discharge_index,
        cycle_type = excluded.cycle_type,

        capacity = excluded.capacity,
        soh = excluded.soh,
        rul = excluded.rul,
        eol_cycle_index = excluded.eol_cycle_index,

        capacity_fade = excluded.capacity_fade,
        delta_soh = excluded.delta_soh,
        fade_rate = excluded.fade_rate,

        voltage_mean = excluded.voltage_mean,
        voltage_std = excluded.voltage_std,
        voltage_min = excluded.voltage_min,
        voltage_max = excluded.voltage_max,
        voltage_drop = excluded.voltage_drop,

        current_mean = excluded.current_mean,
        current_std = excluded.current_std,

        temperature_mean = excluded.temperature_mean,
        temperature_max = excluded.temperature_max,
        temperature_rise = excluded.temperature_rise,

        time_duration = excluded.time_duration,
        n_points = excluded.n_points,

        metadata = excluded.metadata;
    """

    rows = []

    for _, row in features.iterrows():
        metadata = {
            "feature_source": "discharge_timeseries",
            "input_features": [
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
            ],
            "output_labels": [
                "soh",
                "rul",
            ],
        }

        rows.append((
            clean_value(row["dataset_id"]),
            clean_value(row["unit_id"]),
            int(row["cycle_index"]),

            clean_value(row.get("cell_id")),
            clean_value(row.get("discharge_index")),
            clean_value(row.get("cycle_type")),

            clean_value(row.get("capacity")),
            clean_value(row.get("soh")),
            clean_value(row.get("rul")),
            clean_value(row.get("eol_cycle_index")),

            clean_value(row.get("capacity_fade")),
            clean_value(row.get("delta_soh")),
            clean_value(row.get("fade_rate")),

            clean_value(row.get("voltage_mean")),
            clean_value(row.get("voltage_std")),
            clean_value(row.get("voltage_min")),
            clean_value(row.get("voltage_max")),
            clean_value(row.get("voltage_drop")),

            clean_value(row.get("current_mean")),
            clean_value(row.get("current_std")),

            clean_value(row.get("temperature_mean")),
            clean_value(row.get("temperature_max")),
            clean_value(row.get("temperature_rise")),

            clean_value(row.get("time_duration")),
            clean_value(row.get("n_points")),

            json.dumps(metadata),
        ))

    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)

        conn.commit()
        print(f"Uploaded rows to battery_model_features: {len(rows)}")

    except Exception as e:
        conn.rollback()
        print("Upload failed. Rolled back.")
        print(e)
        raise

    finally:
        conn.close()


def main(upload: bool = True):
    features = build_model_features(eol_threshold=0.7)
    save_features_csv(features)

    if upload:
        upload_to_supabase(features)


if __name__ == "__main__":
    main(upload=True)