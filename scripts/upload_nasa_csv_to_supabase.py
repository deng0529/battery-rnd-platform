from pathlib import Path
import json
import toml
import psycopg2
import pandas as pd

from data_adapters.nasa_adapter import (
    DATASET_INFO,
    SIGNAL_MAPPING,
    BATTERY_META,
    build_dataset_metadata,
)


BASE_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

DATASET_ID = DATASET_INFO["dataset_id"]


def load_database_url() -> str:
    secrets = toml.load(SECRETS_PATH)
    return secrets["SUPABASE_DATABASE_URL"]


def clean_value(value):
    if pd.isna(value):
        return None
    return value


def ensure_dataset(conn):
    metadata = build_dataset_metadata(PROCESSED_DATA_DIR)

    sql = """
    insert into datasets (
        dataset_id,
        dataset_name,
        source_name,
        source_type,
        description,
        raw_format,
        metadata
    )
    values (%s, %s, %s, %s, %s, %s, %s::jsonb)
    on conflict (dataset_id)
    do update set
        dataset_name = excluded.dataset_name,
        source_name = excluded.source_name,
        source_type = excluded.source_type,
        description = excluded.description,
        raw_format = excluded.raw_format,
        metadata = excluded.metadata;
    """

    values = (
        DATASET_INFO["dataset_id"],
        DATASET_INFO["dataset_name"],
        DATASET_INFO["source_name"],
        DATASET_INFO["source_type"],
        DATASET_INFO["description"],
        DATASET_INFO["raw_format"],
        json.dumps(metadata),
    )

    with conn.cursor() as cur:
        cur.execute(sql, values)

    print("Dataset registered.")


def upload_battery_units(conn):
    csv_path = PROCESSED_DATA_DIR / "battery_cells.csv"
    df = pd.read_csv(csv_path)

    sql = """
    insert into battery_units (
        dataset_id,
        unit_id,
        unit_type,
        source_unit_id,
        source_unit_type,
        display_name,
        rated_capacity,
        eol_capacity,
        cutoff_voltage,
        initial_capacity,
        metadata
    )
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    on conflict (dataset_id, unit_id)
    do update set
        unit_type = excluded.unit_type,
        source_unit_id = excluded.source_unit_id,
        source_unit_type = excluded.source_unit_type,
        display_name = excluded.display_name,
        rated_capacity = excluded.rated_capacity,
        eol_capacity = excluded.eol_capacity,
        cutoff_voltage = excluded.cutoff_voltage,
        initial_capacity = excluded.initial_capacity,
        metadata = excluded.metadata;
    """

    rows = []

    for _, row in df.iterrows():
        cell_id = row["cell_id"]
        meta = BATTERY_META.get(cell_id, {})

        metadata = {
            "battery_no": clean_value(row.get("battery_no")),
            "source_file": clean_value(row.get("source_file")),
            "dataset_name_from_csv": clean_value(row.get("dataset_name")),
            "source_metadata": meta,
        }

        rows.append((
            DATASET_ID,
            cell_id,
            "cell",
            cell_id,
            "cell",
            f"NASA Battery {cell_id}",
            clean_value(row.get("rated_capacity")),
            clean_value(row.get("eol_capacity")),
            clean_value(row.get("cutoff_voltage")),
            clean_value(row.get("initial_capacity")),
            json.dumps(metadata),
        ))

    with conn.cursor() as cur:
        cur.executemany(sql, rows)

    print(f"Uploaded battery_units: {len(rows)} rows")


def upload_battery_cycles(conn):
    csv_path = PROCESSED_DATA_DIR / "battery_cycles.csv"
    df = pd.read_csv(csv_path)

    sql = """
    insert into battery_cycles (
        dataset_id,
        unit_id,
        cycle_index,
        cycle_type,
        source_cycle_type,
        ambient_temperature,
        matlab_start_time,
        capacity,
        soh,
        rul,
        metadata
    )
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    on conflict (dataset_id, unit_id, cycle_index)
    do update set
        cycle_type = excluded.cycle_type,
        source_cycle_type = excluded.source_cycle_type,
        ambient_temperature = excluded.ambient_temperature,
        matlab_start_time = excluded.matlab_start_time,
        capacity = excluded.capacity,
        soh = excluded.soh,
        rul = excluded.rul,
        metadata = excluded.metadata;
    """

    rows = []

    for _, row in df.iterrows():
        metadata = {}

        rows.append((
            DATASET_ID,
            row["cell_id"],
            int(row["cycle_index"]),
            clean_value(row.get("cycle_type")),
            clean_value(row.get("cycle_type")),
            clean_value(row.get("ambient_temperature")),
            clean_value(row.get("matlab_start_time")),
            clean_value(row.get("capacity")),
            clean_value(row.get("soh")),
            clean_value(row.get("rul")),
            json.dumps(metadata),
        ))

    with conn.cursor() as cur:
        cur.executemany(sql, rows)

    print(f"Uploaded battery_cycles: {len(rows)} rows")


def upload_battery_measurements(
    conn,
    target_cell="B0005",
    max_cycles=10,
    batch_size=5000
):
    csv_path = PROCESSED_DATA_DIR / "battery_timeseries.csv"
    df = pd.read_csv(csv_path)

    # 👉 只保留一个电池
    df = df[df["cell_id"] == target_cell]

    # 👉 只保留前 N 个 cycle
    df = df[df["cycle_index"] <= max_cycles]

    print(f"Uploading measurements for {target_cell}, cycles <= {max_cycles}")
    print(f"Filtered rows: {len(df)}")

    sql_delete = """
    delete from battery_measurements
    where dataset_id = %s;
    """

    sql_insert = """
    insert into battery_measurements (
        dataset_id,
        unit_id,
        cycle_index,
        point_index,
        time_seconds,
        signal_name,
        signal_value,
        signal_unit,
        source_signal_name,
        metadata
    )
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
    """

    # 👉 先删除旧数据
    with conn.cursor() as cur:
        cur.execute(sql_delete, (DATASET_ID,))

    rows = []
    total_inserted = 0

    for _, row in df.iterrows():
        for source_col, mapping in SIGNAL_MAPPING.items():
            if source_col not in df.columns:
                continue

            value = row.get(source_col)
            if pd.isna(value):
                continue

            metadata = {
                "cycle_type": clean_value(row.get("cycle_type")),
                "source_column": source_col,
            }

            rows.append((
                DATASET_ID,
                row["cell_id"],
                int(row["cycle_index"]),
                int(row["point_index"]),
                clean_value(row.get("time_seconds")),
                mapping["signal_name"],
                float(value),
                mapping["signal_unit"],
                mapping["source_signal_name"],
                json.dumps(metadata),
            ))

            if len(rows) >= batch_size:
                with conn.cursor() as cur:
                    cur.executemany(sql_insert, rows)
                total_inserted += len(rows)
                print(f"Inserted: {total_inserted}")
                rows = []

    if rows:
        with conn.cursor() as cur:
            cur.executemany(sql_insert, rows)
        total_inserted += len(rows)

    print(f"Total inserted measurements: {total_inserted}")


def upload_battery_impedance(conn):
    csv_path = PROCESSED_DATA_DIR / "battery_impedance.csv"

    if not csv_path.exists():
        print("battery_impedance.csv not found. Skipped.")
        return

    df = pd.read_csv(csv_path)

    sql_delete = """
    delete from battery_impedance
    where dataset_id = %s;
    """

    sql_insert = """
    insert into battery_impedance (
        dataset_id,
        unit_id,
        cycle_index,
        ambient_temperature,
        matlab_start_time,
        sense_current,
        battery_current,
        current_ratio,
        battery_impedance,
        rectified_impedance,
        re,
        rct,
        metadata
    )
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
    """

    with conn.cursor() as cur:
        cur.execute(sql_delete, (DATASET_ID,))

    rows = []

    for _, row in df.iterrows():
        metadata = {}

        rows.append((
            DATASET_ID,
            row["cell_id"],
            int(row["cycle_index"]),
            clean_value(row.get("ambient_temperature")),
            clean_value(row.get("matlab_start_time")),
            clean_value(row.get("sense_current")),
            clean_value(row.get("battery_current")),
            clean_value(row.get("current_ratio")),
            clean_value(row.get("battery_impedance")),
            clean_value(row.get("rectified_impedance")),
            clean_value(row.get("re")),
            clean_value(row.get("rct")),
            json.dumps(metadata),
        ))

    with conn.cursor() as cur:
        cur.executemany(sql_insert, rows)

    print(f"Uploaded battery_impedance: {len(rows)} rows")


def main():
    database_url = load_database_url()

    conn = psycopg2.connect(database_url)
    conn.autocommit = False

    try:
        ensure_dataset(conn)
        upload_battery_units(conn)
        upload_battery_cycles(conn)

        conn.commit()
        print("Dataset, units and cycles committed.")

        upload_battery_measurements(conn)
        conn.commit()
        print("Measurements committed.")

        upload_battery_impedance(conn)
        conn.commit()
        print("Impedance committed.")

    except Exception as e:
        conn.rollback()
        print("Upload failed. Rolled back.")
        print(e)
        raise

    finally:
        conn.close()

    print("NASA CSV upload completed successfully.")


if __name__ == "__main__":
    main()