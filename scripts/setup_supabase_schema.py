from pathlib import Path
import sys
import toml
import psycopg2


# ============================================================
# 1. 找到项目根目录，并读取 .streamlit/secrets.toml
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"

if not SECRETS_PATH.exists():
    print(f"ERROR: Cannot find secrets file: {SECRETS_PATH}")
    sys.exit(1)

secrets = toml.load(SECRETS_PATH)

DATABASE_URL = secrets.get("SUPABASE_DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: SUPABASE_DATABASE_URL is missing in secrets.toml")
    sys.exit(1)


# ============================================================
# 2. 数据库建表 SQL
# ============================================================

SCHEMA_SQL = """
drop table if exists battery_measurements;
drop table if exists battery_impedance;
drop table if exists battery_cycles;
drop table if exists battery_units;
drop table if exists datasets;

create table datasets (
    dataset_id text primary key,
    dataset_name text not null,
    source_name text,
    source_type text,
    description text,
    raw_format text,
    created_at timestamp default now(),
    metadata jsonb
);

create table battery_units (
    dataset_id text not null references datasets(dataset_id),
    unit_id text not null,
    unit_type text not null,
    source_unit_id text,
    source_unit_type text,
    display_name text,
    rated_capacity double precision,
    eol_capacity double precision,
    cutoff_voltage double precision,
    initial_capacity double precision,
    metadata jsonb,
    primary key (dataset_id, unit_id)
);

create table battery_cycles (
    dataset_id text not null,
    unit_id text not null,
    cycle_index integer not null,
    cycle_type text,
    source_cycle_type text,
    ambient_temperature double precision,
    matlab_start_time text,
    capacity double precision,
    soh double precision,
    rul integer,
    metadata jsonb,
    primary key (dataset_id, unit_id, cycle_index),
    foreign key (dataset_id, unit_id)
        references battery_units(dataset_id, unit_id)
);

create table battery_measurements (
    id bigserial primary key,
    dataset_id text not null,
    unit_id text not null,
    cycle_index integer not null,
    point_index integer not null,
    time_seconds double precision,
    signal_name text not null,
    signal_value double precision,
    signal_unit text,
    source_signal_name text,
    metadata jsonb,
    foreign key (dataset_id, unit_id, cycle_index)
        references battery_cycles(dataset_id, unit_id, cycle_index)
);

create table battery_impedance (
    id bigserial primary key,
    dataset_id text not null,
    unit_id text not null,
    cycle_index integer not null,
    ambient_temperature double precision,
    matlab_start_time text,
    sense_current text,
    battery_current text,
    current_ratio text,
    battery_impedance text,
    rectified_impedance text,
    re double precision,
    rct double precision,
    metadata jsonb,
    foreign key (dataset_id, unit_id, cycle_index)
        references battery_cycles(dataset_id, unit_id, cycle_index)
);

create index idx_cycles_dataset_unit
on battery_cycles(dataset_id, unit_id);

create index idx_measurements_lookup
on battery_measurements(dataset_id, unit_id, cycle_index, signal_name);

create index idx_measurements_signal
on battery_measurements(signal_name);
"""


def get_connection():
    """
    这里就是连接 Supabase PostgreSQL 数据库的地方。

    DATABASE_URL 来自：
    .streamlit/secrets.toml

    格式类似：
    postgresql://postgres.xxxx:你的密码@xxxx.pooler.supabase.com:6543/postgres
    """
    return psycopg2.connect(DATABASE_URL)


def test_connection():
    """
    测试数据库是否能成功连接。
    """
    print("Testing database connection...")

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("select current_database(), current_user;")
            db_name, user_name = cur.fetchone()

        conn.close()

        print("Connection successful.")
        print(f"Database: {db_name}")
        print(f"User: {user_name}")
        return True

    except Exception as e:
        print("Connection failed.")
        print(e)
        return False


def create_schema():
    """
    执行建表 SQL。
    """
    print("Creating database schema...")

    conn = get_connection()
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)

        print("Schema created successfully.")

    except Exception as e:
        print("Schema creation failed.")
        print(e)
        raise

    finally:
        conn.close()


def main():
    print("Project root:", BASE_DIR)
    print("Secrets file:", SECRETS_PATH)

    if not test_connection():
        sys.exit(1)

    create_schema()


if __name__ == "__main__":
    main()