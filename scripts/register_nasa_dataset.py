from pathlib import Path
import json
import toml
import psycopg2

from data_adapters.nasa_adapter import DATASET_INFO, build_dataset_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def load_database_url() -> str:
    secrets = toml.load(SECRETS_PATH)
    return secrets["SUPABASE_DATABASE_URL"]


def register_dataset(conn):
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


def main():
    database_url = load_database_url()

    conn = psycopg2.connect(database_url)
    conn.autocommit = True

    register_dataset(conn)

    conn.close()

    print("NASA dataset registered or updated successfully.")
    print(f"dataset_id: {DATASET_INFO['dataset_id']}")
    print("Check Supabase table: datasets")


if __name__ == "__main__":
    main()