from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATTERY_META = {
    "B0005": {"battery_no": 5, "cutoff_voltage": 2.7, "rated_capacity": 2.0},
    "B0006": {"battery_no": 6, "cutoff_voltage": 2.5, "rated_capacity": 2.0},
    "B0007": {"battery_no": 7, "cutoff_voltage": 2.2, "rated_capacity": 2.0},
    "B0018": {"battery_no": 18, "cutoff_voltage": 2.5, "rated_capacity": 2.0},
}


def to_array(x):
    if isinstance(x, np.ndarray):
        return x.flatten()
    return np.array([x])


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def convert_one_mat(mat_path: Path):
    cell_id = mat_path.stem
    print(f"Processing {cell_id}...")

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    battery = mat[cell_id]
    cycles = battery.cycle

    meta = BATTERY_META.get(cell_id, {
        "battery_no": None,
        "cutoff_voltage": None,
        "rated_capacity": 2.0,
    })

    cells_rows = [{
        "cell_id": cell_id,
        "battery_no": meta["battery_no"],
        "dataset_name": "NASA Battery Aging Dataset",
        "source_file": mat_path.name,
        "rated_capacity": meta["rated_capacity"],
        "eol_capacity": meta["rated_capacity"] * 0.7,
        "cutoff_voltage": meta["cutoff_voltage"],
    }]

    cycle_rows = []
    timeseries_rows = []
    impedance_rows = []

    for cycle_index, cycle in enumerate(cycles, start=1):
        cycle_type = str(cycle.type)
        data = cycle.data

        matlab_start_time = ",".join(map(str, to_array(cycle.time)))

        capacity = None
        if cycle_type == "discharge" and hasattr(data, "Capacity"):
            capacity = safe_float(to_array(data.Capacity)[0])

        cycle_rows.append({
            "cell_id": cell_id,
            "cycle_index": cycle_index,
            "cycle_type": cycle_type,
            "ambient_temperature": safe_float(cycle.ambient_temperature),
            "matlab_start_time": matlab_start_time,
            "capacity": capacity,
            "soh": None,
            "rul": None,
        })

        if cycle_type in ["charge", "discharge"]:
            fields = {
                "time_seconds": "Time",
                "voltage_measured": "Voltage_measured",
                "current_measured": "Current_measured",
                "temperature_measured": "Temperature_measured",
                "current_charge": "Current_charge",
                "voltage_charge": "Voltage_charge",
            }

            arrays = {}
            max_len = 0

            for out_col, mat_col in fields.items():
                if hasattr(data, mat_col):
                    arr = to_array(getattr(data, mat_col))
                    arrays[out_col] = arr
                    max_len = max(max_len, len(arr))

            for point_index in range(max_len):
                row = {
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "cycle_type": cycle_type,
                    "point_index": point_index,
                }

                for col, arr in arrays.items():
                    row[col] = safe_float(arr[point_index]) if point_index < len(arr) else None

                timeseries_rows.append(row)

        elif cycle_type == "impedance":
            row = {
                "cell_id": cell_id,
                "cycle_index": cycle_index,
                "ambient_temperature": safe_float(cycle.ambient_temperature),
                "matlab_start_time": matlab_start_time,
            }

            impedance_fields = {
                "sense_current": "Sense_current",
                "battery_current": "Battery_current",
                "current_ratio": "Current_ratio",
                "battery_impedance": "Battery_impedance",
                "rectified_impedance": "Rectified_impedance",
                "re": "Re",
                "rct": "Rct",
            }

            for out_col, mat_col in impedance_fields.items():
                if hasattr(data, mat_col):
                    arr = to_array(getattr(data, mat_col))
                    row[out_col] = (
                        safe_float(arr[0])
                        if len(arr) == 1
                        else ",".join(map(str, arr.tolist()))
                    )
                else:
                    row[out_col] = None

            impedance_rows.append(row)

    cycles_df = pd.DataFrame(cycle_rows)

    discharge_mask = cycles_df["cycle_type"] == "discharge"
    discharge_indices = cycles_df[discharge_mask].index.tolist()

    if discharge_indices:
        first_capacity = cycles_df.loc[discharge_indices[0], "capacity"]
        total_discharge = len(discharge_indices)

        for n, row_index in enumerate(discharge_indices):
            cap = cycles_df.loc[row_index, "capacity"]
            if pd.notna(cap) and first_capacity:
                cycles_df.loc[row_index, "soh"] = cap / first_capacity
            cycles_df.loc[row_index, "rul"] = total_discharge - n - 1

        cells_rows[0]["initial_capacity"] = first_capacity

    return (
        pd.DataFrame(cells_rows),
        cycles_df,
        pd.DataFrame(timeseries_rows),
        pd.DataFrame(impedance_rows),
    )


def main():

    mat_files = sorted({p.resolve() for p in RAW_DIR.glob("*") if p.suffix.lower() == ".mat"})

    print("RAW_DIR:", RAW_DIR)
    print("MAT files:", [f.name for f in mat_files])

    if not mat_files:
        print("No .mat files found.")
        return

    all_cells = []
    all_cycles = []
    all_timeseries = []
    all_impedance = []

    for mat_file in mat_files:
        cells_df, cycles_df, ts_df, imp_df = convert_one_mat(mat_file)
        all_cells.append(cells_df)
        all_cycles.append(cycles_df)
        all_timeseries.append(ts_df)
        all_impedance.append(imp_df)

    pd.concat(all_cells, ignore_index=True).to_csv(
        OUT_DIR / "battery_cells.csv", index=False
    )
    pd.concat(all_cycles, ignore_index=True).to_csv(
        OUT_DIR / "battery_cycles.csv", index=False
    )
    pd.concat(all_timeseries, ignore_index=True).to_csv(
        OUT_DIR / "battery_timeseries.csv", index=False
    )

    if all_impedance:
        pd.concat(all_impedance, ignore_index=True).to_csv(
            OUT_DIR / "battery_impedance.csv", index=False
        )

    print("Conversion finished.")
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()