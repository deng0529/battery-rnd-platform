from pathlib import Path
import pandas as pd


DATASET_INFO = {
    "dataset_id": "nasa_battery_aging",
    "dataset_name": "NASA Li-ion Battery Aging Dataset",
    "source_name": "NASA",
    "source_type": "experimental",
    "description": (
        "Four Li-ion batteries (#5, #6, #7 and #18) were tested under charge, discharge "
        "and impedance operational profiles at room temperature. Charging used CC mode "
        "at 1.5A until 4.2V, then CV mode until current dropped to 20mA. Discharge used "
        "CC mode at 2A until battery-specific cut-off voltages. Impedance was measured "
        "using EIS from 0.1Hz to 5kHz. The experiments stopped at EOL defined as 30% "
        "capacity fade from 2Ah to 1.4Ah. The dataset supports remaining charge and RUL prediction."
    ),
    "raw_format": "mat",
}


BATTERY_META = {
    "B0005": {
        "battery_no": 5,
        "cutoff_voltage": 2.7,
        "rated_capacity_ahr": 2.0,
        "eol_capacity_ahr": 1.4,
        "source_file": "B0005.mat",
    },
    "B0006": {
        "battery_no": 6,
        "cutoff_voltage": 2.5,
        "rated_capacity_ahr": 2.0,
        "eol_capacity_ahr": 1.4,
        "source_file": "B0006.mat",
    },
    "B0007": {
        "battery_no": 7,
        "cutoff_voltage": 2.2,
        "rated_capacity_ahr": 2.0,
        "eol_capacity_ahr": 1.4,
        "source_file": "B0007.mat",
    },
    "B0018": {
        "battery_no": 18,
        "cutoff_voltage": 2.5,
        "rated_capacity_ahr": 2.0,
        "eol_capacity_ahr": 1.4,
        "source_file": "B0018.mat",
    },
}


OPERATION_PROFILES = {
    "charge": {
        "mode": "CC-CV",
        "cc_current_a": 1.5,
        "voltage_limit_v": 4.2,
        "cv_stop_current_a": 0.02,
    },
    "discharge": {
        "mode": "CC",
        "current_a": 2.0,
        "cutoff_voltage_by_battery": {
            "B0005": 2.7,
            "B0006": 2.5,
            "B0007": 2.2,
            "B0018": 2.5,
        },
    },
    "impedance": {
        "method": "Electrochemical Impedance Spectroscopy",
        "frequency_range_hz": [0.1, 5000],
    },
}


SIGNAL_MAPPING = {
    "voltage_measured": {
        "signal_name": "voltage",
        "signal_unit": "V",
        "source_signal_name": "Voltage_measured",
        "description": "Battery terminal voltage",
    },
    "current_measured": {
        "signal_name": "current",
        "signal_unit": "A",
        "source_signal_name": "Current_measured",
        "description": "Battery output current",
    },
    "temperature_measured": {
        "signal_name": "temperature",
        "signal_unit": "degC",
        "source_signal_name": "Temperature_measured",
        "description": "Battery temperature",
    },
    "current_charge": {
        "signal_name": "current_aux",
        "signal_unit": "A",
        "source_signal_name": "Current_charge",
        "description": "Current measured at charger for charge cycles or load for discharge cycles",
    },
    "voltage_charge": {
        "signal_name": "voltage_aux",
        "signal_unit": "V",
        "source_signal_name": "Voltage_charge",
        "description": "Voltage measured at charger for charge cycles or load for discharge cycles",
    },
}


CYCLE_FIELDS = {
    "type": "charge / discharge / impedance",
    "ambient_temperature": "Ambient temperature in degC",
    "time": "MATLAB date vector for cycle start time",
    "data": "Measurement data structure",
}


IMPEDANCE_FIELDS = {
    "Sense_current": "Current in sense branch, Amps",
    "Battery_current": "Current in battery branch, Amps",
    "Current_ratio": "Ratio of sense current and battery current",
    "Battery_impedance": "Battery impedance in Ohms computed from raw data",
    "Rectified_impedance": "Calibrated and smoothed battery impedance in Ohms",
    "Re": "Estimated electrolyte resistance in Ohms",
    "Rct": "Estimated charge transfer resistance in Ohms",
}


def build_dataset_metadata(processed_data_dir: Path) -> dict:
    """
    Build metadata for the datasets.metadata JSONB column.
    This combines the README knowledge with automatically scanned CSV information.
    """

    metadata = {
        "battery_ids": list(BATTERY_META.keys()),
        "battery_meta": BATTERY_META,
        "operation_profiles": OPERATION_PROFILES,
        "cycle_fields": CYCLE_FIELDS,
        "signal_mapping": SIGNAL_MAPPING,
        "impedance_fields": IMPEDANCE_FIELDS,
        "eol_criterion": {
            "definition": "30% fade in rated capacity",
            "rated_capacity_ahr": 2.0,
            "eol_capacity_ahr": 1.4,
        },
        "prediction_tasks": [
            "remaining charge prediction for a given discharge cycle",
            "remaining useful life prediction",
        ],
        "csv_files": [],
        "csv_columns": {},
        "row_counts": {},
    }

    csv_files = sorted(processed_data_dir.glob("*.csv"))
    metadata["csv_files"] = [f.name for f in csv_files]

    for csv_file in csv_files:
        df_head = pd.read_csv(csv_file, nrows=5)
        metadata["csv_columns"][csv_file.stem] = list(df_head.columns)

        try:
            row_count = sum(1 for _ in open(csv_file, "r", encoding="utf-8")) - 1
        except UnicodeDecodeError:
            row_count = sum(1 for _ in open(csv_file, "r", encoding="latin1")) - 1

        metadata["row_counts"][csv_file.stem] = row_count

    return metadata