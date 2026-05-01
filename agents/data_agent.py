import re

from core.data_service import (
    get_datasets,
    get_units,
    get_cycles,
    get_measurements,
    get_aging_data,
    get_impedance_data,
)


def extract_cycle(prompt: str, default_cycle: int):
    match = re.search(r"cycle\s*(\d+)", prompt.lower())
    if match:
        return int(match.group(1))
    return default_cycle


def extract_signal(prompt: str, default_signal: str | None):
    p = prompt.lower()

    if "voltage" in p:
        return "voltage"
    if "current" in p:
        return "current"
    if "temperature" in p or "temp" in p:
        return "temperature"
    if "charger current" in p:
        return "current_aux"
    if "charger voltage" in p:
        return "voltage_aux"

    return default_signal


def run_data_copilot(prompt: str, context: dict):
    """
    Rule-based Data Agent skeleton.
    Later this can be replaced by an LLM router.
    """

    p = prompt.lower()

    dataset_id = context["dataset_id"]
    unit_id = context["unit_id"]
    default_cycle = context["cycle_index"]
    default_signal = context.get("signal_name")

    # Dataset overview
    if "dataset" in p or "overview" in p or "summary" in p:
        datasets = get_datasets()
        units = get_units(dataset_id)
        cycles = get_cycles(dataset_id, unit_id)

        return {
            "type": "summary",
            "title": "Dataset Summary",
            "message": f"Dataset `{dataset_id}` contains {len(units)} battery units. Selected unit `{unit_id}` has {len(cycles)} cycles.",
            "table": datasets[datasets["dataset_id"] == dataset_id],
        }

    # Aging / capacity / SOH / RUL
    if "capacity" in p or "soh" in p or "rul" in p or "aging" in p or "degradation" in p:
        df = get_aging_data(dataset_id, unit_id)

        return {
            "type": "aging",
            "title": "Aging Analysis",
            "message": f"Showing discharge-cycle aging data for `{unit_id}`.",
            "data": df,
        }

    # Impedance
    if "impedance" in p or "rct" in p or "re" in p:
        df = get_impedance_data(dataset_id, unit_id)

        return {
            "type": "impedance",
            "title": "Impedance Analysis",
            "message": f"Showing impedance data for `{unit_id}`.",
            "data": df,
        }

    # Cycle signal curve
    if "curve" in p or "plot" in p or "show" in p or "draw" in p:
        cycle_index = extract_cycle(prompt, default_cycle)
        signal_name = extract_signal(prompt, default_signal)

        if signal_name is None:
            return {
                "type": "message",
                "title": "Missing Signal",
                "message": "Please specify a signal, for example: voltage, current or temperature.",
            }

        df = get_measurements(
            dataset_id=dataset_id,
            unit_id=unit_id,
            cycle_index=cycle_index,
            signal_name=signal_name,
        )

        return {
            "type": "cycle_curve",
            "title": f"{signal_name} curve",
            "message": f"Showing `{signal_name}` for `{unit_id}`, cycle `{cycle_index}`.",
            "data": df,
            "cycle_index": cycle_index,
            "signal_name": signal_name,
        }

    return {
        "type": "message",
        "title": "Data Copilot",
        "message": (
            "I can help with questions like: "
            "`show voltage curve for cycle 5`, "
            "`show capacity degradation`, "
            "`show SOH trend`, "
            "`show impedance data`, "
            "`summarise this dataset`."
        ),
    }

def handle_data_request(user_input: str):
    """
    Compatibility wrapper for the old router_agent.py interface.
    This keeps the sidebar Copilot working while Data System uses run_data_copilot().
    """
    return (
        "Data Agent is available inside the Data System module. "
        "Please open Data System and ask questions such as: "
        "'show voltage curve for cycle 5' or 'show capacity degradation'."
    )