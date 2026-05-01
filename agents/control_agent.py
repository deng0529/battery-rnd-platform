from core.control_service import run_demo_control_optimisation


def handle_control_request(user_prompt: str) -> str:
    result = run_demo_control_optimisation(objective="Reduce degradation", method="Rule-based")
    rec = result["recommendation"]
    return (
        "Control Agent handled your request. "
        f"Suggested charging current limit: {rec['charging_current_limit']}; "
        f"temperature limit: {rec['temperature_limit']}."
    )
