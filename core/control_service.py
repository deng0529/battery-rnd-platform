def run_demo_control_optimisation(objective: str, method: str) -> dict:
    """Placeholder control optimisation service."""
    return {
        "status": "success",
        "objective": objective,
        "method": method,
        "recommendation": {
            "charging_current_limit": "0.8C",
            "temperature_limit": "35°C",
            "expected_effect": "Reduced degradation risk in demo scenario",
        },
        "message": "Demo control recommendation generated. Real optimisation will be added later.",
    }
