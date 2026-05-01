from core.model_service import run_demo_prediction


def handle_modelling_request(user_prompt: str) -> str:
    result = run_demo_prediction(model_type="Demo Linear Model", train_ratio=0.7)
    return (
        "Modelling Agent handled your request. "
        f"Demo model finished with RMSE = {result['demo_metrics']['RMSE']} and R2 = {result['demo_metrics']['R2']}."
    )
