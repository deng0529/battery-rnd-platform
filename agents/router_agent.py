def route_user_request(user_input: str, current_module: str):
    """
    Route global sidebar Copilot requests to the correct module agent.

    current_module comes from st.session_state.page:
    - home
    - data
    - modelling
    - control
    """

    if current_module == "data":
        return {
            "target_module": "data",
            "needs_context": True,
            "user_input": user_input,
        }

    if current_module == "modelling":
        return {
            "type": "message",
            "title": "Modelling Copilot",
            "message": "Modelling Copilot is not implemented yet.",
        }

    if current_module == "control":
        return {
            "type": "message",
            "title": "Control Copilot",
            "message": "Control Copilot is not implemented yet.",
        }

    return {
        "type": "message",
        "title": "Copilot",
        "message": "Please open a module first, for example Data System.",
    }