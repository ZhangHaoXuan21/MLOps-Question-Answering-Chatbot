def supervisor_route(state):
    supervisor_route_choice = state['supervisor_route_choice']

    if supervisor_route_choice == "vector_store_agent":
        return "mlops_agent"
    elif supervisor_route_choice == "apology_agent":
        return "apology_agent"