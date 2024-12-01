from mlops_agents.utils.nodes import supervisor_agent, mlops_agent, apology_agent
from mlops_agents.utils.conditional_edges import supervisor_route
from mlops_agents.utils.state import GraphState
from IPython.display import Image, display
from langgraph.graph import END, StateGraph


class MlOpsAgent:
    def __init__(self):
        # Initialise the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("supervisor_agent", supervisor_agent)
        workflow.add_node("mlops_agent", mlops_agent)
        workflow.add_node("apology_agent", apology_agent)


        # Add Edges
        workflow.set_entry_point("supervisor_agent")
        workflow.add_conditional_edges(
            source="supervisor_agent",
            path=supervisor_route,
            path_map={
                "mlops_agent": "mlops_agent",
                "apology_agent": "apology_agent",
            }
        )
        workflow.add_edge("mlops_agent", END)
        workflow.add_edge("apology_agent", END)

        self._mlops_graph = workflow.compile()

    def __call__(self, state):
        result_data = self._mlops_graph.invoke(state)
        return result_data
    