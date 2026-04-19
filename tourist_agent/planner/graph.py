"""
Itinerary Planner — compiled subgraph.

This subgraph is added as a node in the main tourist agent graph.
It runs a guided Q&A to collect travel preferences, then generates
a personalized itinerary using LLM.

Flow:
  START → collect_travel_prefs → collect_personal_prefs → generate_itinerary → END
"""
from langgraph.graph import StateGraph, START, END
from tourist_agent.planner.state import PlannerState
from tourist_agent.planner.nodes import (
    node_collect_travel_prefs,
    node_collect_personal_prefs,
    node_generate_itinerary,
)

from tourist_agent.utils import save_graph_diagram


def build_planner_graph():
    builder = StateGraph(PlannerState)

    builder.add_node("collect_travel_prefs", node_collect_travel_prefs)
    builder.add_node("collect_personal_prefs", node_collect_personal_prefs)
    builder.add_node("generate_itinerary", node_generate_itinerary)

    builder.add_edge(START, "collect_travel_prefs")
    builder.add_edge("collect_travel_prefs", "collect_personal_prefs")
    builder.add_edge("collect_personal_prefs", "generate_itinerary")
    builder.add_edge("generate_itinerary", END)

    return builder.compile()


planner_graph = build_planner_graph()
# save_graph_diagram(planner_graph, '/home/esakki1/projects/Travel_Agent/output/planner.png')