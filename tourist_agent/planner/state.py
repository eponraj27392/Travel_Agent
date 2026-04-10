from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage


class PlannerState(TypedDict):
    # Shared with parent AgentState via key name match
    messages: Annotated[list[AnyMessage], add_messages]

    # Collected travel preferences
    planner_travel_type: Optional[str]
    planner_duration: Optional[str]
    planner_destination: Optional[str]
    planner_matched_packages: Optional[list]
    planner_selected_package_id: Optional[str]

    # Personal preferences
    planner_interests: Optional[str]
    planner_fitness: Optional[str]
    planner_special_req: Optional[str]
