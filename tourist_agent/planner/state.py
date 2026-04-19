from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage


class PlannerState(TypedDict):
    # Shared with parent AgentState via key name match
    messages: Annotated[list[AnyMessage], add_messages]

    # Core travel preferences
    planner_destination: Optional[str]       # Ladakh / Himachal / Kashmir
    planner_travel_type: Optional[str]       # Bike / Car / Trek
    planner_duration: Optional[str]          # 3-5 / 5-8 / 9+
    planner_pax: Optional[int]               # 2 / 4 / 6 / 8 / 10 / 12
    planner_month: Optional[str]             # January … December
    planner_min_age: Optional[int]           # youngest traveller
    planner_max_age: Optional[int]           # oldest traveller
    planner_itinerary_type: Optional[str]    # Existing / Custom

    # Package selection
    planner_matched_packages: Optional[list]
    planner_selected_package_id: Optional[str]

    # Personal preferences
    planner_interests: Optional[str]
    planner_fitness: Optional[str]
    planner_special_req: Optional[str]
