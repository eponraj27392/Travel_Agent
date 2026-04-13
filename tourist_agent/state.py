from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage


class BookingDraft(TypedDict, total=False):
    package_id: str
    lead_name: str
    email: str
    phone: str
    travel_date: str
    pax_count: int
    special_requirements: str


class AgentState(TypedDict):
    # Conversation messages (auto-appended)
    messages: Annotated[list[AnyMessage], add_messages]

    # Classified intent: | itinerary planner | booking | cancelation | smalltalk | FAQ's
    intent: Optional[str]

    # Package currently being discussed
    active_package_id: Optional[str]

    # Itinerary day the user is asking about (1-8)
    active_day: Optional[int]

    # Partial booking being assembled
    booking_draft: Optional[BookingDraft]

    # Fields still missing before booking can proceed
    missing_fields: list[str]

    # Whether we are waiting for user's explicit YES/NO to confirm booking
    awaiting_confirmation: bool

    # Booking ID after successful booking
    booking_id: Optional[str]

    # Set by sensitive_guard node: True = user confirmed, False = user cancelled
    sensitive_confirmed: Optional[bool]

    # Itinerary planner subagent state (shared via key name with PlannerState)
    planner_destination: Optional[str]
    planner_travel_type: Optional[str]
    planner_duration: Optional[str]
    planner_pax: Optional[int]
    planner_month: Optional[str]
    planner_min_age: Optional[int]
    planner_max_age: Optional[int]
    planner_itinerary_type: Optional[str]
    planner_matched_packages: Optional[list]
    planner_selected_package_id: Optional[str]
    planner_interests: Optional[str]
    planner_fitness: Optional[str]
    planner_special_req: Optional[str]
