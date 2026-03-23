from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


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
    messages: Annotated[list, add_messages]

    # Classified intent: browse | itinerary | book | confirm | cancel | smalltalk
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
