"""
LangGraph nodes for the tourist chatbot agent.
"""
import re
import json
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.types import interrupt
from tourist_agent.state import AgentState
from tourist_agent.tools import ALL_TOOLS
from tourist_agent.booking_db import create_booking
from tourist_agent.data_loader import get_package

# --------------------------------------------------------------------------- #
# LLM setup – Ollama local model with tool binding                            #
# --------------------------------------------------------------------------- #

OLLAMA_MODEL = "qwen2.5:3b"   # change to "llama3.2:3b" if preferred

def _get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.3,
    ).bind_tools(ALL_TOOLS)


SYSTEM_PROMPT = """You are TravelBot, a friendly assistant for an Indian adventure travel company.
You help users explore tour packages, plan their day-by-day itinerary, and book trips.

Available packages are in Ladakh / Himachal Pradesh. When the user asks about packages,
use the provided tools to fetch real data — never make up details.

Booking rules:
- To book a trip you MUST collect: package_id, lead_name, email, phone, travel_date (DD-MM-YYYY), pax_count.
- Ask for missing fields ONE AT A TIME, in a natural conversational way.
- Before creating a booking, summarise all details and ask the user to confirm with "yes".
- After confirmation, output a JSON block exactly like this (no extra text on same line):
  BOOKING_CONFIRM:{"package_id":"...","lead_name":"...","email":"...","phone":"...","travel_date":"...","pax_count":2}

Itinerary planner:
- When users ask "what do I do on day 3?" use tool_get_day_itinerary.
- When users ask for the full plan use tool_get_full_itinerary.
- Suggest activities, highlight altitudes, and give practical tips.

Keep responses concise and friendly. Use markdown lists where helpful."""


# --------------------------------------------------------------------------- #
# Node: chatbot (main LLM node that can call tools)                           #
# --------------------------------------------------------------------------- #

def node_chatbot(state: AgentState) -> dict:
    llm = _get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Node: tool executor (runs any tool the LLM requested)                       #
# --------------------------------------------------------------------------- #

from langgraph.prebuilt import ToolNode

node_tools = ToolNode(ALL_TOOLS)


# --------------------------------------------------------------------------- #
# Node: booking extractor                                                      #
# Watches for BOOKING_CONFIRM marker in the last AI message and               #
# writes the booking to the DB.                                                #
# --------------------------------------------------------------------------- #

def node_booking_extractor(state: AgentState) -> dict:
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else ""

    match = re.search(r"BOOKING_CONFIRM:(\{.*?\})", content, re.DOTALL)
    if not match:
        return {}

    try:
        draft = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"messages": [AIMessage(content="Sorry, I had trouble processing the booking details. Could you try again?")]}

    pkg = get_package(draft["package_id"])
    pkg_title = pkg["title"] if pkg else draft["package_id"]

    # ------------------------------------------------------------------ #
    # HUMAN-IN-THE-LOOP: interrupt and show booking summary               #
    # Graph pauses here. Resumes when user replies via Command(resume=...) #
    # ------------------------------------------------------------------ #
    summary = (
        f"**Please confirm your booking:**\n\n"
        f"- **Package:** {pkg_title}\n"
        f"- **Traveller:** {draft['lead_name']}\n"
        f"- **Email:** {draft['email']}\n"
        f"- **Phone:** {draft['phone']}\n"
        f"- **Travel Date:** {draft['travel_date']}\n"
        f"- **No. of Travellers:** {draft['pax_count']}\n\n"
        f"Type **yes** to confirm, **no** to cancel, or **why** to know more."
    )
    user_decision = interrupt(summary)

    # Resume point — user_decision is whatever the user typed
    answer = user_decision.strip().lower()

    if answer in ("yes", "y", "confirm"):
        booking_id = create_booking(draft)
        return {
            "booking_id": booking_id,
            "messages": [AIMessage(content=(
                f"Your booking is confirmed!\n\n"
                f"**Booking ID:** `{booking_id}`\n"
                f"**Package:** {pkg_title}\n"
                f"**Traveller:** {draft['lead_name']}\n"
                f"**Travel Date:** {draft['travel_date']}\n"
                f"**Travellers:** {draft['pax_count']}\n"
                f"**Email:** {draft['email']}\n\n"
                f"A confirmation will be sent to your email. Have an amazing trip!"
            ))]
        }

    elif answer in ("no", "n", "cancel"):
        return {
            "messages": [AIMessage(content=(
                "Booking cancelled. No worries! Feel free to ask more questions "
                "or explore other packages anytime."
            ))]
        }

    else:
        # "why" or anything else — explain and note they can restart booking
        return {
            "messages": [AIMessage(content=(
                f"Here's what you're booking:\n\n"
                f"**{pkg_title}** is an 8-day high-altitude road trip from Manali to Leh "
                f"covering Khardung La, Nubra Valley, and Pangong Lake.\n\n"
                f"Your details: {draft['pax_count']} traveller(s) on {draft['travel_date']}.\n\n"
                f"To confirm, just say **yes**. To cancel, say **no**. "
                f"Or ask me anything about the itinerary!"
            ))]
        }


# --------------------------------------------------------------------------- #
# Routing: decide next node after chatbot responds                             #
# --------------------------------------------------------------------------- #

def route_after_chatbot(state: AgentState) -> str:
    last = state["messages"][-1]

    # If LLM requested tool calls → go to tool node
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # If LLM output a booking confirmation marker → extract booking
    content = last.content if hasattr(last, "content") else ""
    if "BOOKING_CONFIRM:" in content:
        return "booking_extractor"

    return "__end__"
