"""
LangGraph nodes for the tourist assistant agent.

Node layout:
  assistant           — main LLM node (tool-calling)
  safe_tools        — executes safe tools directly (no confirmation needed)
  sensitive_guard   — interrupts for human confirmation before sensitive tools
  sensitive_tools   — executes sensitive tools after confirmation
"""
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.config import get_store, get_config
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

import tourist_agent.memory_store as memory_store

from tourist_agent.state import AgentState
from tourist_agent.tools import ALL_TOOLS, SAFE_TOOLS, SENSITIVE_TOOLS, SENSITIVE_TOOL_NAMES, PLANNER_TOOL_NAMES


# --------------------------------------------------------------------------- #
# LLM setup — Ollama local model with all tools bound                         #
# --------------------------------------------------------------------------- #

OLLAMA_MODEL = "qwen3.5:4b"


def _get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.3,
    ).bind_tools(ALL_TOOLS)


SYSTEM_PROMPT = """You are TravelBot, a friendly assistant for an Indian adventure travel company called "HIMALAYAN ROVER".
You help users explore tour packages, plan their day-by-day itinerary (both custom & existing), and book or cancel trips.

Available packages are in Ladakh / Himachal / Kashmir. When the user asks about packages,
use the provided tools to fetch real data — never make up details.

Booking rules:
- To book a trip you MUST collect: package_id, lead_name, email, phone, travel_date (DD-MM-YYYY), pax_count.
- Ask for missing fields ONE AT A TIME, in a natural conversational way.
- Once you have ALL required fields, call tool_validate_booking_info FIRST.
  - If validation fails, tell the user exactly which fields need correction and ask them to re-provide only those fields.
  - Repeat validation after every correction until tool_validate_booking_info returns "All fields valid."
  - Only then call tool_book_trip.
- Phone numbers MUST include country code (e.g. +91 for India).
- Travel date MUST be in DD-MM-YYYY format and must be a future date.

Cancellation rules:
- To cancel a trip, ask the user for their booking_id (format: TRV-YYYYMMDD-XXXX).
- Once you have the booking_id, call tool_cancel_trip.
- The system will ask the user to confirm before the cancellation is processed.

Itinerary planner:
- When the user wants to plan a custom trip or says "plan my itinerary / help me plan a trip / Itinerary planner",
  call tool_start_itinerary_planner — this launches a guided planner that collects all details.

Itinerary questions:
- When users ask "what do I do on day 3?" use tool_get_day_itinerary.
- When users ask for the full plan use tool_get_full_itinerary.
- Suggest activities, highlight altitudes, and give practical tips.

Keep responses concise and friendly. Use markdown lists where helpful.

If the user asks anything unrelated to travel, politely decline and
redirect them. Example: "I'm a travel assistant — I can only help
with travel-related questions. How can I help with your travel ?
"""


# --------------------------------------------------------------------------- #
# Node: assistant                                                                #
# --------------------------------------------------------------------------- #

async def node_assistant(state: AgentState) -> dict:
    config  = get_config()
    user_id = config.get("configurable", {}).get("session_id", "anonymous")
    store   = get_store()

    # ── Read long-term memory ─────────────────────────────────────────────────
    memory_lines = []
    if store is not None:
        profile = await memory_store.get_user_profile(store, user_id)
        prefs   = await memory_store.get_travel_preferences(store, user_id)

        if profile:
            memory_lines.append("## Returning User")
            for field, label in [("lead_name", "Name"), ("email", "Email"), ("phone", "Phone")]:
                if profile.get(field):
                    memory_lines.append(f"- {label}: {profile[field]}")

        if prefs:
            memory_lines.append("## Past Travel Preferences")
            for field, label in [
                ("destination",  "Preferred Destination"),
                ("travel_type",  "Travel Type"),
                ("typical_pax",  "Typical Group Size"),
                ("month",        "Preferred Month"),
                ("interests",    "Interests"),
                ("fitness",      "Fitness Level"),
            ]:
                if prefs.get(field):
                    memory_lines.append(f"- {label}: {prefs[field]}")

    # ── Inject memory into system prompt ─────────────────────────────────────
    if memory_lines:
        system_content = (
            SYSTEM_PROMPT
            + "\n\n---\n"
            + "\n".join(memory_lines)
            + "\n\nUse the above to personalise your responses. "
              "Do not re-ask for details you already know (name, email, phone). "
              "When booking, pre-fill known fields and confirm them with the user."
        )
    else:
        system_content = SYSTEM_PROMPT

    llm = _get_llm()
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)

    # ── Write: save user profile after a confirmed booking ───────────────────
    if store is not None:
        all_msgs = state["messages"] + [response]
        for msg in reversed(all_msgs[-4:]):
            if (
                hasattr(msg, "tool_call_id")
                and msg.content
                and msg.content.startswith("Booking confirmed!")
            ):
                # Find the matching AIMessage tool_call to extract args
                for ai_msg in reversed(all_msgs):
                    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                        for tc in ai_msg.tool_calls:
                            if tc["name"] == "tool_book_trip":
                                args = tc["args"]
                                await memory_store.upsert_user_profile(store, user_id, {
                                    "lead_name": args.get("lead_name", ""),
                                    "email":     args.get("email", ""),
                                    "phone":     args.get("phone", ""),
                                })
                                from tourist_agent.data_loader import get_package
                                pkg        = get_package(args.get("package_id", ""))
                                booking_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                                await memory_store.save_booking_summary(
                                    store, user_id,
                                    f"booking-{booking_ts}",
                                    {
                                        "package_id":    args.get("package_id", ""),
                                        "package_title": pkg["title"] if pkg else args.get("package_id", ""),
                                        "travel_date":   args.get("travel_date", ""),
                                        "pax_count":     args.get("pax_count", 0),
                                        "status":        "CONFIRMED",
                                    },
                                )
                        break
                break

    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Node: safe_tools — runs safe tools without any confirmation                  #
# --------------------------------------------------------------------------- #

node_safe_tools = ToolNode(SAFE_TOOLS)


# --------------------------------------------------------------------------- #
# Node: sensitive_guard — interrupts for human confirmation                   #
#                                                                              #
# If user confirms  → sets sensitive_confirmed=True                           #
#                     graph proceeds to sensitive_tools                        #
# If user cancels   → injects ToolMessages saying "cancelled"                 #
#                     graph goes back to assistant to inform the user            #
# --------------------------------------------------------------------------- #

def node_sensitive_guard(state: AgentState) -> dict:
    last = state["messages"][-1]

    # Find the first sensitive tool call
    sensitive_tc = next(
        (tc for tc in last.tool_calls if tc["name"] in SENSITIVE_TOOL_NAMES),
        None,
    )
    if not sensitive_tc:
        return {"sensitive_confirmed": False}

    # Build a human-readable confirmation summary
    tool_name = sensitive_tc["name"]
    args = sensitive_tc["args"]
    lines = [f"**Confirm action: `{tool_name}`**\n"]
    for k, v in args.items():
        lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    lines.append("\nType **yes** to confirm or **no** to cancel.")
    summary = "\n".join(lines)

    # Pause graph — resumes when user replies
    decision = interrupt(summary)
    confirmed = decision.strip().lower() in ("yes", "y", "confirm")

    if not confirmed:
        # Inject a ToolMessage for every pending tool call so LangGraph's
        # message integrity requirement is satisfied (every tool_call needs a response)
        cancel_msgs = [
            ToolMessage(
                content="Action cancelled by user.",
                tool_call_id=tc["id"],
            )
            for tc in last.tool_calls
        ]
        return {"sensitive_confirmed": False, "messages": cancel_msgs}

    return {"sensitive_confirmed": True}


# --------------------------------------------------------------------------- #
# Node: sensitive_tools — executes sensitive tools after confirmation          #
# --------------------------------------------------------------------------- #

node_sensitive_tools = ToolNode(SENSITIVE_TOOLS)


# --------------------------------------------------------------------------- #
# Routing: after assistant                                                       #
# --------------------------------------------------------------------------- #

def route_after_assistant(state: AgentState) -> str:
    last = state["messages"][-1]
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return "__end__"

    tool_names = {tc["name"] for tc in last.tool_calls}
    if tool_names & PLANNER_TOOL_NAMES:
        return "itinerary_planner"
    if tool_names & SENSITIVE_TOOL_NAMES:
        return "sensitive_guard"
    return "safe_tools"


# --------------------------------------------------------------------------- #
# Routing: after sensitive_guard                                               #
# --------------------------------------------------------------------------- #

def route_after_guard(state: AgentState) -> str:
    if state.get("sensitive_confirmed"):
        return "sensitive_tools"
    return "assistant"
