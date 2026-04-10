"""
Itinerary Planner Subagent — nodes.

Flow:
  node_collect_travel_prefs   → Q&A: travel type, duration, destination, package selection
  node_collect_personal_prefs → Q&A: interests, fitness level, special requirements
  node_generate_itinerary     → LLM builds a personalized day-by-day plan
"""
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from tourist_agent.planner.state import PlannerState
from tourist_agent.data_loader import get_package

DATA_DIR = Path(__file__).parent.parent.parent / "data"
JOURNEY_FILE = DATA_DIR / "journey.json"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_journeys() -> list:
    with open(JOURNEY_FILE) as f:
        return json.load(f)["journeys"]


def _find_packages(travel_type: str, duration: str, destination: str) -> list:
    for j in _load_journeys():
        if (
            j["travel_type"].lower() == travel_type.lower()
            and j["duration"].lower() == duration.lower()
            and j["destination"].lower() == destination.lower()
        ):
            return j["packages"]
    return []


def _normalize_travel_type(answer: str) -> str:
    mapping = {"1": "Car", "2": "Bike", "3": "Flight"}
    a = answer.strip()
    return mapping.get(a, a.title())


def _normalize_duration(answer: str) -> str:
    mapping = {"1": "3-5 days", "2": "7-10 days"}
    a = answer.strip()
    return mapping.get(a, a)


def _normalize_destination(answer: str) -> str:
    mapping = {"1": "Ladakh", "2": "Ladakh & Himachal"}
    a = answer.strip()
    return mapping.get(a, a.title())


def _normalize_interests(answer: str) -> str:
    mapping = {"1": "Adventure", "2": "Culture", "3": "Photography", "4": "All"}
    a = answer.strip()
    return mapping.get(a, a.title())


def _normalize_fitness(answer: str) -> str:
    mapping = {"1": "Beginner", "2": "Experienced"}
    a = answer.strip()
    return mapping.get(a, a.title())


# --------------------------------------------------------------------------- #
# Node 1: Collect travel preferences + package selection                      #
# --------------------------------------------------------------------------- #

def node_collect_travel_prefs(state: PlannerState) -> dict:
    messages = []

    # Close out the tool_start_itinerary_planner call from the parent assistant
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc["name"] == "tool_start_itinerary_planner":
                messages.append(
                    ToolMessage(
                        content="Itinerary planner started.",
                        tool_call_id=tc["id"],
                    )
                )

    # Q1 — Travel type
    travel_type_raw = interrupt(
        "🗺️ **Welcome to the Itinerary Planner!**\n\n"
        "Let's build your perfect trip step by step.\n\n"
        "**Q1. What is your preferred mode of travel?**\n"
        "1. Car\n"
        "2. Bike\n"
        "3. Flight\n\n"
        "_(Type the number or name)_"
    )
    travel_type = _normalize_travel_type(travel_type_raw)

    # Q2 — Duration
    duration_raw = interrupt(
        "**Q2. How many days are you planning for your trip?**\n"
        "1. 3-5 days\n"
        "2. 7-10 days\n\n"
        "_(Type the number or duration)_"
    )
    duration = _normalize_duration(duration_raw)

    # Q3 — Destination
    destination_raw = interrupt(
        "**Q3. Where do you want to go?**\n"
        "1. Ladakh\n"
        "2. Ladakh & Himachal\n\n"
        "_(Type the number or destination)_"
    )
    destination = _normalize_destination(destination_raw)

    # Look up matching packages from journey.json
    packages = _find_packages(travel_type, duration, destination)

    if not packages:
        messages.append(AIMessage(content=(
            f"Sorry, I couldn't find any packages for **{travel_type}** travel, "
            f"**{duration}** in **{destination}**.\n\n"
            "Please try different preferences or ask me to show all available packages."
        )))
        return {"messages": messages}

    # Q4 — Package selection
    pkg_list = "\n".join(
        f"{i + 1}. **{p['title']}** ({p['packageDuration']}) — "
        f"{p.get('starting_price', 'Price on request')} | {p.get('available_months', '')}"
        for i, p in enumerate(packages)
    )
    selection_raw = interrupt(
        f"**Q4. Here are the available packages for your preferences:**\n\n"
        f"{pkg_list}\n\n"
        "Which package would you like to plan for? _(Type the number)_"
    )

    try:
        idx = int(selection_raw.strip()) - 1
        selected = packages[max(0, min(idx, len(packages) - 1))]
    except ValueError:
        selected = packages[0]

    return {
        "messages": messages,
        "planner_travel_type": travel_type,
        "planner_duration": duration,
        "planner_destination": destination,
        "planner_matched_packages": packages,
        "planner_selected_package_id": selected["id"],
    }


# --------------------------------------------------------------------------- #
# Node 2: Collect personal preferences                                         #
# --------------------------------------------------------------------------- #

def node_collect_personal_prefs(state: PlannerState) -> dict:
    # Q5 — Interests
    interests_raw = interrupt(
        "**Q5. What are your main interests for this trip?**\n"
        "1. Adventure (high passes, extreme experiences, treks)\n"
        "2. Culture (monasteries, local villages, food)\n"
        "3. Photography (landscapes, golden hour, unique spots)\n"
        "4. All of the above\n\n"
        "_(Type the number or interest)_"
    )
    interests = _normalize_interests(interests_raw)

    # Q6 — Fitness level
    fitness_raw = interrupt(
        "**Q6. How would you rate your fitness level?**\n"
        "1. Beginner (comfortable with driving & light walking, no extreme treks)\n"
        "2. Experienced (ready for high altitude, long drives, challenging terrain)\n\n"
        "_(Type the number)_"
    )
    fitness = _normalize_fitness(fitness_raw)

    # Q7 — Special requirements
    special_req_raw = interrupt(
        "**Q7. Do you have any special requirements?**\n"
        "_(e.g. vegetarian food, wheelchair access, medical conditions)_\n\n"
        "Type **none** if nothing special."
    )
    special_req = "" if special_req_raw.strip().lower() == "none" else special_req_raw.strip()

    return {
        "planner_interests": interests,
        "planner_fitness": fitness,
        "planner_special_req": special_req,
    }


# --------------------------------------------------------------------------- #
# Node 3: Format itinerary directly from JSON data (no LLM generation)        #
# --------------------------------------------------------------------------- #

# Tags used to highlight activities based on interests
_INTEREST_KEYWORDS = {
    "Adventure":    ["pass", "rafting", "trek", "atv", "camel", "khardung", "chang la", "altitude", "extreme"],
    "Culture":      ["monastery", "gurudwara", "palace", "stupa", "museum", "memorial", "secmol", "village", "langar"],
    "Photography":  ["sunrise", "sunset", "viewpoint", "confluence", "lake", "sand dune", "panoramic", "golden"],
}

def _activity_matches_interests(activity: dict, interests: str) -> bool:
    """Return True if this activity is relevant to the user's interests."""
    if interests == "All":
        return True
    keywords = _INTEREST_KEYWORDS.get(interests, [])
    text = (activity.get("name", "") + " " + activity.get("description", "")).lower()
    return any(kw in text for kw in keywords)


def _format_activity(activity: dict, interests: str, fitness: str) -> str:
    name = activity.get("name", "")
    time = activity.get("time", "")
    duration = activity.get("duration", "")
    description = activity.get("description", "")
    altitude_m = activity.get("altitude_m")
    distance_km = activity.get("distance_km")

    highlight = _activity_matches_interests(activity, interests)
    prefix = "⭐ " if highlight and interests != "All" else ""

    meta_parts = []
    if time:
        meta_parts.append(f"🕐 {time}")
    if duration:
        meta_parts.append(f"⏱ {duration}")
    if altitude_m:
        warning = " ⚠️ High altitude" if altitude_m >= 5000 and fitness == "Beginner" else ""
        meta_parts.append(f"🏔 {altitude_m:,}m{warning}")
    if distance_km:
        meta_parts.append(f"📍 {distance_km} km")

    meta = "  |  ".join(meta_parts)
    lines = [f"**{prefix}{name}**"]
    if meta:
        lines.append(f"_{meta}_")
    if description:
        lines.append(description)
    return "\n".join(lines)


def node_generate_itinerary(state: PlannerState) -> dict:
    package_id = state.get("planner_selected_package_id")
    interests   = state.get("planner_interests", "All")
    fitness     = state.get("planner_fitness", "Beginner")
    special_req = state.get("planner_special_req", "")

    pkg = get_package(package_id)
    if not pkg:
        return {"messages": [AIMessage(content="Sorry, could not load the package data.")]}

    days = pkg.get("detailedItinerary", [])
    sections = []

    for day in days:
        overnight = day.get("overnight", "")
        header = (
            f"## Day {day['day']}: {day['title']}\n"
            f"**Route:** {day.get('route', '')}\n"
            + (f"**Overnight:** {overnight}\n" if overnight else "")
            + f"\n{day.get('description', '')}"
        )
        activity_blocks = [
            _format_activity(act, interests, fitness)
            for act in day.get("activities", [])
        ]
        sections.append(header + "\n\n" + "\n\n---\n\n".join(activity_blocks))

    # Summary header
    summary = (
        f"# 🏔️ Your Itinerary — {pkg['title']}\n"
        f"**{pkg.get('packageDuration','')}**  |  "
        f"**{pkg.get('totalDistance','')}**  |  "
        f"Season: {pkg.get('season','')}\n\n"
        f"_Interests: {interests}  |  Fitness: {fitness}"
        + (f"  |  Special: {special_req}" if special_req else "")
        + "_\n\n"
        + ("⭐ = Activities matching your interests\n\n" if interests != "All" else "")
    )

    # Beginner altitude warning
    if fitness == "Beginner":
        summary += (
            "> ⚠️ **Altitude Note:** You selected Beginner fitness. Activities at 5,000m+ "
            "are marked. Limit exposure at high passes to 45 minutes and descend if you feel unwell.\n\n"
        )

    if special_req:
        summary += f"> 📝 **Special requirement noted:** {special_req}\n\n"

    full_text = summary + "\n\n---\n\n".join(sections)
    return {"messages": [AIMessage(content=full_text)]}
