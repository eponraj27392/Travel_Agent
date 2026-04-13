"""
Itinerary Planner Subagent — nodes.

Flow:
  node_collect_travel_prefs   → 8 guided questions (destination → package selection)
  node_collect_personal_prefs → interests, fitness level, special requirements
  node_generate_itinerary     → formats day-by-day plan from JSON data
"""
import re
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_store, get_config
from langgraph.types import interrupt

import tourist_agent.memory_store as memory_store
from tourist_agent.planner.state import PlannerState
from tourist_agent.data_loader import find_packages, get_package
from tourist_agent.tracing import log_planner_step

# ── Season / availability knowledge ──────────────────────────────────────────

_SEASON_WARNINGS = {
    "ladakh": {
        "closed":    [1, 2, 3, 11, 12],   # Nov–Mar: Khardung La, Chang La closed
        "shoulder":  [4, 10],              # Apr, Oct: unpredictable
        "peak":      [5, 6, 7, 8, 9],     # May–Sep: best season
    },
    "himachal": {
        "closed":    [1, 2],               # Jan–Feb: Rohtang closed, heavy snow
        "shoulder":  [3, 11, 12],
        "peak":      [4, 5, 6, 7, 8, 9, 10],
    },
    "kashmir": {
        "closed":    [],                   # accessible year-round
        "shoulder":  [12, 1, 2],           # cold, some routes restricted
        "peak":      [3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
}

_MONTH_MAP = {
    "jan": 1, "january": 1, "1": 1,
    "feb": 2, "february": 2, "2": 2,
    "mar": 3, "march": 3, "3": 3,
    "apr": 4, "april": 4, "4": 4,
    "may": 5, "5": 5,
    "jun": 6, "june": 6, "6": 6,
    "jul": 7, "july": 7, "7": 7,
    "aug": 8, "august": 8, "8": 8,
    "sep": 9, "september": 9, "9": 9,
    "oct": 10, "october": 10, "10": 10,
    "nov": 11, "november": 11, "11": 11,
    "dec": 12, "december": 12, "12": 12,
}

_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ── Normalizers ───────────────────────────────────────────────────────────────

def _norm_destination(raw: str) -> str:
    mapping = {
        "1": "Ladakh",
        "2": "Himachal",
        "3": "Kashmir",
        "4": "Combination",
    }
    v = raw.strip()
    return mapping.get(v, v.title())


def _norm_travel_type(raw: str) -> str:
    mapping = {"1": "Bike", "2": "Car", "3": "Trek"}
    v = raw.strip()
    return mapping.get(v, v.title())


def _norm_duration(raw: str) -> str:
    mapping = {"1": "3-5", "2": "5-8", "3": "9+"}
    v = raw.strip()
    return mapping.get(v, v.replace(" days", "").strip())


def _norm_pax(raw: str) -> int:
    mapping = {"1": 2, "2": 4, "3": 6, "4": 8, "5": 10, "6": 12}
    v = raw.strip()
    try:
        return mapping.get(v, int(re.search(r"\d+", v).group()))
    except Exception:
        return 2


def _norm_month(raw: str) -> str:
    v = raw.strip().lower()
    num = _MONTH_MAP.get(v[:3], _MONTH_MAP.get(v, 0))
    return _MONTH_NAMES[num] if num else raw.strip().title()


def _norm_age_range(raw: str) -> tuple[int, int]:
    """Parse '18, 55' or '18-55' or '18 to 55' → (18, 55). Single value → same for both."""
    nums = re.findall(r"\d+", raw)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        return int(nums[0]), int(nums[0])
    return 18, 45


def _norm_itinerary_type(raw: str) -> str:
    mapping = {"1": "Existing", "2": "Custom"}
    v = raw.strip()
    return mapping.get(v, v.title())


def _norm_interests(raw: str) -> str:
    mapping = {"1": "Adventure", "2": "Culture", "3": "Photography", "4": "All"}
    return mapping.get(raw.strip(), raw.strip().title())


def _norm_fitness(raw: str) -> str:
    mapping = {"1": "Beginner", "2": "Experienced"}
    return mapping.get(raw.strip(), raw.strip().title())


# ── Intelligence helpers ──────────────────────────────────────────────────────

def _season_advisory(destination: str, month_num: int) -> str | None:
    """Return an advisory string if travel month is risky, else None."""
    if month_num == 0:
        return None
    key    = destination.lower()
    season = _SEASON_WARNINGS.get(key, {})
    if month_num in season.get("closed", []):
        return (
            f"⚠️ **Seasonal Advisory:** {destination} has limited or no access in "
            f"{_MONTH_NAMES[month_num]}. Key mountain passes may be closed due to snow. "
            "We still found matching packages but please confirm road conditions before booking."
        )
    if month_num in season.get("shoulder", []):
        return (
            f"ℹ️ **Seasonal Note:** {_MONTH_NAMES[month_num]} is a shoulder season for "
            f"{destination}. Weather can be unpredictable. Carry extra layers and confirm "
            "pass conditions closer to travel date."
        )
    return None


def _age_advisory(travel_type: str, destination: str, min_age: int, max_age: int) -> list[str]:
    """Return a list of age-based warning strings."""
    warnings = []
    if travel_type == "Bike":
        if min_age < 18:
            warnings.append(
                "⚠️ **Age Advisory:** Bike trips require a minimum age of 18. "
                f"Your group includes travellers under 18 (min age: {min_age}). "
                "Consider a Car trip for this group."
            )
        elif max_age > 60:
            warnings.append(
                f"ℹ️ **Age Note:** Senior travellers ({max_age} years) on a Bike trip at "
                "high altitude require good physical fitness. Ensure a medical check-up "
                "before the trip and carry altitude sickness medication."
            )

    high_altitude_dests = {"ladakh", "himachal"}
    if destination.lower() in high_altitude_dests:
        if max_age > 65:
            warnings.append(
                f"⚠️ **High Altitude Advisory:** Travellers aged {max_age}+ may face "
                "increased risk of altitude sickness above 3,500m. Recommend Standard "
                "category hotels (lower altitude stays), adequate acclimatisation days, "
                "and medical clearance before travel."
            )
        if min_age < 8:
            warnings.append(
                f"ℹ️ **Children Advisory:** Children under 8 (min age: {min_age}) at "
                "high altitude need extra monitoring. Consult a paediatrician before "
                "travelling above 3,500m."
            )
    return warnings


# ── Node 1: Collect travel preferences ───────────────────────────────────────

async def node_collect_travel_prefs(state: PlannerState) -> dict:
    messages = []

    # Close the tool_start_itinerary_planner call from the parent assistant
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

    # ── Q1: Destination ──────────────────────────────────────────────────────
    dest_raw = interrupt(
        "🏔️ **Welcome to the Itinerary Planner!**\n\n"
        "I'll ask you a few quick questions to find the perfect trip.\n\n"
        "**Q1. Where would you like to travel?**\n"
        "1. Ladakh\n"
        "2. Himachal Pradesh\n"
        "3. Kashmir\n"
        "4. Combination _(mix of multiple regions)_\n\n"
        "_(Type a number or the destination name)_"
    )
    destination = _norm_destination(dest_raw)

    # ── Q2: Travel type ──────────────────────────────────────────────────────
    type_raw = interrupt(
        f"**Q2. How would you like to travel to {destination}?**\n"
        "1. Bike  🏍️  (motorcycle tour, self-ridden or guided)\n"
        "2. Car   🚗  (comfortable, SUV / Tempo Traveller)\n"
        "3. Trek  🥾  (trekking expedition)\n\n"
        "_(Type a number or the mode of travel)_"
    )
    travel_type = _norm_travel_type(type_raw)

    # Trek not yet available
    if travel_type == "Trek":
        messages.append(AIMessage(content=(
            "🥾 **Trek packages** for this region are currently being curated and will be "
            "available soon! Meanwhile, we have amazing **Bike** and **Car** packages.\n\n"
            "Switching your selection to **Car** for now so we can continue planning."
        )))
        travel_type = "Car"

    # ── Q3: Duration ─────────────────────────────────────────────────────────
    dur_raw = interrupt(
        "**Q3. How many days are you planning for your trip?**\n"
        "1. 3–5 days  (short getaway)\n"
        "2. 5–8 days  (standard trip)\n"
        "3. 9+ days   (extended expedition)\n\n"
        "_(Type a number or the duration)_"
    )
    duration = _norm_duration(dur_raw)

    # ── Q4: Group size ───────────────────────────────────────────────────────
    pax_raw = interrupt(
        "**Q4. How many people are travelling (including yourself)?**\n"
        "1. 2 persons\n"
        "2. 4 persons\n"
        "3. 6 persons\n"
        "4. 8 persons\n"
        "5. 10 persons\n"
        "6. 12 persons\n\n"
        "_(Type a number 1–6, or the exact count)_"
    )
    pax = _norm_pax(pax_raw)

    # ── Q5: Travel month ─────────────────────────────────────────────────────
    month_raw = interrupt(
        "**Q5. Which month are you planning to travel?**\n"
        "_(e.g. June, July, October — or type the month number)_"
    )
    month     = _norm_month(month_raw)
    month_num = _MONTH_MAP.get(month.lower()[:3], 0)

    # Season advisory
    advisory = _season_advisory(destination, month_num)
    if advisory:
        messages.append(AIMessage(content=advisory))

    # ── Q6: Age range ────────────────────────────────────────────────────────
    age_raw = interrupt(
        "**Q6. What is the age range of your travel group?**\n"
        "_(Enter as: min age, max age — e.g. **18, 55** or **25, 40**)_\n\n"
        "This helps us flag altitude suitability and vehicle recommendations."
    )
    min_age, max_age = _norm_age_range(age_raw)

    # Age-based advisories
    age_advisories = _age_advisory(travel_type, destination, min_age, max_age)
    for adv in age_advisories:
        messages.append(AIMessage(content=adv))

    # ── Q7: Existing vs Custom ───────────────────────────────────────────────
    itin_raw = interrupt(
        "**Q7. What kind of itinerary are you looking for?**\n"
        "1. **Existing** — Choose from our curated packages\n"
        "2. **Custom**   — Tell us your dream trip and we'll tailor it\n\n"
        "_(Type 1 or 2)_"
    )
    itinerary_type = _norm_itinerary_type(itin_raw)

    # ── Custom path ──────────────────────────────────────────────────────────
    if itinerary_type == "Custom":
        custom_desc_raw = interrupt(
            "✨ **Custom Itinerary!**\n\n"
            "Describe your dream trip — places you must visit, experiences you want, "
            "any specific routes or activities in mind:\n\n"
            "_(e.g. 'I want to see Pangong Lake, ride to Khardung La, and spend a night "
            "in a tent at Nubra')_"
        )
        messages.append(AIMessage(content=(
            f"📝 **Custom itinerary request noted!**\n\n"
            f"**Your preferences:**\n"
            f"- Destination: {destination}\n"
            f"- Travel type: {travel_type}\n"
            f"- Duration: {duration} days\n"
            f"- Group: {pax} persons\n"
            f"- Month: {month}\n"
            f"- Age range: {min_age}–{max_age} years\n\n"
            f"**Your vision:** {custom_desc_raw}\n\n"
            "Our team will design a personalised itinerary based on these details. "
            "You'll receive it within 24 hours. In the meantime, would you like to "
            "explore any of our existing packages for inspiration?"
        )))
        # Save partial preferences for Custom path
        _store = get_store()
        if _store is not None:
            _user_id = get_config().get("configurable", {}).get("session_id", "anonymous")
            await memory_store.upsert_travel_preferences(_store, _user_id, {
                "destination": destination,
                "travel_type": travel_type,
                "typical_pax": pax,
                "month":       month,
            })

        log_planner_step(
            "travel_prefs",
            input_data={"path": "Custom"},
            output_data={
                "destination": destination, "travel_type": travel_type,
                "duration": duration, "pax": pax, "month": month,
                "age_range": f"{min_age}–{max_age}", "custom_description": custom_desc_raw,
            },
        )
        return {
            "messages": messages,
            "planner_destination":    destination,
            "planner_travel_type":    travel_type,
            "planner_duration":       duration,
            "planner_pax":            pax,
            "planner_month":          month,
            "planner_min_age":        min_age,
            "planner_max_age":        max_age,
            "planner_itinerary_type": itinerary_type,
        }

    # ── Q8: Package selection (Existing path) ────────────────────────────────
    packages = find_packages(travel_type, destination, duration)

    if not packages:
        # ── Fallback 1: same destination + travel type, any duration ─────────
        alt_packages = find_packages(travel_type, destination, "")
        if not alt_packages:
            # ── Fallback 2: same destination, any travel type ─────────────────
            for alt_type in ("Bike", "Car"):
                if alt_type == travel_type:
                    continue
                alt_packages = find_packages(alt_type, destination, "")
                if alt_packages:
                    travel_type = alt_type   # update so Q8 header is accurate
                    break

        if not alt_packages:
            messages.append(AIMessage(content=(
                f"Sorry, we don't have any **{travel_type}** packages for **{destination}** yet. "
                "You can try a different destination or choose a **Custom** itinerary "
                "and our team will plan one for you."
            )))
            return {
                "messages": messages,
                "planner_destination":    destination,
                "planner_travel_type":    travel_type,
                "planner_duration":       duration,
                "planner_pax":            pax,
                "planner_month":          month,
                "planner_min_age":        min_age,
                "planner_max_age":        max_age,
                "planner_itinerary_type": itinerary_type,
            }

        # Show alternatives via interrupt so user can pick
        alt_list = "\n".join(
            f"{i + 1}. **{p['title']}** ({p['packageDuration']}) "
            f"— {p.get('totalDistance', '')} | {p.get('season', 'Year-round')}"
            for i, p in enumerate(alt_packages)
        )
        selection_raw = interrupt(
            f"No **{duration}-day** packages found for **{destination}**. "
            f"Here are the closest alternatives:\n\n"
            f"{alt_list}\n\n"
            "Type the number to select one, or **skip** to exit the planner."
        )

        if selection_raw.strip().lower() == "skip":
            messages.append(AIMessage(content=(
                "No problem! Let me know if you'd like to explore other options or plan a custom trip."
            )))
            return {
                "messages": messages,
                "planner_destination":    destination,
                "planner_travel_type":    travel_type,
                "planner_duration":       duration,
                "planner_pax":            pax,
                "planner_month":          month,
                "planner_min_age":        min_age,
                "planner_max_age":        max_age,
                "planner_itinerary_type": itinerary_type,
            }

        try:
            idx = int(selection_raw.strip()) - 1
            selected = alt_packages[max(0, min(idx, len(alt_packages) - 1))]
        except (ValueError, IndexError):
            selected = alt_packages[0]

        log_planner_step(
            "travel_prefs",
            input_data={"path": "Existing (fallback)"},
            output_data={
                "destination": destination, "travel_type": travel_type,
                "requested_duration": duration, "selected_package": selected["id"],
            },
        )
        return {
            "messages":               messages,
            "planner_destination":    destination,
            "planner_travel_type":    travel_type,
            "planner_duration":       duration,
            "planner_pax":            pax,
            "planner_month":          month,
            "planner_min_age":        min_age,
            "planner_max_age":        max_age,
            "planner_itinerary_type": itinerary_type,
            "planner_matched_packages":    alt_packages,
            "planner_selected_package_id": selected["id"],
        }

    pkg_list = "\n".join(
        f"{i + 1}. **{p['title']}** ({p['packageDuration']}) "
        f"— {p.get('totalDistance', '')} | {p.get('season', 'Year-round')}"
        for i, p in enumerate(packages)
    )
    selection_raw = interrupt(
        f"**Q8. Available {travel_type} packages for {destination} ({duration} days):**\n\n"
        f"{pkg_list}\n\n"
        "Which package would you like to plan your itinerary for? _(Type the number)_"
    )

    try:
        idx = int(selection_raw.strip()) - 1
        selected = packages[max(0, min(idx, len(packages) - 1))]
    except (ValueError, IndexError):
        selected = packages[0]

    log_planner_step(
        "travel_prefs",
        input_data={"path": "Existing"},
        output_data={
            "destination": destination, "travel_type": travel_type,
            "duration": duration, "pax": pax, "month": month,
            "age_range": f"{min_age}–{max_age}",
            "itinerary_type": itinerary_type,
            "selected_package": selected["id"],
        },
    )
    return {
        "messages":               messages,
        "planner_destination":    destination,
        "planner_travel_type":    travel_type,
        "planner_duration":       duration,
        "planner_pax":            pax,
        "planner_month":          month,
        "planner_min_age":        min_age,
        "planner_max_age":        max_age,
        "planner_itinerary_type": itinerary_type,
        "planner_matched_packages":      packages,
        "planner_selected_package_id":   selected["id"],
    }


# ── Node 2: Collect personal preferences ─────────────────────────────────────

def node_collect_personal_prefs(state: PlannerState) -> dict:
    # Skip if Custom itinerary or no package was found/selected
    if state.get("planner_itinerary_type") == "Custom":
        return {}
    if not state.get("planner_selected_package_id"):
        return {}

    interests_raw = interrupt(
        "**Q9. What are your main interests for this trip?**\n"
        "1. Adventure  🏔️  (passes, rafting, extreme routes)\n"
        "2. Culture    🛕  (monasteries, local villages, food)\n"
        "3. Photography 📸  (landscapes, golden hour, viewpoints)\n"
        "4. All of the above\n\n"
        "_(Type a number)_"
    )
    interests = _norm_interests(interests_raw)

    fitness_raw = interrupt(
        "**Q10. How would you rate your fitness level?**\n"
        "1. Beginner      — comfortable with drives & light walks, no extreme treks\n"
        "2. Experienced   — ready for high altitude, long rides, challenging terrain\n\n"
        "_(Type a number)_"
    )
    fitness = _norm_fitness(fitness_raw)

    special_raw = interrupt(
        "**Q11. Any special requirements?**\n"
        "_(e.g. vegetarian food, wheelchair access, medical conditions, anniversary celebration)_\n\n"
        "Type **none** if nothing special."
    )
    special_req = "" if special_raw.strip().lower() == "none" else special_raw.strip()

    log_planner_step(
        "personal_prefs",
        input_data={},
        output_data={"interests": interests, "fitness": fitness, "special_req": special_req or "none"},
    )
    return {
        "planner_interests":  interests,
        "planner_fitness":    fitness,
        "planner_special_req": special_req,
    }


# ── Node 3: Format itinerary from JSON ───────────────────────────────────────

_INTEREST_KEYWORDS = {
    "Adventure":   ["pass", "rafting", "trek", "atv", "camel", "khardung", "chang la",
                    "altitude", "extreme", "motorcycle", "bike"],
    "Culture":     ["monastery", "gurudwara", "palace", "stupa", "museum", "memorial",
                    "secmol", "village", "langar", "temple", "mosque"],
    "Photography": ["sunrise", "sunset", "viewpoint", "confluence", "lake", "sand dune",
                    "panoramic", "golden", "landscape", "colour"],
}


def _activity_matches_interests(activity, interests: str) -> bool:
    if interests == "All":
        return True
    keywords = _INTEREST_KEYWORDS.get(interests, [])
    if isinstance(activity, str):
        text = activity.lower()
    else:
        text = (activity.get("name", "") + " " + activity.get("description", "")).lower()
    return any(kw in text for kw in keywords)


def _format_activity(activity, interests: str, fitness: str) -> str:
    # Activities in JSON can be plain strings or rich dicts
    if isinstance(activity, str):
        highlight = _activity_matches_interests(activity, interests)
        prefix    = "⭐ " if highlight and interests != "All" else "• "
        return f"{prefix}{activity}"

    name        = activity.get("name", "")
    time        = activity.get("time", "")
    duration    = activity.get("duration", "")
    description = activity.get("description", "")
    altitude_m  = activity.get("altitude_m")
    distance_km = activity.get("distance_km")

    highlight = _activity_matches_interests(activity, interests)
    prefix    = "⭐ " if highlight and interests != "All" else ""

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

    meta  = "  |  ".join(meta_parts)
    lines = [f"**{prefix}{name}**"]
    if meta:
        lines.append(f"_{meta}_")
    if description:
        lines.append(description)
    return "\n".join(lines)


async def node_generate_itinerary(state: PlannerState) -> dict:
    # Custom itinerary was already handled in node_collect_travel_prefs
    if state.get("planner_itinerary_type") == "Custom":
        return {}
    # No package selected means no packages were found — already messaged in travel_prefs node
    if not state.get("planner_selected_package_id"):
        return {}

    package_id  = state.get("planner_selected_package_id")
    interests   = state.get("planner_interests", "All")
    fitness     = state.get("planner_fitness", "Beginner")
    special_req = state.get("planner_special_req", "")
    pax         = state.get("planner_pax", 2)
    month       = state.get("planner_month", "")
    min_age     = state.get("planner_min_age")
    max_age     = state.get("planner_max_age")
    travel_type = state.get("planner_travel_type", "Car")

    pkg = get_package(package_id)
    if not pkg:
        return {"messages": [AIMessage(content="Sorry, could not load the package data.")]}

    # ── Step 1: Show overview from overviewSections ───────────────────────────
    overview_lines = [
        f"# {pkg['title']}\n"
        f"**{pkg.get('packageDuration', '')}**  |  {pkg.get('totalDistance', '')}  |  "
        f"Season: {pkg.get('season', 'Year-round')}\n"
    ]
    for section in pkg.get("overviewSections", []):
        days_label = ", ".join(f"Day {d}" for d in section.get("dayNumbers", []))
        overview_lines.append(f"**{days_label}: {section['title']}**")
        short_desc = section.get("shortDescription", "").strip()
        if short_desc:
            overview_lines.append(short_desc)
        overview_lines.append("")

    overview_text = "\n".join(overview_lines)

    # ── Step 2: Ask if user wants detailed itinerary ──────────────────────────
    want_detail = interrupt(
        overview_text
        + "\nWould you like the detailed day-by-day itinerary with routes and activities?\n"
        "_(Type **yes** or **no**)_"
    )

    messages = []

    if want_detail.strip().lower() not in ("yes", "y"):
        messages.append(AIMessage(content=(
            "No problem! Let me know if you'd like to book this package "
            "or explore other options."
        )))
    else:
        # ── Step 3: Build detailed itinerary ─────────────────────────────────
        sections = []
        for day in pkg.get("detailedItinerary", []):
            overnight = day.get("overnight", "")
            header = (
                f"## Day {day['day']}: {day['title']}\n"
                f"**Route:** {day.get('route', '')}"
                + (f"  |  **Overnight:** {overnight}" if overnight else "")
            )
            activity_lines = [
                _format_activity(act, interests, fitness)
                for act in day.get("activities", [])
            ]
            sections.append(header + "\n\n" + "\n".join(activity_lines))

        advisories = []
        if fitness == "Beginner":
            advisories.append(
                "> ⚠️ Activities at 5,000m+ require caution. Limit exposure to 45 min and descend if unwell."
            )
        if max_age and max_age > 65:
            advisories.append(
                "> ⚠️ Senior travellers: medical clearance recommended before high-altitude travel."
            )
        if special_req:
            advisories.append(f"> 📝 Special requirement noted: {special_req}")

        detail_text = (
            ("\n".join(advisories) + "\n\n" if advisories else "")
            + "\n\n---\n\n".join(sections)
        )
        messages.append(AIMessage(content=detail_text))

    # ── Save travel preferences to long-term memory ───────────────────────────
    _store = get_store()
    if _store is not None:
        _user_id = get_config().get("configurable", {}).get("session_id", "anonymous")
        updates = {}
        for field, key in [
            ("planner_destination", "destination"),
            ("planner_travel_type", "travel_type"),
            ("planner_pax",         "typical_pax"),
            ("planner_month",       "month"),
            ("planner_interests",   "interests"),
            ("planner_fitness",     "fitness"),
        ]:
            val = state.get(field)
            if val is not None:
                updates[key] = val
        if updates:
            await memory_store.upsert_travel_preferences(_store, _user_id, updates)

    log_planner_step(
        "itinerary_generated",
        input_data={"package_id": package_id, "want_detail": want_detail.strip()},
        output_data={"package": pkg.get("title", ""), "detail_shown": want_detail.strip().lower() in ("yes", "y")},
    )
    return {"messages": messages}
