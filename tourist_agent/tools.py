"""
LangChain tools available to the tourist agent.

Safe tools   — executed directly, no confirmation needed.
Sensitive tools — require human-in-the-loop confirmation before execution.
"""
from collections import Counter

from langchain_core.tools import tool
from tourist_agent.data_loader import (
    get_package,
    get_day_itinerary,
    get_packages_summary,
    filter_packages,
    get_package_cost,
    get_cost_categories,
)


# --------------------------------------------------------------------------- #
# Format helpers for tool_list_packages                                        #
# --------------------------------------------------------------------------- #

_TYPE_ICONS = {"Bike": "🏍️", "Car": "🚗", "Trek": "🥾"}


def _fmt_durations(durations: list[str]) -> str:
    """['6D','6D','7D'] → '6D (×2) · 7D'"""
    counts = Counter(durations)
    parts  = []
    for d in sorted(counts, key=lambda x: int(x.replace("D", "") or 0)):
        parts.append(f"{d} (×{counts[d]})" if counts[d] > 1 else d)
    return " · ".join(parts)


def _format_summary(summary: dict) -> str:
    lines = []
    for travel_type in sorted(summary.keys()):
        icon  = _TYPE_ICONS.get(travel_type, "🗺️")
        data  = summary[travel_type]
        total = data["total"]
        lines.append(f"{icon} **{travel_type} Tours** — {total} packages")
        for dest, count in data["by_destination"].items():
            pkg_label = "package" if count == 1 else "packages"
            lines.append(f"   {dest:<24}: {count} {pkg_label}")
        lines.append("")

    if "Trek" not in summary:
        lines.append("🥾 **Trek Expeditions** — Coming soon")
        lines.append("")

    lines.append(
        "_To see specific packages, tell me:_\n"
        "_• **Location** — Ladakh, Himachal, or Kashmir_\n"
        "_• **Type of Tour** — Bike, Car, or Trek_\n"
        "_• **Duration** — 3–5 days, 5–8 days, or 9+ days_"
    )
    return "\n".join(lines)


def _format_package_list(packages: list, travel_type: str, destination: str) -> str:
    header_parts = [p for p in [destination, travel_type, "Tours"] if p]
    header = f"**{' '.join(header_parts)}:**\n\n" if header_parts else "**Matching Packages:**\n\n"

    lines = []
    for i, p in enumerate(packages, 1):
        dist     = p.get("totalDistance", "")
        dist_str = f" — {dist}" if dist else ""
        states   = ", ".join(p.get("state", []))
        pkg_id   = p.get("id", "")
        lines.append(
            f"{i}. **{p['title']}** ({p.get('packageDuration','')}){dist_str}  "
            f"| _{states}_ · `{pkg_id}`"
        )

    lines.append("\n_Ask for full details on any package by name or number._")
    return header + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Safe tools                                                                   #
# --------------------------------------------------------------------------- #

@tool
def tool_list_packages(
    travel_type: str = "",
    destination: str = "",
    duration: str = "",
) -> str:
    """
    List available tour packages with smart progressive disclosure.

    - No filters → compact grouped summary (type → destination → durations).
      Use this when the user asks 'what packages do you have' or 'show all tours'.
    - With filters → short numbered list of matching packages.
      Use this when the user specifies a destination, travel type, or duration.

    travel_type : 'Bike', 'Car', or 'Trek'          (leave blank if not specified)
    destination : 'Ladakh', 'Himachal', or 'Kashmir' (leave blank if not specified)
    duration    : '3-5', '5-8', or '9+'              (leave blank if not specified)
    """
    has_filters = bool(travel_type.strip() or destination.strip() or duration.strip())

    if not has_filters:
        return _format_summary(get_packages_summary())

    packages = filter_packages(
        travel_type=travel_type.strip(),
        destination=destination.strip(),
        duration=duration.strip(),
    )

    if not packages:
        return (
            f"No packages found for the given filters "
            f"(type={travel_type or 'any'}, destination={destination or 'any'}, "
            f"duration={duration or 'any'}).\n\n"
            "Try broadening your search or ask to see the full catalog."
        )

    return _format_package_list(packages, travel_type.strip(), destination.strip())


@tool
def tool_get_package_details(package_id: str) -> str:
    """Get full details of a specific package including all highlights."""
    pkg = get_package(package_id)
    if not pkg:
        return f"Package '{package_id}' not found."
    highlights = "\n".join(f"  • {h}" for h in pkg.get("highlights", []))
    overview = "\n".join(
        f"  Day {', '.join(str(d) for d in s['dayNumbers'])}: {s['title']}"
        for s in pkg.get("overviewSections", [])
    )
    return (
        f"**{pkg['title']}**\n"
        f"Duration: {pkg['packageDuration']} | Distance: {pkg.get('totalDistance','')}\n"
        f"States: {', '.join(pkg.get('state', []))}\n\n"
        f"{pkg.get('description','')}\n\n"
        f"**Highlights:**\n{highlights}\n\n"
        f"**Day-by-Day Overview:**\n{overview}"
    )


@tool
def tool_get_day_itinerary(package_id: str, day: int) -> str:
    """Get the detailed itinerary for a specific day of a package."""
    entry = get_day_itinerary(package_id, day)
    if not entry:
        return f"No itinerary found for day {day} of package '{package_id}'."
    activities = "\n".join(f"  • {a}" for a in entry.get("activities", []))
    return (
        f"**Day {entry['day']}: {entry['title']}**\n"
        f"Route: {entry.get('route','')}\n"
        f"Overnight: {entry.get('overnight','')}\n\n"
        f"{entry.get('description','')}\n\n"
        f"**Activities:**\n{activities}"
    )


@tool
def tool_get_full_itinerary(package_id: str) -> str:
    """Get the complete day-by-day itinerary for a package."""
    pkg = get_package(package_id)
    if not pkg:
        return f"Package '{package_id}' not found."
    days = pkg.get("detailedItinerary", [])
    parts = []
    for entry in days:
        activities = ", ".join(entry.get("activities", []))
        parts.append(
            f"**Day {entry['day']}: {entry['title']}**\n"
            f"Route: {entry.get('route','')}\n"
            f"Overnight: {entry.get('overnight','') or 'Departure'}\n"
            f"Activities: {activities}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def tool_get_package_cost(package_id: str, category: str = "") -> str:
    """
    Get pricing for a specific package.

    - If category is blank, returns the available categories and asks the user
      to pick one before showing pricing.
    - If category is provided, returns a formatted pricing table for that category.

    package_id : exact package ID (e.g. '4D_leh_kardungla')
    category   : 'Standard', 'Deluxe', or 'Premium' — leave blank if not yet chosen
    """
    cost = get_package_cost(package_id)
    if not cost:
        categories = get_cost_categories(package_id)
        if not categories:
            return (
                f"No pricing data found for package `{package_id}`. "
                "Cost sheets are currently available for Bike tours only."
            )

    if not category.strip():
        # Category not chosen yet — list available and ask
        categories = get_cost_categories(package_id)
        cat_list   = " · ".join(f"**{c}**" for c in categories)
        return (
            f"We have **{len(categories)} package categories** for this tour:\n\n"
            f"{cat_list}\n\n"
            "Which category would you prefer?"
        )

    # Filter pricing rows for the requested category
    cat_lower = category.strip().lower()
    rows = [
        r for r in cost["pricing"]
        if r.get("category", "").lower() == cat_lower
    ]

    if not rows:
        available = get_cost_categories(package_id)
        return (
            f"No pricing found for category **{category}**. "
            f"Available: {', '.join(available)}."
        )

    pkg      = get_package(package_id)
    pkg_title = pkg["title"] if pkg else package_id

    lines = [
        f"**{pkg_title}** — {category} Pricing\n",
        f"{'Persons':<12} {'Vehicle':<22} {'Hotels':<12} {'Price/Person':>13} {'Room Split'}",
        f"{'-'*12} {'-'*22} {'-'*12} {'-'*13} {'-'*20}",
    ]
    for r in rows:
        lines.append(
            f"{r.get('persons',''):<12} "
            f"{r.get('car_model',''):<22} "
            f"{r.get('hotels',''):<12} "
            f"₹{r.get('price_per_person',0):>12,} "
            f"{r.get('room_split','')}"
        )

    lines.append(
        f"\n_Prices are per person. GST extra. "
        f"Contact us to confirm availability for your travel date._"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Booking validation tool                                                      #
# --------------------------------------------------------------------------- #

@tool
def tool_validate_booking_info(
    lead_name: str,
    email: str,
    phone: str,
    travel_date: str,
    pax_count: int,
) -> str:
    """
    Validate all booking fields BEFORE calling tool_book_trip.
    Always call this tool first whenever you have all 5 booking fields collected.
    Returns a success message if valid, or a list of specific errors to fix.

    lead_name   : traveller's full name
    email       : email address
    phone       : phone with country code (e.g. +91 98765 43210)
    travel_date : DD-MM-YYYY format
    pax_count   : number of travellers (1–20)
    """
    import re
    from datetime import datetime

    errors = []

    # ── Name ──────────────────────────────────────────────────────────────────
    if not lead_name or len(lead_name.strip()) < 2:
        errors.append("- **Name**: Must be at least 2 characters.")

    # ── Email ─────────────────────────────────────────────────────────────────
    email_pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email.strip()):
        errors.append(
            f"- **Email**: `{email}` is invalid. "
            "Expected format: name@domain.com"
        )

    # ── Phone — must include country code ────────────────────────────────────
    digits = re.sub(r'[\s\-\(\)]', '', phone.strip())
    if digits.startswith('+'):
        if len(digits[1:]) < 10:
            errors.append(
                f"- **Phone**: `{phone}` is too short after the country code. "
                "Example: +91 98765 43210"
            )
    else:
        if len(digits) < 10:
            errors.append(
                f"- **Phone**: `{phone}` has only {len(digits)} digits and is missing "
                "the country code. Example for India: +91 98765 43210"
            )
        else:
            errors.append(
                f"- **Phone**: `{phone}` is missing the country code prefix. "
                "Add +91 for India (e.g. +91 98765 43210)."
            )

    # ── Travel date — DD-MM-YYYY, must be future ──────────────────────────────
    try:
        travel_dt = datetime.strptime(travel_date.strip(), "%d-%m-%Y")
        if travel_dt.date() <= datetime.today().date():
            errors.append(
                f"- **Travel Date**: `{travel_date}` is today or in the past. "
                "Please provide a future date."
            )
    except ValueError:
        errors.append(
            f"- **Travel Date**: `{travel_date}` is not in the correct format. "
            "Use DD-MM-YYYY — example: 25-08-2026."
        )

    # ── Pax count ─────────────────────────────────────────────────────────────
    if not (1 <= pax_count <= 20):
        errors.append(
            f"- **Pax Count**: `{pax_count}` is out of range. Must be between 1 and 20."
        )

    if errors:
        return (
            "⚠️ **Booking validation failed. Please ask the user to correct these fields:**\n\n"
            + "\n".join(errors)
        )

    return "✅ All fields valid. You may now call tool_book_trip."


# --------------------------------------------------------------------------- #
# Sensitive tools — human confirmation required before execution               #
# --------------------------------------------------------------------------- #

@tool
def tool_book_trip(
    package_id: str,
    lead_name: str,
    email: str,
    phone: str,
    travel_date: str,
    pax_count: int,
) -> str:
    """
    Book a trip package for a customer.
    Requires: package_id, lead_name, email, phone, travel_date (DD-MM-YYYY), pax_count.
    Collect all fields conversationally before calling this tool.
    """
    from tourist_agent.booking_db import create_booking
    booking_id = create_booking({
        "package_id": package_id,
        "lead_name": lead_name,
        "email": email,
        "phone": phone,
        "travel_date": travel_date,
        "pax_count": pax_count,
    })
    pkg = get_package(package_id)
    pkg_title = pkg["title"] if pkg else package_id
    return (
        f"Booking confirmed!\n\n"
        f"**Booking ID:** `{booking_id}`\n"
        f"**Package:** {pkg_title}\n"
        f"**Traveller:** {lead_name}\n"
        f"**Travel Date:** {travel_date}\n"
        f"**Travellers:** {pax_count}\n"
        f"**Email:** {email}\n\n"
        f"A confirmation will be sent to your email. Have an amazing trip!"
    )


@tool
def tool_cancel_trip(booking_id: str) -> str:
    """Cancel an existing trip booking by booking ID."""
    from tourist_agent.booking_db import cancel_booking
    return cancel_booking(booking_id)


# --------------------------------------------------------------------------- #
# Tool sets                                                                    #
# --------------------------------------------------------------------------- #

SAFE_TOOLS = [
    tool_list_packages,
    tool_get_package_details,
    tool_get_day_itinerary,
    tool_get_full_itinerary,
    tool_get_package_cost,
    tool_validate_booking_info,
]

SENSITIVE_TOOLS = [
    tool_book_trip,
    tool_cancel_trip,
]

SENSITIVE_TOOL_NAMES = {t.name for t in SENSITIVE_TOOLS}


# --------------------------------------------------------------------------- #
# Planner tools — trigger the itinerary planner subagent                      #
# --------------------------------------------------------------------------- #

@tool
def tool_start_itinerary_planner() -> str:
    """
    Launch the interactive itinerary planner.
    Call this when the user wants to plan a custom trip itinerary step by step.
    """
    return "Starting itinerary planner..."


PLANNER_TOOLS = [tool_start_itinerary_planner]
PLANNER_TOOL_NAMES = {t.name for t in PLANNER_TOOLS}

ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS + PLANNER_TOOLS
