"""
LangChain tools available to the tourist agent.
Each tool is a plain function decorated with @tool.
"""
from langchain_core.tools import tool
from tourist_agent.data_loader import (
    list_packages,
    get_package,
    get_day_itinerary,
    search_packages,
)


@tool
def tool_list_packages(location: str = "") -> str:
    """List all available tour packages. Optionally filter by location/state name."""
    packages = search_packages(location if location else None)
    if not packages:
        return "No packages found."
    lines = []
    for p in packages:
        lines.append(
            f"**{p['title']}** (ID: `{p['id']}`)\n"
            f"  Duration: {p['duration']} | States: {', '.join(p['states'])}\n"
            f"  {p['description']}"
        )
    return "\n\n".join(lines)


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


ALL_TOOLS = [
    tool_list_packages,
    tool_get_package_details,
    tool_get_day_itinerary,
    tool_get_full_itinerary,
]
