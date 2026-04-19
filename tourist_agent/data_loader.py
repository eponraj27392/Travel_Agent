"""
Dynamic itinerary package loader.

Directory layout:
  data/itenary/car/                    ← Car packages (destination via state field)
  data/itenary/bike/ladakh/            ← Bike packages for Ladakh
  data/itenary/bike/himachal/          ← Bike packages for Himachal
  data/itenary/bike/kashmir/           ← Bike packages for Kashmir

Each JSON: { "itinerary": { "id", "state", "packageDuration", ... } }
"""
import json
import re
from pathlib import Path
from functools import lru_cache

DATA_DIR    = Path(__file__).parent.parent / "data/travel"
ITENARY_DIR = DATA_DIR

_DEST_ALIASES = {
    "ladakh":   ["ladakh"],
    "himachal": ["himachal", "himachal pradesh"],
    "kashmir":  ["kashmir", "jammu & kashmir", "jammu and kashmir"],
}


def _parse_days(package_duration: str) -> int:
    """Extract day count from '4D/3N' → 4."""
    m = re.match(r"(\d+)D", str(package_duration), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _days_in_range(days: int, duration_range: str) -> bool:
    r = duration_range.strip()
    if r == "3-5":
        return 3 <= days <= 5
    if r == "5-8":
        return 5 <= days <= 8
    if r.startswith("9"):
        return days >= 9
    return True


def _dest_matches(state_list: list, destination: str) -> bool:
    aliases   = _DEST_ALIASES.get(destination.lower(), [destination.lower()])
    pkg_lower = [s.lower() for s in state_list]
    return any(any(alias in ps for alias in aliases) for ps in pkg_lower)


@lru_cache(maxsize=None)
def _load_all_packages() -> dict:
    """
    Scan all itenary sub-directories and return a flat dict keyed by package id.
    Injects travel_type and dest_folder into each itinerary dict.
    """
    catalog = {}
    for json_file in ITENARY_DIR.rglob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        itinerary = data.get("itinerary", {})
        pkg_id = itinerary.get("id")
        if not pkg_id:
            continue

        rel_parts   = json_file.relative_to(ITENARY_DIR).parts
        travel_type = rel_parts[0].title()                             # "Bike" or "Car"
        dest_folder = rel_parts[1] if len(rel_parts) > 2 else ""      # "ladakh" / "himachal" / ""

        itinerary               = dict(itinerary)
        itinerary["travel_type"] = travel_type
        itinerary["dest_folder"] = dest_folder
        itinerary["file_path"]   = str(json_file)

        catalog[pkg_id] = itinerary

    return catalog


def get_package(package_id: str) -> dict | None:
    return _load_all_packages().get(package_id)


def find_packages(
    travel_type: str,
    destination: str,
    duration_range: str,
) -> list[dict]:
    """
    Return packages matching travel_type + destination + duration range.
    destination can be a single region ("Ladakh") or " & "-separated combination
    ("Ladakh & Himachal") — packages matching ANY of the listed regions are returned.
    Sorted by day count ascending.
    """
    type_lower = travel_type.lower()
    is_combo   = destination.strip().lower() == "combination"
    results    = []

    for pkg in _load_all_packages().values():
        if pkg.get("travel_type", "").lower() != type_lower:
            continue

        state_list = pkg.get("state", [])

        if is_combo:
            # Combination: packages that span more than one distinct region
            regions_covered = sum(
                1 for region in ("ladakh", "himachal", "kashmir")
                if any(region in s.lower() for s in state_list)
            )
            if regions_covered < 2:
                continue
        else:
            dest_folder = pkg.get("dest_folder", "")
            if not (_dest_matches(state_list, destination) or destination.lower() in dest_folder.lower()):
                continue

        days = _parse_days(pkg.get("packageDuration", ""))
        if not _days_in_range(days, duration_range):
            continue

        results.append(pkg)

    results.sort(key=lambda p: _parse_days(p.get("packageDuration", "")))
    return results


def list_packages() -> list[dict]:
    return [
        {
            "id":          p["id"],
            "title":       p.get("title", ""),
            "duration":    p.get("packageDuration", ""),
            "states":      p.get("state", []),
            "travel_type": p.get("travel_type", ""),
            "highlights":  p.get("highlights", [])[:3],
            "description": p.get("description", ""),
        }
        for p in _load_all_packages().values()
    ]


def _short_state(s: str) -> str:
    """Normalise a full state name to a short display label."""
    sl = s.lower()
    if "ladakh" in sl:    return "Ladakh"
    if "himachal" in sl:  return "Himachal"
    if "kashmir" in sl or "jammu" in sl: return "Kashmir"
    return s.strip().title()


def _destination_key(states: list) -> str:
    """
    Build a display key from a list of states, deduplicating and joining.
      ["Ladakh"]                      → "Ladakh"
      ["Himachal Pradesh"]            → "Himachal"
      ["Ladakh", "Himachal"]          → "Ladakh & Himachal"
      ["Ladakh", "Jammu & Kashmir"]   → "Ladakh & Kashmir"
    """
    seen  = []
    for s in states:
        label = _short_state(s)
        if label not in seen:
            seen.append(label)
    return " & ".join(seen) if seen else "Unknown"


def get_packages_summary() -> dict:
    """
    Return a grouped summary for the compact catalog view.

    Structure:
    {
      "Bike": { "total": 19, "by_destination": { "Ladakh": 11, "Himachal": 2, ... } },
      "Car":  { "total": 2,  "by_destination": { "Ladakh": 2 } },
    }
    Values are package *counts* per destination key.
    """
    from collections import defaultdict

    summary: dict = {}

    for pkg in _load_all_packages().values():
        travel_type = pkg.get("travel_type", "Unknown")
        dest        = _destination_key(pkg.get("state", []))

        if travel_type not in summary:
            summary[travel_type] = {"total": 0, "by_destination": defaultdict(int)}

        summary[travel_type]["total"] += 1
        summary[travel_type]["by_destination"][dest] += 1

    # Convert defaultdicts → regular dicts, sort destinations alphabetically
    for type_data in summary.values():
        type_data["by_destination"] = dict(
            sorted(type_data["by_destination"].items())
        )

    return summary


def filter_packages(
    travel_type: str = "",
    destination: str = "",
    duration: str = "",
) -> list[dict]:
    """
    Flexible filter — any combination of travel_type / destination / duration.
    Returns packages sorted by day count ascending.
    """
    results = []
    for pkg in _load_all_packages().values():
        if travel_type and pkg.get("travel_type", "").lower() != travel_type.lower():
            continue
        if destination and not _dest_matches(pkg.get("state", []), destination):
            continue
        if duration:
            days = _parse_days(pkg.get("packageDuration", ""))
            if not _days_in_range(days, duration):
                continue
        results.append(pkg)

    results.sort(key=lambda p: _parse_days(p.get("packageDuration", "")))
    return results


def get_package_cost(package_id: str) -> dict | None:
    """
    Load cost data for a package from data/travel/{travel_type}/cost.json.

    Returns a dict with keys:
      "travel_type", "sheet_name", "itinerary_name", "duration", "pricing"
    or None if cost data is unavailable for this package.
    """
    pkg = get_package(package_id)
    if not pkg:
        return None

    travel_type = pkg.get("travel_type", "").lower()
    cost_file   = DATA_DIR / travel_type / "cost.json"
    if not cost_file.exists():
        return None

    with open(cost_file, encoding="utf-8") as f:
        cost_data = json.load(f)

    # Find the sheet_name by matching package_id in the index
    sheet_name = None
    for entry in cost_data.get("index_sheet", []):
        if entry.get("id", "").lower() == package_id.lower():
            sheet_name = entry.get("sheet_name")
            break

    if not sheet_name or sheet_name not in cost_data:
        return None

    sheet = cost_data[sheet_name]
    return {
        "travel_type":    travel_type,
        "sheet_name":     sheet_name,
        "itinerary_name": sheet.get("itinerary_name", ""),
        "duration":       sheet.get("duration", ""),
        "pricing":        sheet.get("pricing", []),
    }


def get_cost_categories(package_id: str) -> list[str]:
    """Return distinct categories available for a package (e.g. Standard, Deluxe, Premium)."""
    cost = get_package_cost(package_id)
    if not cost:
        return []
    seen = []
    for row in cost["pricing"]:
        cat = row.get("category", "")
        if cat and cat not in seen:
            seen.append(cat)
    return seen


def get_day_itinerary(package_id: str, day: int) -> dict | None:
    pkg = get_package(package_id)
    if not pkg:
        return None
    for entry in pkg.get("detailedItinerary", []):
        if entry["day"] == day:
            return entry
    return None


def search_packages(location: str | None = None) -> list[dict]:
    results = []
    for pkg in _load_all_packages().values():
        if location:
            states = [s.lower() for s in pkg.get("state", [])]
            title  = pkg.get("title", "").lower()
            if not any(location.lower() in s for s in states) and location.lower() not in title:
                continue
        results.append({
            "id":          pkg["id"],
            "title":       pkg.get("title", ""),
            "duration":    pkg.get("packageDuration", ""),
            "states":      pkg.get("state", []),
            "travel_type": pkg.get("travel_type", ""),
            "description": pkg.get("description", ""),
        })
    return results
