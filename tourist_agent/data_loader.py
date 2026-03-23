import json
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_packages() -> dict:
    """Load all package JSON files from the data directory into a dict keyed by package id."""
    catalog = {}
    for file in DATA_DIR.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
        itinerary = data.get("itinerary", {})
        pkg_id = itinerary.get("id")
        if pkg_id:
            catalog[pkg_id] = itinerary
    return catalog


def get_package(package_id: str) -> dict | None:
    return load_packages().get(package_id)


def list_packages() -> list[dict]:
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "duration": p["packageDuration"],
            "states": p.get("state", []),
            "highlights": p.get("highlights", [])[:3],
            "description": p.get("description", ""),
        }
        for p in load_packages().values()
    ]


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
    for pkg in load_packages().values():
        states = [s.lower() for s in pkg.get("state", [])]
        if location and location.lower() not in states:
            title = pkg.get("title", "").lower()
            if location.lower() not in title:
                continue
        results.append({
            "id": pkg["id"],
            "title": pkg["title"],
            "duration": pkg["packageDuration"],
            "states": pkg.get("state", []),
            "description": pkg.get("description", ""),
        })
    return results
