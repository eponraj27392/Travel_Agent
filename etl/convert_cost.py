"""
Generic cost Excel → JSON converter.

Usage:
    python etl/convert_cost.py etl/cost_excel/cost_bike.xlsx

Derives travel type from filename:
    cost_bike.xlsx  →  travel_type = bike
    cost_car.xlsx   →  travel_type = car
    cost_trek.xlsx  →  travel_type = trek

Output:
    data/travel/{travel_type}/cost.json

JSON structure:
{
  "travel_type": "bike",
  "index_sheet": [
    {"detailed_itinerary": 1, "title": "...", "id": "...", "sheet_name": "..."},
    ...
  ],
  "<sheet_name>": {
    "itinerary_name": "...",
    "duration": "...",
    "pricing": [
      {
        "category": "Standard",
        "car_model": "Ertiga + Scorpio",
        "hotels": "Standard",
        "persons": "2 Person",
        "price_per_person": 16000,
        "room_split": "one double room"
      },
      ...
    ]
  },
  ...
}
"""
import json
import math
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

# ── resolve project root (two levels up from etl/) ───────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent


def _derive_travel_type(xlsx_path: Path) -> str:
    """
    Extract travel type from filename.
      cost_bike.xlsx  → 'bike'
      cost_car.xlsx   → 'car'
    Falls back to the stem if pattern doesn't match.
    """
    m = re.match(r"cost_(.+)\.xlsx", xlsx_path.name, re.IGNORECASE)
    return m.group(1).lower() if m else xlsx_path.stem.lower()


def _output_path(travel_type: str) -> Path:
    return PROJECT_ROOT / "data" / "travel" / travel_type / "cost.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(value):
    """Convert pandas NaN / float integers to Python-native types."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value == int(value):
            return int(value)
        return value
    return value


def _parse_index_sheet(df_raw: pd.DataFrame) -> list[dict]:
    header = [str(v).strip() for v in df_raw.iloc[0]]
    records = []
    for _, row in df_raw.iloc[1:].iterrows():
        entry = {header[i]: _clean(row.iloc[i]) for i in range(len(header))}
        if all(v is None for v in entry.values()):
            continue
        records.append(entry)
    return records


def _detect_price_and_room_cols(header: list[str]) -> tuple[int, int]:
    """
    Two layouts:
      8-col: [..., No. of Packs, Price exclude of GST..., Room Split]
      9-col: [..., No. of Packs, Per Person, Price..., room split]
    Price is always the column immediately after 'No. of Packs'.
    Room split is always the last column.
    """
    packs_idx = next(
        (i for i, h in enumerate(header) if "packs" in h.lower()), 5
    )
    return packs_idx + 1, len(header) - 1


def _parse_package_sheet(df_raw: pd.DataFrame) -> dict:
    header = [str(v).strip() for v in df_raw.iloc[0]]
    data   = df_raw.iloc[1:].copy().reset_index(drop=True)
    data   = data.ffill().infer_objects(copy=False)

    itinerary_name = _clean(data.iloc[0, 0]) if not data.empty else None
    duration       = _clean(data.iloc[0, 1]) if not data.empty else None
    price_idx, room_idx = _detect_price_and_room_cols(header)

    pricing = []
    for _, row in data.iterrows():
        price = _clean(row.iloc[price_idx])
        if price is None or price == 0:
            continue
        pricing.append({
            "category":         str(_clean(row.iloc[2]) or "").strip(),
            "car_model":        str(_clean(row.iloc[3]) or "").strip(),
            "hotels":           str(_clean(row.iloc[4]) or "").strip(),
            "persons":          str(_clean(row.iloc[5]) or "").strip(),
            "price_per_person": price,
            "room_split":       str(_clean(row.iloc[room_idx]) or "").strip(),
        })

    return {"itinerary_name": itinerary_name, "duration": duration, "pricing": pricing}


# ── main ─────────────────────────────────────────────────────────────────────

def convert(xlsx_path: Path):
    travel_type = _derive_travel_type(xlsx_path)
    output      = _output_path(travel_type)

    print(f"[ETL] Input  : {xlsx_path}")
    print(f"[ETL] Type   : {travel_type}")
    print(f"[ETL] Output : {output}")

    xl          = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names
    print(f"[ETL] Sheets : {len(sheet_names)} found\n")

    result = {"travel_type": travel_type}

    for name in sheet_names:
        df_raw = xl.parse(name, header=None)
        if name == "index_sheet":
            result["index_sheet"] = _parse_index_sheet(df_raw)
            print(f"  [index_sheet] → {len(result['index_sheet'])} entries")
        else:
            parsed = _parse_package_sheet(df_raw)
            result[name] = parsed
            print(
                f"  [{name}] → {parsed['itinerary_name']!r}, "
                f"{parsed['duration']!r}, "
                f"{len(parsed['pricing'])} pricing rows"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[ETL] Saved → {output}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python etl/convert_cost.py <path/to/cost_<type>.xlsx>")
        sys.exit(1)
    convert(Path(sys.argv[1]))
