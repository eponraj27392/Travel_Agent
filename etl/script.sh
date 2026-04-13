#!/usr/bin/env bash
# ETL runner — converts all cost_*.xlsx files to data/travel/{type}/cost.json
#
# Usage:
#   bash etl/script.sh                  # process all files in etl/cost_excel/
#   bash etl/script.sh cost_bike.xlsx   # process one specific file
#
# Must be run from the project root:
#   cd /home/esakki1/projects/Travel_Agent && bash etl/script.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXCEL_DIR="$SCRIPT_DIR/cost_excel"
CONVERTER="$SCRIPT_DIR/convert_cost.py"

echo "========================================"
echo " Travel Agent — Cost ETL"
echo "========================================"
echo "Excel source : $EXCEL_DIR"
echo "Project root : $PROJECT_ROOT"
echo ""

# ── resolve which files to process ───────────────────────────────────────────
if [ "$#" -ge 1 ]; then
    # Single file passed as argument
    FILES=("$EXCEL_DIR/$1")
else
    # Process all cost_*.xlsx files in the excel directory
    mapfile -t FILES < <(find "$EXCEL_DIR" -maxdepth 1 -name "cost_*.xlsx" | sort)
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No cost_*.xlsx files found in $EXCEL_DIR"
    exit 1
fi

echo "Files to process: ${#FILES[@]}"
for f in "${FILES[@]}"; do
    echo "  - $(basename "$f")"
done
echo ""

# ── run converter for each file ───────────────────────────────────────────────
PASS=0
FAIL=0

for xlsx in "${FILES[@]}"; do
    if [ ! -f "$xlsx" ]; then
        echo "[SKIP] File not found: $xlsx"
        ((FAIL++)) || true
        continue
    fi

    echo "----------------------------------------"
    if uv run python "$CONVERTER" "$xlsx"; then
        ((PASS++)) || true
    else
        echo "[ERROR] Failed to convert: $(basename "$xlsx")"
        ((FAIL++)) || true
    fi
done

# ── summary ───────────────────────────────────────────────────────────────────
echo "========================================"
echo " Done: $PASS succeeded, $FAIL failed"
echo "========================================"
