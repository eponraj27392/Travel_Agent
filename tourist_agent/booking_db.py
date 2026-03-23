import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "bookings.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id   TEXT PRIMARY KEY,
                package_id   TEXT NOT NULL,
                lead_name    TEXT NOT NULL,
                email        TEXT NOT NULL,
                phone        TEXT NOT NULL,
                travel_date  TEXT NOT NULL,
                pax_count    INTEGER NOT NULL,
                special_req  TEXT,
                status       TEXT DEFAULT 'CONFIRMED',
                created_at   TEXT NOT NULL
            )
        """)
        conn.commit()


def create_booking(draft: dict) -> str:
    init_db()
    booking_id = f"TRV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO bookings
               (booking_id, package_id, lead_name, email, phone, travel_date, pax_count, special_req, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                booking_id,
                draft["package_id"],
                draft["lead_name"],
                draft["email"],
                draft["phone"],
                draft["travel_date"],
                draft["pax_count"],
                draft.get("special_requirements", ""),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    return booking_id


def get_booking(booking_id: str) -> dict | None:
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)
        ).fetchone()
    return dict(row) if row else None
