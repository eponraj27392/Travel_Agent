"""
Long-term memory store for TravelBot.

Uses AsyncPostgresStore (same travelbot_db as the checkpointer) to persist
user data across sessions. Three namespaces:

  ("user_profile",       user_id)  → name / email / phone
  ("travel_preferences", user_id)  → destination / travel_type / pax / month / interests / fitness
  ("recent_bookings",    user_id)  → lightweight booking summaries keyed by booking_id

user_id is taken from config["configurable"]["session_id"] at runtime.

All helpers are async and accept a BaseStore instance so they are easy to test
and mock; the module also holds a singleton (_store) initialised at app startup.
"""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langgraph.store.base import BaseStore

load_dotenv()

# Module-level singleton — set by init_store() at app startup
_store = None


async def init_store():
    """
    Async factory called once at startup (inside init_graph).
    Creates an AsyncPostgresStore backed by a connection pool,
    runs setup(), and saves the singleton.
    Returns None when DATABASE_URL is not set.
    """
    global _store

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("[memory_store] DATABASE_URL not set — long-term memory disabled")
        return None

    from psycopg_pool import AsyncConnectionPool
    from langgraph.store.postgres import AsyncPostgresStore

    # Build an explicit conninfo string so psycopg_pool uses the credentials
    # from DATABASE_URL rather than falling back to the OS user.
    from psycopg.conninfo import conninfo_to_dict
    params = conninfo_to_dict(DATABASE_URL)
    conninfo = " ".join(f"{k}={v}" for k, v in params.items() if v is not None)

    # ── Run store migrations manually with autocommit=True ───────────────────
    # AsyncPostgresStore.setup() wraps migrations in a transaction, which makes
    # CREATE INDEX CONCURRENTLY fail and rolls back the whole batch (including
    # the CREATE TABLE). We apply each migration individually on an autocommit
    # connection so CONCURRENTLY is always outside any transaction block.
    import psycopg
    from psycopg.rows import dict_row
    from langgraph.store.postgres import AsyncPostgresStore

    ac = await psycopg.AsyncConnection.connect(
        DATABASE_URL, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    try:
        # Ensure migration tracking table exists
        await ac.execute(
            "CREATE TABLE IF NOT EXISTS store_migrations (v INTEGER PRIMARY KEY)"
        )
        row = await (await ac.execute(
            "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
        )).fetchone()
        current = row["v"] if row else -1

        for v, sql in enumerate(
            AsyncPostgresStore.MIGRATIONS[current + 1:], start=current + 1
        ):
            try:
                await ac.execute(sql)
                await ac.execute(
                    "INSERT INTO store_migrations (v) VALUES (%s) ON CONFLICT DO NOTHING",
                    (v,),
                )
            except Exception as m_err:
                if "already exists" in str(m_err):
                    await ac.execute(
                        "INSERT INTO store_migrations (v) VALUES (%s) ON CONFLICT DO NOTHING",
                        (v,),
                    )
                else:
                    raise
    finally:
        await ac.close()

    # ── Create pool for ongoing store operations ──────────────────────────────
    pool = AsyncConnectionPool(conninfo, min_size=2, max_size=10, open=False)
    await pool.open()
    _store = AsyncPostgresStore(pool)
    print("[memory_store] AsyncPostgresStore initialised → long-term memory active")
    return _store


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# User profile  (name / email / phone)                                        #
# --------------------------------------------------------------------------- #

async def get_user_profile(store: BaseStore, user_id: str) -> dict | None:
    item = await store.aget(("user_profile", user_id), "profile")
    return item.value if item else None


async def upsert_user_profile(
    store: BaseStore, user_id: str, updates: dict
) -> None:
    """Merge-update the user profile; never erases existing keys."""
    existing = await get_user_profile(store, user_id) or {}
    merged   = {**existing, **{k: v for k, v in updates.items() if v}, "updated_at": _now()}
    await store.aput(("user_profile", user_id), "profile", merged)


# --------------------------------------------------------------------------- #
# Travel preferences  (destination / type / pax / month / interests / fitness)#
# --------------------------------------------------------------------------- #

async def get_travel_preferences(store: BaseStore, user_id: str) -> dict | None:
    item = await store.aget(("travel_preferences", user_id), "prefs")
    return item.value if item else None


async def upsert_travel_preferences(
    store: BaseStore, user_id: str, updates: dict
) -> None:
    """Merge-update travel preferences; never erases existing keys."""
    existing = await get_travel_preferences(store, user_id) or {}
    merged   = {**existing, **{k: v for k, v in updates.items() if v is not None}, "updated_at": _now()}
    await store.aput(("travel_preferences", user_id), "prefs", merged)


# --------------------------------------------------------------------------- #
# Recent bookings  (keyed by booking_id)                                      #
# --------------------------------------------------------------------------- #

async def save_booking_summary(
    store: BaseStore, user_id: str, booking_id: str, data: dict
) -> None:
    await store.aput(
        ("recent_bookings", user_id),
        booking_id,
        {**data, "booked_at": _now()},
    )
