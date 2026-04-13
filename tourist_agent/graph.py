"""
Build and compile the LangGraph tourist agent.

Graph layout:
  START → assistant → route_after_assistant
    → safe_tools        → assistant          (safe tool calls, no confirmation)
    → sensitive_guard   → route_after_guard
        → sensitive_tools → assistant        (confirmed sensitive action)
        → assistant                          (cancelled sensitive action)
    → itinerary_planner → assistant          (subagent: guided trip planner)
    → END

Checkpointer:
  - Uses AsyncPostgresSaver when DATABASE_URL is set (must be awaited at startup).
  - Falls back to AsyncSqliteSaver when DATABASE_URL is not set.
  - Call `await init_graph()` once at app startup to build the singleton.
  - Call `await cleanup_graph()` at app shutdown to release connections.
"""
import os
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

DB_PATH = str(Path(__file__).parent.parent / "chat_sessions.db")
DATABASE_URL = os.getenv("DATABASE_URL")

from tourist_agent.state import AgentState
from tourist_agent.nodes import (
    node_assistant,
    node_safe_tools,
    node_sensitive_guard,
    node_sensitive_tools,
    route_after_assistant,
    route_after_guard,
)
from tourist_agent.planner.graph import planner_graph

# Singletons — populated by init_graph() at startup
graph      = None
_exit_stack = AsyncExitStack()   # manages Postgres connection lifetime


def _build_compiled_graph(checkpointer, store=None):
    """Build and compile the StateGraph with the given checkpointer and optional store."""
    builder = StateGraph(AgentState)

    builder.add_node("assistant", node_assistant)
    builder.add_node("safe_tools", node_safe_tools)
    builder.add_node("sensitive_guard", node_sensitive_guard)
    builder.add_node("sensitive_tools", node_sensitive_tools)
    builder.add_node("itinerary_planner", planner_graph)

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {
            "safe_tools": "safe_tools",
            "sensitive_guard": "sensitive_guard",
            "itinerary_planner": "itinerary_planner",
            "__end__": END,
        },
    )

    builder.add_edge("safe_tools", "assistant")

    builder.add_conditional_edges(
        "sensitive_guard",
        route_after_guard,
        {
            "sensitive_tools": "sensitive_tools",
            "assistant": "assistant",
        },
    )

    builder.add_edge("sensitive_tools", "assistant")
    builder.add_edge("itinerary_planner", "assistant")

    return builder.compile(checkpointer=checkpointer, store=store)


async def init_graph():
    """
    Async factory — must be awaited once at application startup.
    Initialises the checkpointer, store, and builds the graph singleton.
    """
    global graph

    if DATABASE_URL:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            conn = await psycopg.AsyncConnection.connect(
                DATABASE_URL,
                autocommit=True,
                prepare_threshold=0,
                row_factory=dict_row,
            )
            checkpointer = AsyncPostgresSaver(conn)

            # setup() creates tables + indexes. CREATE INDEX CONCURRENTLY can fail
            # if the version wraps DDL in a transaction; tables still get created,
            # so we catch that specific error and continue.
            try:
                await checkpointer.setup()
            except Exception as setup_err:
                err_str = str(setup_err)
                if "CONCURRENTLY" in err_str or "already exists" in err_str:
                    print(f"[checkpointer] setup note: {setup_err} (continuing)")
                else:
                    raise

            from tourist_agent.memory_store import init_store
            store = await init_store()

            graph = _build_compiled_graph(checkpointer, store=store)
            print(f"[checkpointer] PostgreSQL (async) → {DATABASE_URL}")
            return graph

        except Exception as e:
            print(f"[checkpointer] PostgreSQL failed ({e}), falling back to AsyncSqlite")

    # SQLite fallback — long-term memory store not available without Postgres
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(DB_PATH)
    checkpointer = AsyncSqliteSaver(conn)
    graph = _build_compiled_graph(checkpointer)
    print(f"[checkpointer] AsyncSQLite → {DB_PATH}")
    return graph


async def cleanup_graph():
    """Release Postgres connections. Call from app shutdown lifespan."""
    await _exit_stack.aclose()
