"""
LangFuse tracing integration.

Setup (one-time):
  Sign up free at https://cloud.langfuse.com  OR  self-host with Docker.
  Then set these env vars (add to .env):

    LANGFUSE_PUBLIC_KEY="pk-lf-..."
    LANGFUSE_SECRET_KEY="sk-lf-..."
    LANGFUSE_BASE_URL="https://cloud.langfuse.com"

If keys are not set, tracing is silently disabled.

Usage:
  Call `init_tracing()` once at startup.
  Wrap each graph invocation with `langfuse_session(session_id)` context manager
  so all observations (including subgraph nodes) are grouped under one session.
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()


def is_tracing_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def init_tracing():
    """
    Verify Langfuse credentials at startup and print status.
    """
    if not is_tracing_enabled():
        print("[LangFuse] Tracing disabled — keys not set.")
        return

    try:
        from langfuse import get_client
        get_client().auth_check()
        print("[LangFuse] Tracing enabled.")
    except Exception as e:
        print(f"[LangFuse] Failed to initialise tracing: {e}")


def get_langfuse_callback() -> list:
    """
    Returns a plain CallbackHandler for the graph config callbacks list.
    Session grouping is handled separately via langfuse_session().
    """
    if not is_tracing_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []


def log_planner_step(name: str, input_data: dict, output_data: dict) -> None:
    """
    Log a planner node's collected data as a Langfuse event.
    Works because propagate_attributes(session_id=...) is active on the
    calling thread — the event inherits session_id automatically.
    Call this at the END of each planner node after all interrupts resolve.
    """
    if not is_tracing_enabled():
        return
    try:
        from langfuse import get_client
        get_client().create_event(
            name=f"planner:{name}",
            input=input_data,
            output=output_data,
        )
    except Exception:
        pass


@asynccontextmanager
async def langfuse_session(session_id: str):
    """
    Async context manager that groups all LLM observations for one WebSocket
    message under a Langfuse session.

    Pattern (from Langfuse docs): create a root span first so there IS an
    active span when propagate_attributes runs — otherwise session_id has
    nowhere to attach and traces appear with no session in the UI.

    Usage:
        async with langfuse_session(session_id):
            await graph.astream(...)
    """
    if not is_tracing_enabled():
        yield
        return

    try:
        from langfuse import propagate_attributes
        async with propagate_attributes(session_id=session_id):
            yield
    except Exception:
        yield
