"""
LangFuse tracing integration.

Setup (one-time):
  Sign up free at https://cloud.langfuse.com  OR  self-host with Docker.
  Then set these env vars (add to ~/.bashrc or .env):

    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"   # or http://localhost:3000 for self-hosted

If keys are not set, tracing is silently disabled.
"""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
def get_langfuse_handler(session_id: str, user_id: Optional[str] = None):
    """
    Returns a LangFuse CallbackHandler if credentials are configured,
    otherwise returns None (tracing disabled).
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler
        # v4: credentials are read from env vars automatically (LANGFUSE_PUBLIC_KEY,
        # LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL). Only trace_context is passed here.
        # LangFuse v4 requires trace_id as 32 lowercase hex chars (no hyphens)
        trace_id = session_id.replace("-", "")
        handler = CallbackHandler(
            trace_context={"trace_id": trace_id}
        )
        return handler
    except Exception as e:
        print(f"[LangFuse] Failed to create handler: {e}")
        return None


def is_tracing_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )
