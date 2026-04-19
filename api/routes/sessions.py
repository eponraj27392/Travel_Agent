import uuid
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import tourist_agent.graph as _graph_module


def _get_graph():
    return _graph_module.graph
from api.models import SessionResponse, HistoryResponse, MessageItem

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _make_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


@router.post("", response_model=SessionResponse, status_code=201)
def create_session():
    """Create a new chat session and return a unique session_id."""
    return SessionResponse(session_id=str(uuid.uuid4()))


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """
    Reset a session by clearing its checkpoint history.
    The session_id can be reused after this call.
    """
    config = _make_config(session_id)
    try:
        await _get_graph().aupdate_state(config, {"messages": []})
    except Exception:
        pass


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Return the full conversation history for a session."""
    config = _make_config(session_id)
    try:
        state = await _get_graph().aget_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append(MessageItem(role="user", content=msg.content or ""))
        elif isinstance(msg, AIMessage):
            content = msg.content or ""
            if content:
                messages.append(MessageItem(role="assistant", content=content))
        elif isinstance(msg, ToolMessage):
            messages.append(MessageItem(role="tool", content=msg.content or ""))

    return HistoryResponse(session_id=session_id, messages=messages)
