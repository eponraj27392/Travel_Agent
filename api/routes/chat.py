import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.types import Command

import tourist_agent.graph as _graph_module
from tourist_agent.tracing import get_langfuse_callback, langfuse_session
from api.models import ChatRequest, ChatResponse

# Suppress OpenTelemetry "Failed to detach context" warning that fires when
# the LangFuse CallbackHandler spans are cleaned up across async task boundaries.
# This is cosmetic — tracing data is captured correctly despite the log noise.
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)


def _get_graph():
    return _graph_module.graph


router = APIRouter(prefix="/chat", tags=["chat"])


def _make_config(session_id: str) -> dict:
    # NOTE: "session_id" in configurable is the long-term memory user_id key.
    # Currently hardcoded for single-user dev. Replace with real authenticated
    # user_id (e.g. JWT sub) once auth is added — all sessions for the same user
    # must share the same value so their profile/preferences persist across tabs.
    return {
        "configurable": {"thread_id": session_id, "session_id": "user-esakki-id"},
        "callbacks": get_langfuse_callback(),
        "metadata": {"langfuse_session_id": session_id},
    }


def _new_message_input(message: str) -> dict:
    """Fresh state for a new user message (resets all planner/intent fields)."""
    return {
        "messages": [HumanMessage(content=message)],
        "intent": None,
        "sensitive_confirmed": None,
        "planner_destination": None,
        "planner_travel_type": None,
        "planner_duration": None,
        "planner_pax": None,
        "planner_month": None,
        "planner_min_age": None,
        "planner_max_age": None,
        "planner_itinerary_type": None,
        "planner_matched_packages": None,
        "planner_selected_package_id": None,
        "planner_interests": None,
        "planner_fitness": None,
        "planner_special_req": None,
    }


async def _is_interrupted(session_id: str) -> bool:
    """Check if the graph is paused at an interrupt() for this session."""
    config = _make_config(session_id)
    try:
        state = await _get_graph().aget_state(config)
        return bool(state.next)
    except Exception:
        return False


# ── REST endpoint (kept for backward compatibility) ───────────────────────────

async def _invoke(session_id: str, message: str) -> ChatResponse:
    config = _make_config(session_id)
    graph = _get_graph()

    if await _is_interrupted(session_id):
        result = await graph.ainvoke(Command(resume=message), config=config)
    else:
        result = await graph.ainvoke(_new_message_input(message), config=config)

    # Graph paused at a new interrupt
    if result.get("__interrupt__"):
        interrupt_msg = result["__interrupt__"][0].value
        return ChatResponse(
            session_id=session_id,
            reply=interrupt_msg,
            interrupted=True,
            interrupt_message=interrupt_msg,
        )

    # Normal completion — return last non-empty AIMessage
    ai_messages = [
        m for m in result.get("messages", [])
        if isinstance(m, AIMessage) and m.content
    ]
    reply = ai_messages[-1].content if ai_messages else "Sorry, I could not process that."
    return ChatResponse(session_id=session_id, reply=reply, interrupted=False)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the travel agent.

    - Automatically resumes the graph if paused at an interrupt()
      (booking confirmation, itinerary planner Q&A, cancel trip).
    - Returns `interrupted: true` when the agent needs a follow-up answer
      before processing can continue.
    """
    return await _invoke(request.session_id, request.message)


def _extract_interrupt_value(state) -> str:
    """
    Pull the interrupt() message from a LangGraph StateSnapshot.

    When interrupt() is called inside a node, LangGraph stores the value in
    state.tasks[i].interrupts[j].value. Fall back to a generic prompt if the
    structure differs across LangGraph versions.
    """
    try:
        for task in state.tasks:
            interrupts = getattr(task, "interrupts", [])
            if interrupts:
                return str(interrupts[0].value)
    except Exception:
        pass
    return "Please provide your answer to continue."


# ── WebSocket endpoint (streaming) ────────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket chat endpoint. Streams LLM tokens to the client in real time.

    Client sends plain text messages; server sends JSON frames:
      {"type": "token",       "content": "..."}   — partial LLM token
      {"type": "node",        "node":    "..."}   — which graph node is running
      {"type": "end",         "content": "..."}   — full reply, stream complete
      {"type": "interrupted", "content": "..."}   — graph paused, waiting for answer
      {"type": "error",       "content": "..."}   — something went wrong

    The connection stays open for the full session — send another message to continue.
    """
    await websocket.accept()
    graph = _get_graph()

    try:
        while True:
            message = await websocket.receive_text()
            config = _make_config(session_id)

            stream_input = (
                Command(resume=message)
                if await _is_interrupted(session_id)
                else _new_message_input(message)
            )

            full_reply = ""

            try:
                async with langfuse_session(session_id):
                    async for stream_mode, data in graph.astream(
                        stream_input,
                        config=config,
                        stream_mode=["messages", "updates"],
                    ):
                        if stream_mode == "messages":
                            chunk, metadata = data
                            if isinstance(chunk, AIMessageChunk) and chunk.content:
                                full_reply += chunk.content
                                await websocket.send_json(
                                    {"type": "token", "content": chunk.content}
                                )

                        elif stream_mode == "updates":
                            # Send which node just ran (for UI status indicators)
                            for node_name in data:
                                if node_name != "__interrupt__":
                                    await websocket.send_json(
                                        {"type": "node", "node": node_name}
                                    )

            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})
                continue

            # Check for interrupt AFTER stream ends — interrupt() pauses the graph
            # and does NOT appear in astream updates; must be read from state.
            state = await graph.aget_state(config)
            if state.next:
                interrupt_msg = _extract_interrupt_value(state)
                await websocket.send_json(
                    {"type": "interrupted", "content": interrupt_msg}
                )
            else:
                await websocket.send_json({"type": "end", "content": full_reply})

    except WebSocketDisconnect:
        pass
