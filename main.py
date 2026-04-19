"""
Tourist Chatbot — Streamlit UI
Connects to the FastAPI backend (api/main.py) via HTTP and WebSocket.

Run backend first:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Run this UI:        streamlit run main.py
"""
import json
import subprocess
import requests
import websocket
import streamlit as st
from tourist_agent.tracing import is_tracing_enabled

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #
API_BASE = "http://localhost:8000"
WS_BASE  = "ws://localhost:8000"

st.set_page_config(
    page_title="TravelBot — Himalayan Rovers",
    page_icon="🏔️",
    layout="centered",
)

# ------------------------------------------------------------------ #
# API helpers                                                          #
# ------------------------------------------------------------------ #

def api_create_session() -> str:
    """POST /sessions → returns a new session_id."""
    resp = requests.post(f"{API_BASE}/sessions", timeout=10)
    resp.raise_for_status()
    return resp.json()["session_id"]


_NODE_STATUS = {
    "assistant":          "🤔 Thinking...",
    "safe_tools":         "🔍 Searching packages...",
    "sensitive_tools":    "⚙️ Processing...",
    "sensitive_guard":    "🔒 Verifying action...",
    "itinerary_planner":  "🗺️ Planning itinerary...",
}


def ws_stream_reply(session_id: str, message: str):
    """
    Generator that connects to WS /chat/ws/{session_id}, sends the message,
    and yields typed dicts:
        {"type": "token",  "content": "..."}   — LLM token chunk
        {"type": "status", "text":    "..."}   — node status label
        {"type": "done"}                        — stream finished normally
        {"type": "interrupted", "content": "..."} — graph paused

    Sets st.session_state.ws_interrupted = True/False after stream ends.
    Sets st.session_state.ws_interrupt_msg when interrupted.
    """
    ws = websocket.WebSocket()
    ws.connect(f"{WS_BASE}/chat/ws/{session_id}")
    ws.send(message)
    st.session_state.ws_interrupted = False
    st.session_state.ws_interrupt_msg = None

    try:
        while True:
            raw = ws.recv()
            frame = json.loads(raw)
            ftype = frame.get("type")

            if ftype == "token":
                yield {"type": "token", "content": frame["content"]}
            elif ftype == "node":
                node_name = frame.get("node", "")
                status_text = _NODE_STATUS.get(node_name, "⚙️ Processing...")
                yield {"type": "status", "text": status_text}
            elif ftype == "interrupted":
                st.session_state.ws_interrupted = True
                st.session_state.ws_interrupt_msg = frame["content"]
                yield {"type": "interrupted", "content": frame["content"]}
                break
            elif ftype == "end":
                yield {"type": "done"}
                break
            elif ftype == "error":
                raise RuntimeError(frame["content"])
    finally:
        ws.close()


def api_clear_session(session_id: str):
    """DELETE /sessions/{id} → resets the session state."""
    requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=10)


def api_get_history(session_id: str) -> list[dict]:
    """GET /sessions/{id}/history → returns list of {role, content}."""
    resp = requests.get(f"{API_BASE}/sessions/{session_id}/history", timeout=10)
    resp.raise_for_status()
    return resp.json()["messages"]


def check_api() -> tuple[bool, str]:
    """Check if the FastAPI backend is reachable."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.ok:
            return True, "API server is running"
        return False, f"API returned {resp.status_code}"
    except requests.ConnectionError:
        return False, "API server not reachable. Run: uvicorn api.main:app --reload"
    except requests.Timeout:
        return False, "API server timed out."


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False, "Ollama not running. Run: `ollama serve`"
        if "qwen3.5:4b" not in result.stdout:
            return False, "Model not found. Run: `ollama pull qwen3.5:4b`"
        return True, "qwen3.5:4b is ready"
    except FileNotFoundError:
        return False, "Ollama not installed."
    except subprocess.TimeoutExpired:
        return False, "Ollama not responding. Run: `ollama serve`"


# ------------------------------------------------------------------ #
# Session state bootstrap                                              #
# ------------------------------------------------------------------ #
if "session_id" not in st.session_state:
    try:
        st.session_state.session_id = api_create_session()
    except Exception:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (role, content)


# ------------------------------------------------------------------ #
# Sidebar                                                              #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("🏔️ TravelBot")
    st.markdown("**Himalayan Rovers — Adventure Planner**")
    st.divider()

    # API status
    api_ok, api_msg = check_api()
    if api_ok:
        st.success(f"✅ {api_msg}")
    else:
        st.error(f"⚠️ {api_msg}")
        st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")

    # Ollama status
    ollama_ok, ollama_msg = check_ollama()
    if ollama_ok:
        st.success(f"✅ {ollama_msg}")
    else:
        st.warning(f"⚠️ {ollama_msg}")

    # LangFuse tracing status
    st.divider()
    if is_tracing_enabled():
        st.success("📊 LangFuse tracing ON")
    else:
        st.warning("📊 LangFuse tracing OFF")
        with st.expander("Enable tracing"):
            st.markdown(
                "Set these env vars to enable:\n"
                "```bash\n"
                "export LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
                "export LANGFUSE_SECRET_KEY=sk-lf-...\n"
                "# Free account: https://cloud.langfuse.com\n"
                "```"
            )

    # Quick question buttons
    st.divider()
    st.markdown("**Quick Questions**")
    quick_questions = [
        "Show me all available packages",
        "Itinerary Planner",
        "Cancellations",
        "I want to book a trip to Manali - Leh",
        "Feedback and Reviews",
        "Product Info",
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_input = q

    # Clear chat
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        api_clear_session(st.session_state.session_id)
        st.session_state.chat_history = []
        try:
            st.session_state.session_id = api_create_session()
        except Exception:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption("Powered by Ollama · qwen3.5:4b · LangGraph · FastAPI")
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ------------------------------------------------------------------ #
# Page header                                                          #
# ------------------------------------------------------------------ #
st.title("🏔️ TravelBot — Himalayan Rovers")
st.caption("Ask about packages, plan your itinerary, or book a trip!")

# ------------------------------------------------------------------ #
# Render conversation history                                          #
# ------------------------------------------------------------------ #
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# ------------------------------------------------------------------ #
# Handle input                                                         #
# ------------------------------------------------------------------ #
user_input = st.chat_input("Ask me anything about Himalayan Rovers...")

if "pending_input" in st.session_state and st.session_state.pending_input:
    user_input = st.session_state.pending_input
    del st.session_state.pending_input

if user_input:
    if not api_ok:
        st.error("API server is not running. Please start it first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream reply via WebSocket
    with st.chat_message("assistant"):
        status_ph = st.empty()   # live status line ("🤔 Thinking…")
        text_ph   = st.empty()   # accumulates streamed tokens
        full_reply = ""
        interrupted = False

        try:
            for item in ws_stream_reply(st.session_state.session_id, user_input):
                itype = item["type"]
                if itype == "status":
                    status_ph.caption(item["text"])
                elif itype == "token":
                    full_reply += item["content"]
                    text_ph.markdown(full_reply + "▌")
                elif itype == "interrupted":
                    full_reply += item["content"]
                    text_ph.markdown(full_reply)
                    interrupted = True
                elif itype == "done":
                    break

            status_ph.empty()                    # remove status line when done
            text_ph.markdown(full_reply)         # final render without cursor
        except Exception as e:
            status_ph.empty()
            full_reply = f"Could not reach the API server: {e}"
            text_ph.markdown(full_reply)

        reply = full_reply

    st.session_state.chat_history.append(("assistant", reply))
