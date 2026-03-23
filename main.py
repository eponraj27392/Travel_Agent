"""
Tourist Chatbot — Streamlit UI (Ollama local model)
Run with:  streamlit run main.py
Requires:  ollama serve  (running in background)
           ollama pull qwen2.5:3b
"""
import uuid
import subprocess
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from tourist_agent.tracing import get_langfuse_handler, is_tracing_enabled

st.set_page_config(
    page_title="TravelBot — Manali Leh Adventure",
    page_icon="🏔️",
    layout="centered",
)


def get_graph():
    from tourist_agent.graph import graph
    return graph


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and qwen2.5:3b is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False, "Ollama is installed but not running. Run: `ollama serve`"
        if "qwen2.5:3b" not in result.stdout:
            return False, "Model not found. Run: `ollama pull qwen2.5:3b`"
        return True, "qwen2.5:3b is ready"
    except FileNotFoundError:
        return False, "Ollama not installed. Install from https://ollama.com then run: `ollama pull qwen2.5:3b`"
    except subprocess.TimeoutExpired:
        return False, "Ollama is not responding. Run: `ollama serve`"


# ------------------------------------------------------------------ #
# Session state                                                        #
# ------------------------------------------------------------------ #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Tracks whether graph is paused at a booking interrupt
if "interrupted" not in st.session_state:
    st.session_state.interrupted = False


# ------------------------------------------------------------------ #
# Sidebar                                                              #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("🏔️ TravelBot")
    st.markdown("**Manali → Leh Adventure Planner**")
    st.divider()

    # Ollama status
    ok, msg = check_ollama()
    if ok:
        st.success(f"✅ {msg}")
    else:
        st.error(f"⚠️ {msg}")
        st.code("# Setup commands\ncurl -fsSL https://ollama.com/install.sh | sh\nollama pull qwen2.5:3b\nollama serve")

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

    st.divider()
    st.markdown("**Quick Questions**")
    quick_questions = [
        "Show me available packages",
        "What is the full itinerary?",
        "Tell me about Day 3",
        "What happens at Pangong Lake?",
        "What should I pack for high altitude?",
        "I want to book a trip",
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_input = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption("Powered by Ollama · qwen2.5:3b · LangGraph · LangFuse")


# ------------------------------------------------------------------ #
# Page header                                                          #
# ------------------------------------------------------------------ #
st.title("🏔️ TravelBot — Manali Leh Chatbot")
st.caption("Ask about the package, plan your itinerary day-by-day, or book a trip — all offline!")

# ------------------------------------------------------------------ #
# Render conversation history                                          #
# ------------------------------------------------------------------ #
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# ------------------------------------------------------------------ #
# Handle input                                                         #
# ------------------------------------------------------------------ #
user_input = st.chat_input("Ask me anything about the Manali Leh tour...")

if "pending_input" in st.session_state and st.session_state.pending_input:
    user_input = st.session_state.pending_input
    del st.session_state.pending_input

if user_input:
    ok, msg = check_ollama()
    if not ok:
        st.error(f"Ollama not ready: {msg}")
        st.stop()

    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (first response may take ~30s on CPU)"):
            graph = get_graph()
            langfuse_handler = get_langfuse_handler(st.session_state.session_id)
            config = {
                "configurable": {"thread_id": st.session_state.session_id},
                "callbacks": [langfuse_handler] if langfuse_handler else [],
            }

            if st.session_state.interrupted:
                # Graph is paused at booking interrupt — resume with user's answer
                result = graph.invoke(Command(resume=user_input), config=config)
            else:
                # Normal invocation
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "intent": None,
                        "active_package_id": None,
                        "active_day": None,
                        "booking_draft": None,
                        "missing_fields": [],
                        "awaiting_confirmation": False,
                        "booking_id": None,
                    },
                    config=config,
                )

        # Check if graph hit a new interrupt (booking confirmation pending)
        if result.get("__interrupt__"):
            interrupt_msg = result["__interrupt__"][0].value
            st.session_state.interrupted = True
            reply = interrupt_msg
        else:
            st.session_state.interrupted = False
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            reply = ai_messages[-1].content if ai_messages else "Sorry, I couldn't process that."

        st.markdown(reply)

    st.session_state.chat_history.append(("assistant", reply))
