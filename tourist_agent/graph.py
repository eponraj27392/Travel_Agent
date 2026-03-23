"""
Build and compile the LangGraph tourist agent.
"""
import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.checkpoint.memory import MemorySaver  # in-memory, resets on app restart

DB_PATH = str(Path(__file__).parent.parent / "chat_sessions.db")

from tourist_agent.state import AgentState
from tourist_agent.nodes import (
    node_chatbot,
    node_tools,
    node_booking_extractor,
    route_after_chatbot,
)



def save_graph_diagram(output_dir: str = ".") -> dict[str, str]:
    """
    Save the graph flow diagram in two formats:
      - <output_dir>/graph.mmd   (Mermaid source, always works offline)
      - <output_dir>/graph.png   (PNG image, requires internet for mermaid.ink API)

    Returns a dict with saved file paths.
    """
    if not output_dir:
        output_dir = "hr_agents/output/graph_png"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}

    # 1. Mermaid text diagram (no dependencies needed)
    mermaid_text = graph.get_graph().draw_mermaid()
    mmd_path = out / "graph.mmd"
    mmd_path.write_text(mermaid_text)
    saved["mermaid"] = str(mmd_path)
    print(f"Mermaid diagram saved → {mmd_path}")

    # 2. PNG via mermaid.ink (needs internet connection)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path = out / "graph.png"
        png_path.write_bytes(png_bytes)
        saved["png"] = str(png_path)
        print(f"PNG diagram saved     → {png_path}")
    except Exception as e:
        print(f"PNG skipped ({e})")

    return saved



def build_graph():
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("chatbot", node_chatbot)
    builder.add_node("tools", node_tools)
    builder.add_node("booking_extractor", node_booking_extractor)

    # Entry point
    builder.add_edge(START, "chatbot")

    # After chatbot: route to tools, booking extractor, or end
    builder.add_conditional_edges(
        "chatbot",
        route_after_chatbot,
        {
            "tools": "tools",
            "booking_extractor": "booking_extractor",
            "__end__": END,
        },
    )

    # After tools: always return to chatbot so it can formulate a response
    builder.add_edge("tools", "chatbot")

    # After booking extraction: done
    builder.add_edge("booking_extractor", END)

    # SQLite checkpointer — persists all session state to disk
    # File: hr_agents/chat_sessions.db
    # Each session_id maps to its full conversation history, even after app restart
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    # checkpointer = MemorySaver()  # swap to this for in-memory (no persistence)
    return builder.compile(checkpointer=checkpointer)


# Singleton graph instance
graph = build_graph()


# save_graph_diagram('/home/esakki/test/hr_agents/output/graph_png')