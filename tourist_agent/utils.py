from pathlib import Path


def save_graph_diagram(compiled_graph, output_path: str = ".") -> dict[str, str]:
    """
    Save a compiled LangGraph diagram in two formats:
      - <output_path>.mmd   (Mermaid source, always works offline)
      - <output_path>.png   (PNG image, requires internet for mermaid.ink API)

    Args:
        compiled_graph: the compiled LangGraph graph instance
        output_path:    path prefix for output files (without extension),
                        or a directory path (uses 'graph' as filename stem)
    """
    out = Path(output_path)

    # If output_path looks like a directory (no suffix), use 'graph' as stem
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        stem = out / "graph"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("")

    saved = {}

    # Mermaid text (no dependencies)
    mermaid_text = compiled_graph.get_graph().draw_mermaid()
    mmd_path = stem.with_suffix(".mmd")
    mmd_path.write_text(mermaid_text)
    saved["mermaid"] = str(mmd_path)
    print(f"Mermaid diagram saved → {mmd_path}")

    # PNG via mermaid.ink (needs internet)
    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        png_path = stem.with_suffix(".png")
        png_path.write_bytes(png_bytes)
        saved["png"] = str(png_path)
        print(f"PNG diagram saved     → {png_path}")
    except Exception as e:
        print(f"PNG skipped ({e})")

    return saved
