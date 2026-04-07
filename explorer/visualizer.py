"""Export app graph to various formats: JSON, Mermaid, stats."""

from __future__ import annotations

from pathlib import Path

from explorer.models import AppGraph


def generate_mermaid(graph: AppGraph) -> str:
    """Generate a Mermaid state diagram from the app graph."""
    lines = ["stateDiagram-v2"]

    # Create safe node names
    name_map: dict[str, str] = {}
    used_names: set[str] = set()
    for node_id, node in graph.nodes.items():
        raw = node.name or node_id[:8]
        safe = raw.replace(" ", "_").replace("-", "_")
        # Deduplicate
        base = safe
        counter = 1
        while safe in used_names:
            safe = f"{base}_{counter}"
            counter += 1
        used_names.add(safe)
        name_map[node_id] = safe
        lines.append(f'    {safe}: {node.name or node_id[:8]}')

    # Mark the first-seen screen as start
    if graph.nodes:
        first_screen = min(graph.nodes.values(), key=lambda n: n.first_seen)
        lines.append(f"    [*] --> {name_map[first_screen.screen_id]}")

    # Deduplicate edges by (source, target, label)
    seen: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        src = name_map.get(edge.source_screen_id)
        tgt = name_map.get(edge.target_screen_id)
        if not src or not tgt:
            continue

        label = edge.action.target_label or edge.action.action_type
        if edge.action.input_category:
            label += f" ({edge.action.input_category})"

        key = (src, tgt, label)
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"    {src} --> {tgt}: {label}")

    return "\n".join(lines)


def generate_stats_report(graph: AppGraph) -> str:
    """Generate a human-readable stats report."""
    lines = [
        "=== App Explorer Report ===",
        f"App: {graph.app_bundle_id}",
        f"Screens discovered: {len(graph.nodes)}",
        f"Transitions recorded: {len(graph.edges)}",
        f"Total actions: {graph.total_actions}",
        "",
        "--- Screens ---",
    ]

    for sid, node in graph.nodes.items():
        lines.append(
            f"  [{sid[:8]}] {node.name or '(unnamed)'} "
            f"({len(node.interactive_elements)} interactive, "
            f"visited {node.visit_count}x)"
        )

    lines.append("")
    lines.append("--- Transitions ---")
    for edge in graph.edges:
        src_name = graph.nodes.get(edge.source_screen_id, None)
        tgt_name = graph.nodes.get(edge.target_screen_id, None)
        src = src_name.name if src_name else edge.source_screen_id[:8]
        tgt = tgt_name.name if tgt_name else edge.target_screen_id[:8]
        action = edge.action.target_label or edge.action.action_type
        if edge.action.input_category:
            action += f" [{edge.action.input_category}]"
        status = "OK" if edge.success else "FAIL"
        lines.append(f"  {src} --({action})--> {tgt} [{status}]")

    return "\n".join(lines)


def export_all(graph: AppGraph, output_dir: str | Path) -> None:
    """Export graph to all formats in the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON graph
    graph.save(out / "graph.json")
    print(f"  graph.json saved ({len(graph.nodes)} screens, {len(graph.edges)} edges)")

    # Mermaid diagram
    mermaid = generate_mermaid(graph)
    (out / "diagram.mmd").write_text(mermaid, encoding="utf-8")
    print(f"  diagram.mmd saved")

    # Stats report
    report = generate_stats_report(graph)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(f"  report.txt saved")

    print(f"\nAll outputs in: {out.resolve()}")
