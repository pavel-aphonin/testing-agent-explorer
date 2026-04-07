"""Data models for the App Explorer graph."""

from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class ElementKind(StrEnum):
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    SWITCH = "switch"
    LABEL = "label"
    IMAGE = "image"
    LINK = "link"
    OTHER = "other"


class ElementSnapshot(BaseModel):
    """Immutable snapshot of a single UI element on a screen."""

    kind: ElementKind
    element_type: str  # raw iOS type (e.g. "Button", "TextField")
    label: str | None = None
    value: str | None = None
    test_id: str | None = None  # maps to label on iOS (via accessibilityIdentifier)
    enabled: bool = True
    frame: dict | None = None  # {x, y, width, height}
    bounds_str: str | None = None  # "[x1,y1][x2,y2]"

    def get_center(self) -> tuple[int, int] | None:
        """Return (x, y) center of the element."""
        if self.frame:
            x = int(self.frame.get("x", 0) + self.frame.get("width", 0) / 2)
            y = int(self.frame.get("y", 0) + self.frame.get("height", 0) / 2)
            return (x, y)
        return None

    def uid(self) -> str:
        """Unique identifier for this element within a screen."""
        return f"{self.element_type}:{self.label or ''}:{self.test_id or ''}"


class ScreenNode(BaseModel):
    """A unique screen state in the app graph."""

    screen_id: str
    name: str = ""
    elements: list[ElementSnapshot] = Field(default_factory=list)
    interactive_elements: list[ElementSnapshot] = Field(default_factory=list)
    screenshot_b64: str | None = None
    first_seen: datetime = Field(default_factory=datetime.now)
    visit_count: int = 0
    app_label: str | None = None  # Application element's label


class ActionType(StrEnum):
    TAP = "tap"
    INPUT = "input"
    SWIPE = "swipe"
    BACK = "back"
    LAUNCH = "launch"


class ActionDetail(BaseModel):
    """Full description of an action that causes a transition."""

    action_type: ActionType
    target_label: str | None = None
    target_test_id: str | None = None
    target_frame: dict | None = None
    input_text: str | None = None
    input_category: str | None = None  # "valid", "empty", "overflow", "xss", etc.

    def signature(self) -> str:
        """Short string for dedup: (type, label, test_id, frame_pos, input_category)."""
        parts = [self.action_type]
        parts.append(self.target_label or "")
        parts.append(self.target_test_id or "")
        # Include frame position to disambiguate unlabeled elements
        if self.target_frame:
            parts.append(f"{int(self.target_frame.get('x',0))},{int(self.target_frame.get('y',0))}")
        if self.input_category:
            parts.append(self.input_category)
        return "|".join(parts)


class GraphEdge(BaseModel):
    """A directed transition between two screen states."""

    edge_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_screen_id: str
    target_screen_id: str
    action: ActionDetail
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    notes: str = ""


class AppGraph(BaseModel):
    """Complete state machine graph of the explored app."""

    app_bundle_id: str
    nodes: dict[str, ScreenNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)
    exploration_start: datetime = Field(default_factory=datetime.now)
    exploration_end: datetime | None = None
    total_actions: int = 0

    def add_node(self, node: ScreenNode) -> bool:
        """Add a node. Returns True if this is a NEW screen."""
        if node.screen_id in self.nodes:
            self.nodes[node.screen_id].visit_count += 1
            return False
        self.nodes[node.screen_id] = node
        return True

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        self.total_actions += 1

    def get_explored_action_signatures(self, screen_id: str) -> set[str]:
        """Return set of action signatures already explored from this screen."""
        sigs = set()
        for e in self.edges:
            if e.source_screen_id == screen_id:
                sigs.add(e.action.signature())
        return sigs

    def get_unexplored_actions(
        self, screen_id: str
    ) -> list[ElementSnapshot]:
        """Return interactive elements on this screen that have no outgoing edge yet."""
        node = self.nodes.get(screen_id)
        if not node:
            return []

        explored_sigs = self.get_explored_action_signatures(screen_id)
        unexplored = []
        for el in node.interactive_elements:
            # Include frame position to disambiguate elements without labels
            frame_key = ""
            if el.frame:
                frame_key = f"{int(el.frame.get('x',0))},{int(el.frame.get('y',0))}"
            action_sig = f"tap|{el.label or ''}|{el.test_id or ''}|{frame_key}"
            if action_sig not in explored_sigs:
                unexplored.append(el)
        return unexplored

    def has_unexplored_screens(self) -> bool:
        """Any screen with interactive elements that lack outgoing edges?"""
        for sid in self.nodes:
            if self.get_unexplored_actions(sid):
                return True
        return False

    def get_screens_with_unexplored(self) -> set[str]:
        """Return set of screen IDs that have unexplored actions."""
        return {sid for sid in self.nodes if self.get_unexplored_actions(sid)}

    def find_path(self, from_id: str, to_id: str) -> list[GraphEdge] | None:
        """BFS shortest path between two screens. Returns list of edges or None."""
        if from_id == to_id:
            return []

        # Build adjacency: screen_id -> [(edge, target_screen_id)]
        adj: dict[str, list[tuple[GraphEdge, str]]] = {}
        for e in self.edges:
            if e.source_screen_id not in adj:
                adj[e.source_screen_id] = []
            adj[e.source_screen_id].append((e, e.target_screen_id))

        visited = {from_id}
        queue: deque[tuple[str, list[GraphEdge]]] = deque()
        queue.append((from_id, []))

        while queue:
            current, path = queue.popleft()
            for edge, target in adj.get(current, []):
                if target in visited:
                    continue
                new_path = path + [edge]
                if target == to_id:
                    return new_path
                visited.add(target)
                queue.append((target, new_path))

        return None  # no path found

    def save(self, path: str | Path) -> None:
        """Serialize graph to JSON file."""
        Path(path).write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> AppGraph:
        """Deserialize graph from JSON file."""
        data = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(data)

    def stats(self) -> str:
        """Return a short stats summary."""
        return (
            f"Screens: {len(self.nodes)}, "
            f"Edges: {len(self.edges)}, "
            f"Actions: {self.total_actions}"
        )
