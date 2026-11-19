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
    is_modal: bool = False  # a sheet or alert covers the rest of the screen


class ActionType(StrEnum):
    TAP = "tap"
    INPUT = "input"
    SWIPE = "swipe"
    BACK = "back"
    LAUNCH = "launch"


def _frame_key(frame: dict | None) -> str:
    if not frame:
        return ""
    return f"{int(frame.get('x', 0))},{int(frame.get('y', 0))}"


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


# Note on the edges that close a written-off screen's actions: bookkeeping
# for "nothing left here", not taps.
WRITTEN_OFF_NOTE = "Written off untried"
UNREACHABLE_NOTE = "Below the end of scrolling"
# Failed replays after which an edge is never used again (see GraphEdge.dead).
MAX_EDGE_FAILURES = 2


class GraphEdge(BaseModel):
    """A directed transition between two screen states."""

    edge_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_screen_id: str
    target_screen_id: str
    action: ActionDetail
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    notes: str = ""
    # Failed to replay: the element it taps is gone. One-shot screens leave
    # such edges behind (Wikipedia: «Enable location» answered once).
    stale: bool = False
    # Replays that failed on this edge since it last worked. At
    # MAX_EDGE_FAILURES it is dead: no path uses it, not even as the only
    # way (Wikipedia run 10: «Add some interests» was gone once the agent
    # had picked interests, and the edge failed 8 times in 4 trips).
    failures: int = 0

    @property
    def dead(self) -> bool:
        return self.failures >= MAX_EDGE_FAILURES

    def replay_failed(self) -> None:
        self.stale = True
        self.failures += 1

    def replay_worked(self) -> None:
        self.stale = False
        self.failures = 0

class AppGraph(BaseModel):
    """Complete state machine graph of the explored app."""

    app_bundle_id: str
    nodes: dict[str, ScreenNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)
    exploration_start: datetime = Field(default_factory=datetime.now)
    exploration_end: datetime | None = None
    total_actions: int = 0
    # Why exploration ended before the step budget, if it did.
    stop_reason: str | None = None
    # Screens the app opened on after a (re)launch, oldest first.
    launch_screens: list[str] = Field(default_factory=list)
    # Screens the Back button did not leave when tried (tab roots): not
    # tried again, a trip from them goes straight to a relaunch. Per run,
    # not saved with the graph.
    back_futile: set[str] = Field(default_factory=set, exclude=True)
    # The edge the last replay failed on (its tap did not work, or the step
    # before it landed off the path), for the engine to mark stale and route
    # around. Per run, not saved.
    failed_edge: GraphEdge | None = Field(default=None, exclude=True)

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

    @property
    def written_off_count(self) -> int:
        return sum(e.notes == WRITTEN_OFF_NOTE for e in self.edges)

    @property
    def unreachable_count(self) -> int:
        return sum(e.notes == UNREACHABLE_NOTE for e in self.edges)

    @property
    def transition_count(self) -> int:
        """Edges that stand for an action actually performed."""
        return len(self.edges) - self.written_off_count

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

        # An element is explored once ANY action on it left this screen.
        # Matching full signatures («tap|…» against «input|…|explored»)
        # never matched for text fields, so a filled form stayed
        # «unexplored» forever — testbed run 2 refilled the login form
        # 20 steps in a row. Marker edges recorded without a frame match
        # on label + test id.
        explored: set[tuple[str, str, str]] = set()
        for e in self.edges:
            if e.source_screen_id == screen_id:
                a = e.action
                explored.add((a.target_label or "", a.target_test_id or "", _frame_key(a.target_frame)))
        unexplored = []
        for el in node.interactive_elements:
            # Include frame position to disambiguate elements without labels
            key = (el.label or "", el.test_id or "", _frame_key(el.frame))
            if key in explored:
                continue
            if el.test_id and (key[0], key[1], "") in explored:
                continue
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

    def find_path(
        self, from_id: str, to_id: str, allow_stale: bool = True
    ) -> list[GraphEdge] | None:
        """BFS shortest path between two screens. Returns list of edges or None.

        Stale edges are avoided while any other way exists; with
        ``allow_stale=False`` they are never used (looking for an alternative
        to a route that just failed). Dead edges are never used."""
        path = self._find_path(from_id, to_id, avoid_stale=True)
        if path or not allow_stale:
            return path
        return self._find_path(from_id, to_id, avoid_stale=False)

    def reachable_from(self, starts: set[str], include_dead: bool = False) -> set[str]:
        """Screens some path of live (not dead, unless ``include_dead``)
        edges leads to from ``starts``, the starts included. Modal closing
        is not modelled: this is an upper bound, for telling apart screens
        no route can ever reach."""
        adj: dict[str, list[str]] = {}
        for e in self.edges:
            if include_dead or not e.dead:
                adj.setdefault(e.source_screen_id, []).append(e.target_screen_id)
        seen = set(starts)
        queue = deque(starts)
        while queue:
            for target in adj.get(queue.popleft(), []):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen

    def _find_path(self, from_id: str, to_id: str, avoid_stale: bool) -> list[GraphEdge] | None:
        if from_id == to_id:
            return []

        # Build adjacency: screen_id -> [(edge, target_screen_id)]
        adj: dict[str, list[tuple[GraphEdge, str]]] = {}
        for e in self.edges:
            if e.dead or (avoid_stale and e.stale):
                continue
            if e.source_screen_id not in adj:
                adj[e.source_screen_id] = []
            adj[e.source_screen_id].append((e, e.target_screen_id))

        # Closing a modal returns to whatever opened it, and one modal can
        # open from several screens: «sheet → history» is no way to get
        # from «benefits» to «history». Only a modal we are on can be left.
        opened = {(e.source_screen_id, e.target_screen_id) for e in self.edges}

        def closes_modal(source: str, target: str) -> bool:
            node = self.nodes.get(source)
            return source != from_id and node is not None and node.is_modal and (target, source) in opened

        visited = {from_id}
        queue: deque[tuple[str, list[GraphEdge]]] = deque()
        queue.append((from_id, []))

        while queue:
            current, path = queue.popleft()
            for edge, target in adj.get(current, []):
                if target in visited or closes_modal(current, target):
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
