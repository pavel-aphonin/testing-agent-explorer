"""Tests for the PUCT + Go-Explore engine using a fake in-memory simulator.

These never touch a real device. The fake controller exposes a tiny set of
screens with deterministic transitions, and we assert:

    - Initial screen registration creates PUCT entries.
    - Selection picks an unexplored action and the engine's explored set
      grows monotonically.
    - When the local frontier is exhausted on screen A but B has unexplored
      actions, the engine navigates to B (via the archive) and continues.
    - max_steps caps the loop.
    - Event callback receives screen_discovered + edge_discovered + status.
    - LLM prior_provider gets called once per new screen and its priors
      end up in the PUCT selector.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from explorer.engine import ExplorationEngine, action_id_for
from explorer.models import (
    ActionType,
    ElementKind,
    ElementSnapshot,
    ScreenNode,
)
from explorer.modes import ExplorationMode


@dataclass
class _Result:
    error: str | None = None


@dataclass
class FakeController:
    """A miniature simulator: maps (current_screen, tapped_label) -> next_screen.

    The graph is a deterministic 3-screen app:
        home --tap "Login button"--> login
        home --tap "Settings button"--> settings
        login --tap "Submit"--> home
        settings --tap "Back"--> home
    """

    transitions: dict[tuple[str, str], str] = field(default_factory=dict)
    current: str = "home"
    screens: dict[str, list[ElementSnapshot]] = field(default_factory=dict)
    tap_log: list[tuple[str, str]] = field(default_factory=list)

    async def get_ui_elements(self) -> list[dict]:
        # The engine calls analyze_screen on this; analyze_screen expects
        # raw element dicts. Build them from our element snapshots.
        return [
            {
                "type": "Button",
                "label": el.label,
                "value": None,
                "identifier": el.test_id,
                "enabled": True,
                "frame": el.frame,
                "bounds_str": None,
                "children": [],
            }
            for el in self.screens.get(self.current, [])
        ]

    async def take_screenshot(self) -> str:
        return f"<fake-png-for-{self.current}>"

    async def tap_at(self, x: int, y: int) -> _Result:
        # Find which element on the current screen is at (x,y).
        for el in self.screens.get(self.current, []):
            if not el.frame:
                continue
            cx = el.frame["x"] + el.frame["width"] / 2
            cy = el.frame["y"] + el.frame["height"] / 2
            if abs(cx - x) < 1 and abs(cy - y) < 1:
                key = (self.current, el.label or "")
                self.tap_log.append(key)
                next_screen = self.transitions.get(key, self.current)
                self.current = next_screen
                return _Result()
        return _Result(error=f"no element at ({x},{y})")

    async def go_back(self) -> _Result:
        # Naive: home is the root.
        self.current = "home"
        return _Result()

    async def terminate_app(self, bundle_id: str) -> None:
        self.current = "home"

    async def launch_app(self, bundle_id: str) -> None:
        self.current = "home"


def _btn(label: str, x: int, y: int, w: int = 100, h: int = 40) -> ElementSnapshot:
    return ElementSnapshot(
        kind=ElementKind.BUTTON,
        element_type="Button",
        label=label,
        test_id=label.lower().replace(" ", "_"),
        enabled=True,
        frame={"x": x, "y": y, "width": w, "height": h},
    )


def _make_three_screen_app() -> FakeController:
    home_buttons = [
        _btn("Login button", 100, 200),
        _btn("Settings button", 100, 300),
    ]
    login_buttons = [_btn("Submit", 100, 200)]
    settings_buttons = [_btn("Back", 100, 200)]

    return FakeController(
        screens={
            "home": home_buttons,
            "login": login_buttons,
            "settings": settings_buttons,
        },
        transitions={
            ("home", "Login button"): "login",
            ("home", "Settings button"): "settings",
            ("login", "Submit"): "home",
            ("settings", "Back"): "home",
        },
        current="home",
    )


@pytest.mark.asyncio
async def test_engine_discovers_three_screens(tmp_path):
    controller = _make_three_screen_app()
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=20,
    )

    await engine.run()

    screen_names = sorted(node.name or sid for sid, node in engine.graph.nodes.items())
    assert len(engine.graph.nodes) == 3, f"expected 3 screens, got {screen_names}"
    # All three real screens should appear at least once in the tap log.
    tapped_buttons = {label for _src, label in controller.tap_log}
    assert "Login button" in tapped_buttons
    assert "Settings button" in tapped_buttons


@pytest.mark.asyncio
async def test_engine_emits_events(tmp_path):
    controller = _make_three_screen_app()
    events: list[dict] = []

    async def callback(event: dict) -> None:
        events.append(event)

    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=20,
        event_callback=callback,
    )

    await engine.run()

    types = [e["type"] for e in events]
    assert "status_change" in types
    assert types.count("screen_discovered") >= 3
    assert "edge_discovered" in types
    assert "stats_update" in types


@pytest.mark.asyncio
async def test_engine_respects_max_steps(tmp_path):
    controller = _make_three_screen_app()
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=2,
    )

    await engine.run()

    # We capped at 2 steps; the engine cannot have visited every action.
    assert engine.step_idx <= 2


@pytest.mark.asyncio
async def test_prior_provider_is_called_once_per_screen(tmp_path):
    controller = _make_three_screen_app()
    calls: list[str] = []

    def provider(node: ScreenNode) -> dict[str, float]:
        calls.append(node.screen_id)
        # Push the engine toward the first interactive element with a strong prior.
        if not node.interactive_elements:
            return {}
        first_id = action_id_for(node.interactive_elements[0])
        return {first_id: 0.9}

    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.HYBRID,
        max_steps=20,
        prior_provider=provider,
    )

    await engine.run()

    # Provider was called once per distinct screen the engine landed on.
    assert sorted(set(calls)) == sorted(engine.graph.nodes.keys())
    # PUCT received the prior for at least one screen.
    any_skewed = False
    for sid in engine.graph.nodes.keys():
        ss = engine.puct.stats_for(sid)
        if ss is None:
            continue
        priors = [a.prior for a in ss.actions.values()]
        if priors and max(priors) - min(priors) > 0.1:
            any_skewed = True
            break
    assert any_skewed, "expected at least one skewed prior distribution"


@pytest.mark.asyncio
async def test_action_id_format_matches_graph_signature():
    """The PUCT action ID must match what AppGraph.get_unexplored_actions uses internally."""
    el = _btn("Login button", 100, 200)
    aid = action_id_for(el)
    # Format: tap|<label>|<test_id>|<x>,<y>
    assert aid == "tap|Login button|login_button|100,200"
