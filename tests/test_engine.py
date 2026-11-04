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


# ----------------------------------------------------------------- PER-207


@dataclass
class _DeadEndController:
    """A simulator stuck on ONE screen whose every tap is a self-loop.

    Models run bbbb2222's tail: the agent circles a screen that opens no
    new screens. Tapping records a self-loop edge, so the local frontier
    empties; the engine then keeps asking Go-Explore for a frontier,
    navigating, and re-landing on the same screen — never growing the
    graph. Without the watchdog this circles until max_steps; with it,
    the run aborts ~MAX_NO_PROGRESS_STEPS in.
    """

    current: str = "stuck"
    tap_log: list = field(default_factory=list)

    async def get_ui_elements(self) -> list[dict]:
        # Two buttons, neither of which transitions anywhere.
        return [
            {
                "type": "Button", "label": "Войти", "value": None,
                "identifier": "voiti", "enabled": True,
                "frame": {"x": 100, "y": 200, "width": 100, "height": 40},
                "children": [],
            },
            {
                "type": "Button", "label": "Стать клиентом", "value": None,
                "identifier": "become", "enabled": True,
                "frame": {"x": 100, "y": 300, "width": 100, "height": 40},
                "children": [],
            },
        ]

    async def take_screenshot(self) -> str:
        return "<fake-png-stuck>"

    async def tap_at(self, x: int, y: int) -> _Result:
        self.tap_log.append((x, y))
        return _Result()  # ok, but screen never changes

    async def go_back(self) -> _Result:
        return _Result()

    async def terminate_app(self, bundle_id: str) -> None:
        pass

    async def launch_app(self, bundle_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_engine_aborts_on_dead_end_loop(tmp_path):
    """PER-207: a single-screen dead end aborts well before max_steps
    instead of circling the exhausted frontier to the budget limit
    (run bbbb2222's failure mode)."""
    from explorer.engine import MAX_NO_PROGRESS_STEPS

    controller = _DeadEndController()
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=200,
    )

    await engine.run()

    # Only ever saw one screen…
    assert len(engine.graph.nodes) == 1
    # …and stopped near the no-progress threshold, NOT at max_steps.
    # Allowance: a few steps to register the screen + exhaust its actions
    # before the watchdog's consecutive-no-growth counter trips.
    assert engine.step_idx < 200
    assert engine.step_idx <= MAX_NO_PROGRESS_STEPS + 5


@pytest.mark.asyncio
async def test_engine_dead_end_abort_emits_log_event(tmp_path, monkeypatch):
    """The watchdog's abort branch must emit its log event and stop cleanly.

    The test above ends by frontier exhaustion at step 2, so it never
    reached this branch — which is how a call to the non-existent
    ``_safe_emit`` crashed testbed run baseline-1 with AttributeError.
    A threshold of 0 forces the branch on the very first step (explored
    actions count as progress, so no real step here stalls).
    """
    monkeypatch.setattr("explorer.engine.MAX_NO_PROGRESS_STEPS", 0)
    events: list[dict] = []
    engine = ExplorationEngine(
        controller=_DeadEndController(),
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=200,
        event_callback=events.append,
    )

    await engine.run()

    assert any(
        e.get("type") == "log" and "Тупик" in e.get("message", "")
        for e in events
    )


@dataclass
class _FlakyTreeController(_DeadEndController):
    """Every other ``describe-ui`` read comes back empty, like the AXe
    «remote automation session» timeouts seen in testbed baseline-1."""

    reads: int = 0

    async def get_ui_elements(self) -> list[dict]:
        self.reads += 1
        if self.reads % 2 == 1:
            return []
        return await super().get_ui_elements()


@pytest.mark.asyncio
async def test_engine_retries_empty_ui_tree_instead_of_new_screen(tmp_path, monkeypatch):
    """An empty AX tree is a failed read, not a screen: the engine retries
    and the graph keeps only the real screen."""
    monkeypatch.setattr("explorer.engine.EMPTY_CAPTURE_RETRY_DELAYS", (0.0, 0.0))
    controller = _FlakyTreeController()
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=20,
    )

    await engine.run()

    assert controller.reads > 2
    assert len(engine.graph.nodes) == 1
    (node,) = engine.graph.nodes.values()
    assert len(node.interactive_elements) == 2


def _app(label: str, *children: dict) -> list[dict]:
    return [{"type": "Application", "label": label, "value": None,
             "identifier": "", "enabled": True, "frame": {}, "children": []},
            *children]


def _button(label: str, y: int) -> dict:
    return {"type": "Button", "label": label, "value": None,
            "identifier": label, "enabled": True,
            "frame": {"x": 100, "y": y, "width": 100, "height": 40},
            "children": []}


@dataclass
class _LeavesAppController:
    """One button opens another app (like «Открыть настройки iOS»)."""

    foreground: str = "app"
    launches: int = 0

    async def get_ui_elements(self) -> list[dict]:
        if self.foreground == "settings":
            return _app("Settings", _button("General", 200), _button("About", 300))
        return _app("Testbed", _button("Открыть настройки iOS", 200), _button("Обновить", 300))

    async def take_screenshot(self) -> str:
        return "<fake-png>"

    async def tap_at(self, x: int, y: int) -> _Result:
        if self.foreground == "app" and y == 220:
            self.foreground = "settings"
        return _Result()

    async def go_back(self) -> _Result:
        return _Result()

    async def terminate_app(self, bundle_id: str) -> None:
        pass

    async def launch_app(self, bundle_id: str) -> None:
        self.launches += 1
        self.foreground = "app"


@pytest.mark.asyncio
async def test_engine_returns_to_app_instead_of_exploring_another(tmp_path, monkeypatch):
    monkeypatch.setattr("explorer.engine.SETTLE_DELAY", 0)
    controller = _LeavesAppController()
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="com.markov.testbed",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=20,
    )

    await engine.run()

    assert controller.launches >= 1
    assert {n.app_label for n in engine.graph.nodes.values()} == {"Testbed"}


@dataclass
class _PopupController:
    """Fixed screen; records every tap's y coordinate."""

    tree: list[dict] = field(default_factory=list)
    taps: list[int] = field(default_factory=list)

    async def get_ui_elements(self) -> list[dict]:
        return self.tree

    async def take_screenshot(self) -> str:
        return "<fake-png>"

    async def tap_at(self, x: int, y: int) -> _Result:
        self.taps.append(y)
        return _Result()


def _popup_engine(tmp_path, controller) -> ExplorationEngine:
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="com.markov.testbed",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
    )
    engine._app_label = "Testbed"
    return engine


@pytest.mark.asyncio
async def test_auto_dismiss_leaves_the_apps_own_cancel_alone(tmp_path):
    # Testbed run 3: the profile-edit sheet's «Отмена» was auto-tapped as
    # if it were a system popup, 25 times in a row.
    controller = _PopupController(tree=_app(
        "Testbed", _button("Отмена", 100), _button("Сохранить", 100),
    ))

    await _popup_engine(tmp_path, controller)._dismiss_system_popup()

    assert controller.taps == []


@pytest.mark.asyncio
async def test_auto_dismiss_closes_springboard_alert(tmp_path, monkeypatch):
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    # iOS 27 notification prompt: SpringBoard root (blank label), typographic apostrophe.
    controller = _PopupController(tree=_app(
        " ", _button("Don’t Allow", 500), _button("Allow", 560),
    ))

    await _popup_engine(tmp_path, controller)._dismiss_system_popup()

    assert controller.taps == [520]


async def _no_sleep(_seconds):
    return None


@dataclass
class _ForegroundController(_PopupController):
    launches: int = 0

    async def launch_app(self, bundle_id: str) -> None:
        self.launches += 1
        self.tree = _app("Testbed", _button("Выйти", 400))


@pytest.mark.asyncio
async def test_returns_to_app_before_acting_when_another_app_is_in_front(tmp_path, monkeypatch):
    # Testbed run 4: the switch to iOS Settings landed after the capture,
    # and the next action («Выйти») was tapped inside iOS Settings.
    monkeypatch.setattr("explorer.engine.SETTLE_DELAY", 0)
    controller = _ForegroundController(tree=_app("Settings", _button("General", 200)))
    engine = _popup_engine(tmp_path, controller)

    await engine._dismiss_system_popup()

    assert controller.launches == 1
    assert controller.taps == []
    assert engine.graph.nodes[engine.current_screen_id].app_label == "Testbed"


@dataclass
class _SwitchController(_PopupController):
    taps_by_id: list[str] = field(default_factory=list)

    async def tap_by_id(self, test_id: str) -> _Result:
        self.taps_by_id.append(test_id)
        return _Result()


@pytest.mark.asyncio
async def test_switch_is_tapped_by_id_and_does_not_count_as_stuck(tmp_path, monkeypatch):
    # Testbed run 5: coordinate taps never flipped the toggle, and its
    # self-loops tripped MAX_STUCK_COUNT before «Выйти» was tried.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    toggle = {"type": "CheckBox", "label": "Уведомления", "value": "0",
              "test_id": "settings.notifications", "enabled": True,
              "frame": {"x": 20, "y": 242, "width": 362, "height": 52}, "children": []}
    controller = _SwitchController(tree=_app("Testbed", toggle, _button("Выйти", 400)))
    engine = _popup_engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id
    switch, logout = sorted(node.interactive_elements, key=lambda e: e.frame["y"])

    await engine._explore_tap(switch)
    assert controller.taps_by_id == ["settings.notifications"]
    assert controller.taps == []
    assert engine._stuck_count == 0

    await engine._explore_tap(logout)  # a button that changes nothing still counts
    assert engine._stuck_count == 1


@dataclass
class _SlowPushController(_PopupController):
    """«Счета» renders its table only after the first post-tap capture."""

    reads_after_tap: int = -1

    async def get_ui_elements(self) -> list[dict]:
        if self.reads_after_tap < 0:
            return _app("Testbed", _button("Счета", 200))
        self.reads_after_tap += 1
        if self.reads_after_tap == 1:
            return _app("Testbed", _button("Назад", 60))
        return _app("Testbed", _button("Назад", 60), _button("Текущий счёт", 300))

    async def tap_at(self, x: int, y: int) -> _Result:
        self.reads_after_tap = 0
        return _Result()


@pytest.mark.asyncio
async def test_new_screen_is_recaptured_until_it_settles(tmp_path, monkeypatch):
    # Testbed run 7 recorded «Счета» before its table rendered.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _SlowPushController()
    engine = _popup_engine(tmp_path, controller)
    home = await engine._capture_screen()
    engine.graph.add_node(home)
    engine.current_screen_id = home.screen_id

    await engine._explore_tap(home.interactive_elements[0])

    accounts = engine.graph.nodes[engine.current_screen_id]
    assert {el.label for el in accounts.interactive_elements} == {"Назад", "Текущий счёт"}
    assert len(engine.graph.nodes) == 2  # the half-rendered capture is not a node


@dataclass
class _PromptController(_PopupController):
    """A button raises a SpringBoard permission prompt; «Don’t Allow» closes it."""

    def __post_init__(self):
        self.app_tree = _app("Testbed", _button("Включить уведомления", 200))
        self.prompt_tree = _app(" ", _button("Don’t Allow", 500), _button("Allow", 560))
        self.tree = self.app_tree

    async def tap_at(self, x: int, y: int) -> _Result:
        self.taps.append(y)
        if self.tree is self.app_tree and y == 220:
            self.tree = self.prompt_tree
        elif self.tree is self.prompt_tree and y == 520:
            self.tree = self.app_tree
        return _Result()


@pytest.mark.asyncio
async def test_system_prompt_after_action_is_not_a_screen(tmp_path, monkeypatch):
    # Testbed run 9 recorded the notification prompt as a screen, then spent
    # 50 min navigating back to it after it had been answered.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _PromptController()
    engine = _popup_engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id

    await engine._explore_tap(node.interactive_elements[0])

    assert controller.taps == [220, 520]  # the action, then «Don’t Allow»
    assert {n.app_label for n in engine.graph.nodes.values()} == {"Testbed"}


def test_unreachable_frontier_is_written_off(tmp_path):
    engine = _popup_engine(tmp_path, _PopupController())
    target = ScreenNode(screen_id="prompt", interactive_elements=[
        ElementSnapshot(kind=ElementKind.BUTTON, element_type="Button", label="Allow",
                        frame={"x": 100, "y": 560, "width": 100, "height": 40}),
    ])
    engine.graph.add_node(target)

    engine._note_unreachable("prompt", step=1)
    assert engine.graph.get_unexplored_actions("prompt")

    engine._note_unreachable("prompt", step=2)
    assert engine.graph.get_unexplored_actions("prompt") == []


@pytest.mark.asyncio
async def test_forbidden_elements_are_never_actions(tmp_path):
    # aweassist: «Switch to test» opens the environment switcher, where
    # exploring would select «Production».
    controller = _PopupController(tree=_app(
        "Деньги", _button("Switch to test", 78), _button("Зайти", 500),
    ))
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
        forbidden=["Switch to test"],
    )

    node = await engine._capture_screen()

    assert [el.label for el in node.interactive_elements] == ["Зайти"]


@dataclass
class _LaunchAlertsController(_PopupController):
    """aweassist on first launch: permission prompt, then an icon-change
    alert with no AX elements (Return confirms it), then the app."""

    returns: int = 0
    launches: int = 0

    def __post_init__(self):
        self.tree = _app(" ", _button("Don’t Allow", 500), _button("Allow", 560))

    async def tap_at(self, x: int, y: int) -> _Result:
        self.taps.append(y)
        if y == 520:
            self.tree = _app(" ")
        return _Result()

    async def press_return(self) -> bool:
        self.returns += 1
        self.tree = _app("Деньги", _button("Зайти", 500))
        return True

    async def launch_app(self, bundle_id: str) -> None:
        self.launches += 1


@pytest.mark.asyncio
async def test_launch_alerts_are_cleared_before_first_capture(tmp_path, monkeypatch):
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _LaunchAlertsController()
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
    )

    await engine._clear_system_ui_at_start()
    first = await engine._capture_screen()

    assert controller.taps == [520] and controller.returns == 1
    assert engine._app_label == "Деньги"
    assert first.app_label == "Деньги"


@pytest.mark.asyncio
async def test_springboard_without_alert_relaunches_the_app(tmp_path, monkeypatch):
    # Return did not bring the app back (it crashed / went home): relaunch.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _LaunchAlertsController()
    controller.tree = _app(" ")

    async def press_return_noop():
        controller.returns += 1
        return True

    controller.press_return = press_return_noop
    engine = _popup_engine(tmp_path, controller)

    await engine._dismiss_system_popup()

    assert controller.returns == 1
    assert controller.launches == 1


@pytest.mark.asyncio
async def test_tree_without_application_root_is_not_springboard(tmp_path, monkeypatch):
    # A tree with no Application element says nothing about SpringBoard;
    # treating it as one relaunched the app on every step.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _LaunchAlertsController()
    controller.tree = [_button("Зайти", 500)]
    engine = _popup_engine(tmp_path, controller)

    await engine._dismiss_system_popup()

    assert controller.returns == 0 and controller.launches == 0


def _keypad_tree(label: str = "Деньги") -> list[dict]:
    """aweassist's PIN screen: ten digit buttons with ids 0–9, no submit."""
    digits = [
        {"type": "Button", "label": d, "value": None, "identifier": d, "test_id": d,
         "enabled": True, "frame": {"x": 45 + 114 * ((int(d) - 1) % 3 if d != "0" else 1),
                                    "y": 334 + 100 * ((int(d) - 1) // 3 if d != "0" else 3),
                                    "width": 85, "height": 85}, "children": []}
        for d in "0123456789"
    ]
    return _app(label, *digits, _button("Забыли код?", 650))


@dataclass
class _KeypadController(_SwitchController):
    """Accepts the PIN after the 4th digit; any other code is rejected."""

    pin: str = "1234"
    typed: str = ""

    def __post_init__(self):
        self.tree = _keypad_tree()

    async def tap_by_id(self, test_id: str) -> _Result:
        self.taps_by_id.append(test_id)
        self.typed += test_id
        if len(self.typed) == 4:
            if self.typed == self.pin:
                self.tree = _app("Деньги", _button("Главная", 200))
            self.typed = ""
        return _Result()


def _keypad_engine(tmp_path, controller, test_data) -> ExplorationEngine:
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
        test_data=test_data, forbidden=["Забыли код?"],
    )
    engine._app_label = "Деньги"
    return engine


@pytest.mark.asyncio
async def test_keypad_digits_are_never_explored_one_by_one(tmp_path):
    engine = _keypad_engine(tmp_path, _KeypadController(), test_data={})

    node = await engine._capture_screen()

    assert node.screen_id in engine._keypad_screens
    assert node.interactive_elements == []  # digits hidden, «Забыли код?» forbidden


@pytest.mark.asyncio
async def test_pin_from_test_data_is_entered_as_one_action(tmp_path, monkeypatch):
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _KeypadController()
    engine = _keypad_engine(tmp_path, controller, test_data={"pin": "1234"})
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id

    assert await engine._enter_pin_here(step=1)

    assert controller.taps_by_id == list("1234")
    landed = engine.graph.nodes[engine.current_screen_id]
    assert [el.label for el in landed.interactive_elements] == ["Главная"]
    (edge,) = [e for e in engine.graph.edges if e.source_screen_id == node.screen_id]
    assert edge.action.target_label == "PIN" and edge.action.input_text == "***"


@pytest.mark.asyncio
async def test_wrong_pin_stops_after_failure_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _KeypadController()
    engine = _keypad_engine(tmp_path, controller, test_data={"pin": "9999"})
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id

    assert await engine._enter_pin_here(step=1)       # 1st rejection: may retry
    assert not await engine._enter_pin_here(step=2)   # 2nd: budget spent
    assert not await engine._enter_pin_here(step=3)   # never types again
    assert len(controller.taps_by_id) == 8


@dataclass
class _FormController(_PopupController):
    terminations: int = 0
    typed: dict = field(default_factory=dict)

    async def terminate_app(self, bundle_id: str) -> None:
        self.terminations += 1

    async def launch_app(self, bundle_id: str) -> None:
        pass

    async def set_text_in_field(self, test_id, label, text) -> bool:
        self.typed[test_id] = text
        return True

    async def press_enter(self) -> None:
        pass


@pytest.mark.asyncio
async def test_form_fill_does_not_restart_the_app(tmp_path, monkeypatch):
    # Testbed run 4: a «clean form» relaunch logged the agent out in the
    # middle of the profile-edit sheet.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    fields = [
        {"type": "TextField", "label": "", "value": "", "identifier": f"profile_edit.{n}",
         "test_id": f"profile_edit.{n}", "enabled": True,
         "frame": {"x": 24, "y": y, "width": 354, "height": 34}, "children": []}
        for n, y in (("name", 273), ("email", 325))
    ]
    controller = _FormController(tree=_app("Testbed", *fields, _button("Сохранить", 82)))
    engine = _popup_engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id
    text_fields = [el for el in node.interactive_elements if el.element_type == "TextField"]

    await engine._fill_form_happy_path(text_fields, text_fields[0])

    assert controller.terminations == 0
    assert set(controller.typed) == {"profile_edit.name", "profile_edit.email"}


@pytest.mark.asyncio
async def test_engine_dead_end_does_not_fire_on_healthy_app(tmp_path):
    """PER-207 guard: the watchdog must NOT abort a normal exploration
    that keeps discovering screens. The 3-screen app finishes by frontier
    exhaustion with all screens found."""
    controller = _make_three_screen_app()
    engine = ExplorationEngine(
        controller=controller,
        app_bundle_id="test.app",
        output_dir=str(tmp_path),
        mode=ExplorationMode.MC,
        max_steps=50,
    )

    await engine.run()

    # All three screens discovered — the watchdog didn't cut it short.
    assert len(engine.graph.nodes) == 3
