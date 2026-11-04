"""Forms in the graph: explored-marking and path replay (testbed run 2).

Run 2 refilled the login form 20 steps in a row (the field was never
counted as explored), and after a relaunch it could not get past login
(replaying an input edge typed one redacted value into one field).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from explorer.models import (
    ActionDetail,
    ActionType,
    AppGraph,
    ElementKind,
    ElementSnapshot,
    GraphEdge,
    ScreenNode,
)
from explorer.navigator import replay_path

EMAIL = ElementSnapshot(
    kind=ElementKind.TEXT_FIELD, element_type="TextField",
    test_id="login.email", frame={"x": 24, "y": 258, "width": 354, "height": 34},
)
SUBMIT = ElementSnapshot(
    kind=ElementKind.BUTTON, element_type="Button", label="Войти",
    test_id="login.submit", frame={"x": 164, "y": 358, "width": 72, "height": 34},
)


def _login_graph() -> AppGraph:
    graph = AppGraph(app_bundle_id="com.markov.testbed")
    graph.add_node(ScreenNode(screen_id="login", name="Вход", interactive_elements=[EMAIL, SUBMIT]))
    graph.add_node(ScreenNode(screen_id="home", name="Главная"))
    return graph


@pytest.mark.parametrize("frame", [EMAIL.frame, None], ids=["with-frame", "legacy-no-frame"])
def test_input_marker_edge_marks_field_explored(frame):
    graph = _login_graph()
    graph.add_edge(GraphEdge(
        source_screen_id="login", target_screen_id="login",
        action=ActionDetail(
            action_type=ActionType.INPUT, target_test_id="login.email",
            target_frame=frame, input_category="explored",
        ),
    ))

    unexplored = graph.get_unexplored_actions("login")

    assert [el.test_id for el in unexplored] == ["login.submit"]


@dataclass
class _Controller:
    typed: list[str] = field(default_factory=list)

    async def tap_at(self, x, y):
        return type("R", (), {"error": None})()

    async def erase_text(self, n):
        return True

    async def type_text(self, text):
        self.typed.append(text)
        return True


@pytest.mark.asyncio
async def test_replay_uses_form_callback_for_input_edges(monkeypatch):
    monkeypatch.setattr("explorer.navigator.SETTLE_DELAY", 0)
    login_submit = GraphEdge(
        source_screen_id="login", target_screen_id="home",
        action=ActionDetail(
            action_type=ActionType.INPUT, target_test_id="login.password",
            target_frame=EMAIL.frame, input_text="***<9>", input_category="valid",
        ),
    )
    controller = _Controller()
    replayed: list[ActionDetail] = []

    async def replay_form(action):
        replayed.append(action)
        return True

    async def capture():
        return ("home", [])

    landed = await replay_path(controller, [login_submit], capture, replay_input=replay_form)

    assert landed == "home"
    assert replayed == [login_submit.action]
    assert controller.typed == []  # the redacted «***<9>» is never typed
