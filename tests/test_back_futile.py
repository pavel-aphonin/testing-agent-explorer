"""A screen Back does not leave is not tried again (wikipedia-5: 46 trips,
~19 of 94 minutes spent pressing Back on tab roots that ignore it)."""

from __future__ import annotations

from dataclasses import dataclass

from explorer.models import AppGraph
from explorer.navigator import go_back_to_screen


@dataclass
class _Controller:
    backs: int = 0
    launches: int = 0

    async def go_back(self):
        self.backs += 1
        return True

    async def terminate_app(self, bundle_id):
        pass

    async def launch_app(self, bundle_id):
        self.launches += 1


async def _capture():
    return ("root", [])


async def _no_sleep(*_a, **_k):
    pass


async def test_second_trip_from_a_screen_back_ignores_goes_straight_to_relaunch(monkeypatch):
    monkeypatch.setattr("explorer.navigator.asyncio.sleep", _no_sleep)
    graph = AppGraph(app_bundle_id="com.example")
    controller = _Controller()

    await go_back_to_screen(controller, graph, "root", "elsewhere", _capture)
    assert controller.backs == 2 and controller.launches == 1

    await go_back_to_screen(controller, graph, "root", "elsewhere", _capture)
    assert controller.backs == 2          # no Back pressed this time
    assert controller.launches == 2


async def test_other_screens_still_try_back_first(monkeypatch):
    monkeypatch.setattr("explorer.navigator.asyncio.sleep", _no_sleep)
    graph = AppGraph(app_bundle_id="com.example")
    graph.back_futile.add("some-other-screen")
    controller = _Controller()

    await go_back_to_screen(controller, graph, "root", "elsewhere", _capture)
    assert controller.backs == 2


def test_the_memory_is_not_saved_with_the_graph():
    graph = AppGraph(app_bundle_id="com.example")
    graph.back_futile.add("root")
    assert "back_futile" not in graph.model_dump()
