"""A relaunch waits for the app's first screen, not a fixed 4 s.

Wikipedia run 10: 63 relaunches, each 1 s after terminate and 3 s after
launch — 4.2 of 57.7 minutes.
"""

from __future__ import annotations

from explorer import navigator
from explorer.navigator import LAUNCH_DELAY, _await_launch

SPRINGBOARD = [{"type": "Application", "label": " "}, {"type": "Button", "label": "Safari"}]
SPLASH = [{"type": "Application", "label": "Wikipedia"}, {"type": "Image", "label": "logo"}]
FEED = [{"type": "Application", "label": "Wikipedia"}, {"type": "Button", "label": "Search"},
        {"type": "RadioButton", "label": "For You"}]


class _Reads:
    def __init__(self, *trees):
        self.trees, self.reads = list(trees), 0

    async def get_ui_elements(self):
        self.reads += 1
        return self.trees[min(self.reads, len(self.trees)) - 1]


def _no_sleep(monkeypatch):
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(navigator.asyncio, "sleep", sleep)
    return slept


async def test_the_first_screen_ends_the_wait_once_it_holds(monkeypatch):
    slept = _no_sleep(monkeypatch)
    phone = _Reads(SPRINGBOARD, SPLASH, FEED, FEED)

    await _await_launch(phone)

    assert phone.reads == 4  # SpringBoard and the splash do not count
    assert sum(slept) < 1 + 3


async def test_a_screen_that_never_settles_is_given_up_on(monkeypatch):
    _no_sleep(monkeypatch)
    phone = _Reads(SPLASH)

    await _await_launch(phone)

    assert phone.reads == round((navigator.LAUNCH_MAX_S - navigator.LAUNCH_MIN_S) / navigator.LAUNCH_POLL_S)


async def test_a_controller_that_cannot_read_waits_the_fixed_delay(monkeypatch):
    slept = _no_sleep(monkeypatch)

    await _await_launch(object())

    assert slept == [LAUNCH_DELAY]
