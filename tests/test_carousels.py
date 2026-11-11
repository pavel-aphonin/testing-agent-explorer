"""Carousels (PER-896): a card past the screen edge needs a sideways swipe.

aweassist's main screen has whole rows of cards at x=372 and beyond, and the
agent skipped them all: vertical scrolling never brings them on screen.
"""

from __future__ import annotations

from tests.test_aweassist_run1 import _client, _el


def _card(x, y=300, label="Отели"):
    return _el("GenericElement", label, y, x=x, width=120)


def _at(x, y=300):
    return {"x": x, "y": y, "width": 120, "height": 60}


def _swipes(calls):
    return [c for c in calls if "swipe" in c]


def _xs(call):
    return float(call[call.index("--start-x") + 1]), float(call[call.index("--end-x") + 1])


async def test_card_past_the_right_edge_is_swiped_into_view(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(540)], [_card(140)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(540))

    assert frame["x"] == 140
    (swipe,) = _swipes(calls)
    start, end = _xs(swipe)
    assert end < start
    assert swipe[swipe.index("--start-y") + 1] == swipe[swipe.index("--end-y") + 1] == "330"


async def test_card_past_the_left_edge_is_swiped_back(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(-400)], [_card(100)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(-400))

    assert frame["x"] == 100
    start, end = _xs(_swipes(calls)[0])
    assert start == 40  # clear of the left edge, where a swipe means «back»
    assert end > start


async def test_swipes_stay_clear_of_the_screen_edges(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(5000)], [_card(4700)], [_card(4400)]])

    await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(5000))

    start, end = _xs(_swipes(calls)[0])
    assert (start, end) == (362, 40)


async def test_the_row_is_brought_into_view_first(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(540, 1500)], [_card(540, 400)], [_card(140, 400)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(540, 1500))

    assert (frame["x"], frame["y"]) == (140, 400)
    first, second = _swipes(calls)
    assert _xs(first)[0] == _xs(first)[1]
    assert _xs(second)[0] > _xs(second)[1]


async def test_row_under_the_tab_bar_that_cannot_rise_is_swiped_where_it_is(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(540, 780)], [_card(540, 780)], [_card(140, 780)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(540, 780))

    assert frame["x"] == 140
    assert not client.scroll_stuck
    up, sideways = _swipes(calls)
    assert _xs(up)[0] == _xs(up)[1]
    assert _xs(sideways)[0] > _xs(sideways)[1]


async def test_end_of_a_carousel_is_not_the_end_of_the_page(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(540)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(540))

    assert frame is None
    assert not client.scroll_stuck
    assert len(_swipes(calls)) == 2


async def test_a_dropped_sideways_swipe_is_retried(monkeypatch):
    client, _, _ = _client(monkeypatch, [[_card(540)], [_card(540)], [_card(140)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(540))

    assert frame["x"] == 140


async def test_identical_cards_in_one_row_are_told_apart_by_x(monkeypatch):
    # «до 5%» twice in one row on aweassist's main screen.
    row = lambda shift: [_card(374 - shift, label="до 5%"), _card(537 - shift, label="до 5%")]
    client, _, _ = _client(monkeypatch, [row(0), row(300)])

    frame = await client.scroll_into_view("GenericElement", "до 5%", "до 5%", _at(537))

    assert frame["x"] == 237


async def test_the_ordinal_does_not_depend_on_tree_order(monkeypatch):
    left, right = (lambda shift: _card(374 - shift, label="до 5%")), (lambda shift: _card(537 - shift, label="до 5%"))
    client, _, _ = _client(monkeypatch, [[right(0), left(0)], [left(300), right(300)]])

    frame = await client.scroll_into_view("GenericElement", "до 5%", "до 5%", _at(537))

    assert frame["x"] == 237


def _duration(call):
    return call[call.index("--duration") + 1]


def _length(call):
    start, end = _xs(call)
    return abs(start - end)


async def test_sideways_swipes_are_slow_enough_not_to_fling(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_card(540)], [_card(140)]])

    await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(540))

    assert float(_duration(_swipes(calls)[0])) >= 2


async def test_an_overshoot_is_corrected_in_shorter_steps(monkeypatch):
    # Testbed, live: the card flew from 471 past the left edge and back,
    # seven times, until the swipes ran out.
    client, calls, _ = _client(monkeypatch, [[_card(500)], [_card(-300)], [_card(450)], [_card(150)]])

    frame = await client.scroll_into_view("GenericElement", "Отели", "Отели", _at(500))

    assert frame["x"] == 150
    first, back, again = _swipes(calls)
    assert _length(back) < _length(first)
    assert _length(again) < _length(back) or _length(again) == 60
