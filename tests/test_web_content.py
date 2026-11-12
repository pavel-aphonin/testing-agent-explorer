"""Web content (PER-898): what a WKWebView shows is missing from the full
tree; point queries reach it."""

from __future__ import annotations

import json

from explorer import axe_client, web_content
from explorer.axe_client import AXeExplorerClient


def _node(type_, label="", children=(), frame=(0, 0, 402, 874), value=""):
    x, y, w, h = frame
    return {"type": type_, "AXLabel": label, "AXValue": value, "AXUniqueId": "",
            "frame": {"x": x, "y": y, "width": w, "height": h}, "children": list(children)}


SCROLL_BARS = [_node("Slider", "Vertical scroll bar, 1 page", frame=(369, 154, 30, 658)),
               _node("Slider", "Horizontal scroll bar, 1 page", frame=(28, 807, 346, 30))]


def _web_view(*extra):
    return _node("ScrollArea", children=[_node("Group", frame=(0, 154, 402, 686)), *SCROLL_BARS, *extra],
                 frame=(0, 154, 402, 686))


def _app(*children):
    return [_node("Application", "Testbed", children=[_node("Group", children=list(children))])]


def test_an_empty_scroll_view_is_a_web_page():
    web = _web_view()
    assert web_content.hollow_areas(_app(_node("Heading", "Веб"), web), 402, 874) == [web]


def test_a_scroll_view_with_content_is_not():
    native = _web_view(_node("Button", "Профиль", frame=(0, 200, 402, 44)))
    assert web_content.hollow_areas(_app(native), 402, 874) == []


def test_a_small_empty_scroll_view_is_not_worth_probing():
    tiny = _node("ScrollArea", children=list(SCROLL_BARS), frame=(0, 154, 402, 60))
    assert web_content.hollow_areas(_app(tiny), 402, 874) == []


def test_empty_groups_are_not_web_pages():
    # Every iOS 27 screen ends with full-screen empty Groups.
    assert web_content.hollow_areas(_app(_node("Group")), 402, 874) == []


def test_grid_stays_inside_the_area_and_the_screen():
    points = web_content.grid({"x": 0, "y": 154, "width": 2000, "height": 2000}, 402, 874)
    assert points
    assert all(0 < x < 402 and 154 < y < 874 for x, y in points)


LINK = _node("Link", "Открыть деталь", frame=(16, 227, 146, 26))
TEXT = _node("StaticText", "", value="Содержимое веб-страницы", frame=(16, 172, 223, 22))


async def test_scan_keeps_what_the_probes_find_once():
    calls = []

    async def probe(x, y):
        calls.append((x, y))
        if y < 220:
            return TEXT
        if y < 300:
            return LINK
        return _node("Group", frame=(0, 154, 402, 686))  # the page body

    found = await web_content.scan(_web_view(), probe, 402, 874)

    assert [(n["type"], n["AXLabel"]) for n in found] == [
        ("StaticText", "Содержимое веб-страницы"), ("Link", "Открыть деталь"),
    ]


async def test_points_inside_found_elements_are_skipped(monkeypatch):
    monkeypatch.setattr(web_content, "PARALLEL_PROBES", 1)
    big = _node("Button", "Большая кнопка", frame=(0, 154, 402, 686))
    calls = []

    async def probe(x, y):
        calls.append((x, y))
        return big

    await web_content.scan(_web_view(), probe, 402, 874)

    assert len(calls) == 1


async def test_scroll_bars_and_containers_are_not_elements():
    async def probe(x, y):
        return SCROLL_BARS[0] if x > 300 else _node("ScrollArea", frame=(0, 154, 402, 686))

    assert await web_content.scan(_web_view(), probe, 402, 874) == []


async def test_the_client_puts_web_elements_into_the_tree(monkeypatch):
    tree = _app(_node("Heading", "Веб-страница", frame=(16, 116, 152, 26)), _web_view())

    async def fake_run(args, timeout=15):
        if "--point" in args:
            x, y = map(int, args[args.index("--point") + 1].split(","))
            return 0, json.dumps(LINK if 220 < y < 300 else _node("Group")), ""
        return 0, json.dumps(tree), ""

    monkeypatch.setattr(axe_client, "_run", fake_run)
    client = AXeExplorerClient()
    client._udid, client._width, client._height = "SIM", 402, 874

    labels = [e["label"] for e in await client.get_ui_elements()]
    assert "Открыть деталь" in labels

    client.scan_web_content = False
    assert "Открыть деталь" not in [e["label"] for e in await client.get_ui_elements()]


def test_springboard_is_never_probed():
    # Its root label is blank; testbed run 22 explored the status bar there.
    springboard = [_node("Application", " ", children=[_web_view()])]
    assert web_content.hollow_areas(springboard, 402, 874) == []


def test_no_probes_in_the_status_bar():
    points = web_content.grid({"x": 0, "y": 0, "width": 402, "height": 874}, 402, 874)
    assert min(y for _, y in points) > web_content.STATUS_BAR_PT


async def test_status_bar_hits_are_dropped():
    clock = _node("Button", "01:54", frame=(30, 15, 50, 24))

    async def probe(x, y):
        return clock

    full = _node("ScrollArea", children=list(SCROLL_BARS), frame=(0, 0, 402, 874))
    assert await web_content.scan(full, probe, 402, 874) == []


def test_a_scroll_container_does_not_change_the_screen():
    from explorer.screen_id import compute_screen_id

    rows = [{"type": "Cell", "label": label, "frame": {"x": 0, "y": 200 + 50 * i, "width": 402, "height": 44}}
            for i, label in enumerate(["Текущий счёт", "Накопительный", "Валютный"])]
    with_container = [{"type": "ScrollArea", "label": None, "frame": {"x": 0, "y": 0, "width": 402, "height": 874}}, *rows]

    assert compute_screen_id(rows) == compute_screen_id(with_container)
