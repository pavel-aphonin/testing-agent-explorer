"""Tab bar items are found again after the tree stops naming them.

wikipedia-7: after a relaunch «Saved», «Activity» and «Search» could not be
brought on screen: the graph knew them as «Root Tab Saved Button», the live
tree gave the same RadioButtons no AXUniqueId, and the lookup by exact id
found nothing. Steps and a whole trip were lost, and six failed trips in a
row stopped the run as «environment unavailable».
"""

from __future__ import annotations

from explorer import axe_client
from explorer.axe_client import AXeExplorerClient

TAB = {"x": 162.5, "y": 795, "width": 77, "height": 54}


def _client(monkeypatch, elements):
    client = AXeExplorerClient()
    client._udid, client._width, client._height = "SIM", 402, 874
    client._tab_bar_visible = True

    async def tree(reuse=False, web=True):
        return list(elements)

    async def no_input(*_a, **_k):
        raise AssertionError("the tab bar does not scroll: no swipe expected")

    monkeypatch.setattr(client, "get_ui_elements", tree)
    monkeypatch.setattr(axe_client, "_run", no_input)
    return client


def _radio(label, test_id=None, frame=TAB):
    return {"type": "RadioButton", "kind": "RadioButton", "label": label,
            "test_id": test_id, "frame": dict(frame), "obscured": False}


async def test_a_tab_is_found_when_the_live_tree_has_no_id_for_it(monkeypatch):
    client = _client(monkeypatch, [_radio("Saved")])
    frame = await client.scroll_into_view("RadioButton", "Saved", "Root Tab Saved Button", TAB)
    assert frame == TAB


async def test_a_tab_on_screen_is_not_scrolled(monkeypatch):
    client = _client(monkeypatch, [_radio("Saved", "Root Tab Saved Button")])
    frame = await client.scroll_into_view("RadioButton", "Saved", "Root Tab Saved Button", TAB)
    assert frame == TAB


async def test_an_element_with_another_id_is_still_not_taken_for_it(monkeypatch):
    other = {"type": "Button", "kind": "Button", "label": "Close", "test_id": "Other",
             "frame": {"x": 10, "y": 300, "width": 40, "height": 40}, "obscured": False}
    client = _client(monkeypatch, [other])
    assert await client.scroll_into_view("Button", "Close", "Close", other["frame"]) is None
