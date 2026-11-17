"""Verified text entry tolerates the keyboard's own edits.

wikipedia-5: iOS turned «test query» into «Test query» and
«' OR 1=1 --» into «‘ OR 1=1 —». The text was in the field, yet the
strict comparison called 74 entries failed, three attempts each.
"""

from __future__ import annotations

import pytest

from explorer.axe_client import AXeExplorerClient


def _client(monkeypatch, value):
    client = AXeExplorerClient()

    async def tree(web=True):
        return [{"test_id": "Search Field", "value": value}]

    monkeypatch.setattr(client, "get_ui_elements", tree)
    return client


@pytest.mark.parametrize("typed, shown", [
    ("test query", "test query"),
    ("test query", "Test query"),
    ("test query", "Test Query"),
    ("' OR 1=1 --", "‘ OR 1=1 —"),
    ('say "hi"...', "Say “hi”…"),
    ("it's", "It’s"),
])
async def test_keyboard_edits_do_not_fail_the_entry(monkeypatch, typed, shown):
    assert await _client(monkeypatch, shown)._field_holds("Search Field", typed)


@pytest.mark.parametrize("typed, shown", [
    ("password123", "test@test.compassword123"),
    ("test query", "test quer"),
    ("test query", "Search Wikipedia"),
    ("test query", ""),
])
async def test_wrong_text_still_fails_the_entry(monkeypatch, typed, shown):
    assert not await _client(monkeypatch, shown)._field_holds("Search Field", typed)


async def test_secure_fields_are_checked_by_length(monkeypatch):
    assert await _client(monkeypatch, "•••")._field_holds("Search Field", "abc")
    assert not await _client(monkeypatch, "••")._field_holds("Search Field", "abc")
