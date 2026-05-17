"""Regression tests for the goal-node value_source contract (PER-111).

Covers T1–T9 from the research plan. Each test wires a minimal fake
controller + fake LLM to ``ScenarioRunner._run_goal_node`` so we can
assert what value the worker actually typed without booting iOS.

The fake LLM returns a queue of pre-canned decisions; the fake
controller records every ``set_text_in_field`` / ``tap_by_label`` /
``go_back`` / ``swipe`` call so a test can introspect exactly what
the worker did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from explorer.scenario_runner import ScenarioRunner


# ----------------------------------------------------------------- fakes


@dataclass
class _TapResult:
    ok: bool = True


@dataclass
class FakeController:
    """Records every controller call; returns scripted UI elements.

    Elements are dicts (the format ``_goal_decide`` reads). To simulate
    "AXLabel is null" pass ``label=None`` — that's what made the
    DevKnife overlay break on the real demo and the test_data lookup
    must still work in that case.
    """

    elements: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"label": "Введите телефон", "kind": "AXTextField", "test_id": "phone_field"},
            {"label": "Зайти", "kind": "AXButton", "test_id": "submit_btn"},
        ]
    )
    set_text_calls: list[tuple[str, str]] = field(default_factory=list)
    tap_calls: list[str] = field(default_factory=list)
    back_calls: int = 0
    swipe_calls: list[tuple] = field(default_factory=list)
    typed_text: list[str] = field(default_factory=list)

    async def get_ui_elements(self) -> list[dict]:
        return list(self.elements)

    async def set_text_in_field(self, test_id: str, value: str) -> bool:
        self.set_text_calls.append((test_id, value))
        return True

    async def tap_by_label(self, label: str) -> _TapResult:
        self.tap_calls.append(label)
        return _TapResult(ok=True)

    async def go_back(self) -> bool:
        self.back_calls += 1
        return True

    async def swipe(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        self.swipe_calls.append((x1, y1, x2, y2))
        return True

    async def type_text(self, value: str) -> bool:
        self.typed_text.append(value)
        return True


@dataclass
class FakeLLMClient:
    """Returns one canned response per ``chat`` call.

    Each response is a JSON string (so the existing parser exercises
    the same code path it does on real llama-server output). When the
    queue is exhausted, raises so a test that expected fewer calls
    sees the discrepancy.
    """

    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    # If set, the next chat() call raises this exception once (then
    # the queue keeps serving). Used to simulate llama-server 400 on
    # response_format (T8).
    raise_once: Exception | None = None

    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
        response_format: dict | None = None,
        screenshot_b64: str | None = None,
    ) -> str | None:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if self.raise_once is not None:
            exc = self.raise_once
            self.raise_once = None
            raise exc
        if not self.responses:
            raise AssertionError("FakeLLMClient called more times than scripted")
        return self.responses.pop(0)


# ----------------------------------------------------------------- helpers


def _make_runner(
    controller: FakeController,
    llm: FakeLLMClient,
    test_data: dict[str, str] | None = None,
) -> ScenarioRunner:
    """Build a ScenarioRunner wired for goal-node tests.

    ``rag_base_url`` is left empty so ``_load_prompt`` skips the HTTP
    call and the runner uses the baked-in fallback templates from
    scenario_runner — keeps tests offline.
    """
    events: list[dict] = []

    async def event_sink(evt: dict) -> None:
        events.append(evt)

    runner = ScenarioRunner(
        controller=controller,
        scenarios=[],
        test_data=test_data or {},
        event_callback=event_sink,
        llm_client=llm,
    )
    runner._test_events = events  # type: ignore[attr-defined]
    return runner


def _decision(
    action: str = "tap",
    *,
    done: bool = False,
    reason: str | None = None,
    element_label: str = "",
    value_source: str = "none",
    value_literal: str | None = None,
    reasoning: str = "test",
) -> str:
    import json

    return json.dumps(
        {
            "done": done,
            "reason": reason,
            "action": action,
            "element_label": element_label,
            "value_source": value_source,
            "value_literal": value_literal,
            "reasoning": reasoning,
        },
        ensure_ascii=False,
    )


# ----------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_T1_uses_test_data_phone_value() -> None:
    """T1: LLM picks test_data.phone → worker types the workspace's
    actual phone, regardless of what value_literal contains."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_label="Введите телефон",
                value_source="test_data.phone",
                value_literal=None,
            ),
            _decision(done=True, reason="logged in"),
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055"}
    )
    ok, _reason = await runner._run_goal_node(
        scenario_id="t1",
        step_idx=0,
        data={"description": "Авторизуйся номером {{phone}}", "max_steps": 5},
        node_id="g1",
    )
    assert ok is True
    assert controller.set_text_calls == [("phone_field", "+79051543055")]


@pytest.mark.asyncio
async def test_T2_response_format_is_sent_with_test_data_enum() -> None:
    """T2: llm_client.chat receives a response_format whose enum
    contains every workspace key as ``test_data.<key>`` plus the three
    special markers. This is what makes fabrication impossible at the
    grammar level on llama-server."""
    controller = FakeController()
    llm = FakeLLMClient(responses=[_decision(done=True, reason="trivial")])
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055", "iban": "DE1"}
    )
    await runner._run_goal_node(
        scenario_id="t2", step_idx=0,
        data={"description": "go", "max_steps": 2}, node_id="g1",
    )
    rf = llm.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    enum = rf["json_schema"]["schema"]["properties"]["value_source"]["enum"]
    assert "test_data.phone" in enum
    assert "test_data.iban" in enum
    assert "goal_literal" in enum
    assert "improvised" in enum
    assert "none" in enum


@pytest.mark.asyncio
async def test_T3_goal_literal_typed_verbatim() -> None:
    """T3: when LLM picks goal_literal, the worker types value_literal
    as-is (used for constants embedded in the goal text like '1000')."""
    controller = FakeController(
        elements=[{"label": "Сумма", "kind": "AXTextField", "test_id": "amount"}]
    )
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_label="Сумма",
                value_source="goal_literal",
                value_literal="1000",
            ),
            _decision(done=True, reason="ok"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _ = await runner._run_goal_node(
        scenario_id="t3", step_idx=0,
        data={"description": "Введи 1000", "max_steps": 5}, node_id="g1",
    )
    assert ok is True
    assert controller.set_text_calls == [("amount", "1000")]


@pytest.mark.asyncio
async def test_T4_improvised_value_remembered_across_steps() -> None:
    """T4: when the model invents a value for an element_label, the
    second visit to the same label produces the same string — even if
    the model itself contradicts itself on the second turn."""
    controller = FakeController(
        elements=[
            {"label": "Название", "kind": "AXTextField", "test_id": "name"},
        ]
    )
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_label="Название",
                value_source="improvised",
                value_literal="Тестовая 1",
            ),
            _decision(
                action="input",
                element_label="Название",
                value_source="improvised",
                # Model changed its mind — but worker MUST stick with
                # the first answer to make the run reproducible.
                value_literal="ABSOLUTELY DIFFERENT",
            ),
            _decision(done=True, reason="ok"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _ = await runner._run_goal_node(
        scenario_id="t4", step_idx=0,
        data={"description": "Придумай имя", "max_steps": 5}, node_id="g1",
    )
    assert ok is True
    assert controller.set_text_calls == [
        ("name", "Тестовая 1"),
        ("name", "Тестовая 1"),
    ]


@pytest.mark.asyncio
async def test_T5_tap_does_not_type_anything() -> None:
    """T5: action=tap with value_source=none — set_text_in_field is
    never called; tap_by_label is."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="tap",
                element_label="Зайти",
                value_source="none",
                value_literal=None,
            ),
            _decision(done=True, reason="ok"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={"phone": "+79051543055"})
    ok, _ = await runner._run_goal_node(
        scenario_id="t5", step_idx=0,
        data={"description": "Нажми Зайти", "max_steps": 5}, node_id="g1",
    )
    assert ok is True
    assert controller.set_text_calls == []
    assert controller.tap_calls == ["Зайти"]


@pytest.mark.asyncio
async def test_T6_done_true_terminates_node() -> None:
    """T6: when LLM reports done=true on the first step, the inner
    loop exits and the goal is reported completed."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[_decision(done=True, reason="главный экран виден")]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, reason = await runner._run_goal_node(
        scenario_id="t6", step_idx=0,
        data={"description": "Авторизуйся", "max_steps": 5}, node_id="g1",
    )
    assert ok is True
    assert reason == "главный экран виден"
    assert len(llm.calls) == 1  # one chat call, then stopped


@pytest.mark.asyncio
async def test_T7_missing_test_data_does_not_fabricate() -> None:
    """T7: when the goal needs a phone but workspace has no phone key,
    the model is supposed to return value_source=none and a reasoning
    that flags missing data — worker neither types nor crashes."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="back",
                element_label="Зайти",
                value_source="none",
                value_literal=None,
                reasoning="missing test_data: phone",
            ),
            _decision(done=False, reason=None, action="back",
                      value_source="none", value_literal=None,
                      reasoning="нечего делать дальше"),
            _decision(done=False, reason=None, action="back",
                      value_source="none", value_literal=None,
                      reasoning="нечего делать"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    # max_steps small so the test finishes quickly.
    ok, _ = await runner._run_goal_node(
        scenario_id="t7", step_idx=0,
        data={"description": "Авторизуйся номером", "max_steps": 3},
        node_id="g1",
    )
    assert ok is False  # never reaches done
    assert controller.set_text_calls == []  # critical: no fabrication


@pytest.mark.asyncio
async def test_T8_graceful_fallback_when_grammar_unsupported() -> None:
    """T8: when llama-server 400s on response_format, llm_client
    retries WITHOUT the format and the run keeps going. We simulate
    this at the level of ``llm_client.chat`` — but ScenarioRunner just
    sees a successful second call. So instead we check that even a
    pure-text JSON response (no grammar enforcement) parses cleanly
    through raw_decode."""
    controller = FakeController()
    # Trailing chatter after the JSON object — this is exactly what
    # broke on the first demo run (Extra data error). raw_decode must
    # tolerate it.
    fake_response = (
        _decision(done=True, reason="ok") + "\n\nI hope this helps!"
    )
    llm = FakeLLMClient(responses=[fake_response])
    runner = _make_runner(controller, llm, test_data={})
    ok, _ = await runner._run_goal_node(
        scenario_id="t8", step_idx=0,
        data={"description": "go", "max_steps": 2}, node_id="g1",
    )
    assert ok is True


@pytest.mark.asyncio
async def test_T9_workspace_agnostic_key_iban() -> None:
    """T9: the contract works for any workspace key, not just phone /
    email — e.g. iban. Confirms there's no category-specific code path
    left over from PER-110."""
    controller = FakeController(
        elements=[{"label": "IBAN", "kind": "AXTextField", "test_id": "iban"}]
    )
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_label="IBAN",
                value_source="test_data.iban",
                value_literal=None,
            ),
            _decision(done=True, reason="ok"),
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"iban": "DE89370400440532013000"},
    )
    ok, _ = await runner._run_goal_node(
        scenario_id="t9", step_idx=0,
        data={"description": "Введи IBAN", "max_steps": 5}, node_id="g1",
    )
    assert ok is True
    assert controller.set_text_calls == [("iban", "DE89370400440532013000")]
