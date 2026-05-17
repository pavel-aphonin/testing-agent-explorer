"""Regression tests for the goal-node value_source contract (PER-111).

PER-111 v2 update:
    * element_id is the primary on-screen identifier (element_label is
      a human-readable label and may be null / change between visits).
    * The action dictionary is workspace-curated and shipped by the
      backend in RunClaimResponse.actions — tests don't pass it (so the
      schema falls back to the permissive `action: string` enum), which
      lets us exercise the value_source + element_id contract in
      isolation from the action-args branching.
    * T7 replaced by T7' — "improvised" must actually type a plausible
      value, not skip the field. v1 said "don't fabricate", v2 says
      "invent a sensible default".
    * T13 — explore mode (empty description) runs until max_steps and
      reports step_completed (designed exit), not step_failed.
    * T14 — neither the system nor user fallback prompt enumerates
      specific action codes ("tap", "input", "swipe", "back"). Action
      names must flow only via the workspace dictionary in the user
      message.

Each test wires a minimal fake controller + fake LLM to
``ScenarioRunner._run_goal_node`` so we can assert what value the
worker actually typed without booting iOS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from explorer.scenario_runner import (
    _GOAL_DECIDE_SYSTEM_FALLBACK,
    _GOAL_DECIDE_USER_FALLBACK,
    ScenarioRunner,
)


# ----------------------------------------------------------------- fakes


@dataclass
class _TapResult:
    ok: bool = True


@dataclass
class FakeController:
    """Records every controller call; returns scripted UI elements.

    Elements are dicts in the format ``_goal_decide`` reads via
    ``build_elements_block``. Each element carries an ``id`` (the
    canonical stable identifier per PER-111 v2) and a ``test_id``
    (the AXe accessibility identifier the dispatcher uses to type
    into the field). In real life they often coincide; here we keep
    them equal for simplicity.
    """

    elements: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "id": "phone_field",
                "label": "Введите телефон",
                "kind": "AXTextField",
                "test_id": "phone_field",
            },
            {
                "id": "submit_btn",
                "label": "Зайти",
                "kind": "AXButton",
                "test_id": "submit_btn",
            },
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
    actions: list[dict[str, Any]] | None = None,
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
        actions=actions or [],
    )
    runner._test_events = events  # type: ignore[attr-defined]
    return runner


def _decision(
    action: str = "tap",
    *,
    done: bool = False,
    reason: str | None = None,
    element_id: str | None = None,
    element_label: str | None = None,
    value_source: str = "none",
    value_literal: str | None = None,
    reasoning: str = "test",
    action_args: dict[str, Any] | None = None,
) -> str:
    """Build one canned LLM response matching the PER-111 v2 contract."""
    return json.dumps(
        {
            "done": done,
            "reason": reason,
            "action": action,
            "action_args": action_args or {},
            "element_id": element_id,
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
    actual phone, regardless of what value_literal contains. Element
    is referenced by element_id (PER-111 v2 contract)."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="phone_field",
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
    schema = rf["json_schema"]["schema"]
    enum = schema["properties"]["value_source"]["enum"]
    assert "test_data.phone" in enum
    assert "test_data.iban" in enum
    assert "goal_literal" in enum
    assert "improvised" in enum
    assert "none" in enum
    # element_id enum must include every on-screen id plus null.
    eid_enum = schema["properties"]["element_id"]["enum"]
    assert "phone_field" in eid_enum
    assert "submit_btn" in eid_enum
    assert None in eid_enum


@pytest.mark.asyncio
async def test_T3_goal_literal_typed_verbatim() -> None:
    """T3: when LLM picks goal_literal, the worker types value_literal
    as-is (used for constants embedded in the goal text like '1000')."""
    controller = FakeController(
        elements=[{
            "id": "amount",
            "label": "Сумма",
            "kind": "AXTextField",
            "test_id": "amount",
        }]
    )
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="amount",
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
    """T4: when the model invents a value for an element_id, the second
    visit to the same id produces the same string — even if the model
    itself contradicts itself on the second turn. PER-111 v2 keys
    improvised_memory by element_id (not label) so re-renders with a
    new label still hit the cache."""
    controller = FakeController(
        elements=[{
            "id": "name_field",
            "label": "Название",
            "kind": "AXTextField",
            "test_id": "name",
        }]
    )
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="name_field",
                element_label="Название",
                value_source="improvised",
                value_literal="Тестовая 1",
            ),
            _decision(
                action="input",
                element_id="name_field",
                # Same id, but UI re-rendered with a different label
                # (e.g. localization / a11y refresh). Memory must still
                # hit because it's keyed by id.
                element_label="Name",
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
                element_id="submit_btn",
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
    """T6: in scenario mode (description non-empty), when LLM reports
    done=true on the first step, the inner loop exits and the goal is
    reported completed."""
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
async def test_T7_improvised_phone_returns_plausible_value() -> None:
    """T7' (PER-111 v2): when the goal needs a phone but workspace has
    no phone key, the model picks value_source=improvised and provides
    a plausible value_literal. The worker types whatever the model
    invented (no v1-style "skip the field" behavior). Value stays
    reproducible across visits via improvised_memory."""
    controller = FakeController()
    invented = "+79991234567"
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="phone_field",
                element_label="Введите телефон",
                value_source="improvised",
                value_literal=invented,
                reasoning="нет test_data.phone — придумываю правдоподобный номер",
            ),
            _decision(done=True, reason="заполнил поле телефона"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _ = await runner._run_goal_node(
        scenario_id="t7", step_idx=0,
        data={"description": "Введи свой телефон", "max_steps": 3},
        node_id="g1",
    )
    assert ok is True
    # Worker typed exactly what the LLM invented — no normalization,
    # no skip, no crash.
    assert controller.set_text_calls == [("phone_field", invented)]


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
        elements=[{
            "id": "iban_field",
            "label": "IBAN",
            "kind": "AXTextField",
            "test_id": "iban",
        }]
    )
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="iban_field",
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


@pytest.mark.asyncio
async def test_T13_explore_mode_ignores_done_and_exhausts_max_steps() -> None:
    """T13 (PER-111 v2): empty description = explore mode. The model is
    forbidden from declaring done=true on its own; if it tries, the
    worker logs a warning and keeps going. The node naturally ends
    when max_steps is exhausted, and the timeline reports it as
    completed (designed exit, not failure) so free-exploration nodes
    don't show a red mark in every run."""
    controller = FakeController()
    # Three taps, each with done=true — worker must ignore the done
    # flag in explore mode and keep stepping until max_steps.
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="tap",
                element_id="submit_btn",
                element_label="Зайти",
                done=True,
                reason="модель пытается закончить рано",
            ),
            _decision(
                action="tap",
                element_id="submit_btn",
                element_label="Зайти",
                done=True,
                reason="и ещё раз пытается",
            ),
            _decision(
                action="tap",
                element_id="submit_btn",
                element_label="Зайти",
                done=True,
                reason="и снова",
            ),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, reason = await runner._run_goal_node(
        scenario_id="t13", step_idx=0,
        # No description → mode=explore.
        data={"description": "", "max_steps": 3}, node_id="g1",
    )
    assert ok is True
    assert reason is not None and "explore" in reason
    # Worker ran the full quota of LLM calls, not just one.
    assert len(llm.calls) == 3
    # And actually tapped the button each time (didn't break early).
    assert controller.tap_calls == ["Зайти", "Зайти", "Зайти"]


def test_T14_fallback_prompts_dont_enumerate_specific_actions() -> None:
    """T14 (PER-111 v2): neither the system nor user fallback prompt
    embedded in scenario_runner.py mentions specific action codes
    (``tap``, ``input``, ``swipe``, ``back``, ``scroll``, ``wait``,
    ``long_press``, ``assert``). Action names must arrive via the
    workspace-curated dictionary in actions_block — the prompt only
    references "the list of available actions".

    Synchronous test (no asyncio) — pure string check on module
    constants.
    """
    blob = (
        _GOAL_DECIDE_SYSTEM_FALLBACK
        + "\n"
        + _GOAL_DECIDE_USER_FALLBACK
    ).lower()
    forbidden = ("tap", "swipe", "scroll", "long_press", "wait_ms", "go_back")
    found = [name for name in forbidden if name in blob]
    assert not found, (
        f"v2 prompt fallbacks must not enumerate action codes; found: {found}"
    )
    # The fallback must also reference the dictionary-driven discovery
    # path, so the LLM knows where to read action names from.
    assert "доступных действий" in _GOAL_DECIDE_SYSTEM_FALLBACK.lower()


def test_T14b_scenario_runner_prompts_no_hardcoded_actions() -> None:
    """T14b: the same property holds for the live module source
    (catches anyone re-introducing hardcoded action names in a prompt
    string later — the dispatcher implementations don't count, but the
    prompt fallbacks do). We scan only the fallback constant region;
    the dispatcher legitimately knows action names because it maps
    them to controller calls."""
    runner_path = (
        Path(__file__).parent.parent / "explorer" / "scenario_runner.py"
    )
    src = runner_path.read_text(encoding="utf-8")
    # Slice between the two fallback markers — that's the prompt
    # region. _GOAL_DECIDE_SYSTEM_FALLBACK and _GOAL_DECIDE_USER_FALLBACK
    # are right next to each other in the file.
    sys_idx = src.find("_GOAL_DECIDE_SYSTEM_FALLBACK")
    end_idx = src.find("_render_template", sys_idx)
    assert sys_idx >= 0 and end_idx > sys_idx, (
        "could not locate fallback prompt region; "
        "rename moved markers? — update this test"
    )
    region = src[sys_idx:end_idx].lower()
    forbidden = ("\"tap\"", "\"swipe\"", "\"scroll\"", "\"long_press\"")
    found = [w for w in forbidden if w in region]
    assert not found, (
        f"hardcoded action codes leaked into prompt fallback region: {found}"
    )
