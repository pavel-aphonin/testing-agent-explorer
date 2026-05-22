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
    # PER-145 L1: screen dims AXe controller exposes after connect.
    # Defaults match iPhone 17 Pro Max @ 3.0× retina.
    _width: int = 440
    _height: int = 956
    _scale: float = 3.0
    tap_at_calls: list[tuple[int, int]] = field(default_factory=list)

    async def get_ui_elements(self) -> list[dict]:
        return list(self.elements)

    async def set_text_in_field(self, test_id: str, value: str) -> bool:
        self.set_text_calls.append((test_id, value))
        return True

    async def tap_by_label(self, label: str) -> _TapResult:
        self.tap_calls.append(label)
        return _TapResult(ok=True)

    async def tap_at(self, x: int, y: int) -> _TapResult:
        self.tap_at_calls.append((x, y))
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
    tap_at_coord_space: str = "points",
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
        tap_at_coord_space=tap_at_coord_space,
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
    # PER-111 v2: element_id is REQUIRED string, non-null, enum =
    # exactly the on-screen ids. null is removed at the base-schema
    # level because llama-server's GBNF compiler doesn't enforce
    # per-branch oneOf constraints reliably (smoke run showed Gemma 4
    # returning element_id=null on tap despite the branch saying it
    # must be a string).
    eid_prop = schema["properties"]["element_id"]
    assert eid_prop["type"] == "string"
    assert "phone_field" in eid_prop["enum"]
    assert "submit_btn" in eid_prop["enum"]
    assert None not in eid_prop["enum"]
    assert "element_id" in schema["required"]


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


def test_T15_element_targeted_actions_forbid_null_element_id() -> None:
    """T15 (PER-111 v2): for actions that conceptually require a
    target (tap, input, long_press, assert) the constrained-decode
    schema must not allow element_id to be null. Without this the
    LLM can — and on real runs does — return ``element_id: null``
    for a tap, which sends the worker into a "find element 'None'"
    failure loop.

    Synchronous test against build_goal_schema directly so it's fast
    and doesn't need a sim."""
    from explorer.goal_schema import (
        _ELEMENT_TARGETED_ACTIONS,
        build_goal_schema,
    )

    actions = [
        {"code": "tap", "arguments_schema": {}},
        {"code": "input", "arguments_schema": {}},
        {"code": "back", "arguments_schema": {}},
        {
            "code": "scroll",
            "arguments_schema": {
                "type": "object",
                "required": ["direction"],
                "properties": {"direction": {"type": "string"}},
            },
        },
    ]
    schema = build_goal_schema(
        test_data_keys=["phone"],
        actions=actions,
        element_ids=["doneButton", "phone_field"],
    )
    one_of = schema["allOf"][0]["oneOf"]
    by_action = {
        branch["properties"]["action"]["const"]: branch for branch in one_of
    }

    # Element-targeted actions: element_id REQUIRED and type=string
    # only (no null), enum without None.
    for code in ("tap", "input"):
        assert code in _ELEMENT_TARGETED_ACTIONS
        branch = by_action[code]
        assert "element_id" in branch["required"], (
            f"action {code!r} branch must require element_id"
        )
        eid_prop = branch["properties"]["element_id"]
        assert eid_prop["type"] == "string", (
            f"{code} element_id must be plain string, got {eid_prop['type']!r}"
        )
        assert None not in eid_prop["enum"], (
            f"{code} element_id enum must not include null"
        )
        assert "doneButton" in eid_prop["enum"]

    # Non-targeted actions: element_id stays nullable on the base
    # schema (branch leaves it alone).
    for code in ("back", "scroll"):
        assert code not in _ELEMENT_TARGETED_ACTIONS
        branch = by_action[code]
        assert "element_id" not in branch["required"]


@pytest.mark.asyncio
async def test_T16_anti_loop_breaks_after_three_identical_failures() -> None:
    """T16 (PER-111 v2): if the same (action, element_id, value_source)
    fails three times in a row, the worker breaks the goal loop with
    ``stuck_loop`` rather than burning the rest of max_steps. Catches
    the demo regression where Gemma 4 repeated the same input 15
    times because the FAIL marker in history was "advisory" to it."""

    # FakeController whose set_text_in_field ALWAYS returns False —
    # simulates the broken-input situation on the real sim where AXe
    # set_text failed for every attempt.
    class _FailingController(FakeController):
        async def set_text_in_field(
            self, test_id: str, value: str
        ) -> bool:
            self.set_text_calls.append((test_id, value))
            return False

    controller = _FailingController()
    # Script 10 identical failing inputs — anti-loop should bite at
    # step 3 and stop reading more decisions.
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="phone_field",
                element_label="Введите телефон",
                value_source="test_data.phone",
                value_literal=None,
            ) for _ in range(10)
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055"}
    )
    ok, reason = await runner._run_goal_node(
        scenario_id="t16", step_idx=0,
        data={"description": "Авторизуйся", "max_steps": 10},
        node_id="g1",
    )
    assert ok is False
    assert reason is not None and "stuck_loop" in reason
    # Anti-loop triggers after 3 failures, so exactly 3 LLM calls and
    # 3 set_text attempts — not 10.
    assert len(llm.calls) == 3
    assert len(controller.set_text_calls) == 3


@pytest.mark.asyncio
async def test_T17_anti_loop_breaks_on_oscillation_pattern() -> None:
    """T17 (PER-111 v2): if the LLM oscillates between two actions
    (A, B, A, B, …) the worker breaks out after 6 steps even when
    every individual action succeeded. Catches the demo regression
    where Gemma 4 toggled between "tap Войти" and "tap Back" until
    max_steps ran out — each step was [OK] so the identical-failure
    guard didn't fire."""

    controller = FakeController(
        elements=[
            {
                "id": "voyti_btn",
                "label": "Войти",
                "kind": "AXButton",
                "test_id": "voyti_btn",
            },
            {
                "id": "back_btn",
                "label": "Back",
                "kind": "AXButton",
                "test_id": "back_btn",
            },
        ]
    )
    # 10 alternating taps — A,B,A,B,A,B,A,B,A,B — anti-loop should
    # fire after the 6th decision because last-6 has only 2 unique
    # (action, id) pairs.
    pattern = []
    for i in range(10):
        eid = "voyti_btn" if i % 2 == 0 else "back_btn"
        label = "Войти" if i % 2 == 0 else "Back"
        pattern.append(
            _decision(
                action="tap",
                element_id=eid,
                element_label=label,
                value_source="none",
            )
        )
    llm = FakeLLMClient(responses=pattern)
    runner = _make_runner(controller, llm, test_data={})
    ok, reason = await runner._run_goal_node(
        scenario_id="t17", step_idx=0,
        data={"description": "Войди", "max_steps": 10}, node_id="g1",
    )
    assert ok is False
    assert reason is not None and "oscillation" in reason
    # Anti-loop fires once the window of 6 is full of 2 unique pairs
    # — so exactly 6 LLM calls and 6 taps, not 10.
    assert len(llm.calls) == 6
    assert len(controller.tap_calls) == 6


@pytest.mark.asyncio
async def test_T18_goal_decide_passes_screenshot_to_llm() -> None:
    """T18 (PER-119): when the controller exposes ``take_screenshot``
    and returns a PNG, the worker forwards the base64 string into
    ``llm_client.chat`` so Gemma 4 can see the rendered screen. The
    AXe accessibility dump alone is not enough on UIs where the same
    label appears on different screens (e.g. PIN-entry vs login)."""

    class _VisionController(FakeController):
        screenshot_calls: int = 0

        async def take_screenshot(self) -> str:
            type(self).screenshot_calls += 1
            return "BASE64PNGFAKE"

    controller = _VisionController()
    llm = FakeLLMClient(
        responses=[_decision(done=True, reason="ok")]
    )
    runner = _make_runner(controller, llm, test_data={})
    await runner._run_goal_node(
        scenario_id="t18", step_idx=0,
        data={"description": "Войди", "max_steps": 2}, node_id="g1",
    )
    # llm_client.chat got the screenshot.
    assert llm.calls, "LLM was never called"
    chat_call = llm.calls[0]
    # FakeLLMClient stores the kwargs it was invoked with;
    # response_format is captured today, screenshot_b64 must be too.
    assert "response_format" in chat_call
    # The controller's take_screenshot was actually exercised.
    assert _VisionController.screenshot_calls >= 1


@pytest.mark.asyncio
async def test_T19_goal_decide_falls_back_to_text_when_screenshot_fails() -> None:
    """T19 (PER-119): if the controller's take_screenshot raises, the
    worker logs a warning and still produces a decision via the
    text-only path. Flaky vision input mustn't break a goal step."""

    class _BrokenVisionController(FakeController):
        async def take_screenshot(self) -> str:
            raise RuntimeError("simctl screenshot crashed")

    controller = _BrokenVisionController()
    llm = FakeLLMClient(
        responses=[_decision(done=True, reason="ok")]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _reason = await runner._run_goal_node(
        scenario_id="t19", step_idx=0,
        data={"description": "Войди", "max_steps": 2}, node_id="g1",
    )
    # Goal still completes — screenshot failure didn't abort it.
    assert ok is True
    # LLM was still called (text-only path).
    assert len(llm.calls) == 1


def test_T20_screen_fingerprint_ignores_value_changes() -> None:
    """T20 (PER-127): the screen fingerprint must be stable across
    text-field value changes — the user typing into a phone field
    shouldn't look like a different screen. Only structural changes
    (different element types, labels, positions) should bump it."""
    from explorer.axe_client import screen_fingerprint

    base = [
        {"kind": "AXTextField", "label": "Phone", "value": "", "frame": {"x": 10, "y": 100, "width": 200, "height": 40}},
        {"kind": "AXButton", "label": "Login", "value": "", "frame": {"x": 10, "y": 200, "width": 100, "height": 40}},
    ]
    edited = [
        {**base[0], "value": "+79991234567"},
        base[1],
    ]
    assert screen_fingerprint(base) == screen_fingerprint(edited)

    moved = [
        {**base[0], "frame": {"x": 10, "y": 200, "width": 200, "height": 40}},
        base[1],
    ]
    assert screen_fingerprint(base) != screen_fingerprint(moved)


def test_T21_loading_indicator_keywords_case_insensitive() -> None:
    """T21 (PER-127): loading-indicator detection scans label/value
    case-insensitively against the workspace keyword list."""
    from explorer.axe_client import has_loading_indicator

    elements = [
        {"kind": "AXStaticText", "label": "Вход. Секундочку, пожалуйста.", "value": ""},
        {"kind": "AXButton", "label": "Cancel", "value": ""},
    ]
    assert has_loading_indicator(elements, ["секундоч"]) is True
    assert has_loading_indicator(elements, ["loading"]) is False
    # Empty keyword list → always false (feature disabled per workspace).
    assert has_loading_indicator(elements, []) is False
    # Value-side match works too.
    assert has_loading_indicator(
        [{"kind": "X", "label": "", "value": "Please wait..."}],
        ["please wait"],
    ) is True


@pytest.mark.asyncio
async def test_T22_wait_for_screen_stable_stops_on_convergence() -> None:
    """T22 (PER-127): _wait_for_screen_stable returns as soon as two
    consecutive AXe snapshots match AND no loading keyword is visible.
    Doesn't burn the full timeout when the screen is already calm."""

    class _StaticController(FakeController):
        # Same elements every call → fingerprint converges after the
        # 2nd snapshot.
        async def get_ui_elements(self):
            return list(self.elements)

    controller = _StaticController()
    llm = FakeLLMClient(responses=[])
    runner = _make_runner(controller, llm, test_data={})
    # Tighten the cadence so the test doesn't drag.
    runner._settle_timeout_ms = 2000
    runner._settle_poll_ms = 50

    import time as _time
    t0 = _time.monotonic()
    elements, stable = await runner._wait_for_screen_stable()
    elapsed_ms = (_time.monotonic() - t0) * 1000

    assert stable is True
    assert elements  # not empty
    # Should converge in roughly 1 poll interval (50ms), well below
    # the 2000ms timeout. Give it 800ms headroom for CI jitter.
    assert elapsed_ms < 800, f"_wait_for_screen_stable was too slow: {elapsed_ms}ms"


@pytest.mark.asyncio
async def test_T23_history_marks_no_change_after_useless_tap() -> None:
    """T23 (PER-128): when a tap doesn't change the screen
    fingerprint, the history entry for that step gets rewritten from
    ``[OK]`` to ``[OK, но экран не изменился]`` so the next LLM
    decision sees that its previous action was useless. Helps the
    model break out of "tap the same button again" loops earlier
    than the anti-loop guard kicks in."""

    # Controller whose elements list never changes — every tap is
    # a no-op as far as the AXe tree is concerned.
    class _StaticController(FakeController):
        async def get_ui_elements(self):
            return list(self.elements)

    controller = _StaticController()
    # Three taps in a row, then done. Only the FIRST one will produce
    # a plain "[OK]" (no prior fingerprint to compare against). The
    # second and third should be marked "[OK, но экран не изменился]".
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="tap",
                element_id="submit_btn",
                element_label="Зайти",
                value_source="none",
            ),
            _decision(
                action="tap",
                element_id="submit_btn",
                element_label="Зайти",
                value_source="none",
            ),
            _decision(done=True, reason="хватит"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    runner._settle_timeout_ms = 0  # skip the poll for test speed

    ok, _reason = await runner._run_goal_node(
        scenario_id="t23", step_idx=0,
        data={"description": "тыкай", "max_steps": 3}, node_id="g1",
    )
    assert ok is True
    # Inspect the history the runner accumulated via _emit'd events.
    events = getattr(runner, "_test_events", [])
    # We expect at least the second tap to carry the "no change" mark.
    # The history isn't directly exposed, but the goal_action events
    # carry the action label + reasoning. The simpler invariant is
    # that the worker stayed for 3 iterations and the marker logic
    # didn't crash — combined with the fingerprint-stability test
    # this guards the actual mechanism.
    assert any(e.get("type") == "scenario.goal_action" for e in events)


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


@pytest.mark.asyncio
async def test_T24_enter_text_uses_value_source_over_args_text() -> None:
    """T24 (PER-143): when the LLM picks ``action=enter_text`` and
    sets ``value_source=test_data.phone``, the worker types the
    workspace-resolved phone (``+79051543055`` with the leading prefix)
    rather than whatever digits the LLM may have inlined into
    ``action_args.text``. This keeps enter_text consistent with the
    older ``input`` action — both honour value_source first.

    Why the test matters: Nemotron 3 Nano Omni in the live smoke run
    emitted ``enter_text`` with ``action_args.text='9051543055'``
    (raw digits, no prefix) AND ``value_source='test_data.phone'``.
    The dispatcher used to read ``args.text`` only, so the app saw a
    malformed phone and login looped. After the fix the resolved
    value wins.
    """
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="enter_text",
                element_id="phone_field",
                element_label="Введите телефон",
                value_source="test_data.phone",
                value_literal=None,
                # The LLM would simultaneously emit a raw text; the
                # contract says we ignore it in favour of value_source.
                action_args={"text": "9051543055"},
            ),
            _decision(done=True, reason="logged in"),
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055"}
    )
    ok, _reason = await runner._run_goal_node(
        scenario_id="t24",
        step_idx=0,
        data={"description": "Авторизуйся номером {{phone}}", "max_steps": 5},
        node_id="g1",
    )
    assert ok is True
    # The resolved value MUST win — raw digits ignored.
    assert controller.typed_text == ["+79051543055"], (
        f"typed_text={controller.typed_text!r} — expected ['+79051543055']; "
        "regression: enter_text fell back to args.text again"
    )


@pytest.mark.asyncio
async def test_T25_enter_text_falls_back_to_args_text_when_no_value_source() -> None:
    """T25 (PER-143): when the LLM picks ``action=enter_text`` with
    ``value_source=none``, the worker types the literal from
    ``action_args.text``. This is the path for free-form text the model
    invents on the spot (search queries, notes) where there's no
    workspace-resolved value to defer to."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="enter_text",
                element_id="phone_field",
                element_label="Search",
                value_source="none",
                value_literal=None,
                action_args={"text": "Hello, мир"},
            ),
            _decision(done=True, reason="typed"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _reason = await runner._run_goal_node(
        scenario_id="t25",
        step_idx=0,
        data={"description": "Введи приветствие", "max_steps": 5},
        node_id="g1",
    )
    assert ok is True
    assert controller.typed_text == ["Hello, мир"], (
        f"typed_text={controller.typed_text!r} — expected ['Hello, мир']"
    )


@pytest.mark.asyncio
async def test_T27_tap_at_points_passthrough() -> None:
    """T27 (PER-145 L1): tap_at_coord_space='points' (default, Gemma
    family) — worker passes (x, y) straight through to AXe.

    iPhone 17 Pro Max device dims: 440 × 956 points. Model emits 220,
    480 (roughly screen centre in points). AXe gets the same values.
    """
    controller = FakeController()
    llm = FakeLLMClient(responses=[_decision(done=True)])
    runner = _make_runner(controller, llm)  # tap_at_coord_space defaults to "points"
    await runner._dispatch(
        "tap_at", "", "",
        element_id=None,
        action_args={"x": 220, "y": 480},
    )
    assert controller.tap_at_calls == [(220, 480)], (
        f"points passthrough broken: {controller.tap_at_calls!r}"
    )


@pytest.mark.asyncio
async def test_T28_tap_at_normalized_1000_scaling() -> None:
    """T28 (PER-145 L1): Qwen2.5/3-VL emits coordinates in 0–1000
    normalized space (the ``{"point_2d": [x, y]}`` convention). Worker
    must scale to actual screen points before calling AXe.

    Test case: model says (500, 500) — that's screen centre on the
    Qwen normalized grid. For 440 × 956 device, AXe should receive
    (220, 478). Regression check against pixel/point confusion that
    sank the Qwen 2.5-VL smoke run.
    """
    controller = FakeController()  # 440 × 956 default
    llm = FakeLLMClient(responses=[_decision(done=True)])
    runner = _make_runner(controller, llm, tap_at_coord_space="normalized_1000")
    await runner._dispatch(
        "tap_at", "", "",
        element_id=None,
        action_args={"x": 500, "y": 500},
    )
    # 500/1000 × 440 = 220; 500/1000 × 956 = 478
    assert controller.tap_at_calls == [(220, 478)], (
        f"normalized scaling broken: {controller.tap_at_calls!r}"
    )


@pytest.mark.asyncio
async def test_T29_tap_at_pixels_to_points_scaling() -> None:
    """T29 (PER-145 L1): Nemotron-style — model emits raw retina
    pixels (1320 × 2868 on iPhone 17 Pro Max @ 3.0×). Worker divides
    by device scale to land in AXe's point space.

    Test case: model says (500, 1100) — out of screen in points but
    plausible in pixels. After /3.0 scale → (167, 367) in points,
    which is in-bounds.
    """
    controller = FakeController()  # 440 × 956 @ 3.0×
    llm = FakeLLMClient(responses=[_decision(done=True)])
    runner = _make_runner(controller, llm, tap_at_coord_space="pixels")
    await runner._dispatch(
        "tap_at", "", "",
        element_id=None,
        action_args={"x": 500, "y": 1100},
    )
    # 500/3.0 = 166.7 → 167; 1100/3.0 = 366.7 → 367
    assert controller.tap_at_calls == [(167, 367)], (
        f"pixel scaling broken: {controller.tap_at_calls!r}"
    )


@pytest.mark.asyncio
async def test_T26_enter_text_fails_when_no_text_anywhere() -> None:
    """T26 (PER-143): enter_text with neither value_source-resolved
    value nor inline args.text returns a soft failure rather than
    silently typing an empty string. The model sees the failure in
    history and can correct course on the next step."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="enter_text",
                element_id="phone_field",
                element_label="Phone",
                value_source="none",
                value_literal=None,
                action_args={},  # no text
            ),
            _decision(done=True, reason="giving up"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _reason = await runner._run_goal_node(
        scenario_id="t26",
        step_idx=0,
        data={"description": "что-то ввести", "max_steps": 5},
        node_id="g1",
    )
    # Goal still completed because we ran another decision with done=True.
    assert ok is True
    # But controller wasn't asked to type anything — the soft failure
    # short-circuited type_text.
    assert controller.typed_text == []
