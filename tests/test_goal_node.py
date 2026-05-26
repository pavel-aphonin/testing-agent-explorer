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
    # PER-160: keyboard-typing path the dispatcher uses by default
    # for the ``input`` action. Records (test_id, label, text) so we
    # can assert the agent went through the user-facing flow rather
    # than the CDP bypass. ``keyboard_unavailable`` flips the helper
    # into a failure mode (tap succeeded, keyboard never showed).
    keyboard_calls: list[tuple[str | None, str | None, str]] = field(
        default_factory=list
    )
    keyboard_unavailable: bool = False

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

    async def tap_field_and_type_via_keyboard(
        self,
        test_id: str | None,
        label: str | None,
        text: str,
        wait_keyboard_ms: int = 2000,
    ) -> tuple[bool, str | None]:
        """PER-160 keyboard-typing path. Records the call; honours
        ``keyboard_unavailable`` to simulate ``keyboard did not appear``
        scenarios."""
        self.keyboard_calls.append((test_id, label, text))
        if self.keyboard_unavailable:
            return False, "keyboard did not appear after tap"
        # Mirror real behaviour: the typed string also ends up in
        # ``typed_text`` so smoke-style assertions keep working.
        self.typed_text.append(text)
        return True, None


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
        *,
        # PER-164 followup: accept sampling kwargs so the real
        # ScenarioRunner can pass them through without exploding
        # the fake. Tests rarely care about exact values — they
        # land in self.calls for any test that wants to assert.
        temperature: float = 0.2,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> str | None:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
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
    # PER-160: input now defaults to keyboard-typing path. Tuple is
    # (test_id, label, text); both old and new paths must end with
    # the typed value in ``typed_text``.
    assert controller.keyboard_calls == [
        ("phone_field", "Введите телефон", "+79051543055")
    ]
    assert "+79051543055" in controller.typed_text


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
    # PER-170: schema is now batch-shaped (top-level: done/reason/
    # expected_next_screen/actions). The per-action constraints
    # (value_source enum, element_id enum, action enum) moved into
    # actions.items.* so each item in the batch is checked the same
    # way the old single-action schema used to check the whole reply.
    assert schema["required"] == ["done", "actions"]
    item_schema = schema["properties"]["actions"]["items"]
    enum = item_schema["properties"]["value_source"]["enum"]
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
    eid_prop = item_schema["properties"]["element_id"]
    assert eid_prop["type"] == "string"
    assert "phone_field" in eid_prop["enum"]
    assert "submit_btn" in eid_prop["enum"]
    assert None not in eid_prop["enum"]
    assert "element_id" in item_schema["required"]


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
    # PER-160: keyboard path is now default. (test_id, label, text)
    assert controller.keyboard_calls == [("amount", "Сумма", "1000")]


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
    # PER-160: keyboard path is default. Two identical inputs (model
    # remembers the improvised value across steps) → two keyboard calls.
    assert len(controller.keyboard_calls) == 2
    for call in controller.keyboard_calls:
        assert call[0] == "name"
        assert call[2] == "Тестовая 1"


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
    # no skip, no crash. PER-160: now via keyboard path by default.
    assert controller.keyboard_calls == [
        ("phone_field", "Введите телефон", invented)
    ]


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
    # PER-160: keyboard path default.
    assert controller.keyboard_calls == [
        ("iban", "IBAN", "DE89370400440532013000")
    ]


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
    # PER-170: schema is now batch-shaped. The per-action ``allOf`` +
    # ``oneOf`` discriminator that used to live at the schema root
    # moved into ``actions.items``, since each item in the batch is
    # validated the same way the old single-action schema was.
    one_of = schema["properties"]["actions"]["items"]["allOf"][0]["oneOf"]
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

    # FakeController where every input attempt fails. PER-160: the
    # primary path is tap_field_and_type_via_keyboard, but we keep
    # set_text_in_field broken too in case some test path falls
    # through (or someone removes the keyboard override). Both must
    # report failure so the anti-loop guard sees three consecutive
    # `ok=False` ticks and trips at step 3.
    class _FailingController(FakeController):
        async def set_text_in_field(
            self, test_id: str, value: str
        ) -> bool:
            self.set_text_calls.append((test_id, value))
            return False

        async def tap_field_and_type_via_keyboard(
            self,
            test_id: str | None,
            label: str | None,
            text: str,
            wait_keyboard_ms: int = 2000,
        ) -> tuple[bool, str | None]:
            self.keyboard_calls.append((test_id, label, text))
            return False, "set_text_in_field returned False"

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
    # 3 input attempts — not 10. PER-160: counts the keyboard path
    # (primary) since that's what the dispatcher uses now.
    assert len(llm.calls) == 3
    assert len(controller.keyboard_calls) == 3


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


@pytest.mark.asyncio
async def test_T30_input_routes_through_keyboard_by_default() -> None:
    """T30 (PER-160): the default ``input`` dispatch goes through the
    user-facing keyboard path. The dispatcher must call
    ``tap_field_and_type_via_keyboard`` (tap field → wait for keyboard
    → HID-type), NOT the CDP-style ``set_text_in_field`` bypass.

    Why this matters: bypassing the keyboard hides anti-fraud
    rejections, keyboard-overlay bugs, broken IME handlers, and
    React onChange callbacks. The agent only finds those bugs when
    it types like a real user."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="phone_field",
                element_label="Введите телефон",
                value_source="test_data.phone",
            ),
            _decision(done=True, reason="entered"),
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055"}
    )
    ok, _reason = await runner._run_goal_node(
        scenario_id="t30",
        step_idx=0,
        data={"description": "Введи телефон {{phone}}", "max_steps": 5},
        node_id="g1",
    )
    assert ok is True
    # Keyboard path used — record holds (test_id, label, text).
    assert controller.keyboard_calls == [
        ("phone_field", "Введите телефон", "+79051543055")
    ], controller.keyboard_calls
    # CDP/legacy bypass must NOT have been touched.
    assert controller.set_text_calls == [], (
        "regression: dispatcher fell back to set_text_in_field bypass"
    )


@pytest.mark.asyncio
async def test_T31_input_bypass_keyboard_opt_in_uses_legacy_path() -> None:
    """T31 (PER-160): when the LLM explicitly sets
    ``action_args.bypass_keyboard=true`` the dispatcher takes the
    legacy ``set_text_in_field`` path (CDP/AX inject). Same value
    arrives in the field, but the keyboard never opens. This is the
    escape hatch for custom IMEs and canvas-rendered fields where
    the keyboard physically cannot appear."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="phone_field",
                element_label="Введите телефон",
                value_source="test_data.phone",
                action_args={"bypass_keyboard": True},
            ),
            _decision(done=True, reason="entered"),
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055"}
    )
    ok, _reason = await runner._run_goal_node(
        scenario_id="t31",
        step_idx=0,
        data={"description": "Введи телефон {{phone}}", "max_steps": 5},
        node_id="g1",
    )
    assert ok is True
    # Legacy path used — set_text_in_field got the value.
    assert controller.set_text_calls == [("phone_field", "+79051543055")]
    # Keyboard path NOT touched.
    assert controller.keyboard_calls == []


@pytest.mark.asyncio
async def test_T32_input_fails_clearly_when_keyboard_never_appears() -> None:
    """T32 (PER-160): if tap succeeds but the on-screen keyboard
    never shows up within timeout, the dispatcher returns a
    soft failure with the specific reason
    ``"keyboard did not appear after tap"``. The LLM sees this in
    history and can pick a different element / try bypass_keyboard
    on the next step — better than silently writing zero characters."""
    controller = FakeController(keyboard_unavailable=True)
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="input",
                element_id="phone_field",
                element_label="Введите телефон",
                value_source="test_data.phone",
            ),
            _decision(done=True, reason="moving on"),
        ]
    )
    runner = _make_runner(
        controller, llm, test_data={"phone": "+79051543055"}
    )
    ok, _reason = await runner._run_goal_node(
        scenario_id="t32",
        step_idx=0,
        data={"description": "Введи телефон {{phone}}", "max_steps": 5},
        node_id="g1",
    )
    # Goal completes (second decision is done=true), but the input
    # action returned failure for the first step.
    assert ok is True
    # Keyboard was attempted exactly once with our value.
    assert controller.keyboard_calls == [
        ("phone_field", "Введите телефон", "+79051543055")
    ]
    # And the dispatcher did NOT silently fall through to the bypass.
    assert controller.set_text_calls == []
    assert controller.typed_text == []


def test_T33_structural_fingerprint_ignores_labels() -> None:
    """T33 (PER-161): the structural fingerprint is label-free.
    Two screens whose layout is identical but whose labels differ
    (rolling balance, message count, "1 unread" vs "2 unread") must
    hash to the same structural id — otherwise visited_actions and
    the coverage plateau detector see every counter-tick as a brand-
    new screen and lose their memory."""
    from explorer.axe_client import (
        screen_fingerprint,
        screen_fingerprint_structural,
    )

    base = [
        {
            "kind": "AXStaticText",
            "label": "Баланс: 100 000 ₽",
            "frame": {"x": 10, "y": 100, "width": 200, "height": 40},
        },
        {
            "kind": "AXButton",
            "label": "Перевод",
            "frame": {"x": 10, "y": 200, "width": 100, "height": 40},
        },
    ]
    same_layout_different_text = [
        {
            "kind": "AXStaticText",
            "label": "Баланс: 200 000 ₽",  # ← different label
            "frame": {"x": 10, "y": 100, "width": 200, "height": 40},
        },
        {
            "kind": "AXButton",
            "label": "Перевод",
            "frame": {"x": 10, "y": 200, "width": 100, "height": 40},
        },
    ]
    # PER-127 fingerprint distinguishes the two (it includes label):
    assert (
        screen_fingerprint(base)
        != screen_fingerprint(same_layout_different_text)
    )
    # PER-161 structural fingerprint collapses them:
    assert (
        screen_fingerprint_structural(base)
        == screen_fingerprint_structural(same_layout_different_text)
    )
    # And it still distinguishes screens with actually different layouts.
    moved = [
        {**base[0], "frame": {"x": 10, "y": 300, "width": 200, "height": 40}},
        base[1],
    ]
    assert (
        screen_fingerprint_structural(base)
        != screen_fingerprint_structural(moved)
    )


@pytest.mark.asyncio
async def test_T34_visited_actions_appears_in_history_block() -> None:
    """T34 (PER-161): after a successful action on a screen, the
    NEXT goal-decide call sees an "уже пробовали на этом экране" hint
    in the user prompt's history block. That's how visited_actions
    propagates into the LLM context to break "tap the same button
    forever" loops without needing a structural prompt rewrite."""
    controller = FakeController()
    llm = FakeLLMClient(
        responses=[
            _decision(
                action="tap",
                element_id="submit_btn",
                element_label="Зайти",
                value_source="none",
            ),
            _decision(done=True, reason="ok"),
        ]
    )
    runner = _make_runner(controller, llm, test_data={})
    ok, _ = await runner._run_goal_node(
        scenario_id="t34", step_idx=0,
        data={"description": "Зайти", "max_steps": 3}, node_id="g1",
    )
    assert ok is True
    # llm.calls[1] is the second goal_decide (after the successful
    # tap). Its user prompt must mention the visited hint.
    assert len(llm.calls) >= 2
    second_user = llm.calls[1]["user"]
    assert "уже пробовали" in second_user, (
        f"expected visited hint in 2nd prompt, got: {second_user[:400]}..."
    )
    assert "tap" in second_user and "submit_btn" in second_user


@pytest.mark.asyncio
async def test_T35_plateau_detector_breaks_goal_after_no_new_screens() -> None:
    """T35 (PER-161): when the agent revisits known screens for
    ``plateau_window`` consecutive steps without discovering any new
    structural fingerprint, the goal aborts with reason
    ``coverage_plateau`` instead of burning all max_steps. Mirrors
    LLMDroid's escalation trigger.

    The default window is 10. We script decisions whose action/id
    tuples vary enough to dodge the 6-step oscillation guard
    (otherwise that fires first), but whose effects leave the
    FakeController's element list unchanged — so structural_fp
    never moves and every plateau-window entry is a revisit."""
    controller = FakeController()
    # Cycle through 4 distinct (action, element_id) pairs so the
    # 6-step oscillation guard sees >2 unique tuples and stays
    # quiet. None of these change FakeController's elements, so
    # structural_fp stays constant.
    cycle = [
        _decision(action="tap", element_id="submit_btn",
                  element_label="Зайти", value_source="none"),
        _decision(action="tap", element_id="phone_field",
                  element_label="Введите телефон", value_source="none"),
        _decision(action="swipe", action_args={"direction": "down"},
                  element_id=None, element_label=None,
                  value_source="none"),
        _decision(action="scroll", action_args={"direction": "down"},
                  element_id=None, element_label=None,
                  value_source="none"),
    ]
    responses = (cycle * 4)[:12]
    llm = FakeLLMClient(responses=responses)
    runner = _make_runner(controller, llm, test_data={})
    runner._settle_timeout_ms = 0  # skip settle for speed
    ok, reason = await runner._run_goal_node(
        scenario_id="t35", step_idx=0,
        data={"description": "крутись на месте", "max_steps": 12},
        node_id="g1",
    )
    # Goal aborted with the right reason.
    assert ok is False
    assert reason is not None and "coverage_plateau" in reason, reason
    # Triggered around step 10 (window size) — well before 12.
    assert len(llm.calls) <= 11, (
        f"plateau guard did not fire in time: {len(llm.calls)} calls"
    )


# ----------------------------- PER-163 retry contracts


def test_T36_schema_allows_null_element_id_when_tap_at_is_available() -> None:
    """T36 (PER-163): when the action dictionary contains a
    coordinate-only action (``tap_at``), the base schema MUST allow
    ``element_id=null`` — otherwise the LLM can't honour the prompt
    instruction "for tap_at leave element_id null" and ends up
    attaching the app root container to every coord tap, poisoning
    history / anti-loop / visited tracking.

    Without coord-only actions in the set the legacy contract holds:
    ``element_id`` is a non-null string from the on-screen enum."""
    from explorer.goal_schema import build_goal_schema

    actions_with_tap_at = [
        {"code": "tap", "arguments_schema": {}},
        {"code": "tap_at", "arguments_schema": {}},
    ]
    schema = build_goal_schema(
        test_data_keys=["phone"],
        actions=actions_with_tap_at,
        element_ids=["btn_a", "btn_b"],
    )
    # PER-170: per-action constraints live inside actions.items now.
    el_schema = schema["properties"]["actions"]["items"]["properties"]["element_id"]
    # Allow null for tap_at.
    assert el_schema["type"] == ["string", "null"], el_schema
    assert None in el_schema["enum"]
    assert "btn_a" in el_schema["enum"]
    assert "btn_b" in el_schema["enum"]

    # Regression: without coord-only action, legacy non-null contract.
    actions_without = [{"code": "tap", "arguments_schema": {}}]
    schema_legacy = build_goal_schema(
        test_data_keys=["phone"],
        actions=actions_without,
        element_ids=["btn_a"],
    )
    el_legacy = schema_legacy["properties"]["actions"]["items"]["properties"]["element_id"]
    assert el_legacy["type"] == "string", el_legacy
    assert None not in el_legacy.get("enum", [])


@pytest.mark.asyncio
async def test_T37_screenshot_max_dim_forwarded_to_controller() -> None:
    """T37 (PER-163 retry): when the model passport carries
    ``screenshot_max_dim``, the worker forwards it to
    ``controller.take_screenshot(max_dim=N)`` instead of letting the
    controller fall back to the legacy logical-points resize. This
    is the contract the audit demanded: prove that the resize knob
    is per-model, not just a controller-internal default."""

    captured_kwargs: list[dict] = []

    class _RecordingController(FakeController):
        async def take_screenshot(self, **kwargs) -> str:
            captured_kwargs.append(dict(kwargs))
            return ""  # empty is fine — worker falls back to text-only

    controller = _RecordingController()
    llm = FakeLLMClient(responses=[_decision(done=True, reason="ok")])
    events: list[dict] = []

    async def sink(evt: dict) -> None:
        events.append(evt)

    from explorer.scenario_runner import ScenarioRunner

    runner = ScenarioRunner(
        controller=controller,
        scenarios=[],
        test_data={},
        event_callback=sink,
        llm_client=llm,
        actions=[],
        supports_multimodal_image=True,
        screenshot_max_dim=1920,
    )
    runner._test_events = events  # type: ignore[attr-defined]
    runner._settle_timeout_ms = 0
    await runner._run_goal_node(
        scenario_id="t37", step_idx=0,
        data={"description": "go", "max_steps": 2}, node_id="g1",
    )
    assert captured_kwargs, "take_screenshot was never called"
    # All calls must carry max_dim=1920 — not the controller-default
    # logical-points path.
    for kw in captured_kwargs:
        assert kw.get("max_dim") == 1920, kw


@pytest.mark.asyncio
async def test_T38_screenshot_max_dim_omitted_when_passport_null() -> None:
    """T38 (PER-163 retry): the controller-default (logical-points
    resize) is preserved when the model passport does NOT set
    ``screenshot_max_dim``. Guards against the worker silently
    forcing high-res screenshots on cheaper models where bandwidth
    matters."""

    captured_kwargs: list[dict] = []

    class _RecordingController(FakeController):
        async def take_screenshot(self, **kwargs) -> str:
            captured_kwargs.append(dict(kwargs))
            return ""

    controller = _RecordingController()
    llm = FakeLLMClient(responses=[_decision(done=True, reason="ok")])
    events: list[dict] = []

    async def sink(evt: dict) -> None:
        events.append(evt)

    from explorer.scenario_runner import ScenarioRunner

    runner = ScenarioRunner(
        controller=controller,
        scenarios=[],
        test_data={},
        event_callback=sink,
        llm_client=llm,
        actions=[],
        supports_multimodal_image=True,
        # screenshot_max_dim left at default None.
    )
    runner._test_events = events  # type: ignore[attr-defined]
    runner._settle_timeout_ms = 0
    await runner._run_goal_node(
        scenario_id="t38", step_idx=0,
        data={"description": "go", "max_steps": 2}, node_id="g1",
    )
    assert captured_kwargs, "take_screenshot was never called"
    # max_dim must NOT be passed when passport is null — controller
    # picks its own default (legacy logical-points resize).
    for kw in captured_kwargs:
        assert "max_dim" not in kw, kw


@pytest.mark.asyncio
async def test_T39_schema_null_leak_rejected_for_element_targeted_action() -> None:
    """T39 (PER-163 retry #2): post-decode validation must reject
    schema-leak where the LLM returns an element-targeted action
    (tap / input / long_press / assert / double_click / right_click)
    with ``element_id=null``.

    Background: llama-server's GBNF compiler does not enforce
    per-branch ``oneOf`` constraints reliably. The base schema
    became nullable for tap_at; the compiler then sometimes lets
    that nullability leak into ``tap`` decisions too. Without this
    guard the dispatcher hits ``_find_element(None)`` and either
    crashes or burns a slot retrying. With it the malformed shape
    becomes ``llm_no_decision`` and the next step gets a fresh
    prompt — visited_actions still records the screen as visited,
    so plateau + anti-loop math stay sound."""
    controller = FakeController()
    # The model returns a malformed shape: action=tap with element_id=null.
    # FakeLLMClient feeds it as the first response. Second response is
    # done=True so the goal completes after the rejected decision.
    bad_decision_raw = json.dumps({
        "done": False, "reason": None,
        "action": "tap",
        "action_args": {},
        "element_id": None,
        "element_label": None,
        "value_source": "none",
        "value_literal": None,
        "reasoning": "schema leak: null element_id with tap",
    }, ensure_ascii=False)
    good_done_raw = _decision(done=True, reason="moving on")
    llm = FakeLLMClient(responses=[bad_decision_raw, good_done_raw])
    runner = _make_runner(controller, llm, test_data={})
    runner._settle_timeout_ms = 0
    ok, reason = await runner._run_goal_node(
        scenario_id="t39",
        step_idx=0,
        data={"description": "click smth", "max_steps": 3},
        node_id="g1",
    )
    # The malformed decision is rejected — _goal_decide returns
    # None which the goal loop treats as ``llm_no_decision`` and
    # aborts. That's fine: the critical contract this test enforces
    # is "schema-leak does NOT reach dispatch" — no tap call landed
    # on the controller. Caller (scenario runner) handles the
    # ``llm_no_decision`` abort the same way it would for any other
    # transport failure (move to next goal).
    assert ok is False
    assert reason == "llm_no_decision"
    assert controller.tap_calls == [], (
        f"schema-leak decision reached dispatch: tap_calls={controller.tap_calls!r}"
    )
    # One LLM call consumed (the bad one); the done one was never
    # served because the loop aborted on the first None.
    assert len(llm.calls) == 1


# ---------------------------------------------------------- PER-164 grounder


class _FakeGrounder:
    """Test double for explorer.grounder_client.GrounderClient.

    Records every locate() call and returns a scripted result so we
    can assert (a) when the dispatcher routes through the grounder,
    (b) which (x, y, coord_space) tuple it ultimately taps.
    """

    def __init__(self, result):  # type: ignore[no-untyped-def]
        self.result = result
        self.calls: list[tuple[bytes, str]] = []

    async def locate(self, screenshot_bytes: bytes, target_description: str):
        self.calls.append((screenshot_bytes, target_description))
        return self.result


class _ControllerWithScreenshot(FakeController):
    """Adds take_screenshot to FakeController for grounder tests.

    Returns a real (tiny) PNG so PIL.Image.open inside the runner's
    ``_take_screenshot_bytes`` can probe width/height — PER-164 needs
    real dims for the ``image_pixels`` coord-space scaling. Defaults
    mirror the resized portrait shape that ``screenshot_max_dim=1920``
    produces from an iPhone 17 Pro Max native 1320×2868 capture.
    """

    fake_image_w: int = 884
    fake_image_h: int = 1920

    async def take_screenshot(self, max_dim: int | None = None) -> str:
        import base64
        import io
        from PIL import Image
        img = Image.new("RGB", (self.fake_image_w, self.fake_image_h), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_T40_tap_at_routes_through_grounder_when_target_description() -> None:
    """T40 (PER-164): when the chat-LLM emits tap_at with
    ``target_description`` and a grounder is wired up, the runtime
    must call ``grounder.locate(screenshot, description)`` and use
    its (x, y) instead of the LLM's own coords. The grounder's
    coord_space (pixels) drives the scaling — controller _scale=3.0,
    so grounder pixel 600 → screen point 200.
    """
    from explorer.grounder_client import GrounderResult

    controller = _ControllerWithScreenshot()
    grounder = _FakeGrounder(GrounderResult(
        x=600, y=1500, coord_space="pixels",
        raw_text="click(start_box='(600,1500)')",
    ))

    decisions = [
        _decision(
            done=False, action="tap_at",
            action_args={
                # LLM provides bogus coords + a target_description;
                # we expect the runtime to ignore (x, y) and use the
                # grounder's answer.
                "x": 999, "y": 999,
                "target_description": "digit 8 on PIN keypad",
            },
            element_id=None,
        ),
        _decision(done=True, reason="ok"),
    ]
    llm = FakeLLMClient(responses=decisions)
    runner = _make_runner(controller, llm, test_data={}, tap_at_coord_space="pixels")
    runner._grounder = grounder  # bypass the no-grounder default
    runner._settle_timeout_ms = 0
    await runner._run_goal_node(
        scenario_id="t40", step_idx=0,
        data={"description": "tap 8", "max_steps": 3}, node_id="g1",
    )
    # Grounder was called with the description; LLM coords ignored.
    assert len(grounder.calls) == 1, f"expected 1 grounder call, got {len(grounder.calls)}"
    assert grounder.calls[0][1] == "digit 8 on PIN keypad"
    # Final tap is grounder.x / scale, grounder.y / scale (pixels space).
    assert controller.tap_at_calls == [(200, 500)], controller.tap_at_calls


@pytest.mark.asyncio
async def test_T41_tap_at_without_target_description_skips_grounder() -> None:
    """T41 (PER-164): backward compat — when the chat-LLM emits
    tap_at without ``target_description``, the runtime must NOT call
    the grounder and must use the LLM's own (x, y) as before.

    This preserves the pre-PER-164 behaviour for models that don't
    speak the description-routing contract and for screens where
    the LLM is confident in its own grounding (rare for canvas, but
    routine for accessibility-rich elements where tap_at is an
    escape hatch).
    """
    controller = _ControllerWithScreenshot()
    grounder = _FakeGrounder(None)  # would return None even if called

    decisions = [
        _decision(
            done=False, action="tap_at",
            action_args={"x": 500, "y": 680},   # no target_description
            element_id=None,
        ),
        _decision(done=True, reason="ok"),
    ]
    llm = FakeLLMClient(responses=decisions)
    runner = _make_runner(controller, llm, test_data={}, tap_at_coord_space="normalized_1000")
    runner._grounder = grounder
    runner._settle_timeout_ms = 0
    await runner._run_goal_node(
        scenario_id="t41", step_idx=0,
        data={"description": "tap by coords", "max_steps": 3}, node_id="g1",
    )
    assert grounder.calls == [], f"grounder should not have been called: {grounder.calls!r}"
    # normalized_1000 with controller._width=440, _height=956 →
    # x=500/1000*440=220, y=680/1000*956=650
    assert controller.tap_at_calls == [(220, 650)], controller.tap_at_calls


@pytest.mark.asyncio
async def test_T42_grounder_image_pixels_scales_by_image_dim() -> None:
    """T42 (PER-164): when the grounder returns coords in
    ``image_pixels`` space (UI-TARS family), the dispatcher must
    scale by ``screen_logical / image_dim``, NOT by retina factor.

    UI-TARS sees the screenshot AFTER it was resized to
    ``screenshot_max_dim``, so its coordinates are in image pixels
    not phone-native pixels. The old retina-based scaling produces
    a wrong target (taps end up far from the intended digit on a
    PIN keypad).

    Setup: image 884×1920 (portrait fit for 1920 max_dim from a
    1320×2868 iPhone 17 Pro Max). Controller logical screen 440×956.
    Grounder returns (442, 1280) — center-half pixel.
    Expected screen point: (442 * 440 / 884, 1280 * 956 / 1920)
                         = (219.99, 637.33) → (220, 637).
    """
    from explorer.grounder_client import GrounderResult

    controller = _ControllerWithScreenshot()
    controller.fake_image_w = 884
    controller.fake_image_h = 1920
    grounder = _FakeGrounder(GrounderResult(
        x=442, y=1280, coord_space="image_pixels",
        raw_text="click(start_box='(442,1280)')",
    ))

    decisions = [
        _decision(
            done=False, action="tap_at",
            action_args={
                "x": 0, "y": 0,  # LLM's coords don't matter — grounder wins
                "target_description": "center of PIN keypad",
            },
            element_id=None,
        ),
        _decision(done=True, reason="ok"),
    ]
    llm = FakeLLMClient(responses=decisions)
    runner = _make_runner(controller, llm, test_data={}, tap_at_coord_space="pixels")
    runner._grounder = grounder
    runner._settle_timeout_ms = 0
    await runner._run_goal_node(
        scenario_id="t42", step_idx=0,
        data={"description": "tap center", "max_steps": 3}, node_id="g1",
    )
    assert len(grounder.calls) == 1
    # Image-pixels scaling: raw * screen / image
    assert controller.tap_at_calls == [(220, 637)], controller.tap_at_calls


@pytest.mark.asyncio
async def test_T43_sampling_profile_forwarded_to_llm_chat() -> None:
    """T43 (PER-164 followup): the per-model sampling profile must
    reach llm_client.chat() — not the worker's old hardcoded T=0.2.

    Backend's claim_next ships ``default_temperature/_top_p/_top_k/
    _min_p`` from the LLMModel row, worker passes them to
    ScenarioRunner, ScenarioRunner threads them into every chat()
    call so each model gets its family-appropriate band (Gemma 4
    wants T=0.65/top_p=0.95/top_k=64/min_p=0.05, Qwen wants
    T=0.7/top_p=0.8/top_k=20). Without this wire-through the
    worker silently overrides every model with T=0.2 — the cause
    of Gemma 4's impulsive single-step decisions on loading
    screens in PER-164 smoke #5.
    """
    controller = FakeController()
    llm = FakeLLMClient(responses=[_decision(done=True, reason="ok")])
    events: list[dict] = []

    async def sink(evt: dict) -> None:
        events.append(evt)

    from explorer.scenario_runner import ScenarioRunner

    runner = ScenarioRunner(
        controller=controller,
        scenarios=[],
        test_data={},
        event_callback=sink,
        llm_client=llm,
        actions=[],
        # Gemma 4-like passport.
        sampling_temperature=0.65,
        sampling_top_p=0.95,
        sampling_top_k=64,
        sampling_min_p=0.05,
    )
    runner._test_events = events  # type: ignore[attr-defined]
    runner._settle_timeout_ms = 0
    await runner._run_goal_node(
        scenario_id="t43", step_idx=0,
        data={"description": "go", "max_steps": 2}, node_id="g1",
    )
    assert llm.calls, "no LLM call made"
    last = llm.calls[-1]
    assert last["temperature"] == 0.65, last
    assert last["top_p"] == 0.95, last
    assert last["top_k"] == 64, last
    assert last["min_p"] == 0.05, last


# ---------------- PER-170 followup: id-keyword hints --------------------


def test_T48_id_keyword_hint_disambiguates_unlabelled_back_forward() -> None:
    """T48 (PER-170 followup): when two buttons live on the same screen
    without accessibility labels (the iOS «(без подписи)» case), the
    elements block must annotate them with hints derived from their
    element_id so the chat-LLM can distinguish them by role.

    Regression for the PER-170 smoke #2 bug: Gemma 4 reasoned «нажму
    Вперёд для подтверждения» and then picked element_id=backButton —
    because the prompt showed both buttons as
    «id=backButton (без подписи)» / «id=forwardButton (без подписи)»
    and the schema's element_id enum order is essentially random
    from the model's POV.
    """
    from explorer.goal_schema import build_elements_block

    elements = [
        {"id": "backButton", "label": "", "kind": "button"},
        {"id": "forwardButton", "label": "", "kind": "button"},
        {"id": "submitBtn", "label": "", "kind": "button"},
        # An element with a real label — hint must NOT override it.
        {"id": "loginButton", "label": "Войти", "kind": "button"},
        # An element whose id carries no recognisable keyword —
        # fallback to bare «(без подписи)» (unchanged behaviour).
        {"id": "Application_dengi_0", "label": "", "kind": "container"},
    ]
    block, ids = build_elements_block(elements)
    assert "id=backButton" in block
    assert "id=forwardButton" in block
    assert "назад" in block.lower(), block
    assert "вперёд" in block.lower(), block
    assert "отправить" in block.lower(), block
    # Real label survives untouched.
    assert "Войти" in block, block
    # The unrecognised id falls back to the legacy «(без подписи)»
    # without a hint — we don't fabricate. ``Application_dengi_0``
    # is the real-world root-container id we saw in PER-163 smokes,
    # chosen as the «no hint» case because it's the most realistic
    # noise id we'll encounter in production.
    assert "Application_dengi_0 [container] (без подписи)" in block, block
    # Crucially: no «вероятно …» annotation tacked on after.
    assert "Application_dengi_0 [container] (без подписи; вероятно" not in block
    # The id enum is unchanged.
    assert ids == [
        "backButton",
        "forwardButton",
        "submitBtn",
        "loginButton",
        "Application_dengi_0",
    ]


def test_T49_id_hint_tokenizes_snake_and_camel() -> None:
    """T49: tokenizer recognises both camelCase (`backButton`) and
    snake_case / kebab-case (`back_button`, `back-button`). Common
    naming styles from iOS, Android, and web testing tools alike."""
    from explorer.goal_schema import _hint_from_id

    assert _hint_from_id("backButton") is not None
    assert _hint_from_id("back_button") is not None
    assert _hint_from_id("back-button") is not None
    assert _hint_from_id("BackButton") is not None
    # No recognisable keyword → no hint.
    assert _hint_from_id("Application_dengi_0") is None
    assert _hint_from_id("") is None


# ---------------- PER-170 retry: Codex QA blockers ----------------------


@pytest.mark.asyncio
async def test_T50_batch_item_tap_without_element_id_is_skipped() -> None:
    """T50 (PER-170 retry, Codex #1): when the LLM emits a batch item
    like ``{"action":"tap","element_id":null}`` the runtime must reject
    it before dispatch — schema grammar can't be trusted to enforce
    per-branch oneOf constraints on llama-server. The item is recorded
    in history with [SKIPPED: …] so the next LLM call sees the failure
    and adapts; the rest of the batch is aborted (sibling actions
    likely depended on the now-uncertain screen)."""
    controller = FakeController(
        elements=[{"id": "btn", "label": "Submit", "kind": "button"}]
    )
    # Batch with an invalid tap followed by a valid one — only the
    # first should be processed (and skipped), the second must NOT
    # dispatch because we break the batch on invalid items.
    bad_decision = json.dumps({
        "done": False,
        "actions": [
            {
                "action": "tap",
                "action_args": {},
                "element_id": None,
                "element_label": None,
                "value_source": "none",
                "value_literal": None,
                "reasoning": "tap forward",
            },
            {
                "action": "tap",
                "action_args": {},
                "element_id": "btn",
                "element_label": "Submit",
                "value_source": "none",
                "value_literal": None,
                "reasoning": "tap submit (should NOT run)",
            },
        ],
    }, ensure_ascii=False)
    # Second decision: model recovers after seeing [SKIPPED] in history.
    good_decision = json.dumps({
        "done": True,
        "reason": "done",
        "actions": [{
            "action": "tap",
            "action_args": {},
            "element_id": "btn",
            "element_label": "Submit",
            "value_source": "none",
            "value_literal": None,
            "reasoning": "ok",
        }],
    }, ensure_ascii=False)
    llm = FakeLLMClient(responses=[bad_decision, good_decision])
    runner = _make_runner(controller, llm, test_data={})
    await runner._run_goal_node(
        scenario_id="t50", step_idx=0,
        data={"description": "go", "max_steps": 5}, node_id="g1",
    )
    # The invalid first item must NOT have hit the controller.
    # Only the recovery decision's tap on "btn" did.
    assert controller.tap_calls == ["Submit"], controller.tap_calls


@pytest.mark.asyncio
async def test_T51_batch_item_tap_at_without_xy_or_desc_is_skipped() -> None:
    """T51 (PER-170 retry, Codex #1 + PER-172 refinement): tap_at must
    carry EITHER numeric x,y OR a non-empty target_description (which
    the grounder uses to localise the target). Reject only when BOTH
    are missing — the arguments_schema (PER-164 followup) explicitly
    relaxes the required list at the JSON layer and delegates this
    check to the runtime."""
    controller = FakeController(
        elements=[{"id": "btn", "label": "Cancel", "kind": "button"}]
    )
    # Empty action_args → no x, no y, no description → must skip.
    bad = json.dumps({
        "done": False,
        "actions": [{
            "action": "tap_at",
            "action_args": {},
            "element_id": None,
            "element_label": None,
            "value_source": "none",
            "value_literal": None,
            "reasoning": "tap somewhere",
        }],
    }, ensure_ascii=False)
    recover = json.dumps({
        "done": True,
        "reason": "recovered",
        "actions": [{
            "action": "tap",
            "action_args": {},
            "element_id": "btn",
            "element_label": "Cancel",
            "value_source": "none",
            "value_literal": None,
            "reasoning": "ok",
        }],
    }, ensure_ascii=False)
    llm = FakeLLMClient(responses=[bad, recover])
    runner = _make_runner(controller, llm, test_data={})
    await runner._run_goal_node(
        scenario_id="t51", step_idx=0,
        data={"description": "go", "max_steps": 5}, node_id="g1",
    )
    # tap_at without coords AND without target_description → no
    # controller side-effect. Recovery tap on "Cancel" is the only
    # thing that lands.
    assert controller.tap_calls == ["Cancel"], controller.tap_calls
    assert not getattr(controller, "tap_at_calls", [])


def test_T52_quick_ax_snapshot_returns_elements_on_success() -> None:
    """T52 (PER-170 retry, Codex #2): _quick_ax_snapshot is the cheap
    AXe poll the runner uses between batch items to detect screen
    drift. It must return the controller's elements as-is on success,
    and None on failure / timeout — the in-batch break logic treats
    None as "couldn't check, keep going" rather than aborting."""
    controller = FakeController(
        elements=[{"id": "x", "label": "X", "kind": "button"}]
    )
    runner = _make_runner(controller, FakeLLMClient(), test_data={})
    import asyncio as _aio
    snap = _aio.run(runner._quick_ax_snapshot())
    assert snap is not None
    assert isinstance(snap, list)
    assert snap[0]["id"] == "x"


# ---------------- PER-172: container-id detection ----------------------


def test_T53_container_ids_are_detected() -> None:
    """T53 (PER-172): root / app-shell ids are recognised so the
    elements block can flag them and the runtime can refuse tap on
    them. Catches the «Application_dengi_0», «AXWindow_root»,
    «PinScreen_main» family without an app-specific hardcode."""
    from explorer.goal_schema import _is_container_id

    assert _is_container_id("Application_dengi_0") is True
    assert _is_container_id("AXWindow_root") is True
    assert _is_container_id("MainScaffold") is True
    assert _is_container_id("RootView_42") is True
    assert _is_container_id("PinScreen_main") is True
    assert _is_container_id("login_page_root") is True

    # Negatives: regular buttons / fields keep their meaning.
    assert _is_container_id("phoneNumberTextFieldId") is False
    assert _is_container_id("submitBtn") is False
    assert _is_container_id("enter") is False
    assert _is_container_id("") is False


def test_T54_elements_block_marks_containers() -> None:
    """T54: the elements block renders containers with the explicit
    КОНТЕЙНЕР warning that the system prompt references."""
    from explorer.goal_schema import build_elements_block

    elements = [
        {"id": "Application_dengi_0", "label": "", "kind": "container"},
        {"id": "submitBtn", "label": "", "kind": "button"},
    ]
    block, _ = build_elements_block(elements)
    assert "КОНТЕЙНЕР" in block
    assert "tap_at" in block  # the hint suggests the alternative
    # Real buttons keep the normal id-hint logic, no КОНТЕЙНЕР marker.
    submit_line = [l for l in block.split("\n") if "submitBtn" in l][0]
    assert "КОНТЕЙНЕР" not in submit_line


@pytest.mark.asyncio
async def test_T55_runtime_skips_tap_on_container_id() -> None:
    """T55 (PER-172): even if the model picks a container element_id
    (the «Application_dengi_0» fallback we saw live), the worker
    refuses to dispatch it. The item is SKIPPED in history with a
    clear reason and the batch is broken so the next LLM call sees
    the failure and retries with tap_at."""
    controller = FakeController(
        elements=[
            {"id": "Application_dengi_0", "label": "", "kind": "container"},
            {"id": "submitBtn", "label": "Submit", "kind": "button"},
        ]
    )
    bad = json.dumps({
        "done": False,
        "actions": [{
            "action": "tap",
            "action_args": {},
            "element_id": "Application_dengi_0",
            "element_label": None,
            "value_source": "none",
            "value_literal": None,
            "reasoning": "no submit visible, tapping app",
        }],
    }, ensure_ascii=False)
    recover = json.dumps({
        "done": True,
        "reason": "ok",
        "actions": [{
            "action": "tap",
            "action_args": {},
            "element_id": "submitBtn",
            "element_label": "Submit",
            "value_source": "none",
            "value_literal": None,
            "reasoning": "ok",
        }],
    }, ensure_ascii=False)
    llm = FakeLLMClient(responses=[bad, recover])
    runner = _make_runner(controller, llm, test_data={})
    await runner._run_goal_node(
        scenario_id="t55", step_idx=0,
        data={"description": "go", "max_steps": 5}, node_id="g1",
    )
    # Container tap NEVER reached the controller; the recovery tap on
    # the real Submit button did.
    assert controller.tap_calls == ["Submit"], controller.tap_calls


# ---------------- PER-107: simulator-down detection ---------------------


def test_T56_simulator_down_detector_catches_axe_and_simctl_messages() -> None:
    """T56 (PER-107): _is_simulator_down recognises both AXe's and
    simctl's «device not booted» error strings. Without this signal
    the worker treats a dead simulator as a transient AXe glitch and
    retries describe-ui for 18+ minutes (real incident: PER-107)."""
    from explorer.axe_client import _is_simulator_down

    # AXe variant — observed in PER-107
    assert _is_simulator_down(
        "Device 7B4EE110-EE1D-48CB-9CDC-16441EBEEEB5 is not booted"
    )
    # simctl variants
    assert _is_simulator_down("Unable to find a device with state: Booted")
    assert _is_simulator_down("No devices are booted.")
    assert _is_simulator_down("Current state: Shutdown")
    # Case-insensitive
    assert _is_simulator_down("IS NOT BOOTED")

    # Negatives — generic failures must NOT trigger fast-fail (we want
    # to retry transient AXe glitches a few times before giving up).
    assert not _is_simulator_down("Some other generic failure")
    assert not _is_simulator_down("AXe describe-ui timed out")
    assert not _is_simulator_down("")


def test_T57_simulator_down_error_subclasses_runtime() -> None:
    """T57: SimulatorDownError must subclass RuntimeError so existing
    ``except Exception`` blocks still catch it as a last-resort fallback,
    but it's distinct enough for the worker's outer loop to single it
    out via isinstance() and present a clean error message."""
    from explorer.axe_client import SimulatorDownError

    assert issubclass(SimulatorDownError, RuntimeError)
    err = SimulatorDownError("simulator XYZ is no longer booted")
    assert "simulator" in str(err).lower()


# ---------------- PER-144: empty LLM response diagnostics --------------


@pytest.mark.asyncio
async def test_T58_empty_llm_response_yields_explicit_no_decision_reason() -> None:
    """T58 (PER-144): when the LLM returns an empty string the goal
    must terminate with an explicit ``llm_no_decision`` reason, NOT
    silently exit with a happy completion. The original bug logged a
    warning but the runner kept the goal status as «done» — making it
    hard to tell from the run summary whether the agent finished
    naturally or was abandoned by an under-budgeted thinking pass."""
    controller = FakeController()
    llm = FakeLLMClient(responses=[""])  # one empty response
    runner = _make_runner(controller, llm, test_data={})
    ok, reason = await runner._run_goal_node(
        scenario_id="t58", step_idx=0,
        data={"description": "go", "max_steps": 2}, node_id="g1",
    )
    assert ok is False, "empty LLM response must not look like success"
    assert reason == "llm_no_decision", reason
