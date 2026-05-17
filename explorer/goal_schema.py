"""Goal-decide JSON schema + value-resolution helpers (PER-111 v2).

The LLM responds with a strict JSON contract built per-step around
what is **actually available** on this screen, in this workspace,
for this run:

* ``action`` — enum of action codes from the workspace's reference
  dictionary (claim_next ships them in ``actions``).
* ``action_args`` — oneOf branched on ``action`` so the LLM cannot
  e.g. pass ``direction`` to ``tap`` or skip it on ``swipe``. The
  args schema for each action lives in
  ``ref_action_types.arguments_schema`` and arrives in the same
  payload.
* ``element_id`` — enum of the stable ids the worker just observed
  via AXe. For actions that act on a specific element (tap, input,
  long_press, assert) the schema requires a non-null id from the
  enum — the LLM cannot return null and the worker cannot end up
  with "find element 'None'". For navigation/timing actions (back,
  wait, swipe, scroll) null is allowed.
* ``value_source`` — ``test_data.<key>`` for any key the workspace
  has, plus ``goal_literal`` / ``improvised`` / ``none``.

Combined with llama-server's ``response_format=json_schema`` this
makes fabrication impossible at the grammar level: the model can
neither invent an action name nor pass invalid args nor reference an
element that isn't on screen nor "forget" the element on a tap.
"""

from __future__ import annotations

import re
from typing import Any


# value_source special markers (anything not in test_data.* must be
# one of these).
GOAL_LITERAL = "goal_literal"
IMPROVISED = "improvised"
NONE = "none"
SPECIAL_SOURCES: tuple[str, ...] = (GOAL_LITERAL, IMPROVISED, NONE)


# Actions that conceptually MUST target a specific on-screen element.
# For these the JSON Schema forbids ``element_id: null`` and the LLM
# is forced to pick one of the stable ids the worker exposed. Listed
# by reference-dictionary code — this is the worker's contract with
# the action vocabulary, not a hardcode of any one app's behavior:
# tap/input/long_press/assert are universal across iOS/Android UIs.
# Adding a new code to the dictionary that targets an element means
# adding it here too (otherwise the LLM can choose null and the
# worker's _find_element returns "not found").
_ELEMENT_TARGETED_ACTIONS: frozenset[str] = frozenset({
    "tap",
    "input",
    "long_press",
    "assert",
    "double_click",
    "right_click",
})


def _action_args_branch(
    action_code: str,
    args_schema: dict[str, Any],
    element_ids: list[str],
) -> dict:
    """Build one ``oneOf`` branch for a single action.

    Each branch pins ``action`` to a constant and supplies the
    corresponding ``action_args`` schema. For element-targeted
    actions the branch also pins ``element_id`` to the non-null
    enum so the model can't skip the target. ``args_schema`` empty
    / missing → branch demands ``action_args: {}`` (empty object).
    Pure JSON-Schema, no GBNF-specific extensions, so llama-server
    just compiles it on its end.

    ``additionalProperties`` is deliberately omitted from the branch
    so it composes with the base object (which has its own
    additionalProperties: false plus the full set of allowed
    properties). A branch with ``additionalProperties: false`` would
    reject the value_source / reasoning / etc. fields the base
    requires — JSON Schema's allOf semantics intersect the two, so
    the branch only needs to add constraints, not redeclare the
    whole shape.
    """
    if not args_schema:
        args_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }
    branch: dict[str, Any] = {
        "type": "object",
        "required": ["action", "action_args"],
        "properties": {
            "action": {"const": action_code},
            "action_args": args_schema,
        },
    }
    if action_code in _ELEMENT_TARGETED_ACTIONS and element_ids:
        # Force the model to pick an element id from the live enum —
        # null is removed. If element_ids is empty (blank screen),
        # leave element_id alone: a forced non-null enum with no
        # values would make the whole schema unsatisfiable.
        branch["required"].append("element_id")
        branch["properties"]["element_id"] = {
            "type": "string",
            "enum": list(element_ids),
        }
    return branch


def build_goal_schema(
    test_data_keys: list[str],
    actions: list[dict[str, Any]],
    element_ids: list[str],
) -> dict[str, Any]:
    """Return the JSON Schema the LLM must satisfy on this step.

    Parameters mirror what claim_next ships + what the worker just
    captured from AXe. The schema combines several enums into one
    object with ``allOf`` + ``oneOf`` for the action-discriminated
    args. Result is a single schema usable as
    ``{"type":"json_schema","json_schema":{"schema":<this>}}`` in
    llama-server's ``response_format``.

    When ``actions`` is empty (misconfigured workspace) we fall back
    to a permissive schema with ``action`` as a plain string — keeps
    the worker from crashing while the operator fixes the dictionary.
    """
    action_codes = [a["code"] for a in actions if a.get("code")]
    value_source_enum = (
        [f"test_data.{k}" for k in test_data_keys] + list(SPECIAL_SOURCES)
    )

    # PER-111 v2: element_id is REQUIRED on every decision, with a
    # non-null string type and an enum of the live on-screen ids.
    # The first live run made it obvious why: llama-server's JSON
    # Schema → GBNF compiler doesn't enforce per-branch constraints
    # inside allOf+oneOf (so a tap branch saying "element_id must
    # be a string" is silently ignored), and Gemma 4 happily returns
    # element_id=null on tap/input. Pinning the field at the base
    # schema works because that constraint IS compiled into the
    # grammar. For navigation actions (back / swipe / scroll / wait)
    # element_id is technically irrelevant — the worker ignores it
    # in _dispatch — but having the LLM pick *any* visible id is
    # cheap and keeps the grammar single-track.
    #
    # If element_ids is empty (blank screen, or the worker called
    # before AXe stabilised) we leave the field nullable: a
    # non-null enum with no values would make the whole schema
    # unsatisfiable and llama-server would 400.
    if element_ids:
        element_id_schema: dict[str, Any] = {
            "type": "string",
            "enum": list(element_ids),
        }
        required_fields = [
            "done",
            "action",
            "action_args",
            "element_id",
            "value_source",
            "value_literal",
            "reasoning",
        ]
    else:
        element_id_schema = {
            "type": ["string", "null"],
            "enum": [None],
        }
        required_fields = [
            "done",
            "action",
            "action_args",
            "value_source",
            "value_literal",
            "reasoning",
        ]

    # Base object — everything except action / action_args, which we
    # nail down via ``allOf`` + ``oneOf`` so the args constrain by
    # the picked action.
    base: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": required_fields,
        "properties": {
            "done": {"type": "boolean"},
            "reason": {"type": ["string", "null"], "maxLength": 300},
            "action": (
                {"type": "string", "enum": action_codes}
                if action_codes else {"type": "string"}
            ),
            "action_args": {"type": "object"},
            "element_id": element_id_schema,
            "element_label": {"type": ["string", "null"], "maxLength": 300},
            "value_source": {"type": "string", "enum": value_source_enum},
            "value_literal": {"type": ["string", "null"], "maxLength": 300},
            "reasoning": {"type": "string", "maxLength": 400},
        },
    }
    if not action_codes:
        return base

    # Discriminated args: ``oneOf`` for the (action, action_args)
    # pair. Each branch fixes action to one of the dictionary codes
    # and, for element-targeted ones, also pins element_id to the
    # non-null enum.
    base["allOf"] = [
        {
            "oneOf": [
                _action_args_branch(
                    a["code"],
                    a.get("arguments_schema") or {},
                    element_ids,
                )
                for a in actions
                if a.get("code")
            ]
        }
    ]
    return base


def build_actions_block(actions: list[dict[str, Any]]) -> str:
    """Human-readable description of available actions for the user
    prompt. The LLM reads this list to know what's allowed and what
    args each action needs.

    Format (per line):
        - <code> — <name>: <description>  args: <keys or "—">
    """
    if not actions:
        return "  (нет доступных действий — обратитесь к администратору)"
    lines: list[str] = []
    for a in actions:
        code = a.get("code") or "?"
        name = a.get("name") or code
        descr = (a.get("description") or "").strip()
        args_schema = a.get("arguments_schema") or {}
        args_props = (args_schema.get("properties") or {}) if isinstance(args_schema, dict) else {}
        if args_props:
            arg_summary = ", ".join(f"{k}: {(v.get('type') or 'any')}" for k, v in args_props.items())
        else:
            arg_summary = "—"
        head = f"  - {code} — {name}"
        if descr:
            head += f": {descr}"
        head += f"  | args: {arg_summary}"
        lines.append(head)
    return "\n".join(lines)


def build_elements_block(elements: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Render the on-screen elements as a numbered list AND return
    the list of stable ids the LLM may reference.

    Returns ``(text_block, element_ids)``. ``text_block`` is what the
    user prompt shows; ``element_ids`` feeds into the schema enum so
    the model physically cannot point at an element it doesn't see.
    """
    if not elements:
        return "  (экран пуст)", []
    lines: list[str] = []
    ids: list[str] = []
    for i, el in enumerate(elements[:50]):
        if isinstance(el, dict):
            elem_id = (
                el.get("id")
                or el.get("identifier")
                or el.get("test_id")
                or f"el_{i + 1}"
            )
            label = el.get("label") or "(без подписи)"
            kind = el.get("kind") or el.get("type") or "element"
        else:
            elem_id = (
                getattr(el, "id", None)
                or getattr(el, "test_id", None)
                or f"el_{i + 1}"
            )
            label = getattr(el, "label", "") or "(без подписи)"
            kind = getattr(el, "kind", "element")
            if hasattr(kind, "value"):
                kind = kind.value
        elem_id = str(elem_id)
        ids.append(elem_id)
        lines.append(f"  - id={elem_id} [{kind}] {label}")
    if len(elements) > 50:
        lines.append(f"  …и ещё {len(elements) - 50}.")
    return "\n".join(lines), ids


def build_test_data_block(test_data: dict[str, str]) -> str:
    """Render the workspace's test_data as a bullet list.

    Same format as v1 — values shown so the LLM can sanity-check
    what will be substituted. Model is NOT expected to copy them
    into ``value_literal``: the system prompt forbids it and the
    schema enum makes it impossible.
    """
    if not test_data:
        return "  (нет — выдумай данные сам там, где это требуется)"
    return "\n".join(f"  - {k}: {v}" for k, v in test_data.items())


def resolve_value(
    decision: dict[str, Any],
    test_data: dict[str, str],
    improvised_memory: dict[str, str],
) -> str | None:
    """Turn an LLM decision into the actual string the worker types.

    PER-111 v2: ``improvised_memory`` is keyed by ``element_id`` —
    the model's element_label can change between visits (re-render,
    localization), but the AXUniqueId stays. ``element_label`` is
    used only as a fallback when no id is provided.
    """
    vs = (decision.get("value_source") or "").strip()
    if vs == NONE or not vs:
        return None
    if vs.startswith("test_data."):
        key = vs.split(".", 1)[1]
        return test_data.get(key)
    if vs == GOAL_LITERAL:
        return decision.get("value_literal")
    if vs == IMPROVISED:
        memory_key = (
            (decision.get("element_id") or "").strip()
            or (decision.get("element_label") or "").strip()
        )
        if memory_key and memory_key in improvised_memory:
            return improvised_memory[memory_key]
        v = decision.get("value_literal")
        if memory_key and v:
            improvised_memory[memory_key] = v
        return v
    return None
