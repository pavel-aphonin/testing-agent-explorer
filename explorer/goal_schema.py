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


# PER-163: actions that target the screen by coordinates, not by
# accessibility id. For these the prompt explicitly tells the model
# ``element_id`` should be ``null`` — and the schema must allow that
# at the base level, otherwise the model is forced to attach a bogus
# id (typically the app root container) which then poisons history,
# anti-loop and visited tracking. Action dispatch ignores element_id
# for these — only ``action_args.x/y`` matter.
_COORD_ONLY_ACTIONS: frozenset[str] = frozenset({
    "tap_at",
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


def build_action_item_schema(
    test_data_keys: list[str],
    actions: list[dict[str, Any]],
    element_ids: list[str],
) -> dict[str, Any]:
    """JSON Schema for a SINGLE action item — used inside the batch.

    Each action item carries everything needed to dispatch one
    operation: action code + args + element/value source + reasoning.
    This is the inner shape; the top-level batch wrapper
    (``build_goal_schema``) embeds a list of these inside
    ``actions: [...]``.

    Same all-of-the-grammar-tricks as before:
    * ``action`` enum from the workspace's reference dictionary
    * ``element_id`` enum from live AXe ids (nullable for tap_at /
      navigation actions; required non-null for tap/input/etc.)
    * ``value_source`` enum of test_data keys + special sources
    * ``oneOf`` discriminator on (action, action_args) so each
      action's args shape is enforced at the grammar layer

    Extracted from the old ``build_goal_schema`` body — the schema
    body for one action didn't change, we just moved it inside a
    batch wrapper as part of PER-170.
    """
    action_codes = [a["code"] for a in actions if a.get("code")]
    has_coord_only_action = any(
        code in _COORD_ONLY_ACTIONS for code in action_codes
    )
    value_source_enum = (
        [f"test_data.{k}" for k in test_data_keys] + list(SPECIAL_SOURCES)
    )

    # PER-111 v2: element_id is REQUIRED on every action item, with a
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
    # PER-163: when the action set contains a coordinate-only action
    # (tap_at), the base schema MUST allow ``element_id: null`` so
    # the model can honour the prompt instruction. Otherwise the
    # model is forced to attach a real id (typically the app root
    # container) which corrupts visited_actions / anti-loop / history
    # tracking — every coord tap looks like "tap on the same root".
    # Per-branch oneOf still pins element-targeted actions to a
    # non-null string from the enum (mostly enforced by llama-server,
    # but where it isn't the dispatch layer catches it).
    #
    # If element_ids is empty (blank screen, or the worker called
    # before AXe stabilised) we leave the field nullable: a
    # non-null enum with no values would make the whole schema
    # unsatisfiable and llama-server would 400.
    if element_ids:
        if has_coord_only_action:
            element_id_schema: dict[str, Any] = {
                "type": ["string", "null"],
                "enum": list(element_ids) + [None],
            }
        else:
            element_id_schema = {
                "type": "string",
                "enum": list(element_ids),
            }
        required_fields = [
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
            "action",
            "action_args",
            "value_source",
            "value_literal",
            "reasoning",
        ]

    # Base object — everything except action / action_args, which we
    # nail down via ``allOf`` + ``oneOf`` so the args constrain by
    # the picked action.
    item: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": required_fields,
        "properties": {
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
        return item

    # Discriminated args: ``oneOf`` for the (action, action_args)
    # pair. Each branch fixes action to one of the dictionary codes
    # and, for element-targeted ones, also pins element_id to the
    # non-null enum.
    item["allOf"] = [
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
    return item


# PER-170: cap on actions per batch. The LLM may want to chain
# many small steps (4-digit PIN, 11-digit phone keyboard, etc.),
# but two cliffs say "don't go infinite":
#   * grammar compile time grows with array size in llama-server,
#   * a 50-action batch dispatched without re-LLM would blast past
#     anti-loop detection if any single step misfires.
# 12 is enough for "input phone + tap submit + 4-digit PIN + submit"
# end-to-end (a representative login flow) and small enough to keep
# the JSON output under 4KB so streaming + JSON-grammar stay fast.
_MAX_ACTIONS_PER_BATCH: int = 12


def build_goal_schema(
    test_data_keys: list[str],
    actions: list[dict[str, Any]],
    element_ids: list[str],
) -> dict[str, Any]:
    """PER-170: batch schema — LLM returns ALL actions for this screen.

    Top-level shape:

        {
          "done": bool,
          "reason": str | null,
          "expected_next_screen": str | null,
          "actions": [ <action item>, <action item>, ... ]
        }

    ``actions`` is the new core: a list of operations the worker
    will dispatch one after another **without another LLM call**.
    The worker keeps a quick sanity check between items (AX-tree
    hash diff) and breaks the batch early if the screen drifts from
    what the LLM expected — at which point a fresh _goal_decide
    kicks in with a new screenshot.

    Why batch:
        Per-step decisions cost a full Gemma 4 round-trip (~25s)
        per click. For known data like PIN 8520 the LLM already
        knows all 4 taps the moment it sees the keypad — making it
        re-decide each digit is wasted compute. PER-169 smoke
        showed 12+ steps × 25s on auth + PIN; batch collapses that
        to 1-2 LLM calls.

    Top-level vs item:
        * ``done`` / ``reason`` are top-level — they describe the
          batch's terminal verdict, not any single action.
        * ``expected_next_screen`` is a free-form hint the worker
          uses as soft guidance (it doesn't block on mismatches,
          it just shortens the batch when reality diverges).
        * ``actions`` items each look like the old single-decision
          object minus ``done`` / ``reason``.

    Empty workspace fallback (no action types configured): we still
    return a valid schema with a permissive ``action`` field so the
    worker doesn't crash mid-run while the operator fixes seeds.
    """
    item_schema = build_action_item_schema(
        test_data_keys, actions, element_ids
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["done", "actions"],
        "properties": {
            "done": {"type": "boolean"},
            "reason": {"type": ["string", "null"], "maxLength": 300},
            "actions": {
                "type": "array",
                # PER-170 first-smoke fix: minItems=1, always.
                # When we shipped with minItems=0 the model used the
                # loophole — emitted ``actions: []`` with a long
                # English description of what it «would» do, and the
                # worker dispatched nothing for 10 LLM calls in a row
                # until coverage_plateau aborted the goal. Forcing
                # at least one item makes the model commit:
                # «done=true» costs a real action (the last one that
                # achieved the goal); «done=false» requires the next
                # concrete step. There is no «think out loud» branch.
                # ``expected_next_screen`` was also removed for the
                # same reason — it served as a free-form bucket the
                # model preferred to fill instead of ``actions``.
                "minItems": 1,
                "maxItems": _MAX_ACTIONS_PER_BATCH,
                "items": item_schema,
            },
        },
    }


def normalize_decision(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn whatever the LLM emitted into the canonical batch shape.

    Handles three cases:
        1. New batch shape with ``actions: [...]`` — returned as-is.
        2. Legacy single-action shape (top-level action/action_args/
           element_id/...) — wrapped into ``actions: [<that>]``.
        3. ``done=true`` with no actions — actions normalized to [].

    Returning a stable dict means the caller (``_run_goal_node``)
    only has to handle one branch. Legacy support keeps tests and
    older prompts working while the prompts table is migrated.

    ``expected_next_screen`` is intentionally dropped on the way in:
    in PER-170's first live smoke the model used it as a free-form
    bucket to «describe what it would do» instead of populating
    ``actions``. Removing it from the canonical shape (and from the
    JSON schema) closes that escape hatch.
    """
    if not isinstance(raw, dict):
        return {"done": False, "reason": None, "actions": []}
    # Case 1: already batch shape
    if isinstance(raw.get("actions"), list):
        out = {
            "done": bool(raw.get("done", False)),
            "reason": raw.get("reason"),
            "actions": raw["actions"],
        }
        return out
    # Case 3: done with no actions
    if raw.get("done") and "action" not in raw:
        return {
            "done": True,
            "reason": raw.get("reason"),
            "actions": [],
        }
    # Case 2: legacy single action — wrap it
    item_keys = (
        "action", "action_args", "element_id", "element_label",
        "value_source", "value_literal", "reasoning",
    )
    item = {k: raw[k] for k in item_keys if k in raw}
    return {
        "done": bool(raw.get("done", False)),
        "reason": raw.get("reason"),
        "actions": [item] if item.get("action") else [],
    }


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
        # PER-164: when an arg carries its own description (e.g.
        # ``tap_at.target_description`` explains the grounder
        # routing), render it on its own line so the LLM sees the
        # intent — without it the model only sees a type and may
        # ignore the arg entirely. Stays single-line for plain
        # type-only args (the legacy case). Description is
        # truncated to the first sentence to keep the prompt
        # cache-friendly (the full text lives in DB / OpenAPI for
        # admin docs).
        if args_props:
            kv_parts: list[str] = []
            for k, v in args_props.items():
                if not isinstance(v, dict):
                    continue
                arg_type = v.get("type") or "any"
                arg_descr = (v.get("description") or "").strip()
                if arg_descr:
                    # First sentence only (split on ". " so
                    # abbreviations like "e.g." don't truncate
                    # early), cap at 120 chars.
                    first = arg_descr.split(". ")[0].strip().rstrip(".")
                    if len(first) > 120:
                        first = first[:117] + "..."
                    kv_parts.append(f"{k} ({arg_type}: {first})")
                else:
                    kv_parts.append(f"{k}: {arg_type}")
            arg_summary = ", ".join(kv_parts)
        else:
            arg_summary = "—"
        head = f"  - {code} — {name}"
        if descr:
            head += f": {descr}"
        head += f"  | args: {arg_summary}"
        lines.append(head)
    return "\n".join(lines)


# PER-170 followup: dictionary of common UI element id keywords →
# human-readable hint. Used by ``_hint_from_id`` to turn a bare
# camelCase / snake_case element id (``backButton``, ``submit_btn``,
# ``loginEnter``) into a hint the chat-LLM can reason about when the
# accessibility label is empty.
#
# This is NOT app-specific hardcode — keywords are the universal UI
# vocabulary every iOS/Android app borrows from the platform's HIG.
# A custom app that names its «pay» button ``ButtonForBuyingThings``
# gets no hint, but it would have also had a confused LLM with bare
# id — adding more keywords to this dictionary is a no-op everywhere
# else.
#
# Bilingual on purpose: AX labels in iOS apps localised to Russian
# tend to surface English ids (`backButton`, `loginButton`) and vice
# versa; the chat-LLM speaks both. Hints are short noun phrases — the
# LLM reads them inline as a parenthetical.
# PER-172 prevention: substring keywords that mark an element id as
# the root container / app shell, NOT an interactive control. When
# the LLM can't find the «правильную» button it has a habit of
# falling back to one of these (a `tap` requires *some* element_id
# from the enum, so the model picks anything plausible) — which then
# resolves to a hit-test on the entire app frame and does nothing
# visible, looking to the model like a no-op screen.
#
# We surface this as a parenthetical hint in elements_block AND, more
# importantly, instruct the model in the prompt that tapping these is
# forbidden (use tap_at with target_description instead). Both halves
# matter: without the hint the LLM can't tell which ids are containers;
# without the prompt rule it still might tap anyway.
_CONTAINER_ID_KEYWORDS: tuple[str, ...] = (
    "application",  # iOS AXApplication (e.g. Application_dengi_0)
    "window",       # AXWindow root
    "scaffold",     # Flutter Scaffold
    "rootview",
    "rootcontainer",
    "approot",
    "container",    # generic catch-all — slightly fuzzy but safe
    "screen_",      # «MainScreen», «PinScreen», etc. as containers
    "page_",        # React Native / Flutter page wrappers
)


def _is_container_id(elem_id: str) -> bool:
    """Heuristic: does this element_id look like a root / shell
    container rather than an interactive control?

    PER-172 prevention. Catches Application_dengi_0, AXWindow_root,
    PinScreen_main, etc. False positives are OK — the worst case is
    we annotate a legitimate button with «(контейнер)» and the model
    skips it once; the next batch corrects. False negatives are also
    OK — the prompt rule that follows is the real backstop.
    """
    if not elem_id:
        return False
    low = elem_id.lower()
    return any(kw in low for kw in _CONTAINER_ID_KEYWORDS)


_ID_KEYWORD_HINTS: tuple[tuple[str, str], ...] = (
    # Navigation — most important class because Gemma 4 confuses
    # forward / back when both are unlabelled buttons.
    ("back", "назад / возврат"),
    ("close", "закрыть / отменить"),
    ("cancel", "отмена"),
    ("dismiss", "закрыть"),
    ("forward", "вперёд"),
    ("next", "следующий шаг / далее"),
    ("prev", "предыдущий"),
    ("skip", "пропустить"),
    # Confirmation / submission
    ("submit", "отправить / подтвердить"),
    ("confirm", "подтвердить"),
    ("login", "войти / авторизация"),
    ("signin", "войти"),
    ("signup", "регистрация"),
    ("register", "регистрация"),
    ("logout", "выйти из аккаунта"),
    ("enter", "войти / ввести"),
    ("done", "готово / завершить"),
    ("apply", "применить"),
    ("save", "сохранить"),
    ("ok", "ОК"),
    # Permissions / system
    ("allow", "разрешить (системный запрос)"),
    ("deny", "запретить"),
    # Common input fields
    ("phone", "поле телефона"),
    ("email", "поле e-mail"),
    ("password", "поле пароля"),
    ("pin", "поле PIN-кода"),
    ("search", "поле поиска"),
    # Common actions
    ("buy", "купить / оплатить"),
    ("pay", "оплатить"),
    ("send", "отправить"),
    ("delete", "удалить"),
    ("edit", "редактировать"),
    ("settings", "настройки"),
    ("menu", "меню"),
)


def _hint_from_id(elem_id: str) -> str | None:
    """Best-effort human hint for an element_id whose AX label is empty.

    Splits the id by camelCase / underscores / dashes and scans for any
    keyword from ``_ID_KEYWORD_HINTS``. Returns the first match's hint,
    or None if nothing recognisable is in the id. Case-insensitive.

    The point is to give Gemma 4 enough signal to NOT confuse a
    ``backButton`` (наблюдалось: модель писала «нажимаю Вперёд» и
    выбирала backButton) with a ``forwardButton``. The schema's
    element_id enum is opaque to a model that sees ``id=backButton
    label=(без подписи)`` and has no way to tell the two apart by
    semantic role — only by id substring. Surfacing that hint
    inline in the elements block lets the model decide on label,
    not on enum order.
    """
    if not elem_id:
        return None
    # Tokenize: split camelCase boundaries + underscores + dashes
    tokens: list[str] = []
    cur = ""
    for ch in elem_id:
        if ch in "_-":
            if cur:
                tokens.append(cur)
                cur = ""
        elif ch.isupper() and cur:
            tokens.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        tokens.append(cur)
    lowered = {t.lower() for t in tokens}
    # Also consider the full lowercased id (catches e.g. `loginbutton`
    # where there's no separator).
    full_lower = elem_id.lower()
    for keyword, hint in _ID_KEYWORD_HINTS:
        if keyword in lowered or keyword in full_lower:
            return hint
    return None


def build_elements_block(elements: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Render the on-screen elements as a numbered list AND return
    the list of stable ids the LLM may reference.

    Returns ``(text_block, element_ids)``. ``text_block`` is what the
    user prompt shows; ``element_ids`` feeds into the schema enum so
    the model physically cannot point at an element it doesn't see.

    PER-170 followup: when the accessibility label is empty
    («(без подписи)») we tack on a parenthetical hint derived from
    the element id — e.g. ``id=backButton`` becomes
    ``«(без подписи; вероятно «назад / возврат»)»``. This kills the
    "Gemma reasons «нажму Вперёд» but element_id=backButton" pattern
    we saw in PER-170 smoke #2: with no label the model had no way
    to disambiguate two unlabelled buttons by role, only by id, and
    the id enum order was effectively random.
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
            raw_label = el.get("label") or ""
            kind = el.get("kind") or el.get("type") or "element"
        else:
            elem_id = (
                getattr(el, "id", None)
                or getattr(el, "test_id", None)
                or f"el_{i + 1}"
            )
            raw_label = getattr(el, "label", "") or ""
            kind = getattr(el, "kind", "element")
            if hasattr(kind, "value"):
                kind = kind.value
        elem_id = str(elem_id)
        ids.append(elem_id)

        if raw_label.strip():
            label_display = raw_label
        else:
            hint = _hint_from_id(elem_id)
            if hint:
                label_display = f"(без подписи; вероятно «{hint}»)"
            else:
                label_display = "(без подписи)"
        # PER-172: container marker. ``[container]`` annotation
        # warns the model that this id is an app shell / root view —
        # tapping it would do nothing meaningful. The system prompt
        # tells the model to use tap_at with target_description
        # instead. We keep the id in the enum so the schema doesn't
        # break, but flag it visually right next to the label.
        container_marker = ""
        if _is_container_id(elem_id):
            container_marker = " [КОНТЕЙНЕР — не тапать; используй tap_at]"
        lines.append(f"  - id={elem_id} [{kind}] {label_display}{container_marker}")
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
