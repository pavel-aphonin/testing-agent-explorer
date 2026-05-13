"""Goal-decide JSON schema + value-resolution helpers (PER-111).

Goal-узлы используют value-by-reference контракт: модель не возвращает
голое ``value``, а указывает источник (``value_source``) — либо ключ из
``test_data``, либо «литерал из текста цели», либо «импровизация».
Подставляет реальное значение код, а не модель. Это:

* убирает галлюцинации идентифицирующих данных (PER-110) — модель
  физически не может выдать ``+71790515430`` в поле phone, если в
  schema ``value_source`` ограничен enum'ом ``[test_data.phone, ...]``;
* делает действия модели прозрачными — в логах видно, какой ключ
  она привязала к какому полю;
* работает на любые workspace-ключи без хардкода категорий —
  enum собирается из ``self.test_data.keys()`` при каждом вызове.

The schema is shipped to llama.cpp through ``response_format`` —
llama-server compiles it to GBNF automatically, so we get constrained
decoding without writing the grammar by hand.
"""

from __future__ import annotations

from typing import Any


#: Special value_source markers that don't correspond to a test_data key.
GOAL_LITERAL = "goal_literal"
IMPROVISED = "improvised"
NONE = "none"
SPECIAL_SOURCES: tuple[str, ...] = (GOAL_LITERAL, IMPROVISED, NONE)


def build_goal_schema(test_data_keys: list[str]) -> dict[str, Any]:
    """Return a JSON schema describing one goal-decide LLM response.

    The schema enumerates ``test_data.<key>`` for every available
    workspace key + the three special markers. Llama-server compiles
    this to GBNF, so the model cannot emit a token combination outside
    the enum. ``additionalProperties=False`` + ``required`` block any
    "extra" keys / dropped fields.

    Keep field order stable — humans read schemas left-to-right in
    failure traces, and a consistent order makes diffs readable.
    """
    enum = [f"test_data.{k}" for k in test_data_keys] + list(SPECIAL_SOURCES)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "done",
            "action",
            "element_label",
            "value_source",
            "value_literal",
            "reasoning",
        ],
        "properties": {
            "done": {"type": "boolean"},
            "reason": {"type": ["string", "null"], "maxLength": 300},
            "action": {
                "type": "string",
                "enum": ["tap", "input", "back", "swipe"],
            },
            "element_label": {"type": "string", "maxLength": 300},
            "value_source": {"type": "string", "enum": enum},
            "value_literal": {"type": ["string", "null"], "maxLength": 300},
            "reasoning": {"type": "string", "maxLength": 400},
        },
    }


def build_value_sources_list(test_data_keys: list[str]) -> str:
    """Human-readable rendering of the allowed value_source enum for
    the {{value_sources_list}} placeholder in the user prompt. Lets the
    model see the same constraint twice — in the schema and in the
    prompt text — which reduces "misalignment risk" (the model picks
    the closest in-schema value rather than the best one) noted by
    Aidan Cooper in the constrained-decoding guide cited in PER-111."""
    sources = [f"test_data.{k}" for k in test_data_keys] + list(SPECIAL_SOURCES)
    return ", ".join(sources)


def build_test_data_block(test_data: dict[str, str]) -> str:
    """Render the test_data dict as a bullet list for the user prompt.

    Values are echoed in full so the model can sanity-check what will
    be substituted, but the model is NOT expected to copy them into
    ``value_literal`` — the system prompt forbids that explicitly and
    the schema enum makes it impossible. Marked ``(нет)`` when the
    workspace ships no values."""
    if not test_data:
        return "  (нет — выдумай данные сам там, где это требуется)"
    return "\n".join(f"  - {k}: {v}" for k, v in test_data.items())


def resolve_value(
    decision: dict[str, Any],
    test_data: dict[str, str],
    improvised_memory: dict[str, str],
) -> str | None:
    """Turn an LLM decision into the actual string the worker types.

    ``value_source`` cases:

    * ``test_data.<key>`` — substitute ``test_data[key]`` verbatim,
      ignoring whatever ``value_literal`` the model put there
      (schema forces it to ``null``, but be defensive).
    * ``goal_literal`` — the goal text contains a constant the model
      copied into ``value_literal``; use it as-is.
    * ``improvised`` — the model made up a value. Cache it under the
      element_label so the next visit to the same field produces the
      same string (the "придумай и запомни" half of Pavel's contract).
    * ``none`` — non-input action (tap/back/swipe) or input with no
      sensible value; return None and let the dispatcher decide.

    Returns the resolved string or None.
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
        label = (decision.get("element_label") or "").strip()
        if label and label in improvised_memory:
            return improvised_memory[label]
        v = decision.get("value_literal")
        if label and v:
            improvised_memory[label] = v
        return v
    return None
