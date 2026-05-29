"""PER-203 Phase 3b: shared planner-prompt assembler.

``build_planner_inputs`` turns a screen state + goal into the
(user_prompt, schema, element_ids) the planner LLM needs — applying the
exact same hint stack (PER-172/198/200) as the synchronous
``scenario_runner`` path, via the shared ``planning.hints`` functions.

The bus planner-runner calls this from the ``screen.captured`` payload;
the sync path keeps its inline assembly (behaviour-identical because
both pull the same hints). Pure + sync — no I/O, no ``self`` — so it's
trivially testable and reusable across transports. Async/agent-derived
inputs (the Context Identifier verdict, the Reflection recommendation)
are passed in pre-computed by the caller.
"""

from __future__ import annotations

import re
from typing import Any

from explorer.goal_schema import build_elements_block, build_goal_schema
from explorer.planning.hints import (
    count_digit_taps,
    credential_routing_hint,
    loop_breaker_hint,
    pin_submit_hint,
)


def _render_template(template: str, values: dict[str, str]) -> str:
    """``{{key}}`` substitution; unknown placeholders stay verbatim.
    Mirror of scenario_runner._render_template (kept local to avoid a
    heavy import cycle)."""
    if not template or "{{" not in template:
        return template

    def _repl(m: "re.Match[str]") -> str:
        v = values.get(m.group(1).strip())
        return v if v is not None else m.group(0)

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", _repl, template)


def build_planner_inputs(
    *,
    goal_text: str,
    mode: str,
    success_criteria: str,
    elements: list[dict[str, Any]],
    history: list[str],
    step_idx: int,
    max_steps: int,
    user_template: str,
    actions: list[dict],
    actions_block: str,
    test_data_block: str,
    test_data_keys: list[str],
    visited_summary: str = "",
    memory_block: str = "",
    context_is_pin: bool = False,
    reflection_text: str | None = None,
) -> dict[str, Any]:
    """Assemble the planner's user prompt + constrained-decode schema.

    Returns ``{"user_prompt", "schema", "element_ids"}``. Hint
    prepend order matches the sync path exactly so the final prompt is
    top→bottom: memory → credentials → loop(+reflection) → pin →
    template body.
    """
    elements_block, element_ids = build_elements_block(elements)

    history_block = (
        "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(history[-8:]))
        if history else "  (пока ничего)"
    )
    if visited_summary:
        history_block = (
            f"  (на этом экране уже пробовали: {visited_summary})\n" + history_block
        )

    schema = build_goal_schema(
        test_data_keys=list(test_data_keys),
        actions=actions,
        element_ids=element_ids,
    )

    user_prompt = _render_template(
        user_template,
        {
            "mode": mode,
            "goal": goal_text,
            "step_idx": str(step_idx + 1),
            "max_steps": str(max_steps),
            "success_criteria": success_criteria,
            "elements_block": elements_block,
            "actions_block": actions_block,
            "test_data_block": test_data_block,
            "history_block": history_block,
        },
    )

    # Prepend hints in the same sequence as scenario_runner (each goes
    # to the front, so the last prepended ends up on top).
    if context_is_pin:
        pin = pin_submit_hint(count_digit_taps(history))
        if pin:
            user_prompt = pin + "\n\n" + user_prompt

    loop = loop_breaker_hint(history)
    if loop:
        if reflection_text:
            loop = loop + f"\n🧭 РЕКОМЕНДАЦИЯ РЕФЛЕКСИИ: {reflection_text}"
        user_prompt = loop + "\n\n" + user_prompt

    cred = credential_routing_hint(set(test_data_keys))
    if cred:
        user_prompt = cred + "\n\n" + user_prompt

    if memory_block:
        user_prompt = (
            "Память о ранее выполненных действиях в этой цели "
            "(используй чтобы НЕ повторять уже сделанное):\n"
            f"{memory_block}\n\n" + user_prompt
        )

    return {"user_prompt": user_prompt, "schema": schema, "element_ids": element_ids}
