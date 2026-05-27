"""PER-196: Planner agent.

Ranks UI elements on a screen and suggests the next action. Used by
the hot path inside ``llm_loop`` (formerly the inline llm_client
call). The system prompt and JSON schema match what
``goal_decide`` expects so downstream parsers don't have to change.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from explorer.agents.base import RoleAgent, RoleAgentResult
from explorer.role_resolver import ModuleRole

logger = logging.getLogger("explorer.agents.planner")


SYSTEM_PROMPT = """Ты — автоматизированный тестировщик мобильных приложений.
На каждом шаге получаешь скрин и список элементов экрана. Выбери ОДИН
элемент, который имеет смысл тапнуть прямо сейчас для прохождения сценария.

Отвечай строго JSON:
{
  "action": "tap" | "tap_at" | "input" | "wait",
  "element_id": "<id из списка элементов>" | null,
  "target_description": "<краткое описание для grounder если action=tap_at>" | null,
  "value": "<текст для input>" | null,
  "reasoning": "<одно предложение>"
}

Правила:
- НИКОГДА не тапай контейнерные элементы (root view, screen, page и т.п.)
- Если кнопка визуально видна но её нет в списке — используй tap_at с target_description
- Если экран загружается — action="wait"
"""


class PlannerAgent(RoleAgent):
    role = ModuleRole.PLANNER

    async def decide(
        self,
        screen_description: str,
        history_block: str = "",
        elements_block: str = "",
    ) -> dict | None:
        """Return a parsed plan dict, or ``None`` if the role is unassigned
        or the model returned junk.

        Caller (the legacy llm_loop / scenario_runner code) decides
        whether to use this dict or fall back to its own LLM call.
        """
        user_msg = "\n".join(
            part for part in [screen_description, elements_block, history_block] if part
        )
        result = await self.call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        if result is None:
            return None
        return _safe_json_parse(result)


def _safe_json_parse(result: RoleAgentResult) -> dict | None:
    """Strip code fences and parse JSON. None on failure."""
    text = result.content.strip()
    # llama-server with grammar usually returns clean JSON; without
    # grammar some models wrap in ```json blocks. Strip them.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Planner JSON parse failed (%s): %s", exc, text[:200])
        return None
    if not isinstance(parsed, dict):
        logger.warning("Planner returned non-object JSON: %s", text[:200])
        return None
    return parsed
