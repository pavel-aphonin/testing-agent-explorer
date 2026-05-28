"""PER-200: Reward Critic + Reflection agents.

Both multiplex onto the PLANNER model (GUI-Owl) per the roster — same
llama-server, different prompts, different ModuleRole so the operator
*could* point them at a dedicated model later.

  RewardCriticAgent.score(goal, action, before, after) -> float 0..1
      Dense per-step progress signal. Used by scenario_runner to detect
      "this action achieved nothing" (score ~0) across several steps →
      the circling the user complained about.

  ReflectionAgent.review(goal, history) -> ReflectionNote
      Periodic "are we stuck, what's been tried, what to try next".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from explorer.agents.base import RoleAgent
from explorer.role_resolver import ModuleRole

logger = logging.getLogger("explorer.agents.critic")


# ─────────────────────────────────────────────────────────────────────
# Reward Critic
# ─────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """Ты — критик прогресса в автоматизированном тестировании.
Оцени, насколько ПОСЛЕДНЕЕ действие продвинуло агента к цели.

Верни строго JSON:
{
  "progress": 0.0-1.0,
  "advanced": true|false,
  "reason": "<кратко: что изменилось или почему застряли>"
}

0.0 = экран не изменился / то же состояние / явный откат назад.
1.0 = заметное продвижение к цели (новый экран, нужный элемент появился).
Если экран до и после практически одинаковый — progress близок к 0."""


@dataclass
class RewardScore:
    progress: float
    advanced: bool
    reason: str


class RewardCriticAgent(RoleAgent):
    role = ModuleRole.REWARD_CRITIC

    async def score(
        self,
        goal_text: str,
        action_description: str,
        before_screen: str,
        after_screen: str,
    ) -> RewardScore | None:
        user = (
            f"Цель: {goal_text}\n\n"
            f"Действие: {action_description}\n\n"
            f"Экран ДО:\n{before_screen[:1500]}\n\n"
            f"Экран ПОСЛЕ:\n{after_screen[:1500]}"
        )
        result = await self.call(
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=160,
            response_format={"type": "json_object"},
        )
        if result is None:
            return None
        parsed = _safe_json(result.content)
        if parsed is None:
            return None
        return RewardScore(
            progress=float(parsed.get("progress", 0.0)),
            advanced=bool(parsed.get("advanced", False)),
            reason=str(parsed.get("reason", "")),
        )


# ─────────────────────────────────────────────────────────────────────
# Reflection
# ─────────────────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """Ты — модуль рефлексии в автоматизированном тестировании.
Агент работает над целью и, возможно, застрял. Проанализируй историю и
дай КОНКРЕТНУЮ рекомендацию, что попробовать иначе.

Верни строго JSON:
{
  "stuck": true|false,
  "diagnosis": "<что агент делает не так>",
  "recommendation": "<одно конкретное следующее действие>"
}"""


@dataclass
class ReflectionNote:
    stuck: bool
    diagnosis: str
    recommendation: str


class ReflectionAgent(RoleAgent):
    role = ModuleRole.REFLECTION

    async def review(
        self, goal_text: str, history: list[str]
    ) -> ReflectionNote | None:
        hist = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-12:]))
        user = f"Цель: {goal_text}\n\nПоследние действия:\n{hist}"
        result = await self.call(
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=220,
            response_format={"type": "json_object"},
        )
        if result is None:
            return None
        parsed = _safe_json(result.content)
        if parsed is None:
            return None
        return ReflectionNote(
            stuck=bool(parsed.get("stuck", False)),
            diagnosis=str(parsed.get("diagnosis", "")),
            recommendation=str(parsed.get("recommendation", "")),
        )


def _safe_json(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```$", "", t)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        logger.debug("critic/reflection JSON parse failed: %s", t[:160])
        return None
