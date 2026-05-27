"""PER-196: Ambiguity Resolver — once-per-scenario disambiguation.

Given a task description + initial screen, picks a canonical path
when the task could be interpreted multiple ways («войти в приложение»
→ через биометрию или через PIN?). Called once at scenario start,
the result is stashed and consulted by downstream planners.

Russian-first because the banking UI is Russian — Qwen3-4B-Instruct-2507
was specifically chosen for this role for IFEval 83.4 + verified
multi-language support including ru.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from explorer.agents.base import RoleAgent
from explorer.role_resolver import ModuleRole

logger = logging.getLogger("explorer.agents.ambiguity")


SYSTEM_PROMPT = """Ты — модуль разрешения неоднозначностей в автоматизированном
тестировании мобильных приложений.

Получаешь задачу пользователя и описание текущего экрана. Если задача
имеет несколько разумных интерпретаций — выбери ОДНУ каноническую,
дай краткое объяснение и список альтернатив.

Отвечай строго JSON:
{
  "is_ambiguous": true | false,
  "canonical_path": "<краткое описание выбранного пути>",
  "alternatives": ["<альтернативный путь 1>", ...],
  "confidence": 0.0-1.0,
  "reasoning": "<одно-два предложения на русском>"
}

Если задача однозначная — is_ambiguous=false, canonical_path=пересказ
задачи, alternatives=[]."""


@dataclass
class CanonicalPath:
    is_ambiguous: bool
    canonical_path: str
    alternatives: list[str]
    confidence: float
    reasoning: str


class AmbiguityAgent(RoleAgent):
    role = ModuleRole.AMBIGUITY_RESOLVER

    async def resolve(
        self,
        task_description: str,
        screen_description: str,
    ) -> CanonicalPath | None:
        """Returns a chosen canonical path, or ``None`` when the role
        is unassigned. Callers safely skip ambiguity resolution and
        proceed with the raw task in that case.
        """
        user_msg = (
            f"Задача: {task_description}\n\n"
            f"Текущий экран:\n{screen_description}"
        )
        result = await self.call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,  # mostly deterministic, slight room for nuance
            max_tokens=384,
            response_format={"type": "json_object"},
        )
        if result is None:
            return None

        text = result.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Ambiguity JSON parse failed (%s): %s", exc, text[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        return CanonicalPath(
            is_ambiguous=bool(parsed.get("is_ambiguous", False)),
            canonical_path=str(parsed.get("canonical_path", "")),
            alternatives=list(parsed.get("alternatives") or []),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", "")),
        )
