"""Loop-breaking strategies for the LLM exploration agent.

When the agent gets stuck (e.g. ping-ponging between Profile and Settings
without finding a "Logout" button), we need to push it out. Five tactics
are exposed; the agent itself picks one each time it's offered:

  1. skip_step      — declare the current scenario step undoable, advance
  2. swipe_to_find  — element might be off-screen, scroll and re-look
  3. cycle_break    — pick a deliberately-untried element to break monotony
  4. fuzzy_lookup   — re-evaluate elements by intent ("logout-like") not text
  5. abandon_scenario — scenario doesn't match the app, switch to free mode

The detector here only watches the transition history and decides WHEN
to surface the toolbox to the LLM. Per-strategy execution lives in
llm_loop.py because each strategy needs the controller / current screen.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal


Strategy = Literal[
    "skip_step", "swipe_to_find", "cycle_break", "fuzzy_lookup", "abandon_scenario",
]


@dataclass
class LoopVerdict:
    """Output of CycleDetector.check.

    `is_stuck` flips True once we detect a pathological transition pattern.
    `pattern_description` is human-readable text we feed to the LLM so it
    knows WHY we think it's stuck. `suggested_strategies` is the toolbox
    we offer — order doesn't imply preference, the LLM picks.
    """

    is_stuck: bool
    pattern_description: str
    suggested_strategies: list[Strategy]


# Window size = how many recent transitions we look at when deciding
# "is this a loop". Small enough that one-off retries don't trigger,
# large enough to catch A↔B↔A↔B oscillation across 4-6 steps.
_WINDOW = 8

# How many times the SAME (source_hash, target_hash) pair must repeat in
# the window before we shout "loop". 3 is a good balance: 2 might be a
# normal back-navigation, 4+ wastes too many steps.
_REPEAT_THRESHOLD = 3


class CycleDetector:
    """Sliding window over recent (source_id, target_id) transitions."""

    def __init__(self) -> None:
        self._buffer: deque[tuple[str, str]] = deque(maxlen=_WINDOW)
        # Steps we already nagged about, so we don't surface the toolbox
        # over and over for the same loop instance.
        self._last_alerted_at: int = -100

    def record(self, source_id: str, target_id: str) -> None:
        self._buffer.append((source_id, target_id))

    def check(self, current_step: int) -> LoopVerdict:
        """Inspect the buffer, return a verdict."""
        # Don't re-trigger within 5 steps of the last alert — the LLM
        # needs time to actually try the strategy it picked.
        if current_step - self._last_alerted_at < 5:
            return LoopVerdict(False, "", [])

        if len(self._buffer) < _REPEAT_THRESHOLD:
            return LoopVerdict(False, "", [])

        # Count occurrences of each transition.
        counts: dict[tuple[str, str], int] = {}
        for pair in self._buffer:
            counts[pair] = counts.get(pair, 0) + 1

        worst = max(counts.values())
        if worst < _REPEAT_THRESHOLD:
            return LoopVerdict(False, "", [])

        # Find the repeating pair for the description.
        offending = next(p for p, c in counts.items() if c == worst)
        # Detect ping-pong A↔B specifically.
        if (offending[1], offending[0]) in counts and counts[(offending[1], offending[0])] >= 2:
            description = (
                f"ты {worst} раза подряд переходил между двумя экранами "
                f"({_short(offending[0])} ↔ {_short(offending[1])}). "
                "Скорее всего, ожидаемого элемента на этих экранах нет."
            )
        else:
            description = (
                f"один и тот же переход ({_short(offending[0])} → "
                f"{_short(offending[1])}) повторился {worst} раз — это "
                "циклическое поведение."
            )

        self._last_alerted_at = current_step
        return LoopVerdict(
            is_stuck=True,
            pattern_description=description,
            suggested_strategies=[
                "skip_step",
                "swipe_to_find",
                "cycle_break",
                "fuzzy_lookup",
                "abandon_scenario",
            ],
        )


def _short(screen_id: str) -> str:
    return screen_id[:8] if len(screen_id) > 8 else screen_id


_STRATEGY_DESCRIPTIONS: dict[Strategy, str] = {
    "skip_step": (
        "skip_step — пропусти текущий шаг сценария и переходи к следующему. "
        "Используй, если шаг невозможно выполнить (нет нужного элемента)."
    ),
    "swipe_to_find": (
        "swipe_to_find — элемент может быть скрыт ниже видимой области. "
        "Сделай свайп вверх (action=\"swipe\", value=\"up\") и проверь снова."
    ),
    "cycle_break": (
        "cycle_break — выбери элемент, который ты ещё НЕ пробовал на этом "
        "экране, чтобы выйти из круга. Игнорируй кнопки навигации (Назад)."
    ),
    "fuzzy_lookup": (
        "fuzzy_lookup — ищи элемент по смыслу, а не по точному тексту. "
        "Например, «Выйти» может называться «Выход», «Logout», «Sign out»."
    ),
    "abandon_scenario": (
        "abandon_scenario — сценарий не подходит для этого приложения. "
        "Переключайся на свободное исследование, прекрати следовать сценарию."
    ),
}


def render_toolbox_for_prompt(strategies: list[Strategy], why: str) -> str:
    """Format the strategy toolbox as a prompt block. Returned text is
    appended to the system prompt the next time the LLM is asked for an
    action."""
    lines = [
        "",
        "## ⚠ ВНИМАНИЕ: похоже, ты застрял.",
        f"Причина: {why}",
        "",
        "Выбери ОДНУ из стратегий выхода и применяй её. Просто продолжать "
        "то же самое — не вариант:",
        "",
    ]
    for s in strategies:
        lines.append(f"- {_STRATEGY_DESCRIPTIONS[s]}")
    lines.append("")
    lines.append(
        "В поле reasoning КРАТКО объясни, какую стратегию ты выбрал и почему."
    )
    return "\n".join(lines)
