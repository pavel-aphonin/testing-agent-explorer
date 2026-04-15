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

    `is_stuck`: a pathological transition pattern detected.
    `pattern_kind`: what TYPE of loop — self_loop (A→A: action didn't
       change screen) vs ping_pong (A↔B: bouncing between two screens)
       vs repeat_edge (A→B→A→B→…: long cycle repeating).
    `offending_ids`: the screen ids involved, so the caller can look
       up pretty names before rendering.
    `pattern_description`: pre-formatted Russian text for logs. The
       caller typically replaces the ids with real screen names
       before showing it to the user.
    `suggested_strategies`: toolbox offered to the LLM.
    `escalation`:
       - "warn"  : first time in a while — soft nudge, let the LLM pick.
       - "force" : loop persisted despite the previous warn — the
                   caller should stop asking the LLM and take a
                   deterministic action (pick an unused element /
                   swipe / back-navigate).
    """

    is_stuck: bool
    pattern_description: str
    suggested_strategies: list[Strategy]
    pattern_kind: Literal["self_loop", "ping_pong", "repeat_edge"] = "repeat_edge"
    offending_ids: tuple[str, ...] = ()
    repeat_count: int = 0
    escalation: Literal["warn", "force"] = "warn"


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
        self._last_alerted_at: int = -100
        self._last_alerted_pattern: tuple[str, ...] = ()

    def record(self, source_id: str, target_id: str) -> None:
        self._buffer.append((source_id, target_id))

    def check(self, current_step: int) -> LoopVerdict:
        """Inspect the buffer, return a verdict."""
        # Don't re-trigger within 3 steps of the last alert — gives the
        # LLM time to try its chosen strategy. Shorter than the original
        # 5 because ESCALATION to force-break needs to happen faster than
        # the user can lose patience.
        if current_step - self._last_alerted_at < 3:
            return LoopVerdict(False, "", [])

        if len(self._buffer) < _REPEAT_THRESHOLD:
            return LoopVerdict(False, "", [])

        counts: dict[tuple[str, str], int] = {}
        for pair in self._buffer:
            counts[pair] = counts.get(pair, 0) + 1

        worst = max(counts.values())
        if worst < _REPEAT_THRESHOLD:
            return LoopVerdict(False, "", [])

        offending = next(p for p, c in counts.items() if c == worst)

        # Classify the pattern — three kinds need different responses.
        if offending[0] == offending[1]:
            kind = "self_loop"
            description = (
                f"ты {worst} раз подряд оставался на одном экране — "
                f"действие не меняет состояние приложения."
            )
            offending_ids = (offending[0],)
        elif (
            (offending[1], offending[0]) in counts
            and counts[(offending[1], offending[0])] >= 2
        ):
            kind = "ping_pong"
            description = (
                f"ты {worst} раз подряд прыгал между двумя экранами. "
                f"Скорее всего, нужного элемента ни на одном из них нет."
            )
            offending_ids = (offending[0], offending[1])
        else:
            kind = "repeat_edge"
            description = (
                f"один и тот же переход повторился {worst} раз — "
                f"похоже, ты в цикле."
            )
            offending_ids = (offending[0], offending[1])

        # ESCALATION: if we alerted the same pattern recently, we already
        # gave the LLM a chance and it didn't work. Tell the caller to
        # stop asking and take over deterministically.
        escalation: Literal["warn", "force"] = "warn"
        if offending_ids == self._last_alerted_pattern:
            escalation = "force"

        self._last_alerted_at = current_step
        self._last_alerted_pattern = offending_ids
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
            pattern_kind=kind,
            offending_ids=offending_ids,
            repeat_count=worst,
            escalation=escalation,
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
