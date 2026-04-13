"""Exploration strategies — how the agent picks elements, actions, and data.

Each strategy implements the same interface so the exploration loop stays
clean and simple regardless of whether we're doing random Monte Carlo,
PUCT tree search, or LLM-guided exploration.

    element = strategy.select_element(screen, elements)
    action  = strategy.select_action(element)
    data    = strategy.select_data(element, action)
    ...
    strategy.update(screen_id, element, action, reward)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from explorer.models import ActionType, ElementKind, ElementSnapshot, ScreenNode


# ─────────────────────────────── data input generation ──

# Map element labels/test_ids to likely data categories
_EMAIL_HINTS = ("email", "e-mail", "почта", "mail")
_PASSWORD_HINTS = ("password", "пароль", "pass")
_PHONE_HINTS = ("phone", "телефон", "тел")
_NAME_HINTS = ("name", "имя", "фамилия", "фио", "lastname", "firstname")

_PBT_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "email": [
        ("test@test.com", "valid"),
        ("", "empty"),
        ("not-an-email", "invalid_format"),
        ("a" * 500 + "@test.com", "overflow"),
        ("<script>alert(1)</script>@test.com", "xss"),
        ("' OR 1=1 --", "sql_injection"),
        ("test@test.com test@test.com", "duplicate"),
        ("ТЕСТ@тест.рф", "unicode"),
    ],
    "password": [
        ("password123", "valid"),
        ("", "empty"),
        ("ab", "too_short"),
        ("a" * 1000, "overflow"),
        ("<script>alert(1)</script>", "xss"),
        ("' OR 1=1 --", "sql_injection"),
    ],
    "phone": [
        ("+7 900 000-00-00", "valid"),
        ("", "empty"),
        ("abc", "invalid_format"),
        ("+7 900 000-00-0" * 20, "overflow"),
    ],
    "name": [
        ("Иван Иванов", "valid"),
        ("", "empty"),
        ("A", "too_short"),
        ("А" * 1000, "overflow"),
        ("<script>alert(1)</script>", "xss"),
    ],
    "generic": [
        ("test value", "valid"),
        ("", "empty"),
        ("a" * 1000, "overflow"),
        ("<script>alert(1)</script>", "xss"),
    ],
}


def _classify_field(element: ElementSnapshot) -> str:
    """Guess what kind of data a text field expects from its label/test_id/value."""
    hints = " ".join(filter(None, [
        element.label,
        element.test_id,
        element.value,
        # Also check element_type for clues
        element.element_type,
    ])).lower()
    if any(h in hints for h in _EMAIL_HINTS):
        return "email"
    if any(h in hints for h in _PASSWORD_HINTS):
        return "password"
    if "secure" in hints:  # SecureTextField = password
        return "password"
    if any(h in hints for h in _PHONE_HINTS):
        return "phone"
    if any(h in hints for h in _NAME_HINTS):
        return "name"
    return "generic"


class PBTInputGenerator:
    """Property-based testing input generator.

    For each text field, generates all variants (valid, empty, overflow,
    XSS, SQL injection...). Tracks which variants have been tried on
    which field so successive calls return the NEXT untried variant.
    """

    def __init__(self) -> None:
        self._tried: dict[str, int] = {}  # element_key → next variant index

    def next_input(self, element: ElementSnapshot) -> tuple[str, str] | None:
        """Return (value, category) for the next untried PBT variant.
        Returns None if all variants exhausted for this field."""
        category = _classify_field(element)
        variants = _PBT_VARIANTS.get(category, _PBT_VARIANTS["generic"])
        key = _element_key(element)
        idx = self._tried.get(key, 0)
        if idx >= len(variants):
            return None  # all variants tried
        self._tried[key] = idx + 1
        return variants[idx]

    def valid_input(self, element: ElementSnapshot) -> str:
        """Return the valid-case value for a field (for form fill)."""
        category = _classify_field(element)
        variants = _PBT_VARIANTS.get(category, _PBT_VARIANTS["generic"])
        return variants[0][0]  # first variant is always "valid"

    def remaining_count(self, element: ElementSnapshot) -> int:
        """How many untried variants remain for this field."""
        category = _classify_field(element)
        variants = _PBT_VARIANTS.get(category, _PBT_VARIANTS["generic"])
        key = _element_key(element)
        idx = self._tried.get(key, 0)
        return max(0, len(variants) - idx)


def generate_input(element: ElementSnapshot) -> str:
    """Return a sensible value to type into a text field (simple version)."""
    category = _classify_field(element)
    return _PBT_VARIANTS.get(category, _PBT_VARIANTS["generic"])[0][0]


# ─────────────────────────────── action selection ──

def available_actions(element: ElementSnapshot) -> list[ActionType]:
    """What actions can we perform on this element?"""
    if element.kind == ElementKind.TEXT_FIELD:
        return [ActionType.INPUT]
    if element.kind == ElementKind.SWITCH:
        return [ActionType.TAP]
    # Buttons, links, labels, images, other — tap
    return [ActionType.TAP]


# ─────────────────────────────── strategy interface ──

class ExplorationStrategy(ABC):
    """Abstract interface for how the agent decides what to do."""

    @abstractmethod
    def select_element(
        self,
        screen: ScreenNode,
        elements: list[ElementSnapshot],
        visited_transitions: set[str],
    ) -> ElementSnapshot | None:
        """Pick an element to interact with. Return None if nothing to do."""
        ...

    def select_action(self, element: ElementSnapshot) -> ActionType:
        """Pick an action type for this element."""
        actions = available_actions(element)
        return actions[0]

    def select_data(self, element: ElementSnapshot, action: ActionType) -> str | None:
        """Pick input data if the action requires it."""
        if action == ActionType.INPUT:
            return generate_input(element)
        return None

    @abstractmethod
    def update(
        self,
        screen_id: str,
        element: ElementSnapshot,
        action: ActionType,
        target_screen_id: str,
        is_new_screen: bool,
    ) -> None:
        """Learn from the outcome of an action."""
        ...


# ─────────────────────────────── MC strategy ──

class MCStrategy(ExplorationStrategy):
    """Pure random exploration. No learning, no memory.

    Selects a random element, skipping transitions we've already recorded.
    Falls back to any element if all transitions are known.
    """

    def select_element(
        self,
        screen: ScreenNode,
        elements: list[ElementSnapshot],
        visited_transitions: set[str],
    ) -> ElementSnapshot | None:
        if not elements:
            return None

        # Prefer elements whose transition from this screen is not yet recorded
        unexplored = [
            el for el in elements
            if _transition_key(screen.screen_id, el) not in visited_transitions
        ]
        if unexplored:
            return random.choice(unexplored)

        # All explored — pick random anyway (maybe different outcome this time)
        return random.choice(elements)

    def update(self, screen_id, element, action, target_screen_id, is_new_screen):
        pass  # MC doesn't learn


# ─────────────────────────────── PUCT strategy ──

@dataclass
class _ActionStats:
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class _ScreenStats:
    visit_count: int = 0
    actions: dict[str, _ActionStats] = field(default_factory=dict)


class PUCTStrategy(ExplorationStrategy):
    """PUCT selection (AlphaZero-style) with self-loop penalties.

    score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

    Rewards:
        +1.0  brand-new screen discovered
        +0.2  known but different screen reached
        -0.5  self-loop (same screen)
    """

    def __init__(self, c_puct: float = 2.0):
        self.c_puct = c_puct
        self._states: dict[str, _ScreenStats] = {}

    def _ensure_state(
        self, screen_id: str, elements: list[ElementSnapshot]
    ) -> _ScreenStats:
        if screen_id not in self._states:
            ss = _ScreenStats()
            n = max(len(elements), 1)
            for el in elements:
                key = _element_key(el)
                ss.actions[key] = _ActionStats(prior=1.0 / n)
            self._states[screen_id] = ss
        return self._states[screen_id]

    def select_element(
        self,
        screen: ScreenNode,
        elements: list[ElementSnapshot],
        visited_transitions: set[str],
    ) -> ElementSnapshot | None:
        if not elements:
            return None

        ss = self._ensure_state(screen.screen_id, elements)
        ss.visit_count += 1
        sqrt_n = math.sqrt(max(ss.visit_count, 1))

        best_el = None
        best_score = float("-inf")

        for el in elements:
            key = _element_key(el)
            stats = ss.actions.get(key)
            if stats is None:
                # New element appeared dynamically — register it
                stats = _ActionStats(prior=0.5)
                ss.actions[key] = stats

            u = self.c_puct * stats.prior * sqrt_n / (1 + stats.visit_count)
            score = stats.q_value + u

            if score > best_score:
                best_score = score
                best_el = el

        return best_el

    def update(self, screen_id, element, action, target_screen_id, is_new_screen):
        ss = self._states.get(screen_id)
        if ss is None:
            return
        key = _element_key(element)
        stats = ss.actions.get(key)
        if stats is None:
            return

        if is_new_screen:
            reward = 1.0
        elif target_screen_id != screen_id:
            reward = 0.2
        else:
            reward = -0.5

        stats.visit_count += 1
        stats.total_value += reward


# ─────────────────────────────── helpers ──

def _element_key(el: ElementSnapshot) -> str:
    """Stable key for an element within a screen."""
    return f"{el.kind}|{el.label or ''}|{el.test_id or ''}"


def _transition_key(screen_id: str, el: ElementSnapshot) -> str:
    """Key for a (screen, element) transition."""
    return f"{screen_id}|{_element_key(el)}"
