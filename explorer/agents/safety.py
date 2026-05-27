"""PER-196: Safety agent — Llama-Guard-3 action filter.

Pre-dispatch check: given a planned action + screen context, ask
Llama-Guard whether it's safe (i.e. not destructive, not crossing
financial boundaries, not leaking PII). Returns a verdict the
caller uses to allow/block/escalate.

Llama-Guard uses a specific chat template ('User'/'Agent' turns).
We follow the official instruct format from Meta — see
https://huggingface.co/meta-llama/Llama-Guard-3-1B
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from explorer.agents.base import RoleAgent
from explorer.role_resolver import ModuleRole

logger = logging.getLogger("explorer.agents.safety")


SAFETY_TEMPLATE = """Task: Check if the action a UI-testing agent is about to take is safe.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Destructive operations (delete account, wipe data, factory reset)
S2: Financial transactions (transfer money, confirm payment, place order)
S3: PII / credentials submission to non-test endpoints
S4: External communication (send email, SMS, share publicly)
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
Agent: I am about to perform: {action_description}
       Current screen: {screen_context}
<END CONVERSATION>

Provide your safety assessment for the Agent in the above conversation:
- First line MUST be 'safe' or 'unsafe'.
- If unsafe, the second line MUST contain a comma-separated list of
  violated categories (e.g. S1,S2)."""


@dataclass
class SafetyVerdict:
    safe: bool
    categories: list[str]
    raw_response: str


class SafetyAgent(RoleAgent):
    role = ModuleRole.SAFETY_GUARD

    async def check(
        self,
        action_description: str,
        screen_context: str = "",
    ) -> SafetyVerdict | None:
        """Returns a verdict, or ``None`` when the role is unassigned
        (caller proceeds without a safety check — current behaviour
        in environments where safety is intentionally disabled).
        """
        prompt = SAFETY_TEMPLATE.format(
            action_description=action_description,
            screen_context=screen_context or "(no extra context)",
        )
        result = await self.call(
            messages=[
                # Llama-Guard expects a plain user turn — no system
                # prompt. Putting the entire template under ``user``
                # matches Meta's reference notebook.
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # safety judgment should be deterministic
            max_tokens=64,
        )
        if result is None:
            return None

        text = result.content.strip()
        # First line: safe/unsafe verdict
        first_line = text.split("\n", 1)[0].strip().lower()
        if first_line.startswith("unsafe"):
            categories = []
            if "\n" in text:
                second_line = text.split("\n", 1)[1].strip()
                categories = [c.strip().upper() for c in re.split(r"[,\s]+", second_line) if c.strip()]
            return SafetyVerdict(safe=False, categories=categories, raw_response=text)
        if first_line.startswith("safe"):
            return SafetyVerdict(safe=True, categories=[], raw_response=text)
        logger.warning(
            "SafetyAgent: ambiguous Llama-Guard response (defaulting to safe): %s",
            text[:120],
        )
        return SafetyVerdict(safe=True, categories=[], raw_response=text)
