"""LLM-powered defect detection for the exploration agent.

Runs AFTER each action. Takes as input:
  - what the agent just did (action + element + value)
  - the screen BEFORE the action (elements + hash)
  - the screen AFTER the action (elements + hash)
  - optionally: the current scenario step (if following one)
  - optionally: RAG spec snippets relevant to this screen

Decides:
  1. Was this an infrastructure issue (network down, screen didn't load)?
     → skip, don't count as a defect
  2. Did the action produce the expected result (or match the spec)?
     → if not, post a defect with priority + kind
  3. Everything looks fine?
     → do nothing

The detector is deliberately conservative — better to miss a defect than
report noise. QA will review the top-priority findings and adjust the
prompt if precision is too low.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a quality analyst reviewing the last action of an \
automated mobile app exploration agent. Your job: decide whether the observed \
result is a real defect and, if so, assign priority and category.

You will respond with ONE JSON object with these fields:
{
    "is_defect": true | false,
    "is_infra": true | false,
    "priority": "P0" | "P1" | "P2" | "P3",
    "kind": "functional" | "ui" | "validation" | "navigation" | "performance" | "crash" | "spec_mismatch" | "infra_noise",
    "title": "<one-line summary in Russian, <= 80 chars>",
    "description": "<reproduction steps + expected vs actual in Russian>"
}

PRIORITY RUBRIC:
  P0 = app crashed, main login/payment flow broken, data loss
  P1 = feature doesn't work, navigation dead-end, clear spec violation
  P2 = works but wrong (edge case, validation lets through invalid input)
  P3 = cosmetic, minor text issues

INFRA RULES (set is_infra=true, is_defect=false):
  - Screen didn't load / network error / timeout
  - Test data missing or invalid for the step
  - App failed to launch
  - Agent's previous action was wrong and the current screen is just
    "recover from that mistake", not a real bug

BE CONSERVATIVE. If unsure, set is_defect=false. Missing a defect is OK;
posting noise is not.

Respond with ONLY the JSON object, no commentary."""


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced {...} substring, or None if not found.

    Walks the string tracking nesting depth and string literals. Used to
    rescue verdicts when the model emits trailing reasoning after the JSON
    object — json.loads alone would raise "Extra data" and we'd lose the
    classification.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, c in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


@dataclass
class DefectVerdict:
    """Parsed LLM response. `is_defect=True` means we should post a defect row."""

    is_defect: bool
    is_infra: bool
    priority: str
    kind: str
    title: str
    description: str


class DefectDetector:
    """Thin wrapper around the RAG LLM for defect verdicts.

    Uses the same Qwen3-8B Instruct used for RAG answers — it's good at
    Russian Q&A and fast enough to run after every step.
    """

    def __init__(
        self,
        llm_base_url: str,
        model: str = "rag-chat",
        timeout: float = 30.0,
    ) -> None:
        self.llm_url = llm_base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def classify(
        self,
        *,
        action: str,
        element_label: str,
        value: str | None,
        screen_name_before: str,
        screen_name_after: str,
        element_count_before: int,
        element_count_after: int,
        expected_result: str | None = None,
        spec_snippet: str | None = None,
        error_message: str | None = None,
    ) -> DefectVerdict | None:
        """Ask the LLM whether this step produced a defect.

        Returns None on transport error — caller should continue without
        defect classification rather than fail the run.
        """
        context_parts = [
            f"Action: {action} on '{element_label}'"
            + (f" with value '{value}'" if value else ""),
            f"Screen before: {screen_name_before} ({element_count_before} elements)",
            f"Screen after: {screen_name_after} ({element_count_after} elements)",
        ]
        if expected_result:
            context_parts.append(f"Expected result (from scenario): {expected_result}")
        if spec_snippet:
            context_parts.append(f"Relevant spec: {spec_snippet}")
        if error_message:
            context_parts.append(f"Error observed: {error_message}")

        user_prompt = "\n".join(context_parts)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.llm_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 500,
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = (data["choices"][0]["message"].get("content") or "").strip()
                # Qwen3 sometimes ignores response_format and emits things like
                #   <think>...</think>
                #   {"is_defect": ...}
                #   "reasoning": "..."
                # We need to (a) strip <think> blocks and (b) extract just the
                # first valid JSON object — json.loads chokes on trailing text.
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                # Extract the first balanced {...} block. Stops at the first
                # closing brace at depth 0; ignores braces inside strings.
                first_obj = _extract_first_json_object(raw)
                if first_obj is None:
                    raise ValueError(f"no JSON object in response: {raw[:200]}")
                parsed = json.loads(first_obj)
                return DefectVerdict(
                    is_defect=bool(parsed.get("is_defect", False)),
                    is_infra=bool(parsed.get("is_infra", False)),
                    priority=str(parsed.get("priority", "P2")),
                    kind=str(parsed.get("kind", "functional")),
                    title=str(parsed.get("title", "")).strip(),
                    description=str(parsed.get("description", "")).strip(),
                )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Defect classification failed: %s", exc)
            return None
        except asyncio.CancelledError:
            raise
