"""PER-203: message envelope + type constants for the bus.

Every message on every stream is an ``Envelope``. The payload is a free
dict serialised to JSON in one Redis-Streams field; the rest are flat
string fields so XADD stays cheap and the stream is greppable with
``XRANGE`` during debugging.

Correlation: ``run_id`` + ``step_id`` together identify one perceive→
act cycle. A consumer that needs to join messages across stages (e.g.
the executor matching ground.produced back to the screen it planned)
keys on this pair.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MsgType(str, Enum):
    """Stream message types = the stages of the pipeline.

    The value is also the stream-name suffix (see streams.stream_for).
    Append-only; renaming a value renames its stream.

    PER-175 full integration — every one of the 12 modules (+ ScreenSeekeR
    as a 13th grounding-refinement stage) has a link, so all are exercised
    in one decision cycle:

        screen.captured     worker emits raw screen state (screenshot + AX)
        → screen.parsed     Screen Parser (OmniParser) — element boxes/types
        → screen.perceived  Dynamic Perceiver (SigLIP2) — novelty / Δ vs last
        → context.classified Context Identifier — screen_type + AffordanceMap (VISION)
        → plan.produced     Planner — INTENTS (what), via Ambiguity/Memory/Reflection
        → plan.critiqued    Reward Critic — pre-operative verdict (Safety side-consumes)
        → actions.resolved  Platform Adapter (12th, pure code) — intent+affordance → batch (how)
        → ground.refined    ScreenSeekeR — region narrowing for small targets
        → ground.produced   Grounder — coordinates
        → ground.verified   Grounding Verifier — confidence gate (worker consumes)
        → action.dispatched worker post-dispatch (Memory side-consumes to record)

    The legacy subset (screen.captured → context.classified → plan.produced
    → ground.produced) stays valid so the pipeline keeps running while
    stages are filled in.
    """

    SCREEN_CAPTURED = "screen.captured"
    SCREEN_PARSED = "screen.parsed"
    SCREEN_PERCEIVED = "screen.perceived"
    CONTEXT_CLASSIFIED = "context.classified"
    PLAN_PRODUCED = "plan.produced"
    PLAN_CRITIQUED = "plan.critiqued"
    ACTIONS_RESOLVED = "actions.resolved"
    GROUND_REFINED = "ground.refined"
    GROUND_PRODUCED = "ground.produced"
    GROUND_VERIFIED = "ground.verified"
    ACTION_DISPATCHED = "action.dispatched"


@dataclass
class Envelope:
    """One message on the bus.

    ``msg_id`` is our own UUID (stable across the whole hop); Redis also
    assigns its native stream entry id on XADD, returned separately by
    the client and used for XACK — we keep ours for end-to-end tracing
    independent of which stream a copy lives on.
    """

    run_id: str
    step_id: int
    type: MsgType
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    def to_fields(self) -> dict[str, str]:
        """Flatten to the Redis-Streams field map (all strings)."""
        return {
            "msg_id": self.msg_id,
            "run_id": self.run_id,
            "step_id": str(self.step_id),
            "type": self.type.value,
            "ts": repr(self.ts),
            "payload": json.dumps(self.payload, ensure_ascii=False),
        }

    @classmethod
    def from_fields(cls, fields: dict[str, str]) -> "Envelope":
        """Rebuild from a Redis-Streams field map. Tolerant of bytes
        keys/values (redis-py returns bytes unless decode_responses)."""
        def _s(v: Any) -> str:
            return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)

        norm = {_s(k): _s(v) for k, v in fields.items()}
        try:
            payload = json.loads(norm.get("payload", "{}"))
        except json.JSONDecodeError:
            payload = {}
        return cls(
            run_id=norm.get("run_id", ""),
            step_id=int(norm.get("step_id", "0") or 0),
            type=MsgType(norm["type"]),
            payload=payload if isinstance(payload, dict) else {},
            msg_id=norm.get("msg_id", uuid.uuid4().hex),
            ts=float(norm.get("ts", "0") or 0.0),
        )
