"""PER-203 Phase 2: generic module-runner — the bus↔model bridge.

One process per pipeline module. It:
  1. connects to the bus, ensures its consumer group on the INPUT type
  2. loops: consume input envelope → run the role handler → publish the
     OUTPUT envelope (same run_id+step_id) → ack input
  3. degrades safely: a handler that returns None publishes nothing and
     acks (the stage is a no-op for that message), so an unassigned /
     down model doesn't wedge the stream.

Run one with:
    python -m explorer.bus.runner --role CONTEXT_IDENTIFIER \
        --backend-url http://localhost:8000 --worker-token <tok>

The chain is linearized — every message carries the full screen state
forward, so each stage joins nothing: screen.captured → context.classified
(echoes screen + label) → plan.produced → ground.produced. Side-consumer
modules (safety/critic) read plan.produced without being in the chain.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from explorer.bus.envelope import Envelope, MsgType
from explorer.bus.streams import BusClient
from explorer.role_resolver import ModuleRole, RoleResolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("explorer.bus.runner")


# Handler: takes the input payload, returns the output payload (or None
# to no-op). Async so it can call the model over HTTP.
Handler = Callable[[dict], Awaitable[dict | None]]


@dataclass
class RoleWiring:
    """How a module sits on the bus: what it consumes, what it emits,
    and the consumer-group name."""

    consumes: MsgType
    produces: MsgType | None  # None = terminal / side-effect only
    group: str


# Per-role bus wiring. The linearized chain + side-consumers.
ROLE_WIRING: dict[ModuleRole, RoleWiring] = {
    ModuleRole.CONTEXT_IDENTIFIER: RoleWiring(
        consumes=MsgType.SCREEN_CAPTURED,
        produces=MsgType.CONTEXT_CLASSIFIED,
        group="g.context",
    ),
    ModuleRole.PLANNER: RoleWiring(
        consumes=MsgType.CONTEXT_CLASSIFIED,
        produces=MsgType.PLAN_PRODUCED,
        group="g.planner",
    ),
    ModuleRole.GROUNDER: RoleWiring(
        consumes=MsgType.PLAN_PRODUCED,
        produces=MsgType.GROUND_PRODUCED,
        group="g.grounder",
    ),
    ModuleRole.SAFETY_GUARD: RoleWiring(
        consumes=MsgType.PLAN_PRODUCED,
        produces=None,  # side-consumer: annotates / vetoes, doesn't advance the chain
        group="g.safety",
    ),
}


class ModuleRunner:
    """Drives one module's consume→handle→publish loop."""

    def __init__(self, role: ModuleRole, backend_url: str, worker_token: str):
        self.role = role
        self.wiring = ROLE_WIRING[role]
        self._backend_url = backend_url
        self._worker_token = worker_token
        self._bus = BusClient(consumer_name=f"{role.value.lower()}-runner")
        self._handler: Handler | None = None

    async def _build_handler(self) -> Handler:
        """Construct the role's handler. Imports are local so a runner
        for one role doesn't drag in every agent's deps."""
        from explorer.worker import BackendClient

        backend = BackendClient(self._backend_url, self._worker_token)
        resolver = RoleResolver(backend)

        if self.role is ModuleRole.CONTEXT_IDENTIFIER:
            from explorer.agents import ContextIdentifierAgent
            agent = ContextIdentifierAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                text = payload.get("elements_block") or payload.get("screen_text") or ""
                res = await agent.classify(text)
                # Echo the screen payload forward + attach the label so
                # the planner stage needs no join.
                out = dict(payload)
                out["context_label"] = res.label if res else None
                out["context_confidence"] = res.confidence if res else 0.0
                out["context_is_pin"] = bool(res and res.is_pin_entry)
                return out
            return handle

        if self.role is ModuleRole.GROUNDER:
            # Batch grounding: resolve coords for EVERY tap_at action in
            # the plan, in order, in one stage (the user's "all coords
            # with sequence" requirement).
            from explorer.agents import GrounderAgent
            agent = GrounderAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                actions = payload.get("actions") or []
                screenshot_b64 = payload.get("screenshot_b64")
                import base64
                shot = base64.b64decode(screenshot_b64) if screenshot_b64 else b""
                grounded: list[dict] = []
                for act in actions:
                    if act.get("action") == "tap_at" and act.get("target_description"):
                        coords = await agent.locate(act["target_description"], shot)
                        grounded.append({**act, "coords": list(coords) if coords else None})
                    else:
                        grounded.append(act)
                out = dict(payload)
                out["grounded_actions"] = grounded
                return out
            return handle

        if self.role is ModuleRole.SAFETY_GUARD:
            from explorer.agents import SafetyAgent
            agent = SafetyAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                for act in payload.get("actions") or []:
                    verdict = await agent.check(str(act), payload.get("elements_block", ""))
                    if verdict and not verdict.safe:
                        logger.warning("SAFETY veto on %s: %s", act, verdict.categories)
                return None  # side-consumer, advances nothing
            return handle

        if self.role is ModuleRole.PLANNER:
            # Phase 3b: real planner — assemble the prompt with the
            # shared build_planner_inputs (same hints as the sync path)
            # and call the resolved PLANNER endpoint. Emits an ACTION
            # ARRAY (batch incl. submit) in plan.produced.
            import json
            import re as _re
            from explorer.agents import PlannerAgent
            from explorer.planning.core import build_planner_inputs
            agent = PlannerAgent(resolver)

            def _strip(text: str) -> str:
                # 8B-Think emits a <think>…</think> block before the JSON.
                text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
                text = _re.sub(r"^```(?:json)?\s*", "", text.strip())
                return _re.sub(r"\s*```$", "", text)

            async def handle(payload: dict) -> dict | None:
                inputs = build_planner_inputs(
                    goal_text=payload.get("goal_text", ""),
                    mode=payload.get("mode", "hybrid"),
                    success_criteria=payload.get("success_criteria", ""),
                    elements=payload.get("elements", []),
                    history=payload.get("history", []),
                    step_idx=int(payload.get("step_idx", 0)),
                    max_steps=int(payload.get("max_steps", 30)),
                    user_template=payload.get("user_template", ""),
                    actions=payload.get("actions", []),
                    actions_block=payload.get("actions_block", ""),
                    test_data_block=payload.get("test_data_block", ""),
                    test_data_keys=payload.get("test_data_keys", []),
                    visited_summary=payload.get("visited_summary", ""),
                    memory_block=payload.get("memory_block", ""),
                    context_is_pin=bool(payload.get("context_is_pin")),
                )
                res = await agent.call(
                    messages=[
                        {"role": "system", "content": payload.get("system_prompt", "")},
                        {"role": "user", "content": inputs["user_prompt"]},
                    ],
                    max_tokens=1024,
                    # Enforce the PER-170 batch schema so the model emits
                    # valid {actions:[...]} with action enums from the
                    # workspace dictionary — not freelanced names like
                    # "input_pin_code" (observed with loose json_object).
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "goal_plan", "schema": inputs["schema"]},
                    },
                )
                if res is None:
                    return None
                try:
                    parsed = json.loads(_strip(res.content))
                except json.JSONDecodeError:
                    logger.warning("planner JSON parse failed: %s", res.content[:160])
                    return None
                # Accept either {actions:[...]} (batch) or a single action object.
                if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list):
                    actions = parsed["actions"]
                elif isinstance(parsed, dict):
                    actions = [parsed]
                else:
                    return None
                out = dict(payload)
                out["actions"] = actions
                return out
            return handle

        raise ValueError(f"No bus handler for role {self.role}")

    async def run(self) -> None:
        await self._bus.connect()
        await self._bus.ensure_group(self.wiring.consumes, self.wiring.group)
        self._handler = await self._build_handler()
        logger.info(
            "ModuleRunner[%s] consuming %s → producing %s",
            self.role.value, self.wiring.consumes.value,
            self.wiring.produces.value if self.wiring.produces else "(none)",
        )
        while True:
            batch = await self._bus.consume(self.wiring.consumes, self.wiring.group, count=4, block_ms=5000)
            for entry_id, env in batch:
                try:
                    out_payload = await self._handler(env.payload)
                    if out_payload is not None and self.wiring.produces is not None:
                        await self._bus.publish(Envelope(
                            run_id=env.run_id, step_id=env.step_id,
                            type=self.wiring.produces, payload=out_payload,
                        ))
                except Exception:
                    logger.exception("handler error on %s (run=%s step=%s)", entry_id, env.run_id, env.step_id)
                finally:
                    await self._bus.ack(self.wiring.consumes, self.wiring.group, entry_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="PER-203 bus module-runner")
    ap.add_argument("--role", required=True, choices=[r.value for r in ROLE_WIRING])
    ap.add_argument("--backend-url", default="http://localhost:8000")
    ap.add_argument("--worker-token", required=True)
    args = ap.parse_args()
    runner = ModuleRunner(ModuleRole(args.role), args.backend_url, args.worker_token)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
