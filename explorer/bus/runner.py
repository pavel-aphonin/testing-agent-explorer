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
            # PER-175 Phase C: the blindness fix in the hot path. Build a
            # VISION AffordanceMap (screen_type via SigLIP2 + element boxes
            # via Screen Parser/OmniParser) instead of classifying AX-tree
            # text. This revives Screen Parser (a previously-dead module)
            # in the same stage. Falls back to the text classifier when
            # there's no screenshot or vision is unavailable, so the chain
            # never wedges.
            import base64
            from explorer.agents import ContextIdentifierAgent, ScreenParserAgent
            from explorer.affordance_builder import build_affordance_map
            from explorer.affordances import AffordanceMap
            agent = ContextIdentifierAgent(resolver)
            parser = ScreenParserAgent(resolver)

            def _editable_regions_from_ax(elements: list) -> list[tuple]:
                """AX-derived text-field rects (pixel space) so the builder
                can mark a vision box editable even when vision can't tell.
                Hybrid perception: vision sees the canvas, AX confirms real
                fields when the tree isn't empty."""
                from explorer.goal_schema import _is_editable_kind
                regions: list[tuple] = []
                for e in elements or []:
                    if not isinstance(e, dict):
                        continue
                    if not _is_editable_kind(e.get("kind") or e.get("type")):
                        continue
                    fr = e.get("frame") or {}
                    try:
                        x, y = int(fr["x"]), int(fr["y"])
                        w, h = int(fr["width"]), int(fr["height"])
                        regions.append((x, y, x + w, y + h))
                    except (KeyError, TypeError, ValueError):
                        continue
                return regions

            async def handle(payload: dict) -> dict | None:
                out = dict(payload)
                shot_b64 = payload.get("screenshot_b64")
                amap: AffordanceMap | None = None
                if shot_b64:
                    shot = base64.b64decode(shot_b64)
                    # screen TYPE from the screenshot (SigLIP2 zero-shot)
                    ctx = await agent.classify_vision(shot)
                    # element BOXES from the screenshot (OmniParser); the
                    # parser returns normalized bboxes → scale to pixels.
                    parsed = await parser.parse(shot)
                    boxes: list[dict] = []
                    if parsed:
                        iw = int(payload.get("screen_w") or 0)
                        ih = int(payload.get("screen_h") or 0)
                        for el in parsed.elements:
                            bb = el.get("bbox") or []
                            if len(bb) == 4 and iw and ih:
                                bb = [int(bb[0] * iw), int(bb[1] * ih),
                                      int(bb[2] * iw), int(bb[3] * ih)]
                            boxes.append({
                                "bbox": bb if len(bb) == 4 else None,
                                "text": el.get("text") or el.get("content") or "",
                                "confidence": el.get("confidence", 0.0),
                            })
                    if ctx is not None or boxes:
                        amap = build_affordance_map(
                            screen_type=(ctx.label if ctx else "unknown"),
                            screen_confidence=(ctx.confidence if ctx else 0.0),
                            boxes=boxes,
                            editable_regions=_editable_regions_from_ax(
                                payload.get("elements") or []
                            ),
                            source="vision",
                        )
                if amap is None:
                    # Fallback: legacy text classifier over the AX block.
                    text = payload.get("elements_block") or payload.get("screen_text") or ""
                    res = await agent.classify(text)
                    amap = AffordanceMap(
                        screen_type=(res.label if res else "unknown"),
                        screen_confidence=(res.confidence if res else 0.0),
                        source="ax_text",
                    )
                # Attach the affordance map (Platform Adapter consumes it)
                # and keep the legacy fields for back-compat consumers.
                out["affordance_map"] = amap.to_dict()
                out["context_label"] = amap.screen_type
                out["context_confidence"] = amap.screen_confidence
                out["context_is_pin"] = bool(amap.is_pin_entry)
                return out
            return handle

        if self.role is ModuleRole.GROUNDER:
            # Batch grounding: resolve coords for EVERY tap_at action in
            # the plan, in order, one stage (the "all coords with
            # sequence" requirement). Uses the real GrounderClient
            # (backend dispatch → UI-TARS); the target_description may
            # live at the top level or inside action_args.
            import base64
            from explorer.grounder_client import GrounderClient
            grounder = GrounderClient(self._backend_url, self._worker_token)

            def _target(act: dict) -> str | None:
                aa = act.get("action_args") if isinstance(act.get("action_args"), dict) else {}
                return act.get("target_description") or aa.get("target_description")

            async def handle(payload: dict) -> dict | None:
                actions = payload.get("actions") or []
                screenshot_b64 = payload.get("screenshot_b64")
                shot = base64.b64decode(screenshot_b64) if screenshot_b64 else b""
                grounded: list[dict] = []
                for act in actions:
                    desc = _target(act)
                    if act.get("action") == "tap_at" and desc and shot:
                        res = await grounder.locate(desc, shot)
                        coords = [res.x, res.y] if res else None
                        grounded.append({**act, "coords": coords})
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
                    # PER-144: batch JSON for a multi-action screen needs
                    # headroom; 1024 truncated mid-array (observed). 2048
                    # fits done+reason+a full action array.
                    max_tokens=2048,
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
                # PER-175 Phase C: Platform Adapter (12th module) resolves
                # the plan against the screen's AffordanceMap — intents →
                # concrete batch, choosing mechanism from what the screen
                # affords (PER-205: enter_text only into editable fields;
                # submit routing). Concrete actions a legacy planner already
                # emitted pass through unchanged.
                #
                # SECRETS: we deliberately do NOT pass a resolve_value here.
                # Expanding a keypad credential into its digit taps needs
                # the actual secret (8-5-2-0), and the codebase keeps secrets
                # in the worker's memory only (PER-111 value_source) — they
                # must not ride the Redis bus. So on the bus path the keypad
                # *digit* expansion is deferred to the worker (it has
                # test_data); here we only ensure the PER-204 submit press
                # is appended after the model's own digit taps. Worker-side
                # bus digit-expansion is a follow-up.
                from explorer.affordances import AffordanceMap
                from explorer.platform_adapter import resolve_plan
                from explorer.planning.hints import append_pin_submit
                amap = AffordanceMap.from_dict(payload.get("affordance_map") or {})
                actions = resolve_plan(
                    actions, amap,
                    test_data_keys=payload.get("test_data_keys", []),
                    resolve_value=None,   # secrets stay worker-side
                )
                # PER-204 submit rule (keypad → ensure a trailing submit).
                # Uses the vision screen_type when present, else the legacy
                # context_is_pin flag.
                is_pin = bool(amap.is_pin_entry) or bool(payload.get("context_is_pin"))
                actions = append_pin_submit(actions, is_pin)
                out = dict(payload)
                out["actions"] = actions
                # Carry the batch terminal verdict forward so the worker
                # (consuming ground.produced) honours done/reason.
                out["done"] = bool(parsed.get("done", False))
                out["reason"] = parsed.get("reason")
                out["expected_next_screen"] = parsed.get("expected_next_screen")
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
