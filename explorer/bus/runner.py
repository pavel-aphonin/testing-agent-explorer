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


# PER-175 full integration: bus wiring for ALL 13 modules. The linearized
# main chain advances one link per stage; side-consumers read a link
# without producing the next. Topology (validated by test_bus_chain):
#
#   screen.captured
#     → SCREEN_PARSER      → screen.parsed
#     → DYNAMIC_PERCEIVER  → screen.perceived
#     → CONTEXT_IDENTIFIER → context.classified
#     → PLANNER            → plan.produced
#     → REWARD_CRITIC      → plan.critiqued      (SAFETY_GUARD side-consumes plan.produced)
#                                                (REFLECTION side-consumes plan.produced)
#     → PLATFORM_ADAPTER   → actions.resolved    (AMBIGUITY side-consumes context.classified)
#     → SCREEN_SEEKER      → ground.refined
#     → GROUNDER           → ground.produced
#     → GROUNDING_VERIFIER → ground.verified     (worker consumes; MEMORY side-consumes)
#
# Every module has exactly one runner. Code-only modules (PLATFORM_ADAPTER,
# SCREEN_SEEKER) run their pure logic in the handler — no model endpoint.
ROLE_WIRING: dict[ModuleRole, RoleWiring] = {
    ModuleRole.SCREEN_PARSER: RoleWiring(
        consumes=MsgType.SCREEN_CAPTURED,
        produces=MsgType.SCREEN_PARSED,
        group="g.screenparser",
    ),
    ModuleRole.DYNAMIC_PERCEIVER: RoleWiring(
        consumes=MsgType.SCREEN_PARSED,
        produces=MsgType.SCREEN_PERCEIVED,
        group="g.perceiver",
    ),
    ModuleRole.CONTEXT_IDENTIFIER: RoleWiring(
        consumes=MsgType.SCREEN_PERCEIVED,
        produces=MsgType.CONTEXT_CLASSIFIED,
        group="g.context",
    ),
    ModuleRole.AMBIGUITY_RESOLVER: RoleWiring(
        consumes=MsgType.CONTEXT_CLASSIFIED,
        produces=None,  # side-consumer: canonicalises the goal, annotates only
        group="g.ambiguity",
    ),
    ModuleRole.PLANNER: RoleWiring(
        consumes=MsgType.CONTEXT_CLASSIFIED,
        produces=MsgType.PLAN_PRODUCED,
        group="g.planner",
    ),
    ModuleRole.SAFETY_GUARD: RoleWiring(
        consumes=MsgType.PLAN_PRODUCED,
        produces=None,  # side-consumer: vetoes unsafe actions, doesn't advance
        group="g.safety",
    ),
    ModuleRole.REFLECTION: RoleWiring(
        consumes=MsgType.PLAN_PRODUCED,
        produces=None,  # side-consumer: stuck-review, annotates only
        group="g.reflection",
    ),
    ModuleRole.REWARD_CRITIC: RoleWiring(
        consumes=MsgType.PLAN_PRODUCED,
        produces=MsgType.PLAN_CRITIQUED,
        group="g.critic",
    ),
    ModuleRole.PLATFORM_ADAPTER: RoleWiring(
        consumes=MsgType.PLAN_CRITIQUED,
        produces=MsgType.ACTIONS_RESOLVED,
        group="g.adapter",
    ),
    ModuleRole.SCREEN_SEEKER: RoleWiring(
        consumes=MsgType.ACTIONS_RESOLVED,
        produces=MsgType.GROUND_REFINED,
        group="g.seeker",
    ),
    ModuleRole.GROUNDER: RoleWiring(
        consumes=MsgType.GROUND_REFINED,
        produces=MsgType.GROUND_PRODUCED,
        group="g.grounder",
    ),
    ModuleRole.GROUNDING_VERIFIER: RoleWiring(
        consumes=MsgType.GROUND_PRODUCED,
        produces=MsgType.GROUND_VERIFIED,
        group="g.verifier",
    ),
    ModuleRole.MEMORY: RoleWiring(
        consumes=MsgType.GROUND_VERIFIED,
        produces=None,  # side-consumer: records the step into episodic memory
        group="g.memory",
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
                    # element BOXES: prefer the Screen Parser stage's output
                    # (it ran first in the chain — no double OmniParser call);
                    # fall back to parsing here if it's absent (legacy 4-link
                    # chain or Screen Parser unassigned).
                    raw_boxes = payload.get("parsed_boxes")
                    if raw_boxes is None:
                        parsed = await parser.parse(shot)
                        raw_boxes = parsed.elements if parsed else []
                    boxes: list[dict] = []
                    iw = int(payload.get("screen_w") or 0)
                    ih = int(payload.get("screen_h") or 0)
                    for el in raw_boxes:
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
                # PER-175: PLANNER emits the RAW plan only. The Reward Critic
                # (next stage) judges this plan, then the Platform Adapter (own
                # runner, consuming plan.critiqued) resolves intents→concrete
                # batch. Keeping planner output raw is what lets the critic
                # judge before the adapter rewrites it.
                out = dict(payload)
                out["actions"] = actions
                out["done"] = bool(parsed.get("done", False))
                out["reason"] = parsed.get("reason")
                out["expected_next_screen"] = parsed.get("expected_next_screen")
                return out
            return handle

        if self.role is ModuleRole.SCREEN_PARSER:
            # PER-175 full integration: element detection (OmniParser) as its
            # own first perception stage. Attaches raw normalized boxes to the
            # payload; the Context Identifier downstream turns them + the
            # screen-type into the AffordanceMap.
            import base64
            from explorer.agents import ScreenParserAgent
            parser = ScreenParserAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                out = dict(payload)
                shot_b64 = payload.get("screenshot_b64")
                if shot_b64:
                    parsed = await parser.parse(base64.b64decode(shot_b64))
                    out["parsed_boxes"] = parsed.elements if parsed else []
                else:
                    out["parsed_boxes"] = []
                return out
            return handle

        if self.role is ModuleRole.DYNAMIC_PERCEIVER:
            # Did the screen actually change vs the previous step? SigLIP2
            # cosine. Annotates ``screen_changed`` so the planner/critic can
            # tell a no-op action from real progress. Keeps last screenshot
            # in closure state (per-runner, single worker → safe).
            import base64
            from explorer.agents import DynamicPerceiverAgent
            perceiver = DynamicPerceiverAgent(resolver)
            last_shot: dict[str, bytes] = {}

            async def handle(payload: dict) -> dict | None:
                out = dict(payload)
                shot_b64 = payload.get("screenshot_b64")
                changed = True
                if shot_b64:
                    shot = base64.b64decode(shot_b64)
                    prev = last_shot.get(payload.get("run_id", ""))
                    if prev:
                        sim = await perceiver.compare(prev, shot)
                        if sim is not None:
                            changed = sim.changed
                    last_shot[payload.get("run_id", "")] = shot
                out["screen_changed"] = changed
                return out
            return handle

        if self.role is ModuleRole.AMBIGUITY_RESOLVER:
            # Side-consumer: canonicalise the goal once it has screen context.
            # Annotates only (logs) — never advances the chain. Cheap, and the
            # planner already has the raw goal, so this is advisory.
            from explorer.agents import AmbiguityAgent
            agent = AmbiguityAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                goal = payload.get("goal_text", "")
                if goal:
                    res = await agent.resolve(goal, payload.get("elements_block", ""))
                    if res and res.is_ambiguous:
                        logger.info(
                            "AMBIGUITY: %r → canonical=%r (conf=%.2f)",
                            goal[:60], res.canonical_path[:80], res.confidence,
                        )
                return None  # side-consumer
            return handle

        if self.role is ModuleRole.REFLECTION:
            # Side-consumer: when history shows the agent looping, log a
            # specific "try this instead" note. Advisory (annotation only) on
            # the bus path; the sync path injects it into the prompt.
            from explorer.agents import ReflectionAgent
            agent = ReflectionAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                hist = payload.get("history") or []
                if len(hist) >= 3:
                    note = await agent.review(payload.get("goal_text", ""), hist)
                    if note and note.stuck:
                        logger.info("REFLECTION: stuck — %s", note.recommendation[:120])
                return None  # side-consumer
            return handle

        if self.role is ModuleRole.REWARD_CRITIC:
            # Pre-operative critic: score whether the planned batch looks like
            # progress before it's executed. Annotates ``critic_progress`` and
            # passes the plan through unchanged (plan.produced → plan.critiqued).
            from explorer.agents import RewardCriticAgent
            agent = RewardCriticAgent(resolver)

            async def handle(payload: dict) -> dict | None:
                out = dict(payload)
                actions = payload.get("actions") or []
                if actions:
                    desc = "; ".join(
                        str(a.get("action") or "") + " " +
                        str((a.get("action_args") or {}).get("target_description")
                            or a.get("element_id") or "")
                        for a in actions[:6]
                    )
                    score = await agent.score(
                        payload.get("goal_text", ""), desc,
                        payload.get("elements_block", ""), "",
                    )
                    if score is not None:
                        out["critic_progress"] = score.progress
                        out["critic_reason"] = score.reason
                        if not score.advanced:
                            logger.info(
                                "CRITIC: plan looks low-progress (%.2f) — %s",
                                score.progress, score.reason[:100],
                            )
                return out
            return handle

        if self.role is ModuleRole.PLATFORM_ADAPTER:
            # 12th module, pure code. Resolves the (critiqued) plan against the
            # screen's AffordanceMap: intents → concrete batch, mechanism from
            # what the screen affords (PER-205 editable guard; PER-204 submit).
            # SECRETS never ride the bus → keypad digit-expansion stays worker-
            # side; here we only ensure the trailing submit press.
            from explorer.affordances import AffordanceMap
            from explorer.platform_adapter import resolve_plan
            from explorer.planning.hints import append_pin_submit

            async def handle(payload: dict) -> dict | None:
                actions = payload.get("actions") or []
                amap = AffordanceMap.from_dict(payload.get("affordance_map") or {})
                actions = resolve_plan(
                    actions, amap,
                    test_data_keys=payload.get("test_data_keys", []),
                    resolve_value=None,  # secrets stay worker-side
                )
                is_pin = bool(amap.is_pin_entry) or bool(payload.get("context_is_pin"))
                actions = append_pin_submit(actions, is_pin)
                out = dict(payload)
                out["actions"] = actions
                return out
            return handle

        if self.role is ModuleRole.SCREEN_SEEKER:
            # ScreenSeekeR — region-narrowing refinement (research: +254% on
            # small targets). Pure code: for each tap_at with a known
            # affordance bbox, attach a ``search_region`` hint the grounder
            # can use to localise within a sub-rectangle. Passes the batch
            # through; never drops actions.
            from explorer.affordances import AffordanceMap

            async def handle(payload: dict) -> dict | None:
                out = dict(payload)
                amap = AffordanceMap.from_dict(payload.get("affordance_map") or {})
                # index affordance bboxes by lowered label/value for a cheap
                # description→region hint
                regions: dict[str, list] = {}
                for a in amap.affordances:
                    if a.bbox:
                        key = (a.label or a.value or "").lower().strip()
                        if key:
                            regions[key] = a.bbox
                actions = []
                for act in payload.get("actions") or []:
                    aa = act.get("action_args") if isinstance(act.get("action_args"), dict) else {}
                    desc = (act.get("target_description") or aa.get("target_description") or "").lower()
                    hint = None
                    for key, bbox in regions.items():
                        if key and key in desc:
                            hint = bbox
                            break
                    if hint:
                        act = {**act, "search_region": hint}
                    actions.append(act)
                out["actions"] = actions
                return out
            return handle

        if self.role is ModuleRole.GROUNDING_VERIFIER:
            # Confidence gate on the grounder's coordinates. The grounder
            # already attaches a confidence (logprob-derived, PER-199); here
            # we flag low-confidence taps so the worker can choose to re-ground
            # or skip. Passes through (ground.produced → ground.verified).
            _MIN_CONF = 0.4

            async def handle(payload: dict) -> dict | None:
                out = dict(payload)
                ga = payload.get("grounded_actions") or payload.get("actions") or []
                verified = []
                for a in ga:
                    conf = a.get("ground_confidence")
                    if conf is not None and conf < _MIN_CONF:
                        a = {**a, "low_confidence": True}
                        logger.info(
                            "VERIFIER: low grounding confidence %.2f for %r",
                            conf, (a.get("target_description") or "")[:60],
                        )
                    verified.append(a)
                out["grounded_actions"] = verified
                return out
            return handle

        if self.role is ModuleRole.MEMORY:
            # Side-consumer: record the resolved+grounded step into episodic
            # memory so the next decision can recall it. Best-effort — a down
            # memory backend never wedges the chain (PER-206 already gates it).
            from explorer.agents import MemoryAgent
            agent = MemoryAgent(resolver)  # noqa: F841 (reserved for embedding recall)

            async def handle(payload: dict) -> dict | None:
                # Episodic write is owned by the worker (it holds the
                # EpisodicMemory singleton + run/goal scope). The bus stage
                # exists so MEMORY is a first-class pipeline citizen and can
                # grow into embedding-based recall; for now it's a no-op
                # side-consumer that keeps the chain shape complete.
                return None  # side-consumer
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
