# PER-175 — Full 12-module integration (+ ScreenSeekeR) — implementation plan

Goal (user, non-negotiable): **all 12 modules genuinely in the work loop,
no cuts, no "optimisations" that drop a module.** Models may change where
research shows a win. A future UI will pick + tune a model per module.

## Why we're here (root cause from run cccc3333)

Screen understanding was **text-only** (DeBERTa over the AX-tree). On a
canvas PIN keypad the AX-tree is empty → classifier guessed "money
transfer (0.36)" → `context_is_pin=False` → PIN logic never fired →
planner tried `enter_text "8520"` into a canvas → no-op → loop. Three of
the perception modules we "built" (Screen Parser, Dynamic Perceiver,
Reward Critic) were instantiated but **never called** (0 call sites).

## What the research said (2025–26, 98 facts)

1. **Vision-only won** for inaccessible UI (canvas/games/WebView). Only
   33% of macOS apps expose full accessibility — blindness is the norm.
2. **Decoupled Planner→(resolver)→Grounder** is the mainstream and beats
   monoliths (Agent S2 +18–32%; P-G dual +3.4%; CODA uses a frozen
   UI-TARS executor under a planner — literally our stack).
3. **Intent must name a visible affordance**, not an abstraction, or the
   grounder misses.
4. **Holo1.5-7B** beats our UI-TARS-1.5-7B (77.3 vs 70.5 avg; 57.9 vs
   39.0 ScreenSpot-Pro), has a UI-QA mode (screen type + state) AND
   llama.cpp quants, Apache-2.0. **GUI-Critic-R1-7B** = pre-operative
   critic (our Reward Critic, done right). **ScreenSeekeR** = +254% on
   small targets via region narrowing (our keypad case).

## Target pipeline — the 13-link chain (every module a stage)

```
screen.captured     worker: screenshot + AX snapshot + goal/test_data
→ screen.parsed     1. Screen Parser (OmniParser)   element boxes/types
→ screen.perceived  2. Dynamic Perceiver (SigLIP2)   novelty / Δ vs last screen
→ context.classified 3. Context Identifier (vision)   screen_type + AffordanceMap
→ plan.produced     4. Planner (GUI-Owl)             INTENTS (what), uses…
                       5. Ambiguity Resolver  (canonicalise goal)
                       6. Memory              (recall prior steps)
                       7. Reflection          (unstick on loops)
→ plan.critiqued    8. Reward Critic (GUI-Critic-R1)  pre-op verdict
                       9. Safety Guard        (side-consumer veto)
→ actions.resolved 12. Platform Adapter (pure code)  intent+affordance → concrete batch (how)
→ ground.refined   13. ScreenSeekeR                  region narrowing for small targets
→ ground.produced  10. Grounder (Holo1.5/UI-TARS)    coordinates
→ ground.verified  11. Grounding Verifier            confidence gate
→ action.dispatched     worker dispatches; Memory side-consumes to record
```

12 modules + ScreenSeekeR (13th stage, algorithmic — no GGUF, but a
first-class configurable runner). The legacy 3-link subset stays valid so
the system runs while stages fill in.

## The key architectural change: intent vs mechanism

- **Planner** outputs an *intent*: `provide_credential(pin_code)`,
  `tap(submit)`, `navigate(back)` — WHAT, referencing a visible affordance.
- **Platform Adapter** (12th, pure code) maps intent + `AffordanceMap` →
  a concrete atomic batch — HOW:
  - credential + on-screen keypad → tap digit keys in order **+ submit**
    (PER-204 becomes a *rule* here, not a prompt hint)
  - credential + editable field → `enter_text` (PER-205 guard: only if
    the target is genuinely editable; else re-route)
  - credential + neither → ask grounder to locate the field by description
- The boolean `context_is_pin` is **deleted**; `screen_type` +
  `AffordanceMap` replace it.

## STATUS (resume here)

- **A — DONE** (commit e3c4333): affordances.py, full bus MsgType chain,
  PLATFORM_ADAPTER+SCREEN_SEEKER roles, Platform Adapter + tests.
- **B — capability DONE, wiring is Phase C.** Done & pushed:
  - `/classify_vision` SigLIP2 zero-shot screen-type endpoint (infra
    commit 37c11c4) — the blindness fix, perception side.
  - `affordance_builder.py` (commit fed2298) — vision boxes→AffordanceMap,
    canvas-keypad + stray-digit rules unit-locked.
  - `ContextIdentifierAgent.classify_vision()` (fed2298).
  - Adapter fix (commit 4787a86): keypad path taps the SECRET's digits via
    a `resolve_value` callable, not 0-9. 158 tests green.
  - NOT yet wired into the decision loop — see Phase C.

## Phase C — IN PROGRESS

DONE & pushed (commit 3cc44e3):
- **Context Identifier (bus) produces an AffordanceMap** — vision
  screen_type (SigLIP2 /classify_vision) + Screen Parser /parse boxes
  (scaled via screen_w/h now on the bus) + AX editable rects (hybrid).
  Revives Screen Parser. Falls back to text classifier with no
  screenshot. Emits `affordance_map` on context.classified.
- **Platform Adapter (resolve_plan) runs in the bus PLANNER handler**
  before grounding (PER-205 routing + PER-204 submit). Legacy concrete
  actions pass through.
- **Secrets stay off the bus** (PER-208 filed): keypad digit-expansion is
  worker-side; bus path only appends submit for now.

REMAINING Phase C:
- **Planner emits intents** (`provide_credential`/`submit`/…) not
  mechanisms — goal_schema + planner prompt + intent vocabulary. (Until
  this lands the resolver mostly passes concrete actions through; the
  PER-205/204 rules still apply.)
- **Sync path** (`_goal_decide`): build AffordanceMap + run resolve_plan
  with a real `resolve_value` bound to test_data (full keypad expansion,
  secrets in-memory). Replace `context_is_pin` usage with AffordanceMap.
- **Grounder → Holo2-8B selectable** (UI-TARS default), ScreenSeekeR
  stage, Grounding Verifier gate.
- Screen Parser /parse returns boxes WITHOUT OCR text — fine for the
  keypad (adapter taps by description), add OCR (ocrmac/easyocr) for
  richer labels later.

Older notes:

1. **Context Identifier produces an AffordanceMap** and puts it on the bus
   (`context.classified` payload carries `affordance_map`): call
   `classify_vision(screenshot)` for screen_type + take Screen Parser
   `/parse` boxes, feed both to `build_affordance_map`. Replace the boolean
   `context_is_pin` everywhere with `AffordanceMap`.
2. **Planner emits intents** (`provide_credential`/`submit`/…) not
   mechanisms — update goal_schema + planner prompt + bus PLANNER handler.
3. **Resolver in the loop**: worker (sync `_goal_decide` and bus
   `_bus_goal_decide`) calls `platform_adapter.resolve_plan(intents,
   amap, test_data_keys, resolve_value=...)` BEFORE grounding. Pass a
   `resolve_value` bound to the run's test_data.
4. Grounder→Holo2-8B selectable (UI-TARS stays default), ScreenSeekeR
   stage, Grounding Verifier gate.
   Note: Screen Parser `/parse` returns boxes WITHOUT text/OCR — fine for
   the keypad (adapter taps by description, grounder localises), but add
   OCR (ocrmac/easyocr) when richer affordance labels are needed.

## Phases (Linear under PER-175; tasks #88–93 locally)

- **A** shared contracts — DONE.
- **B** vision perception capability — DONE (wiring → C).
- **C** planner intents + resolver live + Grounder→Holo1.5 (selectable) +
  ScreenSeekeR + Grounding Verifier gate.
- **D** Reward Critic (GUI-Critic-R1) + Safety + Memory + Reflection all on
  the bus, each a real consumer.
- **E** full ROLE_WIRING (a runner per module) + worker consumes full
  chain + PER-207 dead-end guard ported into goal-node + committed
  `start-module-servers.sh` + live «Для демо» smoke to completed login.
- **F** backend model passports (Holo1.5-7B, GUI-Critic-R1-7B) + role
  assignments; **future**: per-module model-select + tuning UI.

## Decisions locked (user, this session)

- **Holo2-8B → Context Identifier** (vision perception: screen_type +
  affordance map). NOT the grounder. Rationale: Holo2-8B is fine-tuned
  from Qwen3-VL-8B-**Thinking** — the reasoning channel is worth it for
  screen understanding (a rare per-screen call) but adds latency we don't
  want on every grounding call. Verified: `mradermacher/Holo2-8B-GGUF`
  has llama.cpp quants + a vision projector (`mmproj-f16.gguf` 1.3 GB),
  Apache-2.0. Beats Holo1.5-7B (ScreenSpot-Pro 58.9 vs 57.9,
  ScreenSpot-v2 93.2 vs ~92). Holo2 also ships 4B/8B/30B for the future
  per-module model picker.
- **Grounder stays UI-TARS-1.5-7B** (fast, no thinking; the frequent
  coordinate call). Holo2/Holo1.5 kept as selectable alternatives.
- **Specialised models per role** (NOT one shared VLM across 7 roles).

## Models to add (downloads)

- `Holo2-8B` GGUF + mmproj (`mradermacher/Holo2-8B-GGUF`, Q4_K_M ~5 GB +
  mmproj-f16 1.3 GB) — Context Identifier (vision screen-type + affordance
  reasoning via its UI-QA mode).
- `GUI-Critic-R1-7B` (Qwen2.5-VL-7B based) — Reward Critic.
- Keep: GUI-Owl-1.5-8B (Planner/Reflection), UI-TARS-1.5-7B (Grounder),
  OmniParser-v2 (Screen Parser), SigLIP2 (Dynamic Perceiver),
  Qwen3-4B (Ambiguity), Llama-Guard-3 (Safety), Qwen3-Embedding (Memory).
- Holo1.5-7B — optional selectable Grounder/Context alternative.

Sources (titles; URLs were in the 3 verifier agents that didn't finish):
UI-TARS-2, Holo1.5, Aria-UI, UI-Venus 1.5, Ferret-UI Lite, Agent S2,
CODA, OmniParser V2, Set-of-Mark, Screen2AX, ScreenSeekeR, GUI-Critic-R1.

## Phase F — operational steps remaining (code is DONE: A–F committed)

ALL CODE for the 13-module integration is built, tested (91 unit + 162
regression green), committed and pushed through commit 1b06be1. Holo2 is
downloaded: volumes/llm-models/Holo2-8B.Q4_K_M.gguf (4.68 GB) +
Holo2-8B.mmproj-f16.gguf (1.08 GB). What remains is OPERATIONAL — best done
with a working terminal (this session's tool output was broken):

1. Register Holo2 as a model passport (llm_models row): provider=llama_cpp,
   supports_vision=true, supports_json_schema=false (it's Thinking — parse
   tolerates), endpoint_url=http://localhost:8186, family=holo2.
2. Repoint the CONTEXT_IDENTIFIER ModuleAssignment from
   context-identifier-deberta → the new Holo2 passport. (The VisionContextAgent
   activates only when the role resolves to a vision chat model with
   provider != pytorch_microservice — so this swap is what turns it on.)
3. Start Holo2 on :8186 —
   llama-server -m volumes/llm-models/Holo2-8B.Q4_K_M.gguf \
     --mmproj volumes/llm-models/Holo2-8B.mmproj-f16.gguf \
     --host 127.0.0.1 --port 8186 --ctx-size 16384 --n-gpu-layers 99 \
     --alias context --jinja
4. Start the rest of the fleet: docker compose up -d; planner GUI-Owl :8187;
   grounder UI-TARS :8081; perception :8090 (still serves Dynamic Perceiver
   SigLIP); ambiguity Qwen3-4B :8183; safety Llama-Guard :8182; memory
   Qwen3-Embedding :8184. Memory budget ~21 GB of models — check `top`
   (36 GB free seen earlier; OK but tight).
5. ./scripts/start-bus-runners.sh   (all 13 runners)
6. Start worker: TA_BUS_MODE=1 .venv/bin/python -m explorer.worker ... -v
7. Insert a pending run for scenario «Для демо» (630eccd8-...), workspace
   aa274f1b-..., user 50ae7d34-..., bundle com.aweassist.app, app_file_path
   d39a5408-bfef-4589-9e7d-c7851ceee5f7/AWeassist.app, iPhone-17-Pro-Max.
8. Watch /tmp/ta-worker.log: expect classify_vision_vlm → screen_type pin_entry
   (confident, not 0.09) → keypad gate fires on the REAL PIN screen → digit
   taps 8-5-2-0 + submit → login completes.

Risk note: Reward Critic / Safety / Ambiguity side-stages add LLM calls per
step → each step slower than the 4-module chain. If too slow, those three
side-consumers can be left unassigned (they degrade to no-op) for a first
green run, then enabled.

## OOM ROOT CAUSE (smoke eeee5555) — measured, not guessed

The 13-module chain ran END-TO-END (every bus stream advanced; worker
reached ground.verified) — but the machine OOM-thrashed at 96 GB. The
memory math we did (~21 GB of weights) was CORRECT. The blow-up was
operational, three multipliers we missed:

- **All 6 llama-servers resident at once.** Measured Holo2 SOLO = 8.17 GB
  RSS / ~12 GB wired. ×6 models all pinned simultaneously.
- **Metal wired memory.** `--n-gpu-layers 99` on Apple Silicon makes the
  unified-RAM GPU buffers WIRED (non-evictable). Run showed 43 GB wired →
  kernel can't reclaim → thrash. Weights 21 GB became ~80+ GB resident.
- **KV cache bloat from my ctx-size.** I launched at --ctx-size 16384
  (vision) / 32768 (qwen, embed). For these tasks 4096 is plenty; 16-32k
  KV is GBs/model wasted.

### THE FIX (confirmed: system was DESIGNED for this)
- The backend already generates `volumes/llm-models/llama-swap.yaml` with
  `ttl: 300` — models load on demand and auto-unload after 5 min idle, so
  only 1-2 are resident at a time, never 6. We DROPPED this when I hand-
  launched 6 servers. **`llama-swap` binary is NOT installed** (`which
  llama-swap` → not found) — that's why we went manual. Install it
  (build from ggml-org/llama-swap or brew) and route all roles through the
  single swap port; the worker/agents already resolve per-role endpoints.
- The generated yaml is STALE (lists gemma-4-e4b, qwen3-embedding-8b; no
  Holo2/GUI-Owl). Trigger a regen after model-passport changes, and fix the
  generator's baked ctx-size 16384/32768 → 4096.

### Multiplex reminder (answers "11 or 6 models?")
6 unique llama models cover 9 roles via multiplex: GUI-Owl-8B = Planner +
Reflection + Reward Critic; perception(:8090) = Screen Parser + Dynamic
Perceiver. + Holo2 (Context), UI-TARS (Grounder), Qwen3-4B (Ambiguity),
Llama-Guard (Safety), Qwen3-Embedding (Memory). Grounding Verifier =
logprobs (no model); Platform Adapter + ScreenSeeker = pure code.

## NEXT-SESSION ENTRY POINT
1. Install llama-swap binary; point start-host-services at the swap port.
2. Fix llm_swap.py generator: ctx-size 4096 default; ensure it emits ALL
   active passports incl. holo2-8b (passport id 55555555-... already in DB,
   CONTEXT_IDENTIFIER assignment already repointed to it).
3. Diagnose Holo2 "screen_type=unknown conf=0.0": with ONLY Holo2 up (fits
   easily), POST a real PIN screenshot to :8186/v1/chat/completions and read
   the raw reply — is it returning prose, empty, or a parse-miss? (parser
   unit tests pass, so it's the live reply.)
4. Backend ModuleRole enum: add PLATFORM_ADAPTER + SCREEN_SEEKER (worker has
   them; backend 422s on GET /module-assignments/{those} — harmless now,
   but needed for a clean inventory).
5. Then full run via llama-swap (no OOM) → expect Holo2 confident pin_entry
   → keypad gate fires → login completes.
