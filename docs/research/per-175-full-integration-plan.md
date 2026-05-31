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

## Phases (Linear under PER-175; tasks #88–93 locally)

- **A** shared contracts: `affordances.py` ✓, bus chain ✓, add
  `PLATFORM_ADAPTER`+`SCREEN_SEEKER` roles, Platform Adapter + tests.
- **B** vision perception: SigLIP2 screen-type + OmniParser→affordances in
  the perception service; Context Identifier returns `AffordanceMap`;
  Screen Parser + Dynamic Perceiver become live stages.
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
