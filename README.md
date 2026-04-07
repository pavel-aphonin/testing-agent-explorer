# testing-agent-explorer

Core crawler microservice for Testing Agent. Automatically explores mobile and web apps and builds a state graph (screens + transitions).

Part of the Testing Agent MVP stack. See related repos at the bottom.

## What it does

Systematically walks an app's UI using **PUCT** (AlphaZero-style MCTS) with a **Go-Explore** frontier archive to maximize graph coverage. Three modes:

- **Monte Carlo** — uniform priors, fastest, no LLM
- **AI-only** — LLM on every step, highest fidelity, slowest
- **Hybrid** (default) — LLM priors cached per screen, PUCT selection, MC rollouts

Outputs per run: `graph.json`, `diagram.mmd` (Mermaid), `report.txt`, `checkpoint.json`.

## Architecture

```
explorer/
├── models.py          # ScreenNode, GraphEdge, AppGraph (Pydantic)
├── screen_id.py       # Screen fingerprint via accessibility tree hash
├── analyzer.py        # Element classification (button / text_field / switch)
├── form_filler.py     # Hypothesis PBT strategies for text inputs
├── navigator.py       # Backtracking, path replay, app restart
├── engine.py          # Main loop (PUCT + Go-Explore + MC rollouts)
├── visualizer.py      # JSON / Mermaid export
├── cli.py             # Typer CLI entrypoint
│
├── axe_client.py      # AXe CLI + Metro CDP client (React Native)
├── cdp_text_input.py  # React fiber onChangeText via Metro WebSocket
├── appium_client.py   # Appium client (native iOS apps)
└── vision_client.py   # LLM vision fallback (any app)
```

## Modes (`--mode`)

| Mode | Use case | Stack |
|------|---|---|
| `axe` (default) | React Native dev builds | AXe CLI + Metro CDP |
| `native` | Native iOS apps | Appium + XCUITest |
| `vision` | Any app, fallback | simctl screenshot + LLM vision |

## Dependencies

```bash
brew install cameroncooke/axe/axe
pip install -r requirements.txt
```

## Running standalone

### React Native app (TestApp)

```bash
xcrun simctl boot "iPhone 17 Pro Max"
cd ../TestApp && npx react-native run-ios --simulator="iPhone 17 Pro Max"

cd ../testing-agent-explorer
python -m explorer.cli explore \
  org.reactjs.native.example.TestApp \
  --device-id $(xcrun simctl list devices booted | grep -oE '[0-9A-F-]{36}') \
  --output ./explorer_output \
  --mode axe -v
```

### Native iOS app (via Appium)

```bash
appium --address 127.0.0.1 --port 4723 &
python -m explorer.cli explore \
  com.apple.Preferences \
  --device-id <UDID> \
  --output ./explorer_output_settings \
  --mode native -v
```

## Running as a microservice

Normally deployed via the [testing-agent-infra](https://github.com/pavel-aphonin/testing-agent-infra) docker-compose stack. The backend calls the CLI as a subprocess and streams live progress to Postgres + Redis pub/sub.

## Proven results (random-walk baseline, pre-PUCT)

- **TestApp** (React Native 0.76, iOS 26.2): 11 screens / 35 transitions in ~3 minutes
- **iOS Settings** (native): 18 screens / 27 transitions in ~3 minutes

Goal with PUCT + Go-Explore: **2-3× higher coverage** at the same time budget.

## Known limitations

- PBT variants in text fields can loop without an attempt counter (engine-level fix in progress)
- Similar screens (e.g. Profile view vs Profile edit) may share a navigation title and collide
- **iOS 26 + idb_companion 1.1.8** — accessibility API hangs on RN. We use AXe instead.
- **React Native + XCUITest** — `page_source` hangs. We use AXe `describe-ui`.
- **HID type for RN** — does not trigger `onChangeText`. We use Metro CDP.
- iOS Simulator requires macOS host — this container runs in CLI-only mode for headless/vision/web modes; iOS modes run on the host.

## Related repos

- `testing-agent-backend` — FastAPI orchestration + Postgres
- `testing-agent-frontend` — React + Antd dashboard
- `testing-agent-llm` — llama-swap + llama.cpp (Gemma 4, Qwen 3.5)
- `testing-agent-infra` — docker-compose + seed scripts
