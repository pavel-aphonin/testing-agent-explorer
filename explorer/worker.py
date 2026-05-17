"""Explorer worker daemon — runs on the host, executes pending runs.

This worker connects an OUT-OF-PROCESS backend (running in Docker) to
host-side simulator tools (xcrun, AXe, Appium, Metro CDP). It cannot
live inside the backend container because the backend container has
no access to /Applications/Xcode or to a running iOS Simulator.

The worker loop:
    1. Poll POST /api/internal/runs/claim every POLL_INTERVAL seconds.
       The endpoint atomically transitions the oldest pending run to
       running and returns its config.
    2. If a run was claimed, execute it. The worker streams events
       (screen_discovered, edge_discovered, log, error, stats_update,
       status_change) back to POST /api/internal/runs/{id}/event.
    3. On success, post a final status_change to "completed".
       On exception, post status_change to "failed" with error message.

Worker uses a single execution backend — RealExecutor — that drives the
PUCT + Go-Explore engine against a real iOS simulator / Android emulator
through xcrun / Appium / AXe. The synthetic executor that previously
generated fake screens for pipe-validation was removed in PER-48: it
silently swallowed real-app uploads when accidentally left enabled and
confused users about whether their build had actually been explored.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_POLL_INTERVAL = 3.0  # seconds between claim attempts when idle

EventSink = Callable[[dict[str, Any]], Awaitable[None]]


# ----------------------------------------------------------------------------
# Backend client
# ----------------------------------------------------------------------------


class BackendClient:
    """Tiny async HTTP client wrapping the worker's view of the backend.

    PER-51: ``trust_env=False`` is critical on corporate macs with security
    suites (Norton 360, McAfee, Symantec, etc.) that set up a system-wide
    HTTP proxy via ``scutil --proxy`` — even when the proxy env vars are
    cleared, httpx with default ``trust_env=True`` reads the macOS system
    proxy config and routes localhost requests through 127.0.0.1:1082
    (Norton's inline inspector), which then 503s our internal endpoints.

    Setting ``trust_env=False`` makes httpx ignore both env vars AND the
    system proxy config, talking directly to the backend over loopback.
    Confirmed via diagnostic test: with trust_env=True we get 403/503
    from Norton; with trust_env=False we get 200 from FastAPI.

    Optional UDS fallback: if ``TA_BACKEND_UDS`` is set and points at a
    Unix socket on the host (e.g. via socat bridge), the client uses
    that. Useful as defense-in-depth in case some future security suite
    intercepts even direct localhost TCP — UDS is filesystem IPC and
    invisible to network filters.
    """

    def __init__(self, base_url: str, worker_token: str):
        self._base_url = base_url.rstrip("/")
        # ``worker_token`` is exposed on the instance so downstream
        # components (e.g. ScenarioRunner for RAG verification) can
        # reuse the same auth header without re-fetching from CLI/env.
        self.worker_token = worker_token
        self._headers = {"Authorization": f"Bearer {worker_token}"}

        backend_uds = os.environ.get("TA_BACKEND_UDS")
        if backend_uds and Path(backend_uds).exists():
            logger.info("Using UDS transport for backend: %s", backend_uds)
            transport = httpx.AsyncHTTPTransport(uds=backend_uds)
            self._client = httpx.AsyncClient(
                transport=transport,
                timeout=15.0,
                trust_env=False,  # ← bypass macOS system proxy too
            )
            # Override base_url for actual requests — httpx ignores host
            # when transport is UDS, but the URL must still be well-formed.
            self._base_url = "http://backend"
        else:
            # PER-51: trust_env=False is the actual fix for Norton on dev macs.
            self._client = httpx.AsyncClient(timeout=15.0, trust_env=False)

    async def claim_next(self) -> dict[str, Any] | None:
        """Try to claim the oldest pending run. Returns None if nothing available."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/internal/runs/claim",
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            logger.warning("claim request failed: %s", exc)
            return None
        if resp.status_code == 401:
            raise RuntimeError("Worker token rejected by backend")
        if resp.status_code in (200, 204) and not resp.content:
            return None
        if resp.status_code == 200:
            data = resp.json()
            if data is None:
                return None
            return data
        logger.warning("unexpected claim status: %s %s", resp.status_code, resp.text)
        return None

    async def post_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Send an event to the backend.

        Raises RunCancelled when the backend reports 409 with a terminal
        status — the run was deleted / cancelled while we were running.
        The outer loop catches this and stops the executor cleanly.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/internal/runs/{run_id}/event",
                headers=self._headers,
                json=event,
            )
        except httpx.HTTPError as exc:
            logger.warning("post_event failed: %s", exc)
            return
        if resp.status_code == 409:
            # Run is in a terminal state (CANCELLED / DELETED). Stop cleanly.
            raise RunCancelled(run_id, resp.text)
        if resp.status_code >= 300:
            logger.warning(
                "post_event for run %s rejected: %s %s",
                run_id,
                resp.status_code,
                resp.text,
            )

    async def post_heartbeat(self) -> None:
        """Best-effort liveness ping. Backend uses it for the UI's
        "Connected" indicator. Failure is silent — never blocks the loop.
        """
        try:
            await self._client.post(
                f"{self._base_url}/api/internal/worker/heartbeat",
                headers=self._headers,
            )
        except httpx.HTTPError:
            pass

    async def post_defect(self, payload: dict[str, Any]) -> None:
        """Send a detected defect. Best-effort — a failed post shouldn't kill the run."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/internal/defects",
                headers=self._headers,
                json=payload,
            )
        except httpx.HTTPError as exc:
            logger.warning("post_defect failed: %s", exc)
            return
        if resp.status_code >= 300:
            logger.warning(
                "post_defect rejected: %s %s", resp.status_code, resp.text,
            )

    async def aclose(self) -> None:
        await self._client.aclose()


class RunCancelled(Exception):
    """Raised when the backend reports the run was cancelled mid-flight."""

    def __init__(self, run_id: str, detail: str = "") -> None:
        super().__init__(f"Run {run_id} was cancelled: {detail}")
        self.run_id = run_id


# ----------------------------------------------------------------------------
# Real executor — drives the PUCT + Go-Explore engine against iOS/Android.
# Synthetic executor was removed in PER-48: it generated fake trajectories
# that silently swallowed real-app uploads and confused users.
# ----------------------------------------------------------------------------


class RealExecutor:
    """Drives the real PUCT + Go-Explore engine against an iOS simulator.

    The engine emits live progress events (screen_discovered, edge_discovered,
    stats_update, etc.) through its event_callback; we wire that callback
    straight to the worker's event_sink so each event is POSTed to the
    backend's internal API and reaches the browser via Redis + WebSocket.

    Output (graph.json, mermaid diagram, screenshots) is written to
    ./worker_runs/<run_id>/ on the host where the worker is running.
    """

    DEFAULT_OUTPUT_ROOT = Path("./worker_runs")

    def __init__(
        self,
        output_root: str | Path | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        # output_root is overridable for tests; default lives next to the worker.
        self._output_root = Path(output_root) if output_root else self.DEFAULT_OUTPUT_ROOT
        # client_factory is overridable so a test can inject a fake controller
        # without spinning up AXe.
        self._client_factory = client_factory

    async def run(self, config: dict[str, Any], event_sink: EventSink) -> None:
        from explorer.engine import ExplorationEngine
        from explorer.modes import MODE_CONFIGS, ExplorationMode

        run_id = str(config["run_id"])
        bundle_id = config["bundle_id"]
        device_id = config.get("device_id") or "BOOTED"
        # Default mode is AI — most capable for demo / first-run scenarios.
        # Each mode routes to a distinct runtime; see the branch below
        # and explorer/modes.py for the full mode table.
        mode_name = (config.get("mode") or "ai").lower()
        max_steps = int(config.get("max_steps") or 200)
        platform = config.get("platform", "ios")

        try:
            mode = ExplorationMode(mode_name)
        except ValueError:
            logger.warning("Unknown mode %r — falling back to AI", mode_name)
            mode = ExplorationMode.AI

        # ── LLM prior provider ──
        prior_provider = None
        mode_config = MODE_CONFIGS[mode]
        if mode_config.use_llm_priors:
            llm_url = os.environ.get("TA_LLM_BASE_URL", "http://localhost:8080")
            # PER-106 #5: prefer the per-run model name from the claim
            # response. The backend resolves it from the user's pick or
            # AgentSettings default; ``None`` here means "no override",
            # and we fall back to the worker's env var as before.
            llm_model = (
                config.get("llm_model_name")
                or os.environ.get("TA_LLM_MODEL_NAME", "embeddings")
            )
            try:
                from explorer.llm_client import LLMClient, LLMPriorProvider

                llm = LLMClient(base_url=llm_url, model_name=llm_model)
                prior_provider = LLMPriorProvider(
                    llm, vision_enabled=mode_config.vision_enabled
                )
                logger.info(
                    "LLM prior provider enabled (mode=%s, vision=%s)",
                    mode.value, mode_config.vision_enabled,
                )
            except Exception:
                logger.exception("LLM prior provider failed — uniform priors")

        # ── V2: auto-provision simulator/emulator ──
        sim_manager = None
        device_type = config.get("device_type")
        os_version = config.get("os_version")
        app_file_path = config.get("app_file_path")

        uploads_base = os.environ.get(
            "TA_APP_UPLOADS_DIR",
            str(
                Path(__file__).resolve().parent.parent.parent
                / "testing-agent-infra" / "volumes" / "app-uploads"
            ),
        )

        if device_type and os_version:
            from explorer.simulator import (
                AndroidEmulatorManager,
                IOSSimulatorManager,
            )

            try:
                await event_sink({"type": "log", "step_idx": 0, "message": "Creating simulator…"})

                if platform == "android":
                    sim_manager = AndroidEmulatorManager(device_type, os_version)
                else:
                    sim_manager = IOSSimulatorManager(device_type, os_version)

                device_id = await sim_manager.create(run_id)
                await event_sink({"type": "log", "step_idx": 0, "message": f"Booting {device_id}…"})
                await sim_manager.boot()

                if app_file_path:
                    full_app_path = str(Path(uploads_base) / app_file_path)
                    await event_sink({"type": "log", "step_idx": 0, "message": "Installing app…"})
                    await sim_manager.install(full_app_path)

                await event_sink({"type": "log", "step_idx": 0, "message": "Launching app…"})
                await sim_manager.launch(bundle_id)
                await event_sink({"type": "log", "step_idx": 0, "message": "App launched, starting exploration…"})
            except Exception as exc:
                # Setup failed. Clean up the simulator (best effort)
                # and propagate so execute_one_run's exception handler
                # emits the single error event + skips the misleading
                # "completed" status_change. Before this fix the local
                # error event was sent and then execute_one_run posted
                # status_change=completed because run() returned cleanly
                # — backend's runs state-machine then saw an error-then-
                # completed sequence and either rejected the second
                # event with 409 or, worse, marked the run completed
                # despite the upstream error.
                logger.exception("Simulator setup failed for run %s", run_id)
                if sim_manager:
                    try:
                        await sim_manager.cleanup()
                    except Exception:
                        logger.exception(
                            "Simulator cleanup also failed for run %s",
                            run_id,
                        )
                raise RuntimeError(f"Simulator setup failed: {exc}") from exc

        # ── AXe client ──
        if self._client_factory is not None:
            client = self._client_factory()
            connect_fn = getattr(client, "connect", None)
            disconnect_fn = getattr(client, "disconnect", None)
        else:
            from explorer.axe_client import AXeExplorerClient
            client = AXeExplorerClient()
            connect_fn = client.connect
            disconnect_fn = client.disconnect

        output_dir = self._output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        mirror_proc = None
        try:
            if connect_fn is not None:
                await connect_fn(udid=device_id)

            # V1 legacy: launch app via simctl if not using V2 auto-provisioning
            if sim_manager is None and self._client_factory is None and device_id and bundle_id:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "xcrun", "simctl", "launch", device_id, bundle_id,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                    if proc.returncode == 0:
                        logger.info("Launched %s on %s: %s", bundle_id, device_id, stdout.decode().strip())
                except (asyncio.TimeoutError, FileNotFoundError) as exc:
                    logger.warning("simctl launch failed: %s", exc)

            # ── SimMirror sidecar ──
            if self._client_factory is None and platform == "ios":
                mirror_bin = os.environ.get(
                    "TA_SIM_MIRROR_BIN",
                    str(Path(__file__).resolve().parent.parent.parent
                        / "testing-agent-sim-mirror" / ".build" / "release" / "SimMirror"),
                )
                mirror_port = os.environ.get("TA_SIM_MIRROR_PORT", "9999")
                # Capture SimMirror's output to a log so we can debug death.
                # Previously stdout/stderr went to DEVNULL and the process
                # would silently die without telling anyone what window it
                # tried to bind to.
                mirror_log_path = Path("/tmp") / "ta-sim-mirror.log"
                if Path(mirror_bin).exists():
                    # Wait for the simulator window to appear before binding.
                    # ScreenCaptureKit needs an existing on-screen window;
                    # SimMirror exits immediately if it can't find one. Two
                    # seconds is enough on M2 — adjust if you see it racing.
                    await asyncio.sleep(2.0)
                    try:
                        mirror_log = mirror_log_path.open("w")
                        # SimMirror defaults to binding 127.0.0.1
                        # (PER-106 #7). The backend runs in Docker
                        # and reaches the host's SimMirror through
                        # ``host.docker.internal`` — that traffic
                        # arrives on the host's external interface,
                        # NOT loopback, so a loopback-bound mirror
                        # would be unreachable. We override via the
                        # ``TA_SIM_MIRROR_BIND`` env var (default
                        # ``0.0.0.0`` for the docker-backend case);
                        # operators running the backend natively on
                        # the same host can set it to "127.0.0.1"
                        # and stay loopback-only.
                        mirror_bind = os.environ.get(
                            "TA_SIM_MIRROR_BIND", "0.0.0.0"
                        )
                        # PER-111 followup: pin SimMirror to THIS run's
                        # simulator window. Without a selector SimMirror
                        # picked "the first Simulator window", which on
                        # the demo machine ended up being a stray
                        # "Apple TV Remote" 155×515 surface — UI showed
                        # "ждём захват симулятора" forever.
                        # Simulator window titles begin with the device
                        # name (``TA-<run-id-short>``) which we set in
                        # IOSSimulatorManager.create(), so matching by
                        # that substring unambiguously hits the right
                        # window per run.
                        mirror_window_title = f"TA-{run_id[:8]}"
                        mirror_proc = await asyncio.create_subprocess_exec(
                            mirror_bin,
                            "--port", mirror_port,
                            "--bind", mirror_bind,
                            "--max-width", "480",
                            "--fps", "15",
                            "--window-title-substring", mirror_window_title,
                            stdout=mirror_log,
                            stderr=mirror_log,
                        )
                        logger.info(
                            "SimMirror started (pid=%s, port=%s, log=%s)",
                            mirror_proc.pid, mirror_port, mirror_log_path,
                        )
                        # Sanity check: did it actually stay alive?
                        await asyncio.sleep(1.0)
                        if mirror_proc.returncode is not None:
                            logger.warning(
                                "SimMirror died immediately (exit=%s). See %s",
                                mirror_proc.returncode, mirror_log_path,
                            )
                            mirror_proc = None
                    except Exception:
                        logger.exception("SimMirror failed to start")

            # ── Exploration runtime ──
            # Three modes, three execution paths. Each one is the
            # documented runtime for that mode (see modes.py). No
            # silent aliasing.
            llm_url = os.environ.get("TA_LLM_BASE_URL", "http://localhost:8080")
            # PER-106 #5: same per-run override as the prior_provider
            # branch above; keep the two in sync so AI mode actually
            # honours the user's selection.
            llm_model_name = (
                config.get("llm_model_name")
                or os.environ.get("TA_LLM_MODEL_NAME", "embeddings")
            )

            if mode is ExplorationMode.AI:
                # AI: LLM picks every action. Routes through
                # LLMExplorationLoop, which sees the full element list
                # + screenshot + history on every step.
                from explorer.llm_loop import LLMExplorationLoop

                # URL of the RAG LLM — used as the "smart" classifier
                # for defect detection. Falls back to the agent LLM
                # if not set.
                rag_llm_url = os.environ.get(
                    "TA_RAG_LLM_BASE_URL",
                    os.environ.get("TA_LLM_BASE_URL", "http://localhost:8083"),
                )

                # Defect poster injected by execute_one_run from the
                # BackendClient. Falls back to a no-op if missing —
                # keeps tests green when the executor is invoked
                # standalone.
                _post_defect = (
                    config.get("_post_defect")
                    or (lambda _payload: asyncio.sleep(0))
                )

                loop = LLMExplorationLoop(
                    controller=client,
                    app_bundle_id=bundle_id,
                    llm_base_url=llm_url,
                    llm_model=llm_model_name,
                    max_steps=max_steps,
                    event_callback=event_sink,
                    test_data=config.get("test_data") or {},
                    scenarios=config.get("scenarios") or [],
                    defect_detection_enabled=True,
                    defect_llm_base_url=rag_llm_url,
                    defect_callback=_post_defect,
                    run_id=run_id,
                    pbt_enabled=bool(config.get("pbt_enabled", False)),
                    vision_enabled=mode_config.vision_enabled,
                )

            elif mode is ExplorationMode.HYBRID:
                # HYBRID: LLM evaluates each NEW screen once to assign
                # priors over its elements; PUCT picks actions from
                # those priors on subsequent visits. The prior_provider
                # is built above (see the `if mode_config.use_llm_priors`
                # block). ExplorationEngine consumes it via the
                # ``prior_provider=`` constructor parameter and only
                # invokes it when entering an unseen screen.
                from explorer.engine import ExplorationEngine

                loop = ExplorationEngine(
                    controller=client,
                    app_bundle_id=bundle_id,
                    output_dir=str(output_dir),
                    mode=mode,
                    max_steps=max_steps,
                    prior_provider=prior_provider,
                    event_callback=event_sink,
                    test_data=config.get("test_data") or {},
                )

            else:
                # MC: no LLM. PUCT with uniform priors + Monte-Carlo
                # rollouts. Same ExplorationEngine as HYBRID — the
                # mode_config controls c_puct, rollout depth, and the
                # "skip LLM priors" branch. Keeping MC on the engine
                # path (instead of the legacy explorer/loop.py) means
                # form fill, screen identification, event emission,
                # and scenario runner behave identically across modes.
                from explorer.engine import ExplorationEngine

                loop = ExplorationEngine(
                    controller=client,
                    app_bundle_id=bundle_id,
                    output_dir=str(output_dir),
                    mode=mode,
                    max_steps=max_steps,
                    prior_provider=None,
                    event_callback=event_sink,
                    test_data=config.get("test_data") or {},
                )

            # PER-18: explicit scenario runner walks the configured
            # scenario steps deterministically before handing off to
            # free exploration. Runs in HYBRID/AI/MC alike — works
            # purely off the controller and emits scenario.* events.
            scenarios_cfg = config.get("scenarios") or []
            if scenarios_cfg:
                try:
                    from explorer.scenario_runner import ScenarioRunner
                    # PER-37: pass RAG endpoint + token + defect callback
                    # so the runner can auto-verify expected_result against
                    # scenario.rag_document_ids and emit spec_mismatch
                    # defects without waiting for the LLM detector.
                    # Backend URL + worker token are injected into config
                    # by execute_one_run from the BackendClient before the
                    # executor runs. We don't have direct access to
                    # BackendClient in this scope — `client` here is the
                    # AXeExplorerClient that drives the simulator.
                    sr = ScenarioRunner(
                        controller=client,
                        scenarios=scenarios_cfg,
                        # PER-86: linked-only library for sub_scenario
                        # resolution; worker doesn't run these on its own.
                        linked_scenarios=config.get("linked_scenarios") or [],
                        test_data=config.get("test_data") or {},
                        event_callback=event_sink,
                        rag_base_url=config.get("_backend_url") or os.environ.get(
                            "TA_BACKEND_URL", "http://localhost:8000"
                        ),
                        rag_token=config.get("_worker_token") or os.environ.get(
                            "TA_WORKER_TOKEN", ""
                        ),
                        defect_callback=config.get("_post_defect"),
                        run_id=run_id,
                        # PER-85: pass the same LLM client we already
                        # wired up for free-exploration priors. None
                        # is fine — semantic checks just no-op when
                        # the LLM isn't available.
                        llm_client=llm if "llm" in locals() else None,
                        # PER-111 v2: workspace-enabled action dictionary
                        # shipped by /api/internal/runs/claim. The runner
                        # uses it inside goal nodes to build the LLM's
                        # constrained-decode schema and the actions_block
                        # in the user prompt. Empty list = degraded mode
                        # (operator should seed RefActionType + enable
                        # them per workspace).
                        actions=config.get("actions") or [],
                    )
                    sr_summary = await sr.run_all()
                    logger.info("[scenario] summary: %s", sr_summary)
                except RunCancelled:
                    # PER-110 follow-up: when the run is cancelled
                    # while scenarios are still running, propagate
                    # straight to the outer handler (around loop.run()
                    # below) so we don't fall through into free
                    # exploration on a cancelled run.
                    raise
                except Exception:
                    logger.exception("Scenario runner crashed — continuing to free exploration")

            # PER-40 / PER-41: replay a recorded action chain BEFORE
            # free exploration. ``replay_actions`` arrives serialised
            # by app/services/path_finder.py::serialize_action; we
            # rebuild ActionDetail and call navigator.replay_action()
            # one step at a time so each can emit its own log event.
            replay_actions = config.get("replay_actions") or []
            if replay_actions:
                from explorer.models import ActionDetail, ActionType
                from explorer.navigator import replay_action
                logger.info(
                    "[replay] playing %d recorded actions before exploration",
                    len(replay_actions),
                )
                _ALLOWED_ATYPES = {at.value for at in ActionType}
                for i, ra in enumerate(replay_actions):
                    raw_type = (ra.get("action_type") or "tap").lower()
                    atype = ActionType(raw_type) if raw_type in _ALLOWED_ATYPES else ActionType.TAP
                    # Propagate every field path_finder.serialize_action
                    # may have emitted. target_test_id is the most
                    # stable handle (test_id rarely changes between
                    # builds); target_frame lets the controller tap by
                    # coordinate when neither id nor label resolve.
                    # Previous code only carried target_label → the
                    # AXe replay path raised AttributeError (no
                    # tap_element method) every time a frame was
                    # missing.
                    action = ActionDetail(
                        action_type=atype,
                        target_label=ra.get("target_label"),
                        target_test_id=ra.get("target_test_id"),
                        target_frame=ra.get("target_frame"),
                        input_text=ra.get("input_text"),
                    )
                    ok = False
                    try:
                        ok = await asyncio.wait_for(
                            replay_action(client, action), timeout=30
                        )
                    except Exception:
                        logger.exception("[replay] action %d crashed", i)
                    await event_sink({
                        "type": "log",
                        "step_idx": i,
                        "message": (
                            f"replay {i + 1}/{len(replay_actions)}: "
                            f"{atype.value if hasattr(atype, 'value') else atype} "
                            f"'{action.target_label or '—'}' "
                            f"→ {'ok' if ok else 'FAIL'}"
                        ),
                    })
                    # Settle between actions so each gets its own
                    # screen capture later in the exploration loop.
                    await asyncio.sleep(1.2)
                logger.info("[replay] done — switching to free exploration")
                # If max_steps == 0 the caller asked for replay-only
                # mode (PER-40 with continue_after_replay=False).
                # Skip the main loop in that case.
                if max_steps <= 0:
                    logger.info("[replay] max_steps=0 → stopping after replay")
                    return

            try:
                result = await loop.run()
                logger.info("Loop result: %s", result)
            except RunCancelled as exc:
                # User deleted or cancelled the run — stop cleanly, don't mark failed.
                logger.info("Run %s cancelled by user, stopping loop", exc.run_id)
        finally:
            # AXe disconnect
            if disconnect_fn is not None:
                try:
                    await disconnect_fn()
                except Exception:
                    logger.exception("controller disconnect failed")
            # SimMirror teardown
            if mirror_proc is not None and mirror_proc.returncode is None:
                try:
                    mirror_proc.send_signal(signal.SIGINT)
                    await asyncio.wait_for(mirror_proc.wait(), timeout=3.0)
                    logger.info("SimMirror stopped")
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        mirror_proc.kill()
                    except Exception:
                        pass
            # V2: tear down the auto-provisioned simulator/emulator
            if sim_manager is not None:
                await event_sink({"type": "log", "step_idx": 0, "message": "Cleaning up simulator…"})
                await sim_manager.cleanup()


# ----------------------------------------------------------------------------
# Worker loop
# ----------------------------------------------------------------------------


def _make_event_sink(client: BackendClient, run_id: str) -> EventSink:
    """Bind a run_id into a post_event callback for an executor to use."""

    async def sink(event: dict[str, Any]) -> None:
        await client.post_event(run_id, event)

    return sink


async def execute_one_run(
    client: BackendClient,
    config: dict[str, Any],
    executor: RealExecutor,
) -> None:
    run_id = str(config["run_id"])
    logger.info("Executing run %s (%s)", run_id, config.get("bundle_id"))

    sink = _make_event_sink(client, run_id)

    # Inject the BackendClient's defect poster into the config so the
    # executor can forward it to LLMExplorationLoop. Done at this layer
    # because RealExecutor.run() already creates its own AXe client and
    # we don't want to confuse the two — the AXe client talks to AXe,
    # the BackendClient talks to our backend.
    config["_post_defect"] = client.post_defect
    # Same idea for the worker token — ScenarioRunner uses it to call
    # the backend's RAG endpoint to verify expected_result against
    # linked spec documents. Inject so the executor doesn't have to
    # rediscover it from CLI/env (which fails when neither is set,
    # because the Authorization header becomes `Bearer ` and the
    # backend rejects with 401 - silently dropping all RAG checks).
    config["_backend_url"] = client._base_url
    config["_worker_token"] = client.worker_token

    try:
        await executor.run(config, sink)
        await sink(
            {
                "type": "status_change",
                "step_idx": int(config.get("max_steps", 0)),
                "new_status": "completed",
            }
        )
        logger.info("Run %s completed", run_id)
    except NotImplementedError as exc:
        logger.error("Executor not implemented: %s", exc)
        await sink(
            {"type": "error", "step_idx": 0, "message": str(exc)},
        )
    except RunCancelled as exc:
        # PER-110 follow-up: cancellation is not a failure. The backend
        # already moved the run into the terminal state — sending an
        # `error` event back would 409 anyway, and worse, it would
        # show up in the UI as "Run failed" which is misleading. Just
        # log and exit cleanly so the claim loop picks up the next run.
        logger.info("Run %s cancelled mid-execution: %s", run_id, exc)
    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        await sink(
            {"type": "error", "step_idx": 0, "message": str(exc)},
        )


async def worker_loop(
    backend_url: str,
    worker_token: str,
    poll_interval: float,
    stop_event: asyncio.Event,
    executor: RealExecutor | None = None,
) -> None:
    # `executor` is an injection hook for integration tests: pass a
    # pre-built RealExecutor(client_factory=<fake controller>) and the
    # loop will use it instead of constructing a fresh one. Production
    # (main()) never passes this.
    client = BackendClient(backend_url, worker_token)
    if executor is None:
        executor = RealExecutor()

    logger.info(
        "Worker started (backend=%s, poll=%ss)",
        backend_url,
        poll_interval,
    )

    # Report available simulator/emulator configs to the backend so the
    # admin UI can show them in the device management page.
    async def _report_simulator_config() -> bool:
        """Report once. Returns True on success.

        Cached in Redis with 300s TTL on the backend side, so this needs
        to be re-called periodically — see refresh in _heartbeat_loop
        below. Otherwise the admin "Add device" page 503s a few minutes
        after worker startup.
        """
        try:
            from explorer.simulator import (
                AndroidEmulatorManager,
                IOSSimulatorManager,
            )

            ios_runtimes = await IOSSimulatorManager.list_runtimes()
            ios_devices = await IOSSimulatorManager.list_device_types()
            android_images = await AndroidEmulatorManager.list_system_images()
            android_devices = await AndroidEmulatorManager.list_device_types()

            config_payload = {
                "runtimes": ios_runtimes + android_images,
                "device_types": ios_devices + android_devices,
            }
            await client._client.post(
                f"{backend_url}/api/internal/runs/config",
                headers={"Authorization": f"Bearer {worker_token}"},
                json=config_payload,
            )
            return True
        except Exception:
            logger.exception("Failed to report simulator config")
            return False

    if await _report_simulator_config():
        logger.info("Reported simulator config to backend (initial)")

    # Clean up orphan TA-* simulators/AVDs from crashed runs (one-shot
    # at startup; not part of the periodic loop).
    try:
        from explorer.simulator import (
            AndroidEmulatorManager,
            IOSSimulatorManager,
        )
        ios_cleaned = await IOSSimulatorManager.cleanup_orphans()
        android_cleaned = await AndroidEmulatorManager.cleanup_orphans()
        if ios_cleaned or android_cleaned:
            logger.info(
                "Cleaned up %d iOS + %d Android orphan simulators",
                ios_cleaned, android_cleaned,
            )
    except Exception:
        logger.exception("Failed to clean up orphan simulators")

    # Heartbeat runs in its own task so the "Connected" indicator stays
    # green even while a run is executing. Previously it was inline in the
    # claim loop, which meant heartbeats stopped flowing for the entire
    # duration of execute_one_run() — runs take minutes to hours, so the
    # 60s Redis TTL would expire and the UI would show "Не подключено"
    # despite the worker doing real work right at that moment.
    #
    # Also refreshes the simulator-config Redis cache every ~3 minutes
    # (well under its 300s TTL) so the admin UI's "Add device" page
    # keeps working long after worker startup.
    SIMCONFIG_REFRESH_EVERY = 36  # 36 × 5s heartbeat = 3 min

    async def _heartbeat_loop() -> None:
        tick = 0
        while not stop_event.is_set():
            await client.post_heartbeat()
            tick += 1
            if tick % SIMCONFIG_REFRESH_EVERY == 0:
                # Don't await this — failures are best-effort, and the
                # heartbeat shouldn't block on simctl. But we DO want to
                # see exceptions in the log.
                asyncio.create_task(_report_simulator_config())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    try:
        while not stop_event.is_set():
            try:
                config = await client.claim_next()
            except RuntimeError as exc:
                logger.error("Fatal: %s", exc)
                return

            if config is None:
                # Idle — wait POLL_INTERVAL or until told to stop
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue

            await execute_one_run(client, config, executor)
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        await client.aclose()
        logger.info("Worker stopped.")


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Testing Agent explorer worker")
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("TA_BACKEND_URL", DEFAULT_BACKEND_URL),
        help="Backend base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--worker-token",
        default=os.environ.get("TA_WORKER_TOKEN"),
        help=(
            "Worker token for /api/internal/* (required). "
            "Set via env TA_WORKER_TOKEN or pass --worker-token. "
            "There is no default — a public placeholder token was a "
            "P1 security finding (PER-104) because internal endpoints "
            "would have been protected by a known string in env-less runs."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between claim attempts when idle (default: %(default)s)",
    )
    # PER-48: --executor flag removed. Worker has only one mode now (real).
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # PER-118: switch to JSON output when the operator opted in to the
    # centralised log stack via LOGGING_BACKEND. Plain text otherwise
    # so `tail -f /tmp/ta-worker.log` stays human-readable on dev.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging_backend = (os.environ.get("LOGGING_BACKEND") or "none").strip().lower()

    # Quiet the noisiest 3rd-party loggers regardless of mode. Without
    # this, ``-v`` blows the Elasticsearch index up with hundreds of
    # httpcore handshake DEBUG lines per second — useless for
    # diagnosing the agent, ruinous for disk usage. WARNING still
    # surfaces real transport problems (timeouts, refused connections).
    for noisy in ("httpcore", "httpx", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if logging_backend not in ("", "none"):
        try:
            from pythonjsonlogger import jsonlogger
            handler = logging.StreamHandler()
            handler.setFormatter(
                jsonlogger.JsonFormatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s",
                    rename_fields={
                        "asctime": "@timestamp",
                        "levelname": "log.level",
                        "name": "logger",
                    },
                    json_ensure_ascii=False,
                )
            )
            # Tag every record with service=worker so Filebeat's index
            # routing (markov-<service>-<date>) lands them with the
            # backend rows correctly.
            class _ServiceFilter(logging.Filter):
                def filter(self, record):  # noqa: D401
                    record.service = "worker"
                    return True
            handler.addFilter(_ServiceFilter())
            root = logging.getLogger()
            for h in list(root.handlers):
                root.removeHandler(h)
            root.addHandler(handler)
            root.setLevel(log_level)
        except ImportError:
            # python-json-logger not installed in this venv — fall back
            # to plain text. Don't crash the worker over logging config.
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
            logging.getLogger(__name__).warning(
                "python-json-logger not installed; LOGGING_BACKEND=%s ignored",
                logging_backend,
            )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    if not args.worker_token:
        parser.error(
            "Worker token is required. Set TA_WORKER_TOKEN environment "
            "variable or pass --worker-token. Worker refuses to start "
            "with no token rather than fall back to a public default."
        )

    async def _run() -> None:
        stop_event = asyncio.Event()
        _install_signal_handlers(stop_event)
        await worker_loop(
            backend_url=args.backend_url,
            worker_token=args.worker_token,
            poll_interval=args.poll_interval,
            stop_event=stop_event,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
