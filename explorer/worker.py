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

This file ships TWO execution backends:

    SyntheticExecutor    For Block 4a — generates fake screens and
                         edges with delays so the live progress
                         pipeline can be validated end-to-end without
                         a simulator.

    RealExecutor         For Block 4b — calls the actual exploration
                         engine. Stub today; will be filled in once
                         engine.py is refactored to PUCT + Go-Explore.

Both executors implement the same `run(config, event_sink)` interface
so the loop doesn't care which one is in use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
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
    """Tiny async HTTP client wrapping the worker's view of the backend."""

    def __init__(self, base_url: str, worker_token: str):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {worker_token}"}
        self._client = httpx.AsyncClient(timeout=15.0)

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
# Synthetic executor (Block 4a — pipe validation, no simulator)
# ----------------------------------------------------------------------------


class SyntheticExecutor:
    """Generates a deterministic fake exploration trajectory.

    The point isn't to test the explorer — it's to prove the pipe works:
    worker → backend → Postgres → Redis → WebSocket → browser.

    Trajectory:
        - Walks max_steps steps
        - Discovers a new screen with probability ~0.4 each step
        - Records an edge from the previous screen to the current one
        - Sleeps 0.5s between steps so the user actually sees streaming
        - Halfway through, sends a stats_update event
    """

    NAMES = [
        "Login",
        "HomeScreen",
        "ProductList",
        "ProductDetail",
        "Cart",
        "Checkout",
        "Profile",
        "Settings",
        "About",
        "Help",
        "OrderHistory",
        "PaymentMethods",
        "Addresses",
        "Notifications",
        "SearchResults",
    ]

    async def run(self, config: dict[str, Any], event_sink: EventSink) -> None:
        max_steps = max(int(config.get("max_steps", 20)), 5)
        steps_to_run = min(max_steps, 30)

        screens_seen: list[str] = []
        prev_hash: str | None = None
        new_screens = 0
        new_edges = 0

        for step in range(1, steps_to_run + 1):
            await asyncio.sleep(0.5)

            # Decide whether to "discover" a new screen this step
            should_create_new_screen = (
                len(screens_seen) == 0 or random.random() < 0.4
            )

            if should_create_new_screen and len(screens_seen) < len(self.NAMES):
                idx = len(screens_seen)
                name = self.NAMES[idx]
                screen_hash = f"synthetic-{idx:03d}-{name.lower()}"
                screens_seen.append(screen_hash)
                new_screens += 1
                await event_sink(
                    {
                        "type": "screen_discovered",
                        "step_idx": step,
                        "screen_id_hash": screen_hash,
                        "screen_name": name,
                    }
                )
            else:
                # Re-visit a known screen
                screen_hash = random.choice(screens_seen)

            if prev_hash is not None and prev_hash != screen_hash:
                action_type = random.choice(["tap", "swipe", "type"])
                await event_sink(
                    {
                        "type": "edge_discovered",
                        "step_idx": step,
                        "source_screen_hash": prev_hash,
                        "target_screen_hash": screen_hash,
                        "action_type": action_type,
                        "success": True,
                    }
                )
                new_edges += 1

            prev_hash = screen_hash

            # Halfway-through stats update
            if step == steps_to_run // 2:
                await event_sink(
                    {
                        "type": "stats_update",
                        "step_idx": step,
                        "stats": {
                            "screens": new_screens,
                            "edges": new_edges,
                            "step": step,
                            "max_steps": steps_to_run,
                        },
                    }
                )

        # Final stats
        await event_sink(
            {
                "type": "stats_update",
                "step_idx": steps_to_run,
                "stats": {
                    "screens": new_screens,
                    "edges": new_edges,
                    "step": steps_to_run,
                    "max_steps": steps_to_run,
                },
            }
        )


# ----------------------------------------------------------------------------
# Real executor (Block 4b — TODO)
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
        mode_name = (config.get("mode") or "hybrid").lower()
        max_steps = int(config.get("max_steps") or 200)
        platform = config.get("platform", "ios")

        try:
            mode = ExplorationMode(mode_name)
        except ValueError:
            logger.warning("Unknown mode %r — falling back to hybrid", mode_name)
            mode = ExplorationMode.HYBRID

        # ── LLM prior provider ──
        prior_provider = None
        mode_config = MODE_CONFIGS[mode]
        if mode_config.use_llm_priors:
            llm_url = os.environ.get("TA_LLM_BASE_URL", "http://localhost:8080")
            llm_model = os.environ.get("TA_LLM_MODEL_NAME", "embeddings")
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
                logger.exception("Simulator setup failed for run %s", run_id)
                await event_sink({"type": "error", "step_idx": 0, "message": f"Simulator setup failed: {exc}"})
                if sim_manager:
                    await sim_manager.cleanup()
                return

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
                        mirror_proc = await asyncio.create_subprocess_exec(
                            mirror_bin,
                            "--port", mirror_port,
                            "--max-width", "480",
                            "--fps", "15",
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

            # ── Exploration ──
            if mode in (ExplorationMode.HYBRID, ExplorationMode.AI):
                # LLM-driven: model decides every action
                from explorer.llm_loop import LLMExplorationLoop

                llm_url = os.environ.get("TA_LLM_BASE_URL", "http://localhost:8080")
                llm_model_name = os.environ.get("TA_LLM_MODEL_NAME", "embeddings")

                # URL of the RAG LLM — used as the "smart" classifier for
                # defect detection. Falls back to the agent LLM if not set.
                rag_llm_url = os.environ.get(
                    "TA_RAG_LLM_BASE_URL",
                    os.environ.get("TA_LLM_BASE_URL", "http://localhost:8083"),
                )

                # Defect poster injected by execute_one_run from the
                # BackendClient. Falls back to a no-op if missing — keeps
                # tests green when the executor is invoked standalone.
                _post_defect = config.get("_post_defect") or (lambda _payload: asyncio.sleep(0))

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
            else:
                # MC mode: no LLM, random/PUCT selection
                from explorer.loop import ExplorationLoop
                from explorer.strategy import MCStrategy

                loop = ExplorationLoop(
                    controller=client,
                    strategy=MCStrategy(),
                    app_bundle_id=bundle_id,
                    max_steps=max_steps,
                    event_callback=event_sink,
                )

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
    executor: SyntheticExecutor | RealExecutor,
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
    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        await sink(
            {"type": "error", "step_idx": 0, "message": str(exc)},
        )


async def worker_loop(
    backend_url: str,
    worker_token: str,
    poll_interval: float,
    executor_kind: str,
    stop_event: asyncio.Event,
    executor: SyntheticExecutor | RealExecutor | None = None,
) -> None:
    # `executor` is an injection hook for integration tests: pass a
    # pre-built RealExecutor(client_factory=<fake controller>) and the
    # loop will use it instead of constructing a fresh executor from
    # `executor_kind`. Production (main()) never passes this — it sticks
    # to the CLI-string-driven path below.
    client = BackendClient(backend_url, worker_token)
    if executor is None:
        executor = (
            SyntheticExecutor() if executor_kind == "synthetic" else RealExecutor()
        )

    logger.info(
        "Worker started (backend=%s, poll=%ss, executor=%s)",
        backend_url,
        poll_interval,
        executor_kind,
    )

    # Report available simulator/emulator configs to the backend so the
    # admin UI can show them in the device management page.
    if executor_kind == "real":
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
            logger.info(
                "Reported simulator config: %d runtimes, %d device types",
                len(config_payload["runtimes"]),
                len(config_payload["device_types"]),
            )

            # Clean up orphan TA-* simulators/AVDs from crashed runs
            ios_cleaned = await IOSSimulatorManager.cleanup_orphans()
            android_cleaned = await AndroidEmulatorManager.cleanup_orphans()
            if ios_cleaned or android_cleaned:
                logger.info(
                    "Cleaned up %d iOS + %d Android orphan simulators",
                    ios_cleaned, android_cleaned,
                )
        except Exception:
            logger.exception("Failed to report simulator config or cleanup orphans")

    # Heartbeat runs in its own task so the "Connected" indicator stays
    # green even while a run is executing. Previously it was inline in the
    # claim loop, which meant heartbeats stopped flowing for the entire
    # duration of execute_one_run() — runs take minutes to hours, so the
    # 60s Redis TTL would expire and the UI would show "Не подключено"
    # despite the worker doing real work right at that moment.
    async def _heartbeat_loop() -> None:
        while not stop_event.is_set():
            await client.post_heartbeat()
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
        default=os.environ.get(
            "TA_WORKER_TOKEN", "change_me_worker_token_long_random_string"
        ),
        help="Worker token for /api/internal/* (default: env TA_WORKER_TOKEN)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between claim attempts when idle (default: %(default)s)",
    )
    parser.add_argument(
        "--executor",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Which executor to use (default: synthetic)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def _run() -> None:
        stop_event = asyncio.Event()
        _install_signal_handlers(stop_event)
        await worker_loop(
            backend_url=args.backend_url,
            worker_token=args.worker_token,
            poll_interval=args.poll_interval,
            executor_kind=args.executor,
            stop_event=stop_event,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
