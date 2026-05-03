"""Full-mesh integration test: user → backend → worker → backend → DB + WS.

This exercises the HTTP contract between every service in one go:

    1. POST /api/runs            (user, JWT)   — create a pending run
    2. WS  /ws/runs/{id}         (user, JWT)   — subscribe to live events
    3. worker_loop in-process    (worker)      — claims the run, drives
                                                 the RealExecutor, posts
                                                 every engine event to
                                                 POST /api/internal/runs/{id}/event
    4. backend writes events to Postgres and publishes to Redis
    5. WS client receives the same events, in order
    6. GET /api/runs/{id}/results (user, JWT)  — final state matches

The simulator is replaced with ``FakeController`` from ``test_engine.py``
— the point of this test is the *network path*, not real iOS behaviour.
A separate, manually-run test will cover the AXe + simulator combo
later.

Skipped automatically if the backend isn't reachable (see the
``backend_available`` fixture in ``conftest.py``). This keeps
``pytest tests/`` green on a dev machine that isn't running docker
compose at the moment.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import websockets

from explorer.worker import RealExecutor, worker_loop
from tests.test_engine import _make_three_screen_app


# How long we're willing to wait for the whole pipeline to complete.
# The fake controller explores its 3-screen graph in well under a second
# of real work; the extra slack absorbs docker-desktop startup jitter
# and the worker's poll interval.
RUN_COMPLETION_TIMEOUT = 30.0
WORKER_POLL_INTERVAL = 0.2  # tight polling since we're in a test

# Identifier we use for the run so cleanup can find it if something
# goes wrong and the teardown DELETE doesn't run.
TEST_BUNDLE_ID = "integration-test.fake.app"


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest.mark.integration
async def test_full_run_through_http_mesh(
    backend_url: str,
    admin_jwt: str,
    worker_token: str,
    tmp_path,
) -> None:
    headers = {"Authorization": f"Bearer {admin_jwt}"}

    # --- 1. Create the run via the user-facing REST API -------------------
    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.post(
            f"{backend_url}/api/runs",
            headers=headers,
            json={
                "bundle_id": TEST_BUNDLE_ID,
                "device_id": "BOOTED",
                "platform": "ios",
                "mode": "mc",
                "max_steps": 20,
            },
        )
        resp.raise_for_status()
        run = resp.json()

    run_id = run["id"]
    assert run["status"] == "pending"
    assert run["bundle_id"] == TEST_BUNDLE_ID

    # Collected here so the finally block can still clean up if something
    # between steps 2 and 5 raises.
    events: list[dict[str, Any]] = []
    snapshot: dict[str, Any] | None = None
    snapshot_received = asyncio.Event()
    done_event = asyncio.Event()
    ws_task: asyncio.Task[None] | None = None
    worker_task: asyncio.Task[None] | None = None
    worker_stop = asyncio.Event()

    try:
        # --- 2. Open the WebSocket and start draining events -------------
        ws_url = (
            f"{backend_url.replace('http', 'ws')}/ws/runs/{run_id}"
            f"?token={admin_jwt}"
        )

        async def ws_listener() -> None:
            nonlocal snapshot
            async with websockets.connect(ws_url) as ws:
                async for msg in ws:
                    try:
                        ev = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "snapshot":
                        snapshot = ev
                        snapshot_received.set()
                        continue
                    events.append(ev)
                    if (
                        ev.get("type") == "status_change"
                        and ev.get("new_status") == "completed"
                    ):
                        done_event.set()
                        return

        ws_task = asyncio.create_task(ws_listener())

        # Wait for the snapshot frame before kicking the worker. The
        # backend's run_ws handler sends the snapshot *before* calling
        # `pubsub.subscribe`, so snapshot-received alone doesn't
        # guarantee the subscription is live — but it does bound the
        # gap. A short additional pause after that absorbs the
        # subscribe-to-Redis round-trip. Without this, the backend's
        # own `status_change → running` publish during /claim races
        # ahead of our subscriber.
        await asyncio.wait_for(snapshot_received.wait(), timeout=5)
        await asyncio.sleep(0.2)

        # --- 3. Spin up the worker in-process with a fake controller ----
        controller = _make_three_screen_app()
        # FakeController in test_engine.py doesn't define connect /
        # disconnect — RealExecutor expects them to be awaitable.
        # Same shim as test_worker_real_executor.py.
        controller.connect = _noop  # type: ignore[attr-defined]
        controller.disconnect = _noop  # type: ignore[attr-defined]

        executor = RealExecutor(
            output_root=tmp_path,
            client_factory=lambda: controller,
        )

        worker_task = asyncio.create_task(
            worker_loop(
                backend_url=backend_url,
                worker_token=worker_token,
                poll_interval=WORKER_POLL_INTERVAL,
                stop_event=worker_stop,
                executor=executor,
            )
        )

        # --- 4. Wait for completion --------------------------------------
        try:
            await asyncio.wait_for(
                done_event.wait(), timeout=RUN_COMPLETION_TIMEOUT
            )
        except asyncio.TimeoutError:
            pytest.fail(
                f"run did not reach completed status within "
                f"{RUN_COMPLETION_TIMEOUT}s; "
                f"events seen so far: {[e.get('type') for e in events]}"
            )

        # --- 5. Assert final state from both channels --------------------
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"{backend_url}/api/runs/{run_id}/results",
                headers=headers,
            )
            resp.raise_for_status()
            results = resp.json()

        # Postgres-side: run finished, lifecycle timestamps set, screens
        # and edges got written.
        assert (
            results["run"]["status"] == "completed"
        ), f"run did not complete, state: {results['run']}"
        # started_at / finished_at prove the pending → running → completed
        # path went through the claim + status_change handlers.
        assert results["run"]["started_at"] is not None
        assert results["run"]["finished_at"] is not None

        screen_hashes = {s["screen_id_hash"] for s in results["screens"]}
        assert len(screen_hashes) >= 3, (
            f"expected ≥3 distinct screens after exploring a 3-screen app, "
            f"got {len(screen_hashes)}: {screen_hashes}"
        )
        assert len(results["edges"]) >= 2, (
            f"expected ≥2 edges (home→login, home→settings), got "
            f"{len(results['edges'])}"
        )

        # Redis/WS-side: we saw every event type we expect the engine
        # + worker to emit for a successful run. We deliberately do NOT
        # assert on `status_change → running` here because the backend
        # publishes that event from inside /api/internal/runs/claim
        # before the WS subscriber has had a chance to `SUBSCRIBE` on
        # Redis, so the assertion would be a flaky timing race. The DB
        # `started_at` check above is the authoritative signal.
        event_types = {e["type"] for e in events}
        assert "screen_discovered" in event_types
        assert "edge_discovered" in event_types
        assert "stats_update" in event_types
        assert any(
            e["type"] == "status_change"
            and e.get("new_status") == "completed"
            for e in events
        ), "worker should have posted status_change → completed"

        # Contract: every screen the WS told us about ended up in the
        # database. The snapshot may have arrived with 0 screens
        # (we connected before the worker claimed the run), so the
        # DB-side set is a superset of what the live stream carried.
        ws_screen_hashes = {
            e.get("screen_id_hash")
            for e in events
            if e["type"] == "screen_discovered" and e.get("screen_id_hash")
        }
        assert screen_hashes >= ws_screen_hashes, (
            f"DB screens {screen_hashes} missing some WS-reported screens "
            f"{ws_screen_hashes - screen_hashes}"
        )

        # Snapshot sanity — we should have received exactly one, and it
        # should refer to our run. The screens list inside may be empty
        # (see above) which is fine.
        assert snapshot is not None, "never received snapshot frame"
        assert snapshot["run"]["id"] == run_id

    finally:
        # --- 6. Teardown: stop worker, cancel WS, delete test run --------
        worker_stop.set()
        if worker_task is not None:
            try:
                await asyncio.wait_for(worker_task, timeout=5)
            except (asyncio.TimeoutError, Exception):
                worker_task.cancel()
                try:
                    await worker_task
                except (asyncio.CancelledError, Exception):
                    pass

        if ws_task is not None:
            ws_task.cancel()
            try:
                await ws_task
            except (asyncio.CancelledError, Exception):
                pass

        # Best-effort delete so the DB doesn't accumulate test rows.
        try:
            async with httpx.AsyncClient(timeout=5) as http:
                await http.delete(
                    f"{backend_url}/api/runs/{run_id}",
                    headers=headers,
                )
        except Exception:
            pass
