"""CLI entry point for the App Explorer."""

from __future__ import annotations

import asyncio
import logging
import sys

import typer

app = typer.Typer(help="App Explorer: automated state graph builder for mobile apps")


def _setup_logging(verbose: bool = False):
    for noisy in ("hpack", "idb", "grpc", "h2", "httpcore", "httpx",
                   "PIL", "asyncio", "urllib3", "selenium", "appium",
                   "minitap.mobile_use.clients.idb_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command()
def explore(
    bundle_id: str = typer.Argument(help="App bundle ID"),
    device_id: str = typer.Option(..., "--device-id", "-d", help="iOS simulator UDID"),
    output: str = typer.Option("explorer_output", "--output", "-o", help="Output directory"),
    resume: str | None = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    mode: str = typer.Option("axe", "--mode", help="Client mode: axe (default, best for RN), native (Appium, native apps only), vision (LLM, universal)"),
    model: str = typer.Option("qwen3-vl-32b-instruct-mlx", "--model", "-m", help="Vision model (for --mode vision)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Explore an app and build its state machine graph."""
    _setup_logging(verbose)

    if mode == "axe":
        asyncio.run(_explore_axe(bundle_id, device_id, output, resume))
    elif mode == "native":
        asyncio.run(_explore_native(bundle_id, device_id, output, resume))
    elif mode == "vision":
        asyncio.run(_explore_vision(bundle_id, device_id, output, resume, model))
    else:
        print(f"Unknown mode: {mode}. Use: axe, native, vision")
        raise typer.Exit(1)


async def _explore_axe(bundle_id, device_id, output, resume):
    """Explore using AXe CLI + idb. Works with RN and native apps."""
    from explorer.axe_client import AXeExplorerClient
    from explorer.engine import ExplorationEngine
    from explorer.visualizer import export_all

    client = AXeExplorerClient()
    try:
        print(f">>> Connecting via AXe + idb...", flush=True)
        await client.connect(udid=device_id)
        print(f">>> Connected ({client._width}x{client._height} points)", flush=True)

        engine = ExplorationEngine(
            controller=client, app_bundle_id=bundle_id,
            output_dir=output, resume_from=resume,
        )
        graph = await engine.run()

        print(f"\n{'='*50}\nExploration complete!\n{'='*50}", flush=True)
        export_all(graph, output)
    finally:
        await client.disconnect()


async def _explore_native(bundle_id, device_id, output, resume):
    """Explore using Appium + XCUITest. Native apps only."""
    from explorer.appium_client import AppiumExplorerClient
    from explorer.engine import ExplorationEngine
    from explorer.visualizer import export_all

    client = AppiumExplorerClient()
    try:
        print(">>> Starting Appium session...", flush=True)
        await client.start_session(bundle_id=bundle_id, udid=device_id)
        print(">>> Appium session ready", flush=True)

        engine = ExplorationEngine(
            controller=client, app_bundle_id=bundle_id,
            output_dir=output, resume_from=resume,
        )
        graph = await engine.run()

        print(f"\n{'='*50}\nExploration complete!\n{'='*50}", flush=True)
        export_all(graph, output)
    finally:
        await client.stop_session()


async def _explore_vision(bundle_id, device_id, output, resume, model):
    """Explore using idb + vision LLM. Universal fallback."""
    from explorer.engine import ExplorationEngine
    from explorer.vision_client import VisionExplorerClient
    from explorer.visualizer import export_all

    client = VisionExplorerClient(model=model)
    try:
        print(f">>> Connecting via idb + vision ({model})...", flush=True)
        await client.connect(udid=device_id)
        print(f">>> Connected ({client._width}x{client._height} points)", flush=True)

        engine = ExplorationEngine(
            controller=client, app_bundle_id=bundle_id,
            output_dir=output, resume_from=resume,
        )
        graph = await engine.run()

        print(f"\n{'='*50}\nExploration complete!\n{'='*50}", flush=True)
        export_all(graph, output)
    finally:
        await client.disconnect()


@app.command()
def visualize(
    graph_path: str = typer.Argument(help="Path to graph.json"),
    output: str = typer.Option("explorer_output", "--output", "-o"),
):
    """Generate Mermaid diagram and report from an existing graph."""
    from explorer.models import AppGraph
    from explorer.visualizer import export_all
    graph = AppGraph.load(graph_path)
    print(f"Loaded graph: {graph.stats()}")
    export_all(graph, output)


@app.command()
def diagnose(
    bundle_id: str = typer.Argument(help="App bundle ID"),
    device_id: str = typer.Option(..., "--device-id", "-d", help="iOS simulator UDID"),
):
    """Diagnose which accessibility channels work for this app."""
    asyncio.run(_diagnose(bundle_id, device_id))


async def _diagnose(bundle_id: str, device_id: str):
    """Run all accessibility channels and report results."""
    import time

    print(f"\n== iOS Accessibility Channel Diagnostics ==")
    print(f"App: {bundle_id}")
    print(f"Device: {device_id}")
    print()

    # 1. xcrun simctl screenshot
    print("[1/5] xcrun simctl screenshot...", end="  ", flush=True)
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        "xcrun", "simctl", "io", device_id, "screenshot", "/tmp/diag_screenshot.png",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    dt = time.time() - t0
    if proc.returncode == 0:
        print(f"✅ OK ({dt:.1f}s)")
    else:
        print(f"❌ FAIL ({dt:.1f}s)")

    # 2. idb screenshot
    print("[2/5] idb screenshot...", end="  ", flush=True)
    t0 = time.time()
    try:
        from minitap.mobile_use.clients.idb_client import IdbClientWrapper
        idb = IdbClientWrapper(udid=device_id)
        await idb.init_companion()
        data = await idb.screenshot()
        dt = time.time() - t0
        if data and len(data) > 0:
            print(f"✅ OK ({dt:.1f}s, {len(data)} bytes)")
        else:
            print(f"❌ No data ({dt:.1f}s)")
    except Exception as e:
        print(f"❌ ERROR ({time.time()-t0:.1f}s): {e}")
        idb = None

    # 3. AXe describe-ui
    print("[3/5] axe describe-ui...", end="  ", flush=True)
    t0 = time.time()
    try:
        from explorer.axe_client import AXE
        proc = await asyncio.create_subprocess_exec(
            AXE, "describe-ui", "--udid", device_id,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        dt = time.time() - t0
        if proc.returncode == 0 and stdout:
            import json
            tree = json.loads(stdout.decode())
            count = len(stdout.decode().split("\n"))
            print(f"✅ OK ({dt:.1f}s, {count} lines)")
        else:
            print(f"❌ FAIL ({dt:.1f}s): {stderr.decode()[:100]}")
    except asyncio.TimeoutError:
        print(f"❌ TIMEOUT ({time.time()-t0:.1f}s)")
    except FileNotFoundError:
        print(f"❌ axe not installed")

    # 4. idb describe-all
    print("[4/5] idb describe-all (gRPC)...", end="  ", flush=True)
    t0 = time.time()
    if idb and idb._client:
        try:
            info = await asyncio.wait_for(
                idb._client.accessibility_info(point=None, nested=True),
                timeout=15,
            )
            dt = time.time() - t0
            if info and info.json:
                print(f"✅ OK ({dt:.1f}s)")
            else:
                print(f"⚠️ Empty ({dt:.1f}s)")
        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"❌ ERROR ({time.time()-t0:.1f}s): {e}")
    else:
        print("⏭️ SKIPPED (idb not connected)")

    # 5. Appium page_source (shallow)
    print("[5/5] Appium page_source (shallow)...", end="  ", flush=True)
    t0 = time.time()
    try:
        import threading
        from appium import webdriver
        from appium.options.ios import XCUITestOptions

        options = XCUITestOptions()
        options.platform_name = "iOS"
        options.udid = device_id
        options.automation_name = "XCUITest"
        options.bundle_id = bundle_id
        options.no_reset = True
        options.set_capability("appium:forceAppLaunch", False)
        options.set_capability("appium:settings[snapshotMaxDepth]", 8)
        options.set_capability("appium:settings[customSnapshotTimeout]", 5.0)

        driver = await asyncio.get_event_loop().run_in_executor(
            None, lambda: webdriver.Remote("http://127.0.0.1:4723", options=options)
        )
        res = [None]
        def get_src():
            try: res[0] = driver.page_source
            except: pass
        t = threading.Thread(target=get_src); t.start(); t.join(timeout=10)
        dt = time.time() - t0

        if t.is_alive():
            print(f"❌ TIMEOUT ({dt:.1f}s)")
        elif res[0]:
            print(f"✅ OK ({dt:.1f}s, {len(res[0])} chars)")
        else:
            print(f"⚠️ Empty ({dt:.1f}s)")

        try: driver.quit()
        except: pass
    except Exception as e:
        print(f"❌ ERROR ({time.time()-t0:.1f}s): {str(e)[:100]}")

    # Cleanup
    if idb:
        await idb.cleanup()

    # Recommendation
    print()
    print("== Recommendation ==")
    print("Primary: AXe CLI (fastest, works with RN)")
    print("Fallback: Appium shallow (snapshotMaxDepth=8)")
    print("Last resort: Vision-only (screenshot + LLM)")


def main():
    app()


if __name__ == "__main__":
    main()
