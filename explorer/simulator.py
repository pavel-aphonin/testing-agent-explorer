"""Simulator / emulator lifecycle managers.

IOSSimulatorManager wraps ``xcrun simctl`` to create, boot, install,
launch, and tear down iOS Simulators on demand.

AndroidEmulatorManager wraps ``avdmanager``, ``emulator``, and ``adb``
to do the same for Android Virtual Devices.

Both classes follow the same interface so the worker can treat them
polymorphically:

    manager = IOSSimulatorManager(...) | AndroidEmulatorManager(...)
    udid = await manager.create(run_id)
    await manager.boot()
    await manager.install(app_path)
    await manager.launch(bundle_id)
    ...
    await manager.cleanup()   # always in a finally block

Each created simulator/AVD is named ``TA-<run_id[:8]>`` so orphans
are easy to spot and clean up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("explorer.simulator")

ANDROID_SDK = os.environ.get(
    "ANDROID_HOME",
    os.path.expanduser("~/Library/Android/sdk"),
)


# ─────────────────────────────────────────────────── helpers ──

async def _exec(
    *args: str,
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> str:
    """Run a subprocess, raise on non-zero exit, return stdout."""
    merged_env = {**os.environ, **(env or {})}
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        cmd = " ".join(args[:3])
        raise RuntimeError(f"{cmd} failed (rc={proc.returncode}): {stderr.decode().strip()}")
    return stdout.decode().strip()


# ───────────────────────────────────────────── iOS Simulator ──

class IOSSimulatorManager:
    """Lifecycle manager for a single iOS Simulator instance."""

    def __init__(self, device_identifier: str, runtime_identifier: str) -> None:
        self.device_identifier = device_identifier
        self.runtime_identifier = runtime_identifier
        self.udid: str | None = None
        self._name: str | None = None

    async def create(self, run_id: str) -> str:
        self._name = f"TA-{run_id[:8]}"
        self.udid = await _exec(
            "xcrun", "simctl", "create",
            self._name, self.device_identifier, self.runtime_identifier,
            timeout=30.0,
        )
        logger.info("Created iOS simulator %s (udid=%s)", self._name, self.udid)
        return self.udid

    async def boot(self) -> None:
        assert self.udid
        await _exec("xcrun", "simctl", "boot", self.udid, timeout=30.0)
        # Wait for full boot (bootstatus -b blocks until ready)
        await _exec(
            "xcrun", "simctl", "bootstatus", self.udid, "-b",
            timeout=120.0,
        )
        # Open Simulator.app GUI so the window is visible for SimMirror
        # and the user can see what's happening. Without this, simctl boot
        # runs headless and ScreenCaptureKit can't find the window.
        try:
            await _exec(
                "open", "-a", "Simulator",
                "--args", "-CurrentDeviceUDID", self.udid,
                timeout=10.0,
            )
            # Give Simulator.app a moment to create the window
            await asyncio.sleep(3)
        except Exception:
            logger.warning("Could not open Simulator.app GUI — live mirror will be unavailable")
        logger.info("Simulator %s booted", self.udid)

    async def install(self, app_path: str) -> None:
        assert self.udid
        if not Path(app_path).exists():
            raise FileNotFoundError(f"App not found: {app_path}")
        await _exec(
            "xcrun", "simctl", "install", self.udid, app_path,
            timeout=60.0,
        )
        logger.info("Installed %s on %s", Path(app_path).name, self.udid)

    async def launch(self, bundle_id: str) -> None:
        assert self.udid
        result = await _exec(
            "xcrun", "simctl", "launch", self.udid, bundle_id,
            timeout=15.0,
        )
        logger.info("Launched %s: %s", bundle_id, result)

    async def shutdown(self) -> None:
        if not self.udid:
            return
        try:
            await _exec("xcrun", "simctl", "shutdown", self.udid, timeout=15.0)
            logger.info("Simulator %s shut down", self.udid)
        except Exception:
            logger.debug("shutdown failed for %s (may already be off)", self.udid)

    async def delete(self) -> None:
        if not self.udid:
            return
        try:
            await _exec("xcrun", "simctl", "delete", self.udid, timeout=15.0)
            logger.info("Simulator %s deleted", self.udid)
        except Exception:
            logger.warning("Failed to delete simulator %s", self.udid)

    async def cleanup(self) -> None:
        """Shutdown + delete, swallowing errors. Call from finally blocks."""
        await self.shutdown()
        await self.delete()

    # ── static discovery ──

    @staticmethod
    async def list_runtimes() -> list[dict[str, str]]:
        raw = await _exec("xcrun", "simctl", "list", "runtimes", "-j", timeout=10.0)
        data = json.loads(raw)
        return [
            {
                "name": r["name"],
                "identifier": r["identifier"],
                "platform": "ios",
            }
            for r in data.get("runtimes", [])
            if r.get("platform") == "iOS" and r.get("isAvailable", False)
        ]

    @staticmethod
    async def list_device_types() -> list[dict[str, str]]:
        raw = await _exec("xcrun", "simctl", "list", "devicetypes", "-j", timeout=10.0)
        data = json.loads(raw)
        return [
            {
                "name": dt["name"],
                "identifier": dt["identifier"],
                "platform": "ios",
            }
            for dt in data.get("devicetypes", [])
            if "iPhone" in dt.get("name", "") or "iPad" in dt.get("name", "")
        ]

    @staticmethod
    async def cleanup_orphans() -> int:
        """Delete any TA-* simulators left from crashed runs."""
        raw = await _exec("xcrun", "simctl", "list", "devices", "-j", timeout=10.0)
        data = json.loads(raw)
        count = 0
        for runtime, devices in data.get("devices", {}).items():
            for d in devices:
                if d.get("name", "").startswith("TA-"):
                    try:
                        udid = d["udid"]
                        if d.get("state") == "Booted":
                            await _exec("xcrun", "simctl", "shutdown", udid, timeout=10.0)
                        await _exec("xcrun", "simctl", "delete", udid, timeout=10.0)
                        count += 1
                        logger.info("Cleaned up orphan simulator %s (%s)", d["name"], udid)
                    except Exception:
                        logger.warning("Failed to clean orphan %s", d.get("name"))
        return count


# ──────────────────────────────────── Android Emulator ──

class AndroidEmulatorManager:
    """Lifecycle manager for a single Android Virtual Device."""

    def __init__(self, device_name: str, system_image: str) -> None:
        self.device_name = device_name      # e.g. "pixel_9_pro_xl"
        self.system_image = system_image    # e.g. "system-images;android-36;google_apis_playstore;arm64-v8a"
        self.avd_name: str | None = None
        self._emulator_proc: asyncio.subprocess.Process | None = None

    def _sdk_tool(self, name: str) -> str:
        """Resolve path to an Android SDK tool."""
        candidates = [
            f"{ANDROID_SDK}/cmdline-tools/latest/bin/{name}",
            f"{ANDROID_SDK}/tools/bin/{name}",
            f"{ANDROID_SDK}/platform-tools/{name}",
            f"{ANDROID_SDK}/emulator/{name}",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return name  # hope it's on PATH

    async def create(self, run_id: str) -> str:
        self.avd_name = f"TA-{run_id[:8]}"
        avdmanager = self._sdk_tool("avdmanager")
        await _exec(
            avdmanager, "create", "avd",
            "-n", self.avd_name,
            "-k", self.system_image,
            "-d", self.device_name,
            "--force",
            timeout=60.0,
            env={"ANDROID_HOME": ANDROID_SDK, "ANDROID_SDK_ROOT": ANDROID_SDK},
        )
        logger.info("Created AVD %s (device=%s, image=%s)", self.avd_name, self.device_name, self.system_image)
        return self.avd_name

    async def boot(self) -> None:
        assert self.avd_name
        emulator = self._sdk_tool("emulator")
        self._emulator_proc = await asyncio.create_subprocess_exec(
            emulator,
            "-avd", self.avd_name,
            "-no-window",
            "-no-audio",
            "-no-boot-anim",
            "-gpu", "swiftshader_indirect",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, "ANDROID_HOME": ANDROID_SDK, "ANDROID_SDK_ROOT": ANDROID_SDK},
        )
        logger.info("Emulator process started (pid=%s)", self._emulator_proc.pid)

        # Wait for the device to be fully booted
        adb = self._sdk_tool("adb")
        await _exec(adb, "wait-for-device", timeout=120.0)

        # Poll sys.boot_completed
        for _ in range(60):
            try:
                result = await _exec(
                    adb, "shell", "getprop", "sys.boot_completed",
                    timeout=5.0,
                )
                if result.strip() == "1":
                    logger.info("AVD %s fully booted", self.avd_name)
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise RuntimeError(f"AVD {self.avd_name} did not finish booting within 120s")

    async def install(self, apk_path: str) -> None:
        if not Path(apk_path).exists():
            raise FileNotFoundError(f"APK not found: {apk_path}")
        adb = self._sdk_tool("adb")
        await _exec(adb, "install", "-r", apk_path, timeout=60.0)
        logger.info("Installed %s", Path(apk_path).name)

    async def launch(self, package: str) -> None:
        adb = self._sdk_tool("adb")
        # Find the launcher activity
        try:
            dump = await _exec(
                adb, "shell", "cmd", "package", "resolve-activity",
                "--brief", package,
                timeout=10.0,
            )
            # Output format: "priority=0 preferredOrder=0 ...\ncom.example/.MainActivity"
            activity = dump.strip().split("\n")[-1].strip()
        except Exception:
            # Fallback: launch via monkey
            await _exec(
                adb, "shell", "monkey", "-p", package,
                "-c", "android.intent.category.LAUNCHER", "1",
                timeout=10.0,
            )
            logger.info("Launched %s via monkey", package)
            return
        await _exec(
            adb, "shell", "am", "start", "-n", activity,
            timeout=10.0,
        )
        logger.info("Launched %s (%s)", package, activity)

    async def shutdown(self) -> None:
        if self._emulator_proc and self._emulator_proc.returncode is None:
            try:
                adb = self._sdk_tool("adb")
                await _exec(adb, "emu", "kill", timeout=10.0)
            except Exception:
                try:
                    self._emulator_proc.kill()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self._emulator_proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
            logger.info("Emulator process stopped")

    async def delete(self) -> None:
        if not self.avd_name:
            return
        try:
            avdmanager = self._sdk_tool("avdmanager")
            await _exec(
                avdmanager, "delete", "avd", "-n", self.avd_name,
                timeout=15.0,
                env={"ANDROID_HOME": ANDROID_SDK, "ANDROID_SDK_ROOT": ANDROID_SDK},
            )
            logger.info("AVD %s deleted", self.avd_name)
        except Exception:
            logger.warning("Failed to delete AVD %s", self.avd_name)

    async def cleanup(self) -> None:
        """Shutdown + delete, swallowing errors. Call from finally blocks."""
        await self.shutdown()
        await self.delete()

    # ── static discovery ──

    @staticmethod
    async def list_system_images() -> list[dict[str, str]]:
        """List installed Android system images (for admin device config)."""
        sdkmanager = f"{ANDROID_SDK}/cmdline-tools/latest/bin/sdkmanager"
        if not Path(sdkmanager).exists():
            return []
        try:
            raw = await _exec(sdkmanager, "--list", timeout=30.0)
        except Exception:
            return []

        images: list[dict[str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("system-images;") and "google" in line.lower():
                parts = line.split("|")
                identifier = parts[0].strip()
                # Parse "system-images;android-36;google_apis_playstore;arm64-v8a"
                segments = identifier.split(";")
                if len(segments) >= 2:
                    api = segments[1].replace("android-", "Android ")
                    images.append({
                        "name": api,
                        "identifier": identifier,
                        "platform": "android",
                    })
        return images

    @staticmethod
    async def list_device_types() -> list[dict[str, str]]:
        """List available Android device definitions."""
        avdmanager = f"{ANDROID_SDK}/cmdline-tools/latest/bin/avdmanager"
        if not Path(avdmanager).exists():
            return []
        try:
            raw = await _exec(
                avdmanager, "list", "device", "-c",
                timeout=15.0,
                env={"ANDROID_HOME": ANDROID_SDK, "ANDROID_SDK_ROOT": ANDROID_SDK},
            )
        except Exception:
            return []

        devices: list[dict[str, str]] = []
        for line in raw.splitlines():
            name = line.strip()
            if name:
                # avdmanager outputs device identifiers like "pixel_9_pro_xl"
                friendly = name.replace("_", " ").title()
                devices.append({
                    "name": friendly,
                    "identifier": name,
                    "platform": "android",
                })
        return devices

    @staticmethod
    async def cleanup_orphans() -> int:
        """Delete any TA-* AVDs left from crashed runs."""
        avdmanager = f"{ANDROID_SDK}/cmdline-tools/latest/bin/avdmanager"
        if not Path(avdmanager).exists():
            return 0
        try:
            raw = await _exec(
                avdmanager, "list", "avd", "-c",
                timeout=15.0,
                env={"ANDROID_HOME": ANDROID_SDK, "ANDROID_SDK_ROOT": ANDROID_SDK},
            )
        except Exception:
            return 0

        count = 0
        for line in raw.splitlines():
            name = line.strip()
            if name.startswith("TA-"):
                try:
                    await _exec(
                        avdmanager, "delete", "avd", "-n", name,
                        timeout=15.0,
                        env={"ANDROID_HOME": ANDROID_SDK, "ANDROID_SDK_ROOT": ANDROID_SDK},
                    )
                    count += 1
                    logger.info("Cleaned up orphan AVD %s", name)
                except Exception:
                    logger.warning("Failed to clean orphan AVD %s", name)
        return count
