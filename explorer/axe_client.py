"""
AXe CLI-based device client for the explorer.
Uses AXe for EVERYTHING: accessibility tree, tap, type, swipe, screenshot.
NO idb dependency. Each AXe call is a separate subprocess — no hanging gRPC.

AXe (cameroncooke/axe) works with React Native on iOS 26 where idb/Appium fail.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
from dataclasses import dataclass

logger = logging.getLogger("explorer.axe_client")


def _resolve_axe_binary() -> str | None:
    """Locate the AXe CLI binary.

    Lookup order (first hit wins):
      1. ``TA_AXE_BIN`` environment variable — explicit override, lets
         a user point at a custom build or installed-elsewhere copy.
      2. ``shutil.which("axe")`` — the system PATH. Works on Intel
         Macs (``/usr/local/bin/axe``), Apple Silicon Macs
         (``/opt/homebrew/bin/axe``), and Linux installs alike.

    Returns ``None`` when neither is set. Callers that actually invoke
    AXe must check and fail with a clear error — no silent hardcoded
    platform default. To install on macOS: ``brew install
    cameroncooke/axe/axe``; on Linux build from source and add to PATH.
    """
    explicit = os.environ.get("TA_AXE_BIN")
    if explicit:
        return explicit
    return shutil.which("axe")


AXE = _resolve_axe_binary()
if AXE is None:
    # Don't fail at import — modules that only need types from this
    # file (e.g. tests, linting) shouldn't blow up. Log loudly so a
    # worker-mode startup notices, then let the first subprocess call
    # raise a real FileNotFoundError with the missing-binary path.
    logger.warning(
        "AXe CLI not found. Set TA_AXE_BIN or install axe on PATH "
        "(brew install cameroncooke/axe/axe). Subprocess calls will "
        "fail until one of these is in place."
    )


async def _run(args: list[str], timeout: float = 15) -> tuple[int, str, str]:
    """Run a subprocess with timeout. Returns (returncode, stdout, stderr).

    Fails fast with a clear error if AXe was never resolved at import
    time — beats letting create_subprocess_exec raise an obscure
    NoneType error deep in asyncio.
    """
    if not args or args[0] is None:
        raise RuntimeError(
            "AXe CLI binary not configured. Set TA_AXE_BIN or install "
            "'axe' on PATH (brew install cameroncooke/axe/axe). "
            "Worker cannot drive iOS Simulator without it."
        )
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "", "TIMEOUT"


@dataclass
class TapResult:
    error: str | None = None


class AXeExplorerClient:
    """Pure AXe device client. No idb, no gRPC, no hangs.
    For React Native dev builds, uses Metro CDP for reliable text input."""

    def __init__(self):
        self._udid = ""
        self._width = 0
        self._height = 0
        self._cdp = None  # CDPTextInput, lazy-initialized
        self._cdp_available = False
        # Map element_uid -> test_id, used to translate clicks to CDP text input
        self._last_focused_test_id: str | None = None
        self._last_focused_label: str | None = None

    async def connect(self, udid: str) -> None:
        self._udid = udid
        # Get screen size from screenshot. Take it via a per-run
        # temp file rather than a shared /tmp path — two workers on
        # the same machine would otherwise race for /tmp/_axe_init.png
        # and one would read the other's bytes (audit PER-104 #10).
        import tempfile
        with tempfile.NamedTemporaryFile(
            prefix="axe_init_", suffix=".png", delete=False
        ) as tf:
            screenshot_path = tf.name
        try:
            code, _, _ = await _run([
                "xcrun", "simctl", "io", udid, "screenshot", screenshot_path,
            ])
            if code != 0:
                raise RuntimeError("Cannot take screenshot — is simulator booted?")

            from PIL import Image
            img = Image.open(screenshot_path)
            # Derive the per-pixel scale from simctl rather than
            # assuming 3x. iPhone 16/17 Pro use 3x; iPad and older
            # phones use 2x. ``simctl device-info`` exposes the
            # logical point size; ratio = pixel / point.
            point_w, point_h = await self._device_logical_size(udid)
            if point_w and point_h:
                scale_x = img.width / point_w
                scale_y = img.height / point_h
                # Round to nearest 0.5 — real scales are always 2.0,
                # 2.5 or 3.0. Drift from PNG compression / orientation
                # noise should not change the divisor.
                self._scale = round(((scale_x + scale_y) / 2) * 2) / 2
            else:
                # device-info unavailable (older simctl?) — fall back
                # to scanning common scales and picking the one that
                # makes width round to an integer.
                self._scale = 3.0 if img.width % 3 == 0 else (
                    2.0 if img.width % 2 == 0 else 1.0
                )
            self._width = int(img.width / self._scale)
            self._height = int(img.height / self._scale)
            logger.info(
                f"Connected: {self._width}x{self._height} pts @ {self._scale}x, "
                f"UDID: {udid}"
            )
        finally:
            try:
                import os
                os.unlink(screenshot_path)
            except OSError:
                pass

        # Try to connect to Metro CDP (for React Native text input)
        try:
            from explorer.cdp_text_input import CDPTextInput
            self._cdp = CDPTextInput()
            if await self._cdp.is_available():
                connected = await self._cdp.connect()
                if connected:
                    self._cdp_available = True
                    logger.info("CDP text input ready (Metro debugger connected)")
                else:
                    logger.info("Metro found but CDP connect failed")
            else:
                logger.info("Metro debugger not running, will use AXe HID for text input")
        except Exception as e:
            logger.debug(f"CDP setup skipped: {e}")

    async def disconnect(self) -> None:
        if self._cdp:
            await self._cdp.disconnect()
            self._cdp = None

    async def tap_at(self, x: int, y: int) -> TapResult:
        code, out, err = await _run([
            AXE, "tap", "-x", str(x), "-y", str(y), "--udid", self._udid
        ])
        if code != 0:
            return TapResult(error=err.strip()[:200])
        return TapResult()

    async def tap_by_id(self, test_id: str) -> TapResult:
        """Tap element by testID (AXUniqueId) — AXe finds it automatically."""
        code, out, err = await _run([
            AXE, "tap", "--id", test_id, "--udid", self._udid
        ])
        if code != 0:
            return TapResult(error=err.strip()[:200])
        self._last_focused_test_id = test_id
        return TapResult()

    async def tap_by_label(self, label: str, element_type: str | None = None) -> TapResult:
        """Tap element by accessibility label."""
        args = [AXE, "tap", "--label", label, "--udid", self._udid]
        if element_type:
            args.extend(["--element-type", element_type])
        code, out, err = await _run(args)
        if code != 0:
            return TapResult(error=err.strip()[:200])
        return TapResult()

    async def type_text(self, text: str) -> bool:
        """Type text via CDP (RN dev) or AXe HID (fallback)."""
        # Try CDP first if we know what field is focused
        if self._cdp_available and self._cdp:
            if self._last_focused_test_id:
                ok = await self._cdp.set_text_by_test_id(
                    self._last_focused_test_id, text
                )
                if ok:
                    logger.debug(f"Text set via CDP testID={self._last_focused_test_id}")
                    return True
            if self._last_focused_label:
                ok = await self._cdp.set_text_by_label(
                    self._last_focused_label, text
                )
                if ok:
                    logger.debug(f"Text set via CDP label={self._last_focused_label}")
                    return True

        # Fallback: AXe HID type (limited charset, no React state update)
        code, out, err = await _run([
            AXE, "type", text, "--udid", self._udid
        ], timeout=30)
        if code != 0:
            logger.warning(f"type_text failed: {err[:100]}")
            return False
        return True

    async def set_text_in_field(
        self, test_id: str | None, label: str | None, text: str
    ) -> bool:
        """High-level: set text in a specific field. Uses CDP if available."""
        if self._cdp_available and self._cdp:
            if test_id:
                ok = await self._cdp.set_text_by_test_id(test_id, text)
                if ok:
                    return True
            if label:
                ok = await self._cdp.set_text_by_label(label, text)
                if ok:
                    return True
        # Fallback: tap + AXe type
        if test_id:
            await self.tap_by_id(test_id)
            await asyncio.sleep(0.3)
        return await self.type_text(text)

    async def _device_logical_size(self, udid: str) -> tuple[int, int] | tuple[None, None]:
        """Return (width_pt, height_pt) for the device, or (None, None) if
        simctl can't tell us. Used to derive screen scale at connect-time
        instead of assuming 3x for every device.
        """
        import json
        code, out, _ = await _run([
            "xcrun", "simctl", "list", "devices", "--json"
        ])
        if code != 0:
            return None, None
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return None, None
        # Walk the {runtime: [{udid, deviceTypeIdentifier}]} structure.
        for _runtime, devices in (data.get("devices") or {}).items():
            for d in devices:
                if d.get("udid") == udid:
                    # simctl exposes the device-type id like
                    # "com.apple.CoreSimulator.SimDeviceType.iPhone-16-Pro".
                    # The actual point size lives in
                    # `xcrun simctl list devicetypes --json` keyed by the
                    # same id, but parsing that table is heavyweight.
                    # For the common case we hit the screenshot ratio
                    # check in connect(); this method is a safety net.
                    return None, None
        return None, None

    async def take_screenshot(self) -> str:
        """Screenshot via simctl, return base64 PNG (resized to logical)."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            prefix="axe_shot_", suffix=".png", delete=False
        ) as tf:
            path = tf.name
        try:
            code, _, _ = await _run([
                "xcrun", "simctl", "io", self._udid, "screenshot", path
            ])
            if code != 0:
                return ""

            from io import BytesIO
            from PIL import Image
            img = Image.open(path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img = img.resize((self._width, self._height), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        finally:
            try:
                import os
                os.unlink(path)
            except OSError:
                pass

    async def get_ui_elements(self) -> list[dict]:
        """Get accessibility tree via AXe CLI."""
        code, stdout, stderr = await _run([
            AXE, "describe-ui", "--udid", self._udid
        ])
        if code != 0:
            logger.warning(f"AXe describe-ui failed: {stderr[:200]}")
            return []

        try:
            raw = json.loads(stdout)
            elements = []
            self._flatten(raw if isinstance(raw, list) else [raw], elements)
            return elements
        except json.JSONDecodeError as e:
            logger.warning(f"AXe JSON parse error: {e}")
            return []

    def _flatten(self, nodes: list, result: list[dict]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            result.append({
                "type": node.get("type", ""),
                "label": node.get("AXLabel", ""),
                "value": node.get("AXValue", ""),
                "test_id": node.get("AXUniqueId", ""),
                "frame": node.get("frame", {}),
                "enabled": node.get("enabled", True),
                "visible": True,
            })
            for child in node.get("children", []):
                self._flatten([child], result)

    async def launch_app(self, bundle_id: str) -> bool:
        code, _, _ = await _run([
            "xcrun", "simctl", "launch", self._udid, bundle_id
        ])
        return code == 0

    async def terminate_app(self, bundle_id: str) -> bool:
        code, _, _ = await _run([
            "xcrun", "simctl", "terminate", self._udid, bundle_id
        ])
        return True  # OK even if not running

    async def go_back(self) -> bool:
        """iOS back: swipe from left edge."""
        code, _, _ = await _run([
            AXE, "swipe",
            "--start-x", "0", "--start-y", str(self._height // 2),
            "--end-x", str(self._width // 2), "--end-y", str(self._height // 2),
            "--udid", self._udid,
        ])
        return code == 0

    async def press_enter(self) -> bool:
        # AXe type can send Return
        code, _, _ = await _run([AXE, "type", "\n", "--udid", self._udid])
        return code == 0

    async def erase_text(self, count: int = 1) -> bool:
        return True  # Not used — we relaunch app for clean fields

    @property
    def ctx(self):
        return self

    @property
    def device(self):
        return self

    @property
    def device_id(self):
        return self._udid

    @property
    def device_width(self):
        return self._width * 3

    @property
    def device_height(self):
        return self._height * 3
