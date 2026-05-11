"""
Appium-based device client for the explorer.
Replaces idb_companion (which hangs on iOS 26) with Appium + XCUITest driver.

Provides the same interface the explorer engine expects:
  - tap_at(x, y)
  - type_text(text)
  - take_screenshot() -> base64
  - get_ui_elements() -> list[dict]
  - launch_app(bundle_id)
  - go_back()
  - press_enter()
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("explorer.appium_client")


@dataclass
class TapResult:
    error: str | None = None


class AppiumExplorerClient:
    """Thin async wrapper around Appium WebDriver for the explorer."""

    def __init__(self, appium_url: str = "http://127.0.0.1:4723"):
        self.appium_url = appium_url
        self._driver = None
        self._session_id: str | None = None

    async def start_session(
        self,
        bundle_id: str,
        udid: str,
        platform_version: str | None = None,
    ) -> None:
        """Start an Appium session with XCUITest driver.

        ``platform_version`` defaults to whatever simctl reports for
        the booted UDID (instead of a hardcoded "18.2"). This used to
        be a P2 audit finding: the hardcode silently mismatched the
        actual simulator iOS version on machines running 17.x or 26.x,
        and Appium then complained or attached to the wrong runtime.
        """
        from appium import webdriver
        from appium.options.ios import XCUITestOptions

        if platform_version is None:
            platform_version = await self._detect_ios_version(udid) or "18.2"

        options = XCUITestOptions()
        options.platform_name = "iOS"
        options.device_name = "iPhone"
        options.udid = udid
        options.platform_version = platform_version
        options.automation_name = "XCUITest"
        options.bundle_id = bundle_id
        options.no_reset = True
        # Force fresh launch so WDA has correct app context
        options.set_capability("appium:forceAppLaunch", True)
        # Longer command timeout (default 60s is too short for first source call)
        options.set_capability("appium:wdaLaunchTimeout", 120000)
        options.set_capability("appium:wdaConnectionTimeout", 120000)
        options.set_capability("appium:commandTimeouts", '{"default": 120000}')
        # Longer timeout for page source on React Native apps
        options.set_capability("appium:pageSourceTimeout", 120000)

        logger.info(f"Starting Appium session for {bundle_id} on {udid}...")

        # Appium WebDriver is sync, run in executor
        loop = asyncio.get_event_loop()
        self._driver = await loop.run_in_executor(
            None,
            lambda: webdriver.Remote(self.appium_url, options=options),
        )
        self._session_id = self._driver.session_id
        logger.info(f"Appium session started: {self._session_id}")

    async def stop_session(self) -> None:
        if self._driver:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._driver.quit)
            self._driver = None
            self._session_id = None

    def _run_sync(self, fn):
        """Run a sync Appium call in the thread pool."""
        return asyncio.get_event_loop().run_in_executor(None, fn)

    async def tap_at(self, x: int, y: int) -> TapResult:
        try:
            await self._run_sync(
                lambda: self._driver.execute_script(
                    "mobile: tap", {"x": x, "y": y}
                )
            )
            return TapResult()
        except Exception as e:
            return TapResult(error=str(e))

    async def type_text(self, text: str) -> bool:
        try:
            # Use mobile: type which types into the currently focused element
            await self._run_sync(
                lambda: self._driver.execute_script("mobile: type", {"text": text})
            )
            return True
        except Exception:
            # Fallback: find active element and send_keys
            try:
                active = await self._run_sync(lambda: self._driver.switch_to.active_element)
                if active:
                    await self._run_sync(lambda: active.send_keys(text))
                    return True
            except Exception as e:
                logger.warning(f"type_text failed: {e}")
            return False

    async def take_screenshot(self) -> str:
        """Returns base64-encoded PNG screenshot."""
        b64 = await self._run_sync(lambda: self._driver.get_screenshot_as_base64())
        return b64

    async def get_ui_elements(self) -> list[dict]:
        """Get the full accessibility tree as a flat list of elements."""
        import json as json_mod
        try:
            # Use page_source (XML) — more reliable than mobile: source
            xml_source = await self._run_sync(lambda: self._driver.page_source)
            if xml_source:
                elements = self._parse_xml_source(xml_source)
                return elements
        except Exception as e:
            logger.warning(f"page_source failed: {e}, trying mobile: source")

        # Fallback to mobile: source with JSON
        try:
            source = await self._run_sync(
                lambda: self._driver.execute_script(
                    "mobile: source", {"format": "json"}
                )
            )
            if isinstance(source, str):
                source = json_mod.loads(source)
            elements = []
            self._flatten_xcui_tree(source, elements)
            return elements
        except Exception as e:
            logger.error(f"get_ui_elements failed: {e}")
            return []

    def _parse_xml_source(self, xml_str: str) -> list[dict]:
        """Parse XCUITest XML page source into flat element list."""
        import xml.etree.ElementTree as ET
        elements = []
        try:
            root = ET.fromstring(xml_str)
            self._flatten_xml_tree(root, elements)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        return elements

    def _flatten_xml_tree(self, node, result: list[dict]) -> None:
        """Recursively flatten XML element tree."""
        tag = node.tag or ""

        # Map XCUITest XML tags
        type_map = {
            "XCUIElementTypeButton": "Button",
            "XCUIElementTypeTextField": "TextField",
            "XCUIElementTypeSecureTextField": "SecureTextField",
            "XCUIElementTypeStaticText": "StaticText",
            "XCUIElementTypeImage": "Image",
            "XCUIElementTypeSwitch": "Switch",
            "XCUIElementTypeLink": "Link",
            "XCUIElementTypeSearchField": "SearchField",
            "XCUIElementTypeCell": "Cell",
            "XCUIElementTypeOther": "Other",
            "XCUIElementTypeApplication": "Application",
            "XCUIElementTypeWindow": "Window",
            "XCUIElementTypeNavigationBar": "NavigationBar",
            "XCUIElementTypeTabBar": "TabBar",
            "XCUIElementTypeScrollView": "ScrollView",
            "XCUIElementTypeTable": "Table",
            "XCUIElementTypeTextView": "TextArea",
            "XCUIElementTypeGroup": "Group",
            "XCUIElementTypeCollectionView": "CollectionView",
        }

        mapped = type_map.get(tag, tag.replace("XCUIElementType", ""))
        label = node.get("label", "") or node.get("name", "") or ""
        value = node.get("value", "") or ""
        enabled = node.get("enabled", "true") == "true"
        visible = node.get("visible", "true") == "true"

        # Build frame from XML attributes
        frame = {}
        x = node.get("x")
        y_attr = node.get("y")
        w = node.get("width")
        h = node.get("height")
        if all(v is not None for v in [x, y_attr, w, h]):
            frame = {
                "x": int(x), "y": int(y_attr),
                "width": int(w), "height": int(h),
            }

        if mapped and visible:
            result.append({
                "type": mapped,
                "label": label,
                "value": value,
                "frame": frame,
                "enabled": enabled,
                "visible": visible,
            })

        for child in node:
            self._flatten_xml_tree(child, result)

    def _flatten_xcui_tree(self, node: dict, result: list[dict]) -> None:
        """Recursively flatten XCUITest element tree to flat list."""
        if not isinstance(node, dict):
            return

        el_type = node.get("type", "")
        label = node.get("label", "")
        value = node.get("value", "")
        name = node.get("name", "")
        enabled = node.get("enabled", True)
        visible = node.get("visible", True)
        rect = node.get("rect", {})

        # Build frame dict
        frame = {}
        if rect:
            frame = {
                "x": rect.get("x", 0),
                "y": rect.get("y", 0),
                "width": rect.get("width", 0),
                "height": rect.get("height", 0),
            }

        # Map XCUITest types to our standard format
        type_map = {
            "XCUIElementTypeButton": "Button",
            "XCUIElementTypeTextField": "TextField",
            "XCUIElementTypeSecureTextField": "SecureTextField",
            "XCUIElementTypeStaticText": "StaticText",
            "XCUIElementTypeImage": "Image",
            "XCUIElementTypeSwitch": "Switch",
            "XCUIElementTypeLink": "Link",
            "XCUIElementTypeSearchField": "SearchField",
            "XCUIElementTypeCell": "Cell",
            "XCUIElementTypeNavigationBar": "NavigationBar",
            "XCUIElementTypeTabBar": "TabBar",
            "XCUIElementTypeOther": "Other",
            "XCUIElementTypeApplication": "Application",
            "XCUIElementTypeWindow": "Window",
            "XCUIElementTypeGroup": "Group",
            "XCUIElementTypeScrollView": "ScrollView",
            "XCUIElementTypeTable": "Table",
            "XCUIElementTypeCollectionView": "CollectionView",
            "XCUIElementTypeTextView": "TextArea",
        }

        mapped_type = type_map.get(el_type, el_type.replace("XCUIElementType", ""))

        if mapped_type and visible:
            result.append({
                "type": mapped_type,
                "label": label or name or "",
                "value": value or "",
                "frame": frame,
                "enabled": enabled,
                "visible": visible,
            })

        # Recurse into children
        for child in node.get("children", []):
            self._flatten_xcui_tree(child, result)

    async def launch_app(self, bundle_id: str) -> bool:
        try:
            await self._run_sync(
                lambda: self._driver.execute_script(
                    "mobile: launchApp", {"bundleId": bundle_id}
                )
            )
            return True
        except Exception as e:
            logger.warning(f"launch_app failed: {e}")
            return False

    async def terminate_app(self, bundle_id: str) -> bool:
        try:
            await self._run_sync(
                lambda: self._driver.execute_script(
                    "mobile: terminateApp", {"bundleId": bundle_id}
                )
            )
            return True
        except Exception as e:
            logger.warning(f"terminate_app failed: {e}")
            return False

    async def go_back(self) -> bool:
        try:
            await self._run_sync(lambda: self._driver.back())
            return True
        except Exception:
            return False

    async def press_enter(self) -> bool:
        try:
            await self._run_sync(
                lambda: self._driver.execute_script(
                    "mobile: pressButton", {"name": "return"}
                )
            )
            return True
        except Exception:
            return False

    async def erase_text(self, count: int = 1) -> bool:
        """Not needed with Appium — clear via element.clear()."""
        return True

    async def _detect_ios_version(self, udid: str) -> str | None:
        """Return the iOS version string of the booted device, or None.

        Reads simctl's device-list JSON and matches the booted UDID to
        its runtime identifier (e.g.
        ``com.apple.CoreSimulator.SimRuntime.iOS-18-2`` →  ``"18.2"``).
        Falls back to None so the caller can default explicitly rather
        than guess.
        """
        import json
        import asyncio as _asyncio
        proc = await _asyncio.create_subprocess_exec(
            "xcrun", "simctl", "list", "devices", "--json",
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        try:
            data = json.loads(stdout.decode())
        except json.JSONDecodeError:
            return None
        for runtime_id, devices in (data.get("devices") or {}).items():
            for d in devices:
                if d.get("udid") == udid:
                    # "com.apple.CoreSimulator.SimRuntime.iOS-18-2"
                    # → split on "iOS-" → "18-2" → "18.2"
                    if "iOS-" in runtime_id:
                        ver = runtime_id.rsplit("iOS-", 1)[1].replace("-", ".")
                        return ver
                    return None
        return None

    # Context-like properties for compatibility with engine.
    # ``_udid``, ``_width``, ``_height`` are initialised in __init__
    # to None so the properties never raise AttributeError before
    # start_session() runs. Audit PER-104 #10 noted callers reading
    # these too early on an unstarted client.
    @property
    def ctx(self):
        return self

    @property
    def device(self):
        return self

    @property
    def device_id(self) -> str | None:
        return getattr(self, "_udid", None)

    @property
    def device_width(self) -> int | None:
        return getattr(self, "_width", None)

    @property
    def device_height(self) -> int | None:
        return getattr(self, "_height", None)
