"""
CDP (Chrome DevTools Protocol) text input for React Native apps.

Connects to Metro debugger via WebSocket and uses React fiber's onChangeText
to set text values directly. This bypasses the HID input limitation where
keyboard events don't trigger React Native's onChangeText.

Only works for dev builds with Metro running. For release builds, fall back
to AXe HID type (which has limited @ and special character support).
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import websockets

logger = logging.getLogger("explorer.cdp")

METRO_DEFAULT_URL = "http://localhost:8081"


class CDPTextInput:
    """Set text in React Native TextInput components via CDP/React fiber."""

    def __init__(self, metro_url: str = METRO_DEFAULT_URL):
        self.metro_url = metro_url
        self._ws_url: str | None = None
        self._ws = None
        self._msg_id = 0

    async def discover(self) -> bool:
        """Discover the WebSocket URL of the running RN app via Metro."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.metro_url}/json")
                if resp.status_code != 200:
                    return False
                pages = resp.json()
                # Pick the first React Native Bridge page
                for page in pages:
                    title = page.get("title", "")
                    if "React Native" in title or "Bridge" in title:
                        self._ws_url = page.get("webSocketDebuggerUrl")
                        if self._ws_url:
                            return True
                return False
        except Exception as e:
            logger.debug(f"CDP discover failed: {e}")
            return False

    async def connect(self) -> bool:
        """Connect to the discovered WebSocket."""
        if not self._ws_url:
            ok = await self.discover()
            if not ok:
                return False
        try:
            self._ws = await websockets.connect(self._ws_url)
            return True
        except Exception as e:
            logger.warning(f"CDP connect failed: {e}")
            return False

    async def disconnect(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _eval(self, expression: str, timeout: float = 5) -> dict:
        """Evaluate JS in the RN app and return the result."""
        if not self._ws:
            raise RuntimeError("Not connected")
        self._msg_id += 1
        msg = {
            "id": self._msg_id,
            "method": "Runtime.evaluate",
            "params": {
                "expression": expression,
                "returnByValue": True,
            },
        }
        await self._ws.send(json.dumps(msg))
        # Wait for response with matching id (CDP may send other events)
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            data = json.loads(raw)
            if data.get("id") == self._msg_id:
                return data
            # Otherwise it's an event we don't care about

    async def _ensure_connected(self) -> bool:
        """Reconnect if WebSocket was closed (e.g. after app relaunch)."""
        if self._ws is None or self._ws.state.value != 1:  # 1 = OPEN
            self._ws = None
            self._ws_url = None  # Force re-discover (URL changes after relaunch)
            return await self.connect()
        return True

    async def set_text_by_test_id(self, test_id: str, text: str) -> bool:
        """
        Find a TextInput by testID and call its onChangeText with the new value.
        Uses React DevTools hook to traverse the fiber tree.
        """
        if not await self._ensure_connected():
            return False

        # Escape text for JS string literal
        text_js = json.dumps(text)

        expression = f"""
        (function() {{
            try {{
                var hook = globalThis.__REACT_DEVTOOLS_GLOBAL_HOOK__;
                if (!hook || !hook.getFiberRoots) return "no_hook";
                var roots = Array.from(hook.getFiberRoots(1));
                if (roots.length === 0) return "no_roots";
                var root = roots[0];

                function findByTestID(fiber, testID) {{
                    if (!fiber) return null;
                    var props = fiber.memoizedProps || {{}};
                    if (props.testID === testID || props.nativeID === testID) return fiber;
                    var child = findByTestID(fiber.child, testID);
                    if (child) return child;
                    return findByTestID(fiber.sibling, testID);
                }}

                var fiber = findByTestID(root.current, {json.dumps(test_id)});
                if (!fiber) return "fiber_not_found";

                var props = fiber.memoizedProps;
                if (!props || typeof props.onChangeText !== "function") {{
                    return "no_onChangeText";
                }}

                props.onChangeText({text_js});
                return "ok";
            }} catch(e) {{
                return "error: " + e.message;
            }}
        }})()
        """

        try:
            result = await self._eval(expression)
            value = result.get("result", {}).get("result", {}).get("value", "")
            if value == "ok":
                return True
            logger.debug(f"set_text_by_test_id({test_id}) -> {value}")
            return False
        except Exception as e:
            logger.debug(f"set_text_by_test_id failed: {e}, will retry once after reconnect")
            # Retry once after forced reconnect
            self._ws = None
            self._ws_url = None
            if await self.connect():
                try:
                    result = await self._eval(expression)
                    value = result.get("result", {}).get("result", {}).get("value", "")
                    return value == "ok"
                except Exception:
                    pass
            return False

    async def set_text_by_label(self, label: str, text: str) -> bool:
        """Find a TextInput by accessibilityLabel and set its text."""
        if not await self._ensure_connected():
            return False

        text_js = json.dumps(text)
        label_js = json.dumps(label)

        expression = f"""
        (function() {{
            try {{
                var hook = globalThis.__REACT_DEVTOOLS_GLOBAL_HOOK__;
                if (!hook || !hook.getFiberRoots) return "no_hook";
                var roots = Array.from(hook.getFiberRoots(1));
                if (roots.length === 0) return "no_roots";
                var root = roots[0];

                function find(fiber, label) {{
                    if (!fiber) return null;
                    var props = fiber.memoizedProps || {{}};
                    if (props.accessibilityLabel === label && typeof props.onChangeText === "function") return fiber;
                    if (props.placeholder === label && typeof props.onChangeText === "function") return fiber;
                    var child = find(fiber.child, label);
                    if (child) return child;
                    return find(fiber.sibling, label);
                }}

                var fiber = find(root.current, {label_js});
                if (!fiber) return "fiber_not_found";

                fiber.memoizedProps.onChangeText({text_js});
                return "ok";
            }} catch(e) {{
                return "error: " + e.message;
            }}
        }})()
        """

        try:
            result = await self._eval(expression)
            value = result.get("result", {}).get("result", {}).get("value", "")
            return value == "ok"
        except Exception as e:
            logger.warning(f"set_text_by_label failed: {e}")
            return False

    async def is_available(self) -> bool:
        """Quick check if Metro CDP is reachable."""
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{self.metro_url}/json")
                return resp.status_code == 200
        except Exception:
            return False
