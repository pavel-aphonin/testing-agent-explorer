"""
Vision-based device client for the explorer.
Uses idb for device control (tap, type, screenshot) and LLM vision
for UI element detection. Works with ANY app including React Native.

idb accessibility APIs hang on React Native + iOS 26, but tap/type/screenshot
work fine. So we use screenshots + vision model to identify UI elements.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("explorer.vision_client")

VISION_PROMPT_TEMPLATE = """Analyze this iOS app screenshot. List ALL interactive UI elements you see.

For each element, return a JSON object with:
- "type": one of "Button", "TextField", "SecureTextField", "Switch", "Link", "StaticText", "Image"
- "label": the visible text on/near the element
- "frame": the element position as JSON with keys x, y, width, height (integers, in screen points)
- "enabled": true/false

Return ONLY a JSON array. No explanation, no markdown, no thinking.
Example: [{"type": "Button", "label": "Login", "frame": {"x": 50, "y": 400, "width": 300, "height": 50}, "enabled": true}]

Include ALL visible elements: buttons, text fields, switches, labels, links.
Coordinates should be in logical points (not pixels). The screen is SCREEN_WxSCREEN_H points."""


@dataclass
class TapResult:
    error: str | None = None


class VisionExplorerClient:
    """
    Device client that uses idb for control and LLM vision for UI detection.
    Works with any app, no accessibility tree required.
    """

    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1",
        model: str = "qwen3-vl-32b-instruct-mlx",
    ):
        self.lm_studio_url = lm_studio_url
        self.model = model
        self._idb_client = None
        self._width = 0
        self._height = 0
        self._udid = ""
        self._http = httpx.AsyncClient(timeout=300)  # 5 min for 32B vision model
        self._last_screenshot_b64: str | None = None  # Cache for get_ui_elements + take_screenshot

    async def connect(self, udid: str) -> None:
        """Connect to iOS simulator via idb."""
        from minitap.mobile_use.clients.idb_client import IdbClientWrapper

        self._udid = udid
        self._idb_client = IdbClientWrapper(udid=udid)
        await self._idb_client.init_companion()

        # Get screen dimensions
        from io import BytesIO
        from PIL import Image

        screenshot_bytes = await self._idb_client.screenshot()
        if not screenshot_bytes:
            raise RuntimeError("Failed to take initial screenshot")
        img = Image.open(BytesIO(screenshot_bytes))
        # Convert pixel dimensions to logical points (3x scale for Pro Max)
        self._width = img.width // 3
        self._height = img.height // 3
        logger.info(f"Connected: {self._width}x{self._height} points, UDID: {udid}")

    async def disconnect(self) -> None:
        if self._idb_client:
            await self._idb_client.cleanup()
            self._idb_client = None
        await self._http.aclose()

    async def tap_at(self, x: int, y: int) -> TapResult:
        try:
            ok = await self._idb_client.tap(x, y)
            return TapResult() if ok else TapResult(error="tap returned false")
        except Exception as e:
            return TapResult(error=str(e))

    async def type_text(self, text: str) -> bool:
        try:
            return await self._idb_client.text(text)
        except Exception as e:
            logger.warning(f"type_text failed: {e}")
            return False

    async def take_screenshot(self) -> str:
        """Returns base64-encoded PNG (resized to logical resolution)."""
        data = await self._idb_client.screenshot()
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(data))
        # Resize to logical resolution (3x -> 1x) to reduce size
        logical_size = (img.width // 3, img.height // 3)
        img = img.resize(logical_size, Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def get_ui_elements(self) -> list[dict]:
        """Use LLM vision to detect UI elements from screenshot."""
        print("      [vision] Taking screenshot for LLM...", flush=True)
        screenshot_b64 = await self.take_screenshot()
        # Cache so that _capture_screen doesn't take another screenshot
        self._last_screenshot_b64 = screenshot_b64

        print(f"      [vision] Asking {self.model} to identify elements...", flush=True)
        prompt = VISION_PROMPT_TEMPLATE.replace("SCREEN_W", str(self._width)).replace("SCREEN_H", str(self._height))

        try:
            response = await self._http.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{screenshot_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"      [vision] Raw LLM response ({len(content)} chars): {content[:200]}...", flush=True)

            # Extract JSON from response — may be wrapped in markdown, thinking tags, etc.
            elements = self._parse_elements_from_llm(content)

            print(f"      [vision] LLM found {len(elements)} elements", flush=True)
            return elements

        except Exception as e:
            logger.error(f"Vision LLM failed: {e}")
            print(f"      [vision] ERROR: {e}", flush=True)
            return []

    def _parse_elements_from_llm(self, content: str) -> list[dict]:
        """Robustly extract a JSON array of elements from LLM output."""
        import re

        content = content.strip()

        # Remove <think>...</think> blocks (Qwen3 thinking mode)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Remove markdown code fences
        if "```" in content:
            # Find content between first ``` and last ```
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    content = part
                    break

        # Find the JSON array in the response
        # Look for first [ and matching ]
        start = content.find("[")
        if start == -1:
            logger.warning(f"No JSON array found in LLM response")
            return []

        # Find matching bracket
        depth = 0
        end = start
        for i in range(start, len(content)):
            if content[i] == "[":
                depth += 1
            elif content[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_str = content[start:end]

        try:
            elements = json.loads(json_str)
            if not isinstance(elements, list):
                elements = [elements]
            for el in elements:
                el.setdefault("visible", True)
                el.setdefault("enabled", True)
            return elements
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}, raw: {json_str[:200]}")
            return []

    async def launch_app(self, bundle_id: str) -> bool:
        try:
            return await self._idb_client.launch(bundle_id)
        except Exception as e:
            logger.warning(f"launch failed: {e}")
            return False

    async def terminate_app(self, bundle_id: str) -> bool:
        try:
            return await self._idb_client.terminate(bundle_id)
        except Exception as e:
            logger.warning(f"terminate failed: {e}")
            return False

    async def go_back(self) -> bool:
        # iOS: swipe from left edge
        try:
            return await self._idb_client.swipe(0, self._height // 2, self._width // 2, self._height // 2)
        except Exception:
            return False

    async def press_enter(self) -> bool:
        try:
            return await self._idb_client.key(40)  # HID keycode for Return
        except Exception:
            return False

    async def erase_text(self, count: int = 1) -> bool:
        return True  # Not needed in vision mode

    # Compatibility properties for engine
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
        return self._width * 3  # Return pixel dimensions

    @property
    def device_height(self):
        return self._height * 3
