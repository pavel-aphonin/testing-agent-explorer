"""
AXe CLI-based device client for the explorer.
Uses AXe for EVERYTHING: accessibility tree, tap, type, swipe, screenshot.
NO idb dependency. Each AXe call is a separate subprocess — no hanging gRPC.

AXe (cameroncooke/axe) works with React Native on iOS 26 where idb/Appium fail.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger("explorer.axe_client")


class SimulatorDownError(RuntimeError):
    """Raised when AXe / simctl reports the simulator is no longer
    booted. Separated from generic AXe failures (parse errors,
    accidental crashes inside describe-ui) so the worker can react
    differently — repeated transient AXe glitches are worth a retry,
    but a dead simulator means the run is dead too and we should
    fail it fast instead of looping ``describe-ui`` for 18 minutes
    (PER-107). The message string is interchangeable; the type is
    the signal."""


# PER-107: substrings in stderr that mean «the simulator is gone».
# Matched case-insensitively against the captured stderr — broad
# enough to catch both axe's variant («Device * is not booted») and
# simctl's variant («Unable to find a device with state: booted /
# Booted»). Kept narrow on purpose: a generic «failed» does NOT count
# as a dead simulator, only these explicit booted-state messages.
_SIMULATOR_DOWN_MARKERS: tuple[str, ...] = (
    "is not booted",
    "unable to find a device with state",
    "no devices are booted",
    "current state: shutdown",
)


def _is_simulator_down(stderr: str) -> bool:
    if not stderr:
        return False
    low = stderr.lower()
    return any(m in low for m in _SIMULATOR_DOWN_MARKERS)


# On iOS 27, AXe reports the system keyboard as plain Group/Button nodes
# in separate top-level windows of the Application — there is no
# ``Keyboard`` type any more, so the analyzer's KEYBOARD_TYPES filter and
# ``wait_for_keyboard`` silently stopped working. Testbed run baseline-1
# tapped keys («q», «Passwords», «dictation») as app buttons, and the
# keyboard showing/hiding forked the login screen into 5 nodes.
# A top-level window containing one of these ids is the keyboard.
_KEYBOARD_WINDOW_IDS = frozenset({"SystemInputAssistantView", "inputView"})

# Type given to every element inside a keyboard window. It is part of
# KEYBOARD_TYPES, so the analyzer and the screen fingerprint skip it.
_KEYBOARD_ELEMENT_TYPE = "KeyboardKey"


def _contains_id(node: dict, ids: frozenset[str]) -> bool:
    if node.get("AXUniqueId") in ids:
        return True
    return any(
        _contains_id(child, ids)
        for child in node.get("children", [])
        if isinstance(child, dict)
    )


# On iOS 27, AXe reports UITableView / SwiftUI List rows as plain
# ``Group`` children of a ``ScrollArea`` — no ``Cell`` type — and the
# analyzer skips every Group, so testbed run 2 saw no account rows to
# tap. A ScrollArea child Group holding text and no controls of its own
# is a tappable row. Rows with their own buttons (NavigationLink rows,
# rows with switches) stay as they are: those controls are the actions.
_ROW_CONTROL_TYPES = frozenset({"Button", "TextField", "Switch", "Link", "Slider"})
# Disclosure indicator UIKit puts inside a row; decoration, not a control.
_ROW_ACCESSORY_LABELS = frozenset({"chevron"})


# Modal presentation on iOS 27: the app window holds the presenting
# screen and, as a LATER sibling container, the modal on top — a SwiftUI
# sheet, or a ``Sheet`` element for alerts. The AX tree still lists the
# controls underneath; testbed run 4 tapped those three times under the
# «Удалить аккаунт?» alert, got no change, and gave up before trying
# «Отмена»/«Удалить». Only the topmost container's controls are actions.
_MODAL_CONTROL_TYPES = frozenset({
    "Button", "TextField", "SecureTextField", "Switch", "CheckBox", "Link",
})


def _obscured_children(window: dict) -> set[int]:
    """Indices of the window's child containers covered by a later one."""
    children = window.get("children", [])
    with_controls = [
        i for i, child in enumerate(children)
        if isinstance(child, dict) and any(
            d.get("type") in _MODAL_CONTROL_TYPES for d in _descendants(child)
        )
    ]
    return set(with_controls[:-1])


def _reading_order(element: dict) -> tuple[float, float]:
    frame = element.get("frame") or {}
    return frame.get("y", 0), frame.get("x", 0)


def _descendants(node: dict):
    for child in node.get("children", []):
        if isinstance(child, dict):
            yield child
            yield from _descendants(child)


def _is_accessory(node: dict) -> bool:
    return node.get("type") == "Button" and node.get("AXLabel") in _ROW_ACCESSORY_LABELS


def _has_one_accessory(node: dict) -> bool:
    return sum(_is_accessory(d) for d in _descendants(node)) == 1


def _row_label(node: dict) -> str | None:
    """Label for a list row: None if ``node`` is not a tappable row,
    else its own label or the first text inside it."""
    if node.get("type") != "Group":
        return None
    texts = []
    for d in _descendants(node):
        if d.get("type") in _ROW_CONTROL_TYPES and not _is_accessory(d):
            return None
        if d.get("type") == "StaticText" and d.get("AXLabel"):
            texts.append(d["AXLabel"])
    if not texts:
        return None
    return node.get("AXLabel") or texts[0]


# PER-111 v2: a short, alphanumeric slug used inside synthetic
# element ids when AXUniqueId is missing. We transliterate Cyrillic
# (Готово → gotovo) so the generated id stays purely ASCII — keeps
# the LLM's response_format enum compact and predictable. Trim hard
# at 24 chars to bound the per-element enum entry size.
_SLUG_TRIM_LEN = 24
_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")

# unicodedata.normalize("NFKD") strips Latin diacritics but not
# Cyrillic, so we explicitly transliterate the Russian alphabet —
# every screen we run goal-nodes against is bilingual (Russian app
# UI + English worker logs), and a slug like "" for the Готово
# button is useless. Lowercase only; the slug pipeline downstream
# already lower-cases the input.
_CYRILLIC_TRANSLIT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
}


def _slug(text: str) -> str:
    """Lower-case ASCII-only slug for use inside synthetic ids."""
    if not text:
        return ""
    lowered = text.lower()
    transliterated = "".join(_CYRILLIC_TRANSLIT.get(c, c) for c in lowered)
    normalized = unicodedata.normalize("NFKD", transliterated)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    s = _SLUG_INVALID_RE.sub("_", ascii_text).strip("_")
    return s[:_SLUG_TRIM_LEN] if len(s) > _SLUG_TRIM_LEN else s


def screen_fingerprint(elements: list[dict]) -> str:
    """Stable SHA256 hash of an accessibility-tree snapshot.

    PER-127: used by ``_wait_for_screen_stable`` to detect when two
    consecutive snapshots describe the same screen. The hash covers
    the bits that change between screens (kind/label + position) and
    deliberately ignores volatile ones (focus state, AXValue —
    text-field contents change as the user types but the screen is
    still "the same screen").

    Elements are projected to ``(kind, label, x, y, w, h)`` tuples,
    sorted to make the result independent of AXe's traversal order,
    joined into one string, and hashed.
    """
    parts: list[str] = []
    for el in elements or []:
        if not isinstance(el, dict):
            continue
        kind = (el.get("kind") or el.get("type") or "") or ""
        label = (el.get("label") or "") or ""
        frame = el.get("frame") or {}
        try:
            x = int(frame.get("x", 0) or 0)
            y = int(frame.get("y", 0) or 0)
            w = int(frame.get("width", 0) or 0)
            h = int(frame.get("height", 0) or 0)
        except (TypeError, ValueError):
            x, y, w, h = 0, 0, 0, 0
        parts.append(f"{kind}|{label}|{x}|{y}|{w}|{h}")
    # Sort so two snapshots with the same elements in different AXe
    # traversal order still match. SHA256 is overkill for what is
    # effectively a screen equality test, but it's the same primitive
    # we use elsewhere (PER-85 screen-match) and the cost is trivial
    # compared to the AXe describe-ui call we just made.
    parts.sort()
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def screen_fingerprint_structural(elements: list[dict]) -> str:
    """PER-161: label-free structural fingerprint of a screen.

    Like :func:`screen_fingerprint` but drops every text label from
    the projection — keeps only kind + frame. Two screens that
    differ ONLY in labels (e.g. balance card showing
    «100 000 ₽» vs «200 000 ₽», a list of transactions with rolling
    timestamps, an inbox with new message counts) hash to the same
    structural id.

    Why a second function instead of always-strip in the original:

    * ``screen_fingerprint`` is the right "did the screen change"
      signal for the in-step `wait` / `OK-but-no-change` markers —
      we WANT it to notice that a counter ticked.
    * ``screen_fingerprint_structural`` is the right "is this the
      same logical screen" signal for visited_actions / plateau
      tracking — we don't want every counter-tick to look like a
      new screen and reset the agent's memory of what it's tried.

    Both stay stable across AXe traversal order (sorted) and across
    AXValue changes (already ignored).
    """
    parts: list[str] = []
    for el in elements or []:
        if not isinstance(el, dict):
            continue
        kind = (el.get("kind") or el.get("type") or "") or ""
        frame = el.get("frame") or {}
        try:
            x = int(frame.get("x", 0) or 0)
            y = int(frame.get("y", 0) or 0)
            w = int(frame.get("width", 0) or 0)
            h = int(frame.get("height", 0) or 0)
        except (TypeError, ValueError):
            x, y, w, h = 0, 0, 0, 0
        parts.append(f"{kind}|{x}|{y}|{w}|{h}")
    parts.sort()
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def has_loading_indicator(
    elements: list[dict], keywords: list[str] | None,
) -> bool:
    """True when any element's label or value contains one of the
    workspace-configured loading keywords (case-insensitive).

    Empty ``keywords`` list disables the check entirely — the screen
    is considered loaded as long as the fingerprint has converged.
    """
    if not keywords:
        return False
    needles = [k.lower() for k in keywords if k]
    if not needles:
        return False
    for el in elements or []:
        if not isinstance(el, dict):
            continue
        haystacks = (
            (el.get("label") or "").lower(),
            (el.get("value") or "").lower(),
        )
        for h in haystacks:
            if not h:
                continue
            for n in needles:
                if n in h:
                    return True
    return False


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


# PER-891: on Xcode 27, AXe 1.8.0 sends a tap before the simulator's
# dtuhidd has activated its touch service, and the event is silently
# dropped (cameroncooke/AXe#71). AXe waits this long before the first
# event; 600 ms was the smallest value that made taps land 4/4 on
# iOS 27.2 (300 ms: 0/4). Drop once AXe ships the upstream idb fix.
_AXE_HID_STABILIZATION_MS = "600"


# Verified text entry (see AXeExplorerClient.set_text_in_field).
_TYPE_ATTEMPTS = 3
# Extra wait after the keyboard is up: when moving between fields the
# keyboard is already visible, but focus has not moved yet.
_FOCUS_SETTLE_S = 0.8
# Scrolling elements into view (see AXeExplorerClient.scroll_into_view).
# Margins keep taps off the navigation bar and the tab bar; a slow swipe
# avoids flinging the content past the target.
_SCROLL_TOP_MARGIN = 120
_SCROLL_BOTTOM_MARGIN = 110
_SCROLL_BOTTOM_MARGIN_NO_TAB_BAR = 40
_MAX_SCROLL_SWIPES = 14
# Horizontal swipes keep this far from the edges: an edge swipe means «back».
_CAROUSEL_EDGE_PT = 40
# Sideways swipes are slow: at 0.6 s a carousel flings on, 250 pt for a
# 100 pt swipe (testbed), and the card bounced past the screen both ways.
# At 2 s the content moves as far as the finger.
_CAROUSEL_SWIPE_DURATION_S = "2.0"
_SCROLL_SWIPE_DURATION_S = "0.6"
_SCROLL_SETTLE_S = 0.8
_MIN_SCROLL_SWIPE_PT = 60
# A swipe that moves the element less than this hit the end of the content.
_SCROLL_NO_MOVE_PT = 4
_TAB_BAR_TYPES = frozenset({"RadioButton", "Tab"})

# Navigation-bar back buttons (see AXeExplorerClient.go_back): UIKit and
# SwiftUI use the BackButton id with the previous screen's title as label.
_BACK_IDS = frozenset({"BackButton", "backButton"})
_BACK_LABELS = frozenset({"Назад", "Back"})
_BACK_BUTTON_MAX_Y = 200

# Verified switch taps (see AXeExplorerClient.tap_switch). A permission
# prompt can take ~1 s to replace the app in the AX tree.
_SWITCH_ATTEMPTS = 3
_SWITCH_SETTLE_S = 1.2
# USB HID usage ids for ``axe key`` / ``axe key-combo``.
_HID_LEFT_GUI = "227"
_HID_KEY_A = "4"
_HID_KEY_BACKSPACE = "42"
_HID_KEY_RETURN = "40"


# Simulator degradation over long runs (testbed runs 1–3): every axe
# input command opens a fresh HID connection to the simulator's dtuhidd.
# dtuhidd cancels each connection's virtual services but keeps notifying
# later connections about them — the «No matching virtual service» noise
# per connection grew linearly (27 → 717 lines in 11 min), and after
# ~200 connections taps were dropped and describe-ui timed out until the
# simulator was rebooted. Restarting only dtuhidd resets this (732 → 0
# lines per tap) and keeps the app running on the same screen, so do it
# every _DTUHIDD_RESTART_EVERY input commands.
_HID_SUBCOMMANDS = frozenset({
    "tap", "type", "swipe", "drag", "button", "key", "key-sequence",
    "key-combo", "touch", "gesture", "slider", "batch",
})
_DTUHIDD_RESTART_EVERY = 80
_DTUHIDD_LABEL = "system/com.apple.coredevice.dtuhidd"
_hid_connections: dict[str, int] = {}


async def _note_hid_connection(args: list[str]) -> None:
    try:
        udid = args[args.index("--udid") + 1]
    except (ValueError, IndexError):
        return
    count = _hid_connections.get(udid, 0) + 1
    _hid_connections[udid] = count
    if count % _DTUHIDD_RESTART_EVERY:
        return
    logger.info("Restarting dtuhidd on %s after %d input commands", udid, count)
    proc = await asyncio.create_subprocess_exec(
        "xcrun", "simctl", "spawn", udid, "launchctl", "kickstart", "-k", _DTUHIDD_LABEL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=15)
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("dtuhidd restart timed out on %s", udid)
        return
    await asyncio.sleep(2.0)


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
    env = None
    if args[0] == AXE:
        env = dict(os.environ)
        env.setdefault("AXE_HID_STABILIZATION_MS", _AXE_HID_STABILIZATION_MS)
        if len(args) > 1 and args[1] in _HID_SUBCOMMANDS:
            await _note_hid_connection(args)
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
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
        # Whether the last tree read showed an unobscured tab bar (see
        # _bottom_margin). Assume one until the first read.
        self._tab_bar_visible = True
        # The last scroll_into_view hit the end of the content going down.
        self.scroll_stuck = False

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

    async def tap_switch(self, test_id: str) -> TapResult:
        """Flip a switch and verify it flipped.

        Switches ignore coordinate taps; ``axe tap --id`` reaches them with
        a physical touch, but that touch is sometimes silently dropped
        (testbed run 7: «success», toggle unchanged, no permission
        prompt). Read the value back; the switch counts as flipped when
        its value changed or it left the tree (a system prompt took
        over). Retry with a post-touch delay, which landed 100% in the
        tap-style measurements.
        """
        before = await self._switch_value(test_id)
        for attempt in range(1, _SWITCH_ATTEMPTS + 1):
            args = [AXE, "tap", "--id", test_id, "--udid", self._udid]
            if attempt > 1:
                args[2:2] = ["--tap-style", "physical", "--post-delay", "0.5"]
            code, _, err = await _run(args)
            if code != 0:
                return TapResult(error=err.strip()[:200])
            await asyncio.sleep(_SWITCH_SETTLE_S)
            after = await self._switch_value(test_id)
            if after != before:
                return TapResult()
            logger.warning(
                "tap_switch: %r did not flip (attempt %d/%d)",
                test_id, attempt, _SWITCH_ATTEMPTS,
            )
        return TapResult(error=f"switch {test_id!r} did not flip")

    async def _switch_value(self, test_id: str) -> str | None:
        """Current value of the switch, or None when it is not in the tree."""
        for el in await self.get_ui_elements():
            if el.get("test_id") == test_id:
                return el.get("value")
        return None

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
        """Type text via CDP (RN dev) or AXe HID (fallback).

        Used by the older ``input`` dispatch path and by free-form
        ``enter_text`` action. The CDP shortcut writes the value
        directly into the React tree — fast but **bypasses native
        keyboard events** (no onChange). For tests that need to
        validate the real user keyboard path use
        :meth:`type_text_via_hid` instead (see PER-160).
        """
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

    async def type_text_via_hid(self, text: str) -> bool:
        """PER-160: keyboard-typing path that never touches CDP.

        Always goes through AXe HID ``type`` — same hardware keyboard
        events a real user keystroke produces. Triggers React
        ``onChange``/``onInput`` and any native ``UITextField``
        delegate callbacks. Use this whenever the test needs to
        validate the **user-visible** keyboard path: ``input`` action
        defaults here so the agent finds keyboard-specific bugs
        (overlay covering field, broken handler, missing IME) that
        CDP-injection silently hides.
        """
        if not text:
            return True
        code, _, err = await _run(
            [AXE, "type", text, "--udid", self._udid], timeout=30
        )
        if code != 0:
            logger.warning("type_text_via_hid failed: %s", err[:120])
            return False
        return True

    async def wait_for_keyboard(self, timeout_ms: int = 2000) -> bool:
        """PER-160: poll the AX-tree until an on-screen keyboard appears.

        Returns True as soon as any element with ``type`` containing
        ``Keyboard`` shows up. False on timeout. Used right after
        tapping an input field to make sure the OS keyboard is up
        before we start sending HID keystrokes — otherwise the first
        few characters land on whatever was focused before.
        """
        import time as _time
        deadline = _time.monotonic() + (timeout_ms / 1000.0)
        poll_ms = 100
        while _time.monotonic() < deadline:
            elems = await self.get_ui_elements()
            for el in elems:
                kind = (el.get("type") or el.get("kind") or "")
                if "Keyboard" in kind:
                    return True
            await asyncio.sleep(poll_ms / 1000.0)
        return False

    async def tap_field_and_type_via_keyboard(
        self,
        test_id: str | None,
        label: str | None,
        text: str,
        wait_keyboard_ms: int = 5000,
    ) -> tuple[bool, str | None]:
        """PER-160: high-level «как живой пользователь» input.

        1. Tap the input field (by test_id, else by label) so iOS
           focuses it and brings up the keyboard.
        2. Wait until ``XCUIElementTypeKeyboard`` is in the AX-tree.
        3. Type the text via AXe HID — native key events.

        Returns ``(ok, reason)``. Failures get a specific reason
        (``"tap failed"`` / ``"keyboard did not appear"`` /
        ``"type_text_via_hid returned False"``) so the dispatcher can
        propagate it into the LLM's history without guessing.
        """
        logger.info(
            "[keyboard] tap_field_and_type_via_keyboard test_id=%r label=%r text_len=%d",
            test_id, label, len(text),
        )
        if test_id:
            tap = await self.tap_by_id(test_id)
        elif label:
            tap = await self.tap_by_label(label)
        else:
            return False, "no test_id or label to tap"
        if not getattr(tap, "ok", True):
            logger.warning(
                "[keyboard] tap failed: %s",
                getattr(tap, "error", None) or "unknown",
            )
            return False, f"tap failed: {getattr(tap, 'error', None) or 'unknown'}"
        # Cache focus hint so anything that falls back to type_text
        # (e.g. legacy code) still knows where it lives.
        if test_id:
            self._last_focused_test_id = test_id
        if label:
            self._last_focused_label = label
        # PER-160 follow-up: keyboard detection in iOS 26 AXe output
        # is inconsistent — when the keyboard is up it sometimes
        # doesn't show as a separate ``Keyboard``-typed element in
        # describe-ui, and waiting for it would needlessly fail every
        # tap that actually worked. So we wait UP TO ``wait_keyboard_ms``
        # for an explicit signal, but if we don't see one we still
        # type after a short fixed delay (animation buffer). Worst
        # case the HID keystrokes land on whatever's focused, which
        # was just the field we tapped — that's the right field
        # anyway. Failure mode here is no worse than the old
        # CDP-bypass path it replaces.
        ready = await self.wait_for_keyboard(wait_keyboard_ms)
        if not ready:
            logger.info(
                "[keyboard] no explicit Keyboard element in AX-tree after "
                "%d ms — typing anyway with 500ms animation buffer",
                wait_keyboard_ms,
            )
            await asyncio.sleep(0.5)
        else:
            logger.info("[keyboard] detected via AX-tree")
        ok = await self.type_text_via_hid(text)
        if not ok:
            logger.warning("[keyboard] type_text_via_hid returned False")
        else:
            logger.info("[keyboard] type_text_via_hid OK (%d chars)", len(text))
        return ok, None if ok else "type_text_via_hid returned False"

    async def set_text_in_field(
        self, test_id: str | None, label: str | None, text: str
    ) -> bool:
        """High-level: set text in a specific field. Uses CDP if available.

        PER-160: This is the **bypass** path — fast, but writes value
        directly through React/CDP without surfacing the keyboard.
        Used only when the caller explicitly opts in via
        ``action_args.bypass_keyboard=True``. Default ``input``
        dispatch now routes through
        :meth:`tap_field_and_type_via_keyboard` instead.
        """
        if self._cdp_available and self._cdp:
            if test_id:
                ok = await self._cdp.set_text_by_test_id(test_id, text)
                if ok:
                    return True
            if label:
                ok = await self._cdp.set_text_by_label(label, text)
                if ok:
                    return True
        if not test_id:
            return await self.type_text(text)
        if not text:
            # An empty field reports its placeholder as the AX value, so
            # there is nothing to verify — just clear it.
            await self.tap_by_id(test_id)
            await self.wait_for_keyboard(5000)
            await asyncio.sleep(_FOCUS_SETTLE_S)
            await self._clear_focused_field()
            return True
        # Fallback: tap + AXe type, verified. AX trees carry no focus
        # flag, and a fixed 0.3 s sleep lost the race in testbed run
        # baseline-1: the password went into the still-focused email
        # field («test@test.compassword123»). So read the field back and
        # retry until it holds exactly ``text``.
        for attempt in range(1, _TYPE_ATTEMPTS + 1):
            await self.tap_by_id(test_id)
            await self.wait_for_keyboard(5000)
            await asyncio.sleep(_FOCUS_SETTLE_S)
            await self._clear_focused_field()
            await self.type_text_via_hid(text)
            await asyncio.sleep(0.3)
            if await self._field_holds(test_id, text):
                return True
            logger.warning(
                "set_text_in_field: %r does not hold the typed text "
                "(attempt %d/%d)", test_id, attempt, _TYPE_ATTEMPTS,
            )
        return False

    async def _clear_focused_field(self) -> None:
        """Select all (⌘A) and delete in the focused field."""
        await _run([
            AXE, "key-combo", "--modifiers", _HID_LEFT_GUI,
            "--key", _HID_KEY_A, "--udid", self._udid,
        ])
        await _run([AXE, "key", _HID_KEY_BACKSPACE, "--udid", self._udid])

    async def _field_holds(self, test_id: str, text: str) -> bool:
        """True iff the field's AX value equals ``text``. Secure fields
        expose one bullet per character, so only the length is checked."""
        for el in await self.get_ui_elements():
            if el.get("test_id") != test_id:
                continue
            value = el.get("value") or ""
            if value and set(value) == {"•"}:
                return len(value) == len(text)
            return value == text
        return False

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

    async def take_screenshot(self, max_dim: int | None = None) -> str:
        """Screenshot via simctl, return base64 PNG.

        PER-163 retry: ``max_dim`` controls the resize ceiling:

        * ``None`` (legacy) → downscale to logical-points
          ``(self._width, self._height)``. Cheap, loses detail on
          small UI controls (PIN keypad digits, app-icon grids).
        * ``int`` → downscale only if larger than ``max_dim`` on the
          longer side, preserve aspect ratio. Lets vision-precision
          models see near-native pixel detail (iPhone 17 Pro Max is
          1320×2868 — ``max_dim=1920`` keeps full height and 660px
          width, well above the 440 logical points and far below the
          context-blowing native bytes).

        Worker reads the per-model passport ``screenshot_max_dim``
        and forwards it; controller stays single-knob.

        PER-163 retry #2: when the caller does not pass ``max_dim``
        explicitly, the env var ``TA_SCREENSHOT_MAX_DIM`` provides a
        machine-wide override. That covers the legacy
        random / explore / llm-loop callers (engine.py, loop.py,
        llm_loop.py) that don't know about model passports —
        operator sets the env var when running them in
        vision-precision mode. Explicit ``max_dim`` argument always
        wins over the env var, which always wins over the legacy
        logical-points resize.
        """
        import os as _os
        import tempfile
        if max_dim is None:
            env_dim = _os.environ.get("TA_SCREENSHOT_MAX_DIM", "").strip()
            if env_dim.isdigit() and int(env_dim) > 0:
                max_dim = int(env_dim)
        with tempfile.NamedTemporaryFile(
            prefix="axe_shot_", suffix=".png", delete=False
        ) as tf:
            path = tf.name
        try:
            code, _, stderr = await _run([
                "xcrun", "simctl", "io", self._udid, "screenshot", path
            ])
            if code != 0:
                # PER-107: distinguish dead simulator from a transient
                # simctl glitch (it's flaky under load). Worker outer
                # loop catches SimulatorDownError and fails the run
                # fast; empty-string return keeps the legacy soft path
                # for everything else (screenshot loss is non-fatal —
                # next step retries on its own).
                if _is_simulator_down(stderr):
                    raise SimulatorDownError(
                        f"simulator {self._udid} is no longer booted "
                        f"(simctl io screenshot: {stderr[:160]})"
                    )
                return ""

            from io import BytesIO
            from PIL import Image
            img = Image.open(path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            if max_dim is None:
                img = img.resize((self._width, self._height), Image.LANCZOS)
            else:
                longest = max(img.width, img.height)
                if longest > max_dim:
                    scale = max_dim / float(longest)
                    new_w = int(round(img.width * scale))
                    new_h = int(round(img.height * scale))
                    img = img.resize((new_w, new_h), Image.LANCZOS)
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
        """Get accessibility tree via AXe CLI.

        PER-107: a dead simulator (manual shutdown, OS sleep, memory
        pressure under heavy llama load) makes ``axe describe-ui``
        return stderr containing «is not booted» — we used to log a
        warning, return [], and retry forever from the outer loop.
        Now we surface ``SimulatorDownError`` so the worker can fail
        the run quickly instead of looping for 18 minutes.
        """
        code, stdout, stderr = await _run([
            AXE, "describe-ui", "--udid", self._udid
        ])
        if code != 0:
            if _is_simulator_down(stderr):
                raise SimulatorDownError(
                    f"simulator {self._udid} is no longer booted "
                    f"(axe describe-ui: {stderr[:160]})"
                )
            logger.warning(f"AXe describe-ui failed: {stderr[:200]}")
            return []

        try:
            raw = json.loads(stdout)
            elements = []
            self._flatten(raw if isinstance(raw, list) else [raw], elements)
            self._tab_bar_visible = any(
                el["type"] in _TAB_BAR_TYPES and not el.get("obscured")
                and self.is_on_screen(el.get("frame"))
                for el in elements
            )
            return elements
        except json.JSONDecodeError as e:
            logger.warning(f"AXe JSON parse error: {e}")
            return []

    def _flatten(
        self,
        nodes: list,
        result: list[dict],
        in_keyboard: bool = False,
        parent_type: str = "",
        in_row: bool = False,
        obscured: bool = False,
        in_alert: bool = False,
    ) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if in_row and _is_accessory(node):
                continue
            # iOS 27 nests an unlabeled CheckBox inside each toggle's
            # CheckBox; it is the same control, tapped twice otherwise.
            if parent_type == "CheckBox" and node.get("type") == "CheckBox":
                continue
            # Keyboard windows are direct children of the Application.
            node_in_keyboard = in_keyboard or (
                parent_type == "Application"
                and _contains_id(node, _KEYBOARD_WINDOW_IDS)
            )
            row_label = None
            # An alert (``Sheet``) lays its title and message out in a
            # ScrollArea too; that text block is not a list row.
            # A disclosure chevron marks a row wherever it sits: the same
            # table sometimes comes without its ScrollArea (testbed run 15:
            # three accounts seen as plain text, the screen a dead end).
            if not node_in_keyboard and not in_alert and (
                parent_type == "ScrollArea" or (not in_row and _has_one_accessory(node))
            ):
                row_label = _row_label(node)
            test_id = node.get("AXUniqueId", "")
            frame = node.get("frame", {}) or {}
            label = node.get("AXLabel", "") or ""
            if node_in_keyboard:
                kind = _KEYBOARD_ELEMENT_TYPE
            elif row_label is not None:
                kind = "Cell"
                label = row_label
            else:
                kind = node.get("type", "") or "element"
            value = node.get("AXValue", "")
            if value is not None and not isinstance(value, str):
                # Sliders report floats (iOS Settings → Fonts: 0.555…);
                # ElementSnapshot.value is a string.
                value = str(value)
            # PER-111 v2: every element needs a stable id so the LLM
            # can refer to it from the constrained enum and the
            # worker can resolve it back to a frame. Priority is:
            #   1. AXUniqueId (developer-set accessibilityIdentifier)
            #   2. <kind>_<slug(label)>_<y>  (semantic + position)
            #   3. <kind>_<x>_<y>            (last resort)
            # The Y coordinate disambiguates duplicate labels on the
            # same screen (two "Готово" buttons in a stack) without
            # making the id meaningless. ``slug`` trims to 24 chars
            # and keeps only [a-z0-9_], so the id stays short enough
            # for llama-server to enumerate efficiently.
            try:
                fx = int(frame.get("x", 0) or 0)
                fy = int(frame.get("y", 0) or 0)
            except (TypeError, ValueError):
                fx, fy = 0, 0
            if test_id:
                synthetic_id = test_id
            elif label:
                synthetic_id = f"{kind}_{_slug(label)}_{fy}"
            else:
                synthetic_id = f"{kind}_{fx}_{fy}"
            result.append({
                "id": synthetic_id,
                "type": kind,
                "kind": kind,
                "label": label,
                "value": value,
                "test_id": test_id,
                "frame": frame,
                "enabled": node.get("enabled", True),
                "visible": True,
                "obscured": obscured,
            })
            # App windows are the Application's non-keyboard children.
            covered = set()
            if parent_type == "Application" and not node_in_keyboard:
                covered = _obscured_children(node)
            for i, child in enumerate(node.get("children", [])):
                self._flatten(
                    [child], result, node_in_keyboard, node.get("type", ""),
                    in_row or row_label is not None,
                    obscured or i in covered,
                    in_alert or node.get("type") == "Sheet",
                )

    # ─────────────────────────────────── PER-111 v2 action dispatchers
    # The 8 reference actions seeded by migration 20260518_per111_v2
    # — tap / input / back / type already exist above; below are the
    # ones the v2 dispatcher needs. Direction-based gestures use AXe's
    # ``gesture`` preset (full-screen scroll); long_press uses the
    # touch primitive with a delay.

    async def swipe(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Custom coordinate swipe. Used by legacy callers in llm_loop
        and by the scenario dispatcher when explicit coords are given.
        For direction-based swipes prefer :meth:`swipe_direction`."""
        code, _, _ = await _run([
            AXE, "swipe",
            "--start-x", str(int(x1)), "--start-y", str(int(y1)),
            "--end-x", str(int(x2)), "--end-y", str(int(y2)),
            "--udid", self._udid,
        ])
        return code == 0

    async def swipe_direction(self, direction: str) -> bool:
        """PER-111 v2 swipe.direction handler — maps to AXe gesture
        presets. iOS gesture is a one-finger drag, so we use the
        ``swipe-from-*-edge`` presets for edge gestures and
        ``scroll-*`` for the in-screen ones."""
        preset = {
            "up": "scroll-up",
            "down": "scroll-down",
            "left": "scroll-left",
            "right": "scroll-right",
        }.get((direction or "").lower())
        if not preset:
            logger.warning("swipe_direction: unknown direction %r", direction)
            return False
        code, _, _ = await _run([
            AXE, "gesture", preset, "--udid", self._udid,
        ])
        return code == 0

    async def scroll(self, direction: str) -> bool:
        """PER-111 v2 scroll.direction handler — same AXe primitive as
        swipe_direction on iOS. Distinguishing scroll from swipe at the
        action level is useful for the LLM's reasoning, not for the
        gesture engine."""
        return await self.swipe_direction(direction)

    async def long_press(
        self, x: int | None = None, y: int | None = None,
        duration_ms: int = 800,
    ) -> bool:
        """PER-111 v2 long_press handler. With explicit coords —
        touches the point for ``duration_ms``. Without coords, falls
        back to the screen centre (e.g. for a context-menu invocation
        on the current focus); callers that target a specific element
        should pass its frame centre."""
        try:
            ms = max(100, min(int(duration_ms), 5000))
        except (TypeError, ValueError):
            ms = 800
        if x is None or y is None:
            x = self._width // 2
            y = self._height // 2
        delay = ms / 1000.0
        code, _, _ = await _run([
            AXE, "touch",
            "-x", str(int(x)), "-y", str(int(y)),
            "--down", "--up",
            "--delay", f"{delay:.3f}",
            "--udid", self._udid,
        ])
        return code == 0

    async def wait_ms(self, ms: int) -> bool:
        """PER-111 v2 wait.ms handler — plain sleep, no AXe call. Useful
        when the LLM needs the UI to settle before the next step (e.g.
        after kicking off a network request)."""
        try:
            seconds = max(0.1, min(int(ms) / 1000.0, 60.0))
        except (TypeError, ValueError):
            seconds = 0.5
        await asyncio.sleep(seconds)
        return True

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

    def is_on_screen(self, frame: dict | None) -> bool:
        """Center of ``frame`` inside the screen. Navigation-bar and tab-bar
        items sit in the margins and are on screen (testbed run 11: treating
        the margins as off screen skipped every «Назад», «Отмена» and
        «Сохранить»)."""
        if not frame or not self._height:
            return True
        cx = frame.get("x", 0) + frame.get("width", 0) / 2
        cy = frame.get("y", 0) + frame.get("height", 0) / 2
        return 0 <= cx <= self._width and 0 <= cy <= self._height

    def needs_scroll(self, element_type: str | None, frame: dict | None) -> bool:
        """True when tapping ``frame`` as captured would miss: off screen,
        or content under the tab bar."""
        if not frame or not self._height:
            return False
        if not self.is_on_screen(frame):
            return True
        cy = frame.get("y", 0) + frame.get("height", 0) / 2
        return cy > self._height - self._bottom_margin() and element_type not in _TAB_BAR_TYPES

    def _bottom_margin(self) -> int:
        """Room to keep free at the bottom: the tab bar when one is visible
        (as of the last tree read), else just the home indicator. Without a
        tab bar — a sheet or alert covering it — bottom items are fine
        (aweassist run 2 skipped «Сделать фото» and profile rows)."""
        return _SCROLL_BOTTOM_MARGIN if self._tab_bar_visible else _SCROLL_BOTTOM_MARGIN_NO_TAB_BAR

    def _in_scroll_band(self, frame: dict) -> bool:
        """Where a scrolled element must land: clear of the navigation bar
        and the tab bar, which would take the tap otherwise."""
        cy = frame.get("y", 0) + frame.get("height", 0) / 2
        return _SCROLL_TOP_MARGIN <= cy <= self._height - self._bottom_margin()

    async def scroll_into_view(
        self, element_type: str | None, label: str | None, test_id: str | None, frame: dict,
    ) -> dict | None:
        """Scroll until the element is on screen; return its new frame.

        The AX tree lists a whole long screen, including what lies below the
        fold (aweassist run 1: 480 of 978 actions off screen). Tapping those
        coordinates hits nothing or the tab bar. Elements are re-found after
        every swipe by type/label/id and their ordinal among identical ones
        (cards repeat labels like «Кэшбэк 10%»). None when the element
        cannot be brought on screen. A card off to the side in a carousel
        is reached by swiping its row sideways, once the row is in view.

        When the content stops moving with the element still below the
        screen, ``scroll_stuck`` is set: nothing further down is reachable.
        An expanded panel locks scrolling while the page it covers stays in
        the tree (aweassist «Счета»: 93 of 101 actions). A visible text with
        the element's label is then tapped instead: the panel's «Настройки»
        is only a text, the button of that name is far below.
        """
        self.scroll_stuck = False

        def same(el: dict) -> bool:
            return (
                (element_type is None or el.get("type") == element_type)
                and (el.get("label") or None) == (label or None)
                # ElementSnapshot falls back to the label when there is no
                # accessibility id (analyzer.classify_element).
                and (el.get("test_id") or el.get("label") or None) == (test_id or None)
                and not el.get("obscured")
            )

        elements = [el for el in await self.get_ui_elements() if same(el)]
        if not elements:
            return None
        elements.sort(key=_reading_order)
        ordinal = min(
            range(len(elements)),
            key=lambda i: (
                abs((elements[i].get("frame") or {}).get("y", 0) - frame.get("y", 0)),
                abs((elements[i].get("frame") or {}).get("x", 0) - frame.get("x", 0)),
            ),
        )
        x = self._width // 2
        top, bottom = _SCROLL_TOP_MARGIN, self._height - self._bottom_margin()
        no_move_seen = False
        row_in_view = False
        row_scale, last_side = 1.0, 0
        for attempt in range(_MAX_SCROLL_SWIPES):
            current = (elements[ordinal].get("frame") or {}) if ordinal < len(elements) else None
            if current is None:
                return None
            if self._in_scroll_band(current) or row_in_view:
                if self._in_width(current):
                    return current
                side = 1 if current.get("x", 0) + current.get("width", 0) / 2 > self._width else -1
                if last_side and side != last_side:
                    row_scale /= 2  # overshot: come back in shorter steps
                last_side = side
                await self._swipe_row(current, row_scale)
                tree = await self.get_ui_elements()
                elements = sorted((el for el in tree if same(el)), key=_reading_order)
                moved = (elements[ordinal].get("frame") or {}) if ordinal < len(elements) else None
                if moved is None:
                    return None
                if abs(moved.get("x", 0) - current.get("x", 0)) < _SCROLL_NO_MOVE_PT:
                    # The end of this carousel; says nothing about the page.
                    if no_move_seen:
                        return None
                    no_move_seen = True
                else:
                    no_move_seen = False
                continue
            # Bar items live in the margins and are tapped where they are:
            # the navigation bar at the top, the tab bar's own items at the
            # bottom. Content near the bottom edge is under the tab bar, so
            # it gets scrolled into the band like anything off screen.
            cy = current.get("y", 0) + current.get("height", 0) / 2
            if attempt == 0 and self.is_on_screen(current) and (
                cy < top or element_type in _TAB_BAR_TYPES
            ):
                return current
            # Swipe just far enough to bring the element to the middle of the
            # band. A fixed swipe of more than half a screen overshot short
            # distances and bounced the content up and down (aweassist run 3,
            # profile rows). Positive ``needed``: move content up.
            needed = cy - (top + bottom) / 2
            dist = min(max(abs(needed), _MIN_SCROLL_SWIPE_PT), bottom - top - 80)
            start = bottom - 40 if needed > 0 else top + 40
            end = start - dist if needed > 0 else start + dist
            await _run([
                AXE, "swipe", "--start-x", str(x), "--start-y", str(int(start)),
                "--end-x", str(x), "--end-y", str(int(end)),
                "--duration", _SCROLL_SWIPE_DURATION_S, "--udid", self._udid,
            ])
            await asyncio.sleep(_SCROLL_SETTLE_S)
            tree = await self.get_ui_elements()
            elements = sorted((el for el in tree if same(el)), key=_reading_order)
            moved = (elements[ordinal].get("frame") or {}) if ordinal < len(elements) else None
            if moved is None:
                return None
            if abs(moved.get("y", 0) - current.get("y", 0)) < _SCROLL_NO_MOVE_PT:
                # The content cannot scroll any further this way: tap the
                # element where it is if it is on screen at all, instead of
                # spending the remaining swipes on nothing. Off screen, a
                # second try first: AXe drops input, and a dropped swipe
                # looks just like the end of the content.
                if self.is_on_screen(moved):
                    return moved
                if self._in_height(moved):
                    row_in_view = True
                    no_move_seen = False
                    continue
                if not no_move_seen:
                    no_move_seen = True
                    continue
                self.scroll_stuck = needed > 0
                return self._visible_text(tree, label)
            no_move_seen = False
        return None

    def _in_width(self, frame: dict) -> bool:
        return 0 <= frame.get("x", 0) + frame.get("width", 0) / 2 <= self._width

    def _in_height(self, frame: dict) -> bool:
        return 0 <= frame.get("y", 0) + frame.get("height", 0) / 2 <= self._height

    async def _swipe_row(self, frame: dict, scale: float = 1.0) -> None:
        """Swipe the row at ``frame`` far enough to centre it horizontally."""
        needed = (frame.get("x", 0) + frame.get("width", 0) / 2 - self._width / 2) * scale
        dist = min(max(abs(needed), _MIN_SCROLL_SWIPE_PT), self._width - 2 * _CAROUSEL_EDGE_PT)
        start = self._width - _CAROUSEL_EDGE_PT if needed > 0 else _CAROUSEL_EDGE_PT
        end = start - dist if needed > 0 else start + dist
        y = str(int(frame.get("y", 0) + frame.get("height", 0) / 2))
        await _run([
            AXE, "swipe", "--start-x", str(int(start)), "--start-y", y,
            "--end-x", str(int(end)), "--end-y", y,
            "--duration", _CAROUSEL_SWIPE_DURATION_S, "--udid", self._udid,
        ])
        await asyncio.sleep(_SCROLL_SETTLE_S)

    def _visible_text(self, tree: list[dict], label: str | None) -> dict | None:
        """Frame of an uncovered on-screen text reading ``label``."""
        for el in tree:
            frame = el.get("frame") or {}
            if (
                label and el.get("type") == "StaticText" and el.get("label") == label
                and not el.get("obscured") and self._in_scroll_band(frame)
            ):
                return frame
        return None

    async def go_back(self) -> bool:
        """iOS back: tap the visible back button, else swipe from the left edge.

        Many real screens disable the edge-swipe gesture (aweassist run 1:
        11 screens where five swipes did nothing and the app was relaunched),
        but nearly all show a back button in the navigation bar.
        """
        for el in await self.get_ui_elements():
            frame = el.get("frame") or {}
            if (
                el.get("type") == "Button"
                and not el.get("obscured")
                and (el.get("test_id") in _BACK_IDS or (el.get("label") or "").strip() in _BACK_LABELS)
                and 0 <= frame.get("y", -1) < _BACK_BUTTON_MAX_Y
            ):
                await self.tap_at(
                    int(frame.get("x", 0) + frame.get("width", 0) / 2),
                    int(frame.get("y", 0) + frame.get("height", 0) / 2),
                )
                return True
        code, _, _ = await _run([
            AXE, "swipe",
            "--start-x", "0", "--start-y", str(self._height // 2),
            "--end-x", str(self._width // 2), "--end-y", str(self._height // 2),
            "--udid", self._udid,
        ])
        return code == 0

    async def press_return(self) -> bool:
        """Hardware Return key. Confirms the default button of SpringBoard
        alerts that are missing from the AX tree (aweassist's «You have
        changed the icon» alert has no elements at all)."""
        code, _, _ = await _run([AXE, "key", _HID_KEY_RETURN, "--udid", self._udid])
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
