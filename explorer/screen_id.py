"""Screen fingerprinting: compute a stable hash of the current screen state.

The hash should be STABLE across minor UI changes:
- Error messages appearing/disappearing
- Keyboard up/down
- Button enabled/disabled state
- Text field values changing
- Loading spinners
- iOS status bar contents (clock, battery, signal) — see PER-24

It should CHANGE only when the screen STRUCTURE changes:
- Different set of buttons/fields
- Navigation to a completely different screen
- Modal/alert appearing

Strategy: hash only by STRUCTURAL elements (buttons, text fields,
switches, navigation bars) and their labels. Ignore everything else.
Additionally normalise labels and drop status-bar zones so a clock
tick doesn't fork "Главная" into N nodes (PER-24).
"""

import hashlib
import json
import os
import re

# Element types that represent STRUCTURE (included in hash)
STRUCTURAL_TYPES = {
    "Button", "TextField", "SecureTextField", "Switch", "Slider",
    "Tab", "TabBar", "SegmentedControl",
    "NavigationBar", "Toolbar",
    "Cell", "Link",
    "GenericElement",  # React Native renders buttons as GenericElement
}

# Element types to ALWAYS skip (dynamic/decorative)
# Backward compat: analyzer.py imports this name
KEYBOARD_TYPES = {"Key", "Keyboard", "KeyboardKey"}

SKIP_TYPES = {
    "Key", "Keyboard", "KeyboardKey",       # virtual keyboard
    "StaticText",                             # error messages, labels, dynamic text
    "Image", "Icon",                          # decorative
    "ActivityIndicator", "ProgressIndicator", # loading
    "Window", "Application",                  # container
    "ScrollView", "Table", "CollectionView",  # layout containers
    "Group", "Other",                         # generic
}

# iOS status bar lives in roughly the top 50 px on every device. Any
# element that lands fully in that band is dynamic (clock / signal /
# battery / notification badges) and shouldn't push the screen into
# a new node. Configurable via env so weird custom statusbar overlays
# (or Android, where the band is different) can opt out.
_STATUS_BAR_HEIGHT = int(os.environ.get("TA_SCREEN_STATUS_BAR_PX", "50"))

# Labels that match these patterns are *value-like* and should be
# normalized before hashing. e.g. a button labelled "12:34" today and
# "12:35" a minute later should hash as the same button — its identity
# is its position/role, not the changing value text.
_DYNAMIC_LABEL_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|ам|пм)?\s*$", re.IGNORECASE),  # 12:34
    re.compile(r"^\s*\d{1,3}\s*%\s*$"),                                              # 87%
    re.compile(r"^\s*\d+\s*(unread|new|непрочитан\w*)\s*$", re.IGNORECASE),          # "3 unread"
    re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}\s*$"),                                      # countdown timer
)


def _is_in_status_bar(element: dict) -> bool:
    """True iff the element's frame falls inside the status-bar band
    at the top of the screen. ``frame`` may be a dict {x,y,width,height}
    or a list [x,y,w,h] depending on the AXe version — handle both, and
    fall back to ``False`` if neither shape parses."""
    frame = element.get("frame")
    if not frame:
        return False
    try:
        if isinstance(frame, dict):
            y = float(frame.get("y", 0))
            h = float(frame.get("height", 0))
        elif isinstance(frame, (list, tuple)) and len(frame) >= 4:
            y = float(frame[1])
            h = float(frame[3])
        else:
            return False
    except (TypeError, ValueError):
        return False
    # Element fully inside the band, or hugs the very top.
    return y + h <= _STATUS_BAR_HEIGHT


def _normalize_label(label: str) -> str:
    """If ``label`` matches one of the dynamic patterns, replace it
    with a placeholder that's stable across ticks. Otherwise return
    as-is. This keeps the *identity* of a "clock" or "battery" button
    consistent without ignoring genuine label changes."""
    if not label:
        return ""
    for pat in _DYNAMIC_LABEL_PATTERNS:
        if pat.match(label):
            return "<dynamic>"
    return label


def compute_screen_id(elements: list[dict]) -> str:
    """Compute a structural fingerprint hash.

    Only considers STRUCTURAL element types (buttons, fields, switches)
    and their labels. Deliberately ignores:
    - StaticText (error messages, dynamic content)
    - enabled/disabled state
    - element values (user input)
    - frames/positions
    - keyboard presence
    - status-bar elements (clock, battery, signal — PER-24)
    - labels that look like clocks or percentages (PER-24)

    Returns a 16-char hex string.
    """
    fingerprint_parts: list[tuple[str, str]] = []
    for el in elements:
        el_type = el.get("type", "")

        # Status-bar zone is dynamic by construction — clocks tick,
        # battery drains, notification badges flip on and off.
        if _is_in_status_bar(el):
            continue

        # Only include structural elements
        if el_type in SKIP_TYPES:
            continue
        if el_type not in STRUCTURAL_TYPES:
            # Unknown type — include it to be safe but without label
            # (so dynamic labels don't change the hash)
            fingerprint_parts.append((el_type, ""))
            continue

        # For structural elements: use type + (normalised) label.
        el_label = el.get("label", "") or ""
        fingerprint_parts.append((el_type, _normalize_label(el_label)))

    fingerprint_parts.sort()
    canonical = json.dumps(fingerprint_parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def screens_are_same(elements_a: list[dict], elements_b: list[dict]) -> bool:
    """Check if two element lists represent the same screen."""
    return compute_screen_id(elements_a) == compute_screen_id(elements_b)
