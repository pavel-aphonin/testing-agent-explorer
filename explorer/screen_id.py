"""Screen fingerprinting: compute a stable hash of the current screen state."""

import hashlib
import json

# iOS element types that belong to the virtual keyboard
KEYBOARD_TYPES = {"Key", "Keyboard", "KeyboardKey"}


def compute_screen_id(elements: list[dict]) -> str:
    """
    Compute a structural fingerprint hash of the current screen.

    Uses (type, label, enabled) for each element, deliberately excluding
    value and frame (which change with user input or keyboard state).

    Filters out keyboard elements so the same screen with keyboard
    up or down produces the same hash.

    Returns a 16-char hex string (64-bit hash).
    """
    fingerprint_parts = []
    for el in elements:
        el_type = el.get("type", "")
        # Skip keyboard elements
        if el_type in KEYBOARD_TYPES:
            continue
        el_label = el.get("label", "")
        el_enabled = el.get("enabled", False)
        fingerprint_parts.append((el_type, el_label, el_enabled))

    fingerprint_parts.sort()
    canonical = json.dumps(fingerprint_parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def screens_are_same(elements_a: list[dict], elements_b: list[dict]) -> bool:
    """Check if two element lists represent the same screen."""
    return compute_screen_id(elements_a) == compute_screen_id(elements_b)
