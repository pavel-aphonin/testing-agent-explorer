"""Screen fingerprinting: compute a stable hash of the current screen state.

The hash should be STABLE across minor UI changes:
- Error messages appearing/disappearing
- Keyboard up/down
- Button enabled/disabled state
- Text field values changing
- Loading spinners

It should CHANGE only when the screen STRUCTURE changes:
- Different set of buttons/fields
- Navigation to a completely different screen
- Modal/alert appearing

Strategy: hash only by STRUCTURAL elements (buttons, text fields,
switches, navigation bars) and their labels. Ignore everything else.
"""

import hashlib
import json

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


def compute_screen_id(elements: list[dict]) -> str:
    """Compute a structural fingerprint hash.

    Only considers STRUCTURAL element types (buttons, fields, switches)
    and their labels. Deliberately ignores:
    - StaticText (error messages, dynamic content)
    - enabled/disabled state
    - element values (user input)
    - frames/positions
    - keyboard presence

    Returns a 16-char hex string.
    """
    fingerprint_parts = []
    for el in elements:
        el_type = el.get("type", "")

        # Only include structural elements
        if el_type in SKIP_TYPES:
            continue
        if el_type not in STRUCTURAL_TYPES:
            # Unknown type — include it to be safe but without label
            # (so dynamic labels don't change the hash)
            fingerprint_parts.append((el_type, ""))
            continue

        # For structural elements: use type + label
        el_label = el.get("label", "") or ""
        fingerprint_parts.append((el_type, el_label))

    fingerprint_parts.sort()
    canonical = json.dumps(fingerprint_parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def screens_are_same(elements_a: list[dict], elements_b: list[dict]) -> bool:
    """Check if two element lists represent the same screen."""
    return compute_screen_id(elements_a) == compute_screen_id(elements_b)
