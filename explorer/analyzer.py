"""Element classification: convert raw accessibility elements to typed snapshots."""

from explorer.models import ElementKind, ElementSnapshot, ScreenNode
from explorer.screen_id import KEYBOARD_TYPES, SKIP_TYPES, compute_screen_id

# Mapping from iOS accessibility types to ElementKind
IOS_TYPE_MAP: dict[str, ElementKind] = {
    "Button": ElementKind.BUTTON,
    "TextField": ElementKind.TEXT_FIELD,
    "SecureTextField": ElementKind.TEXT_FIELD,
    "SearchField": ElementKind.TEXT_FIELD,
    "TextArea": ElementKind.TEXT_FIELD,
    "Switch": ElementKind.SWITCH,
    "Toggle": ElementKind.SWITCH,
    "StaticText": ElementKind.LABEL,
    "Heading": ElementKind.LABEL,
    "Image": ElementKind.IMAGE,
    "Link": ElementKind.LINK,
    "Cell": ElementKind.BUTTON,  # table/collection cells are tappable
    "Tab": ElementKind.BUTTON,
    "GenericElement": ElementKind.BUTTON,  # RN TouchableOpacity without accessibilityRole
}

# Kinds that represent interactive elements
INTERACTIVE_KINDS = {
    ElementKind.BUTTON,
    ElementKind.TEXT_FIELD,
    ElementKind.SWITCH,
    ElementKind.LINK,
}

# Element types to skip entirely (containers, non-interactive)
SKIP_TYPES = {
    "Application",
    "Window",
    "Group",
    "ScrollView",
    "Table",
    "CollectionView",
    "NavigationBar",
    "TabBar",
    "Toolbar",
    "StatusBar",
    "Other",
}


def classify_element(raw: dict) -> ElementSnapshot:
    """Convert a raw accessibility element dict to an ElementSnapshot."""
    el_type = raw.get("type", "")
    kind = IOS_TYPE_MAP.get(el_type, ElementKind.OTHER)

    label = raw.get("label") or None
    value = raw.get("value") or None
    frame = raw.get("frame") or None
    bounds = raw.get("bounds") or None
    enabled = raw.get("enabled", True)

    # AXe provides test_id directly; other sources may have it in label
    test_id = raw.get("test_id") or None
    if not test_id and label and not label.startswith(" "):
        test_id = label

    return ElementSnapshot(
        kind=kind,
        element_type=el_type,
        label=label,
        value=value,
        test_id=test_id,
        enabled=enabled,
        frame=frame,
        bounds_str=bounds,
    )


def analyze_screen(
    elements: list[dict],
    screenshot_b64: str | None = None,
) -> ScreenNode:
    """
    Analyze raw accessibility elements into a ScreenNode.

    Filters keyboard elements, classifies types, separates interactive elements.
    """
    # Filter out keyboard and skip types
    filtered = [
        el for el in elements
        if el.get("type", "") not in KEYBOARD_TYPES
        and el.get("type", "") not in SKIP_TYPES
    ]

    all_snapshots = [classify_element(el) for el in filtered]
    interactive = [
        s for s in all_snapshots
        if s.kind in INTERACTIVE_KINDS and s.enabled
    ]

    screen_id = compute_screen_id(elements)

    # Try to extract app label from Application element
    app_label = None
    for el in elements:
        if el.get("type") == "Application":
            app_label = el.get("label")
            break

    # Auto-generate name: prefer Heading (usually the screen title), then app label
    name = ""
    for s in all_snapshots:
        if s.element_type == "Heading" and s.label:
            name = s.label
            break
    if not name:
        # Look for a StaticText right after NavigationBar in raw elements
        for i, el in enumerate(elements):
            if el.get("type") == "NavigationBar" and i + 1 < len(elements):
                next_el = elements[i + 1]
                if next_el.get("type") == "StaticText" and next_el.get("label"):
                    name = next_el["label"]
                    break
    if not name:
        # Fallback chain: app label → bundle-id-derived → generic.
        # We never want a raw hex hash to leak into the UI as the screen
        # name (Pavel's feedback: "Обнаружил «4f53cda1»" is unreadable).
        name = app_label or "Главный экран"

    # Improve naming for popup/alert screens:
    # If we have few interactive elements (1-3) and the name is just the
    # app label, this is likely a popup/alert. Use button labels to make
    # the name more descriptive: "TestApp" → "Диалог: OK" or
    # "Диалог: Удалить / Отмена"
    if name == app_label and 0 < len(interactive) <= 3:
        btn_labels = [s.label for s in interactive if s.label]
        if btn_labels:
            name = f"Диалог: {' / '.join(btn_labels)}"

    # NOTE: a previous version appended " (ред.)" to the name whenever
    # a screen had at least one text field AND a Save/Cancel/Apply
    # button. That fragmented the dedup state: the same structural
    # screen would oscillate between "Деньги" and "Деньги (ред.)"
    # depending on whether the keyboard / save button happened to be
    # visible, so ``_filled_per_screen[name]`` ended up split across
    # two buckets and the agent re-typed fields after every popup
    # dismiss. Removing the suffix here costs us a small bit of
    # human-readable disambiguation; the gain is stable identity for
    # forms across keyboard / button visibility changes. If view-vs-
    # edit mode genuinely needs a distinct identity, the structural
    # hash (screen_id) already captures it — the name doesn't have to.

    return ScreenNode(
        screen_id=screen_id,
        name=name,
        elements=all_snapshots,
        interactive_elements=interactive,
        screenshot_b64=screenshot_b64,
        app_label=app_label,
    )
