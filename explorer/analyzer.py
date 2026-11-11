"""Element classification: convert raw accessibility elements to typed snapshots."""

from explorer.models import ElementKind, ElementSnapshot, ScreenNode
from explorer.screen_id import KEYBOARD_TYPES, SKIP_TYPES, compute_screen_id, normalize_label

# Mapping from iOS accessibility types to ElementKind
IOS_TYPE_MAP: dict[str, ElementKind] = {
    "Button": ElementKind.BUTTON,
    "TextField": ElementKind.TEXT_FIELD,
    "SecureTextField": ElementKind.TEXT_FIELD,
    "SearchField": ElementKind.TEXT_FIELD,
    "TextArea": ElementKind.TEXT_FIELD,
    "Switch": ElementKind.SWITCH,
    "Toggle": ElementKind.SWITCH,
    # iOS 27: SwiftUI Toggle / UISwitch come as CheckBox (subrole AXSwitch).
    # Unmapped, the testbed's notification toggle was never tapped.
    "CheckBox": ElementKind.SWITCH,
    "StaticText": ElementKind.LABEL,
    "Heading": ElementKind.LABEL,
    "Image": ElementKind.IMAGE,
    "Link": ElementKind.LINK,
    "Cell": ElementKind.BUTTON,  # table/collection cells are tappable
    "Tab": ElementKind.BUTTON,
    # iOS 27 tab bar items (aweassist: «Главный / Платежи / Выгода /
    # История / Чаты»). Unmapped, the whole bottom menu was invisible.
    "RadioButton": ElementKind.BUTTON,
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


# Tab bar items; RadioButton is how iOS 27 reports them.
TAB_BAR_TYPES = frozenset({"RadioButton", "Tab"})

# Pushed screens show a back button; see _name_from_content.
_BACK_IDS = {"BackButton", "backButton"}
_BACK_LABELS = {"Назад", "Back"}
_TITLE_TOP_Y, _TITLE_BOTTOM_Y = 50, 320
_TITLE_MAX_LEN = 40


# Action overlap above which two nodes can be one screen (see is_variant).
VARIANT_JACCARD = 0.85
# Below this many actions a card more or less is a different screen.
MIN_VARIANT_ACTIONS = 5


def _has_back(node: ScreenNode) -> bool:
    return any(
        s.test_id in _BACK_IDS or (s.label or "").strip() in _BACK_LABELS
        for s in node.interactive_elements
    )


def _title(node: ScreenNode) -> str | None:
    return next((s.label for s in node.elements if s.element_type == "Heading" and s.label), None)


def action_key(element: ElementSnapshot) -> tuple[str, str]:
    return element.element_type, normalize_label(element.label or "")


def _content_actions(node: ScreenNode) -> set[tuple[str, str]]:
    """The screen's own labelled actions; the tab bar is on every screen."""
    return {
        action_key(s) for s in node.interactive_elements
        if s.element_type not in TAB_BAR_TYPES and (s.label or "").strip()
    }


def is_variant(a: ScreenNode, b: ScreenNode) -> bool:
    """Two fingerprints of one screen: a card more, a banner, a title that
    collapsed on scroll, an error with a «reload» button.

    Overlap alone is not enough. The expanded panel on aweassist's main
    screen shares 97 of 101 actions with it, because what it covers stays
    in the tree; its back button is what gives it away. Two sheets can
    offer the same buttons under different titles.
    """
    ka, kb = _content_actions(a), _content_actions(b)
    if min(len(ka), len(kb)) < MIN_VARIANT_ACTIONS or _has_back(a) != _has_back(b):
        return False
    ta, tb = _title(a), _title(b)
    if ta and tb and ta != tb:
        return False
    return len(ka & kb) / len(ka | kb) >= VARIANT_JACCARD


def _name_from_content(snapshots: list[ElementSnapshot]) -> str | None:
    """A readable name for a screen with no Heading.

    Without one, aweassist named 24 of 66 screens after the app («Деньги»).
    A pushed screen (it has a back button) is named by the first text below
    the navigation bar, skipping amounts, dates and button captions. A root
    screen is named by its selected tab, the topmost one when a segmented
    control sits above the tab bar.
    """
    has_back = any(
        s.test_id in _BACK_IDS or (s.label or "").strip() in _BACK_LABELS for s in snapshots
    )
    if has_back:
        controls = {s.label for s in snapshots if s.element_type == "Button"}
        texts = [
            s for s in snapshots
            if s.element_type in ("StaticText", "GenericElement") and s.label and s.frame
            and _TITLE_TOP_Y <= s.frame.get("y", 0) <= _TITLE_BOTTOM_Y
            and s.label not in controls
            and len(s.label) <= _TITLE_MAX_LEN
            and not any(ch.isdigit() for ch in s.label)
        ]
        if texts:
            return min(texts, key=lambda s: s.frame.get("y", 0)).label
    selected = [
        s for s in snapshots
        if s.element_type == "RadioButton" and s.value == "1" and s.label and s.frame
    ]
    if selected:
        return min(selected, key=lambda s: s.frame.get("y", 0)).label
    return None


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
    # Controls covered by a modal stay in the screen (and its fingerprint)
    # but are not actions — see axe_client._obscured_children.
    interactive = [
        s for s, el in zip(all_snapshots, filtered)
        if s.kind in INTERACTIVE_KINDS and s.enabled and not el.get("obscured")
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
        name = _name_from_content(all_snapshots) or ""
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
        is_modal=any(el.get("obscured") for el in elements),
    )
