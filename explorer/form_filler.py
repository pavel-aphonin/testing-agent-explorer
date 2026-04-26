"""Form filling with Property-Based Testing strategies using Hypothesis."""

from __future__ import annotations

import logging
import re
import string

from hypothesis import strategies as st

from explorer.models import ElementSnapshot

logger = logging.getLogger(__name__)

# When ``test_data`` is supplied, look up these key aliases per field
# type before falling back to BUILTIN_VARIANTS. This lets a workspace
# define ``login`` once and have it used for both email and password
# inputs that don't have an exact match.
_TEST_DATA_ALIASES: dict[str, tuple[str, ...]] = {
    "email":    ("email", "login", "username"),
    "password": ("password", "pass", "pwd"),
    "phone":    ("phone", "tel", "mobile"),
    "name":     ("name", "fullname", "user"),
    "search":   ("search", "query", "q"),
    "number":   ("number", "amount", "value"),
    "url":      ("url", "link", "website"),
}

# Field type classification by label/testID keywords
FIELD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"email|mail|e-mail|почта", re.IGNORECASE)),
    ("password", re.compile(r"password|pass|пароль", re.IGNORECASE)),
    ("phone", re.compile(r"phone|tel|телефон|мобильный", re.IGNORECASE)),
    ("name", re.compile(r"name|имя|фамилия|фио|user.?name", re.IGNORECASE)),
    ("search", re.compile(r"search|поиск|найти|query", re.IGNORECASE)),
    ("number", re.compile(r"number|amount|количество|число|age|возраст", re.IGNORECASE)),
    ("url", re.compile(r"url|link|ссылка|адрес|website", re.IGNORECASE)),
]


def classify_field(element: ElementSnapshot) -> str:
    """Classify a text field by its label/testID using heuristics."""
    text_to_check = " ".join(
        filter(None, [element.label, element.test_id, element.value])
    )
    if not text_to_check:
        return "generic"

    for field_type, pattern in FIELD_PATTERNS:
        if pattern.search(text_to_check):
            return field_type

    # Check element type
    if element.element_type == "SecureTextField":
        return "password"

    return "generic"


# Pre-defined test data variants per field type
# Each variant: (value, category_name)
# "valid" is always first — used for happy-path form filling
BUILTIN_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "email": [
        ("test@test.com", "valid"),
        ("", "empty"),
        ("notanemail", "invalid_format"),
        ("a" * 200 + "@test.com", "overflow"),
    ],
    "password": [
        ("password123", "valid"),
        ("", "empty"),
        ("ab", "too_short"),
        ("a" * 200, "overflow"),
    ],
    "phone": [
        ("+7 900 000-00-00", "valid"),
        ("", "empty"),
        ("abc", "non_numeric"),
        ("12", "too_short"),
    ],
    "name": [
        ("Test User", "valid"),
        ("", "empty"),
        ("  ", "whitespace_only"),
        ("A" * 200, "overflow"),
    ],
    "search": [
        ("test query", "valid"),
        ("", "empty"),
    ],
    "number": [
        ("42", "valid"),
        ("", "empty"),
        ("abc", "non_numeric"),
        ("-1", "negative"),
        ("0", "zero"),
    ],
    "url": [
        ("https://example.com", "valid"),
        ("", "empty"),
        ("not a url", "invalid_format"),
    ],
    "generic": [
        ("test value", "valid"),
        ("", "empty"),
        ("a" * 200, "overflow"),
    ],
}


def get_valid_value(
    field_type: str, test_data: dict[str, str] | None = None
) -> str:
    """Get the 'valid' variant for a field type.

    Lookup chain (first match wins):
      1. ``test_data[field_type]`` — exact match (e.g. ``email``).
      2. ``test_data[alias]`` for each alias in ``_TEST_DATA_ALIASES``
         (e.g. ``login`` covers email/password if the workspace named
         it that way).
      3. The first ("valid") entry in ``BUILTIN_VARIANTS`` for this
         field type — what callers used to get unconditionally.

    The function logs which source it picked so a run trace makes it
    obvious whether form filling came from the workspace's test_data
    or fell back to the built-in defaults (e.g. ``test@test.com``).
    """
    if test_data:
        # 1. Exact key match.
        if field_type in test_data:
            value = test_data[field_type]
            logger.info(
                "[form_filler] field=%s source=test_data:%s value=%r",
                field_type, field_type, value,
            )
            return value
        # 2. Alias chain.
        for alias in _TEST_DATA_ALIASES.get(field_type, ()):
            if alias in test_data and alias != field_type:
                value = test_data[alias]
                logger.info(
                    "[form_filler] field=%s source=test_data:%s (alias) value=%r",
                    field_type, alias, value,
                )
                return value
    # 3. Built-in fallback.
    variants = BUILTIN_VARIANTS.get(field_type, BUILTIN_VARIANTS["generic"])
    fallback = variants[0][0]  # First variant is always valid
    logger.info(
        "[form_filler] field=%s source=builtin value=%r",
        field_type, fallback,
    )
    return fallback


def get_test_variants(field_type: str) -> list[tuple[str, str]]:
    """Get all test variants for a field type."""
    return BUILTIN_VARIANTS.get(field_type, BUILTIN_VARIANTS["generic"])


def get_hypothesis_strategy(field_type: str) -> st.SearchStrategy:
    """Return a Hypothesis strategy for generating random test data."""
    match field_type:
        case "email":
            # Generate random valid-ish emails
            return st.builds(
                lambda user, domain: f"{user}@{domain}.com",
                st.text(
                    alphabet=string.ascii_lowercase + string.digits,
                    min_size=1,
                    max_size=20,
                ),
                st.text(
                    alphabet=string.ascii_lowercase,
                    min_size=2,
                    max_size=10,
                ),
            )
        case "password":
            return st.text(min_size=0, max_size=50)
        case "phone":
            return st.builds(
                lambda digits: "+" + digits,
                st.text(alphabet=string.digits, min_size=7, max_size=15),
            )
        case "name":
            return st.text(
                alphabet=string.ascii_letters + " ",
                min_size=0,
                max_size=50,
            )
        case "number":
            return st.integers(min_value=-1000, max_value=10000).map(str)
        case _:
            return st.text(min_size=0, max_size=100)


class FormFiller:
    """Manages test data generation for form fields.

    ``test_data`` (workspace-supplied dict of key→value) is consulted
    by ``get_valid_value_for`` before falling back to BUILTIN_VARIANTS.
    Pass an empty dict (default) to get the legacy behaviour.
    """

    def __init__(self, test_data: dict[str, str] | None = None):
        # Cache: element_uid -> field_type
        self._field_types: dict[str, str] = {}
        # Cache: element_uid -> list of variants already tried
        self._tried_variants: dict[str, set[str]] = {}
        # Workspace-level test_data used as the highest-priority source
        # for happy-path values (see get_valid_value above).
        self.test_data: dict[str, str] = dict(test_data or {})

    def classify(self, element: ElementSnapshot) -> str:
        """Classify a field and cache the result."""
        uid = element.uid()
        if uid not in self._field_types:
            self._field_types[uid] = classify_field(element)
        return self._field_types[uid]

    def get_next_variant(
        self, element: ElementSnapshot
    ) -> tuple[str, str] | None:
        """
        Get the next untried variant for this field.
        Returns (value, category) or None if all variants exhausted.
        """
        uid = element.uid()
        field_type = self.classify(element)
        variants = get_test_variants(field_type)

        if uid not in self._tried_variants:
            self._tried_variants[uid] = set()

        for value, category in variants:
            if category not in self._tried_variants[uid]:
                self._tried_variants[uid].add(category)
                return (value, category)

        return None  # All variants tried

    def get_valid_value_for(self, element: ElementSnapshot) -> str:
        """Get the valid/happy-path value for a field.

        Honours ``self.test_data`` so a workspace's seeded credentials
        win over the hard-coded ``test@test.com`` / ``password123``
        defaults — see :func:`get_valid_value` for the lookup chain.
        """
        field_type = self.classify(element)
        return get_valid_value(field_type, self.test_data)

    def has_untried_variants(self, element: ElementSnapshot) -> bool:
        """Check if there are untried variants for this field."""
        uid = element.uid()
        field_type = self.classify(element)
        variants = get_test_variants(field_type)
        tried = self._tried_variants.get(uid, set())
        return any(cat not in tried for _, cat in variants)
