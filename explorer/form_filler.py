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

# Field type classification by label/testID keywords. Single source of
# truth across the explorer — anything else that needs to map field
# labels to types must call `classify_field()` rather than maintain its
# own regex table. Patterns are ordered first-match-wins.
FIELD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"email|mail|e-mail|почта", re.IGNORECASE)),
    ("password", re.compile(r"password|pass|пароль", re.IGNORECASE)),
    ("phone", re.compile(r"phone|tel\b|тел\b|телефон|мобильный", re.IGNORECASE)),
    ("name", re.compile(r"name|имя|фамилия|фио|lastname|firstname|user.?name", re.IGNORECASE)),
    ("search", re.compile(r"search|поиск|найти|query", re.IGNORECASE)),
    ("number", re.compile(r"number|amount|количество|число|age|возраст", re.IGNORECASE)),
    ("url", re.compile(r"url|link|ссылка|адрес|website", re.IGNORECASE)),
]


# Field-type categories whose values must never appear unmasked in
# logs or persisted run events. Audit (PER-104 #6) flagged the prior
# behaviour where input values were logged verbatim — exposing the
# workspace's password / API token to anyone with log access.
SENSITIVE_FIELD_TYPES: frozenset[str] = frozenset({"password"})

# Substring tokens (case-insensitive) that mark a field as sensitive
# even when its classification is "generic". Catches custom labels
# like "API key" or "auth token" that don't hit FIELD_PATTERNS.
_SENSITIVE_LABEL_TOKENS: tuple[str, ...] = (
    "secret", "token", "api_key", "api-key", "apikey",
    "auth", "credential",
    "пароль", "секрет", "токен",
)


def is_sensitive_field(element: ElementSnapshot) -> bool:
    """True if values for this element should be masked in logs.

    Decision order:
      1. classify_field() returns a type from SENSITIVE_FIELD_TYPES
         (today: ``password``).
      2. The element's label or test_id contains any sensitive token
         (``secret``, ``token``, ``api_key`` and friends). Whitespace
         and hyphen/underscore separators are stripped from the
         haystack so "API Key", "api-key", "api_key" all match the
         same ``apikey`` token.
      3. The element exposes itself as a ``SecureTextField`` — iOS's
         own signal that the value is sensitive.
    """
    if classify_field(element) in SENSITIVE_FIELD_TYPES:
        return True
    raw = " ".join(
        filter(None, [element.label, element.test_id])
    ).lower()
    # Normalise separators so "API Key" / "api-key" / "api_key" all
    # collapse to "apikey".
    normalised = "".join(c for c in raw if c.isalnum())
    haystack = normalised + " " + raw  # match both forms
    if any(tok.replace("_", "").replace("-", "") in normalised for tok in _SENSITIVE_LABEL_TOKENS):
        return True
    if any(tok in haystack for tok in _SENSITIVE_LABEL_TOKENS):
        return True
    if element.element_type == "SecureTextField":
        return True
    return False


def redact_value(element: ElementSnapshot, value: str | None) -> str:
    """Return a safe representation of ``value`` for logs / persistence.

    For sensitive fields returns a mask preserving only the length
    (e.g. ``***<8>``) so an operator can still tell "I typed 8
    characters" without leaking the secret. For non-sensitive fields
    returns the value as-is (so log diagnostics stay useful for
    ФИО / phone / search / etc.).
    """
    if value is None:
        return ""
    if not is_sensitive_field(element):
        return value
    return f"***<{len(value)}>"


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


# Pre-defined test data variants per field type.
#
# Single source of truth for default field values across the explorer.
# When a workspace has not provided a value via test_data, this table
# supplies the placeholder. Anything that needs to know about per-type
# variants (including the legacy PUCT/MC strategy and the PBT prompt
# section in llm_loop) imports from here — no parallel copy lives in
# other modules.
#
# Each variant: (value, category_name). "valid" is always first — used
# for happy-path form filling. Categories beyond "valid" cover negative
# / boundary cases that property-based testing mode probes; the
# "Security" and "Boundary" categories below are what the PBT prompt
# enumerates when PBT mode is enabled.
BUILTIN_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "email": [
        ("test@test.com", "valid"),
        ("", "empty"),
        ("not-an-email", "invalid_format"),
        ("a" * 500 + "@test.com", "overflow"),
        ("<script>alert(1)</script>@test.com", "xss"),
        ("' OR 1=1 --", "sql_injection"),
        ("test@test.com test@test.com", "duplicate"),
        ("ТЕСТ@тест.рф", "unicode"),
    ],
    "password": [
        ("password123", "valid"),
        ("", "empty"),
        ("ab", "too_short"),
        ("a" * 1000, "overflow"),
        ("<script>alert(1)</script>", "xss"),
        ("' OR 1=1 --", "sql_injection"),
    ],
    "phone": [
        ("+7 900 000-00-00", "valid"),
        ("", "empty"),
        ("abc", "non_numeric"),
        ("12", "too_short"),
        ("+7 900 000-00-0" * 20, "overflow"),
    ],
    "name": [
        ("Test User", "valid"),
        ("", "empty"),
        ("  ", "whitespace_only"),
        ("A", "too_short"),
        ("A" * 1000, "overflow"),
        ("<script>alert(1)</script>", "xss"),
    ],
    "search": [
        ("test query", "valid"),
        ("", "empty"),
        ("<script>alert(1)</script>", "xss"),
        ("' OR 1=1 --", "sql_injection"),
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
        ("a" * 1000, "overflow"),
        ("<script>alert(1)</script>", "xss"),
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
    # All log calls below redact the value when field_type is in
    # SENSITIVE_FIELD_TYPES (today: 'password') — operators see the
    # source and length but not the secret itself. Non-sensitive
    # fields (name / phone / search / etc.) log verbatim because
    # debugging form fills needs to see what was typed.
    is_secret = field_type in SENSITIVE_FIELD_TYPES

    def _log_value(v: str) -> str:
        return f"***<{len(v)}>" if is_secret else repr(v)

    if test_data:
        # 1. Exact key match.
        if field_type in test_data:
            value = test_data[field_type]
            logger.info(
                "[form_filler] field=%s source=test_data:%s value=%s",
                field_type, field_type, _log_value(value),
            )
            return value
        # 2. Alias chain.
        for alias in _TEST_DATA_ALIASES.get(field_type, ()):
            if alias in test_data and alias != field_type:
                value = test_data[alias]
                logger.info(
                    "[form_filler] field=%s source=test_data:%s (alias) value=%s",
                    field_type, alias, _log_value(value),
                )
                return value
    # 3. Built-in fallback.
    variants = BUILTIN_VARIANTS.get(field_type, BUILTIN_VARIANTS["generic"])
    fallback = variants[0][0]  # First variant is always valid
    logger.info(
        "[form_filler] field=%s source=builtin value=%s",
        field_type, _log_value(fallback),
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
