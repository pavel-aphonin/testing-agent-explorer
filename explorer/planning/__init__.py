"""PER-203 Phase 3: shared planning core.

The planning *intelligence* (prompt hints, schema assembly) lives here
as pure, transport-agnostic functions so BOTH the synchronous
``scenario_runner`` path and the bus ``planner-runner`` produce
identical plans. Extracting it is what lets the planner move onto the
message bus without duplicating or regressing the PER-172/198/200
behavioural fixes.
"""

from explorer.planning.hints import (
    append_pin_submit,
    count_digit_taps,
    credential_routing_hint,
    loop_breaker_hint,
    pin_keypad_hint,
    pin_submit_hint,
)

__all__ = [
    "append_pin_submit",
    "count_digit_taps",
    "credential_routing_hint",
    "loop_breaker_hint",
    "pin_keypad_hint",
    "pin_submit_hint",
]
