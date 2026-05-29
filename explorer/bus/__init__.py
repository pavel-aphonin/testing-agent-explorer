"""PER-203: message-bus layer for the 12-module pipeline (Redis Streams).

Built ALONGSIDE the synchronous path and gated by ``TA_BUS_MODE=1`` — the
worker keeps its existing synchronous orchestration when the flag is off,
so the system stays runnable at every step of the migration.

Public surface:
    Envelope          — the message shape on every stream
    BusClient         — async publish / consume / ack over Redis Streams
    stream_for / GROUP — naming helpers + the message-type constants
"""

from explorer.bus.envelope import Envelope, MsgType
from explorer.bus.streams import BusClient, stream_for

__all__ = ["BusClient", "Envelope", "MsgType", "stream_for"]
