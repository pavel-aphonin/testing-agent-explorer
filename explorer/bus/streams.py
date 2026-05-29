"""PER-203: Redis Streams client for the bus.

Thin async wrapper over redis-py's stream commands. One ``BusClient``
per process (worker or a module-runner). Consumer groups give us
at-least-once delivery + load-balancing if we ever run >1 instance of
a module; XACK marks a message done; un-acked messages are reclaimable
(dead-letter handling is Phase 6, not here).

Stream naming: ``ta:<msg-type>`` e.g. ``ta:plan.produced``. The ``ta:``
prefix keeps our streams clear of the backend's existing run-event
pub/sub channels on the same Redis.
"""

from __future__ import annotations

import logging
import os

import redis.asyncio as aioredis

from explorer.bus.envelope import Envelope, MsgType

logger = logging.getLogger("explorer.bus")

STREAM_PREFIX = "ta:"


def stream_for(msg_type: MsgType) -> str:
    """Stream key for a message type."""
    return f"{STREAM_PREFIX}{msg_type.value}"


def _redis_url() -> str:
    # Worker is a host process; Redis is exposed by docker on localhost.
    # Overridable for container / remote deploys.
    return os.environ.get("TA_REDIS_URL", "redis://localhost:6379/0")


class BusClient:
    """Async publish/consume/ack over Redis Streams."""

    def __init__(self, url: str | None = None, *, consumer_name: str | None = None):
        self._url = url or _redis_url()
        self._consumer = consumer_name or f"c-{os.getpid()}"
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        if self._redis is None:
            # decode_responses=False — Envelope.from_fields tolerates
            # bytes; keeping raw avoids a decode pass on the payload.
            self._redis = aioredis.from_url(self._url)
            await self._redis.ping()
            logger.info("BusClient connected to %s (consumer=%s)", self._url, self._consumer)

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def publish(self, env: Envelope) -> str:
        """XADD the envelope to its type's stream. Returns Redis entry id."""
        assert self._redis is not None, "call connect() first"
        stream = stream_for(env.type)
        entry_id = await self._redis.xadd(stream, env.to_fields())
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    async def ensure_group(self, msg_type: MsgType, group: str) -> None:
        """Create the consumer group (and the stream) if absent.

        MKSTREAM creates the stream so a consumer can subscribe before
        any producer has published. BUSYGROUP (group already exists) is
        swallowed — idempotent bootstrap.
        """
        assert self._redis is not None, "call connect() first"
        stream = stream_for(msg_type)
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
            logger.info("created consumer group %s on %s", group, stream)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume(
        self,
        msg_type: MsgType,
        group: str,
        *,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[str, Envelope]]:
        """XREADGROUP one batch. Returns [(entry_id, Envelope)].

        ``entry_id`` is Redis's native stream id — pass it back to
        ``ack()``. Empty list on timeout (block_ms elapsed with nothing
        new), so callers loop.
        """
        assert self._redis is not None, "call connect() first"
        stream = stream_for(msg_type)
        resp = await self._redis.xreadgroup(
            group, self._consumer, {stream: ">"}, count=count, block=block_ms,
        )
        out: list[tuple[str, Envelope]] = []
        if not resp:
            return out
        # resp: [(stream, [(entry_id, {field:val}), ...])]
        for _stream, entries in resp:
            for entry_id, fields in entries:
                eid = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                try:
                    out.append((eid, Envelope.from_fields(fields)))
                except Exception as exc:
                    logger.warning("undecodable bus message %s: %s — acking to skip", eid, exc)
                    await self.ack(msg_type, group, eid)
        return out

    async def ack(self, msg_type: MsgType, group: str, entry_id: str) -> None:
        assert self._redis is not None, "call connect() first"
        await self._redis.xack(stream_for(msg_type), group, entry_id)
