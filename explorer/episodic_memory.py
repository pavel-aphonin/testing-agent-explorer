"""Episodic memory layer for the agent (PER-169).

The chat-LLM (Gemma 4 / Qwen3.6) is unable to track multi-step
progress from screenshots alone — empirically it loops on "tap
digit 8" five times because it cannot tell from the PNG that
N out of 4 PIN dots are already filled. Sampling tweaks
(PER-164 followup) gave us patience on loading screens but not
memory of past actions.

Fix per PER-166 research: hook every dispatched action into a
**knowledge graph** (Graphiti + FalkorDB), and before each
goal-decide step, recall the relevant history and inject a
plain-text "memory" block into the prompt. Then the model sees
explicit progress like:

    Memory: in this goal you have tapped digit 8 at (445,1315)
    and digit 5 at (445,1152). 2/4 PIN digits done. Remaining
    target: digits 2, 0.

This module wraps the Graphiti API into something the worker
can use without leaking heavy deps. Key design choices:

* **Async fire-and-forget on add**. ``Graphiti.add_episode``
  triggers entity-extraction LLM calls that take ~15s each on
  Qwen3-8B. Blocking the agent on that would double end-to-end
  latency. We dispatch into a background task and return
  immediately; failures are logged but don't fail the action.
* **Search is awaited inline**. The agent's next decision
  needs the memory NOW, so we await — but with a tight timeout
  (3s) and graceful fallback to "" so a slow recall doesn't
  cascade into ``llm_no_decision``.
* **Lazy connect**. We don't pay graph-startup cost on workers
  that don't have an episodic-memory config (no FalkorDB on
  this host, no chat-server running, etc.). First ``add`` or
  ``recall`` triggers init; failures are remembered to skip
  retries.
* **Extraction LLM ≠ chat-LLM**. Graphiti wants a text-only
  model that reliably emits structured JSON. Our chat-LLM
  (Gemma 4) is vision-heavy and slow; the worker passes a
  separate ``extraction_endpoint`` (defaults to the RAG
  Qwen3-8B on :8083) so extraction is fast and reliable.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodicMemoryConfig:
    """Where to talk to FalkorDB + extraction LLM + embedder."""

    # FalkorDB graph backend
    falkordb_host: str = "localhost"
    falkordb_port: int = 6380
    # Per-run database namespace so concurrent runs don't see each
    # other's episodes. ``None`` lets the worker pick (typically
    # ``run_{run_id}``).
    database: str | None = None
    # Extraction LLM endpoint. Graphiti calls it ~3-6 times per
    # add_episode for entity/edge extraction; latency adds up.
    # Default points at the RAG Qwen3-8B on :8083 — fast (~30 TPS),
    # reliably emits JSON, lighter than the chat-LLM on :8080.
    extraction_endpoint: str = "http://localhost:8083/v1"
    extraction_model: str = "rag-chat"
    # Embedder. Qwen3-Embedding-8B on :8082 (PER-118).
    embedding_endpoint: str = "http://localhost:8082/v1"
    embedding_model: str = "embeddings"
    # Time budget for recall — too long = chat-LLM call is delayed.
    recall_timeout_sec: float = 3.0
    # Max bytes of memory block injected into prompt. Bumped from
    # 500→1500 after PER-169 smoke #1: a chronological action trail
    # of 8 steps × ~80 chars per line was getting truncated to 4
    # lines, hiding the most recent decisions. 1500 fits ~15 lines
    # which covers a typical goal without flooding the prompt.
    memory_block_max_chars: int = 1500


class EpisodicMemory:
    """Per-worker episodic memory layer over Graphiti + FalkorDB.

    Construct once at worker boot, share across runs. Each call
    takes ``run_id`` + ``goal_id`` to scope episodes — different
    runs / goals live in different ``group_id`` namespaces so
    recall queries don't mix them.
    """

    def __init__(self, config: EpisodicMemoryConfig | None = None) -> None:
        self._config = config or EpisodicMemoryConfig()
        self._graphiti: Any | None = None
        self._init_lock = asyncio.Lock()
        self._init_failed = False  # latched on first error to skip retries
        # Strong references to in-flight fire-and-forget add_episode tasks.
        # Python 3.13+ asyncio.create_task() only keeps a *weak* ref to the
        # task — without our own strong ref the GC can reap a task before
        # it runs. Empirically we saw 0/4 episodes land in FalkorDB when
        # add_action_fire_and_forget'd from a short-lived script (PER-169
        # smoke). We keep a set, add a done_callback to discard, so the
        # set stays bounded.
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        # PER-206: latch so a dead extraction endpoint logs ONE warning
        # with a traceback instead of one per dispatched action.
        self._add_warn_logged = False

    @staticmethod
    async def _probe_http(base_url: str, timeout: float = 3.0) -> bool:
        """PER-206: quick reachability probe for an OpenAI-compatible
        endpoint. GET ``{base}/models``; True iff it answers (any
        non-5xx). FalkorDB being healthy is NOT enough — ``add_episode``
        also calls the extraction LLM + embedder, and when those are
        down every write throws an httpx ConnectError with a full
        traceback. Probing here lets us disable memory cleanly.
        """
        import httpx

        url = base_url.rstrip("/") + "/models"
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
                r = await c.get(url)
            return r.status_code < 500
        except Exception:
            return False

    async def _ensure_init(self) -> Any | None:
        """Lazy-connect to FalkorDB + Graphiti. Returns instance or None on failure."""
        if self._graphiti is not None:
            return self._graphiti
        if self._init_failed:
            return None
        async with self._init_lock:
            if self._graphiti is not None:
                return self._graphiti
            if self._init_failed:
                return None
            try:
                from graphiti_core import Graphiti
                from graphiti_core.driver.falkordb_driver import FalkorDriver
                from graphiti_core.llm_client import LLMConfig
                from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
                from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
                from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            except ImportError as exc:
                logger.warning(
                    "graphiti-core not installed (%s) — episodic memory disabled. "
                    "pip install 'graphiti-core[falkordb]' to enable.", exc,
                )
                self._init_failed = True
                return None
            # OpenAI client constructors trip on missing OPENAI_API_KEY
            # env var even when we pass an explicit api_key. Set a
            # benign placeholder if not already set.
            os.environ.setdefault("OPENAI_API_KEY", "local-llamacpp")
            cfg = self._config
            # PER-206: probe the extraction LLM + embedder before we
            # commit to memory. If either is unreachable, disable cleanly
            # with a single actionable warning instead of letting every
            # add_episode throw an httpx ConnectError + traceback.
            extraction_ok = await self._probe_http(cfg.extraction_endpoint)
            embedding_ok = await self._probe_http(cfg.embedding_endpoint)
            if not (extraction_ok and embedding_ok):
                logger.warning(
                    "Episodic memory disabled: extraction LLM %s reachable=%s, "
                    "embedder %s reachable=%s. Start these llama-servers or set "
                    "TA_EPISODIC_MEMORY=0. Agent runs without memory recall.",
                    cfg.extraction_endpoint, extraction_ok,
                    cfg.embedding_endpoint, embedding_ok,
                )
                self._init_failed = True
                return None
            llm_cfg = LLMConfig(
                base_url=cfg.extraction_endpoint,
                model=cfg.extraction_model,
                api_key="local-llamacpp",
                temperature=0.0,
            )
            emb_cfg = OpenAIEmbedderConfig(
                base_url=cfg.embedding_endpoint,
                embedding_model=cfg.embedding_model,
                api_key="local-llamacpp",
            )
            try:
                g = Graphiti(
                    graph_driver=FalkorDriver(
                        host=cfg.falkordb_host,
                        port=cfg.falkordb_port,
                        database=cfg.database or "default_db",
                    ),
                    llm_client=OpenAIGenericClient(config=llm_cfg),
                    embedder=OpenAIEmbedder(config=emb_cfg),
                    cross_encoder=OpenAIRerankerClient(config=llm_cfg),
                )
                await g.build_indices_and_constraints()
            except Exception as exc:
                logger.warning(
                    "Episodic memory init failed (%s: %s) — agent runs without memory recall",
                    type(exc).__name__, exc,
                )
                self._init_failed = True
                return None
            self._graphiti = g
            logger.info(
                "Episodic memory ready: FalkorDB %s:%d, extraction=%s, embedder=%s",
                cfg.falkordb_host, cfg.falkordb_port,
                cfg.extraction_endpoint, cfg.embedding_endpoint,
            )
            return g

    @staticmethod
    def _sanitize(s: str) -> str:
        """FalkorDB's RediSearch full-text indexer treats ``-`` (and
        a few other chars) as query operators. UUIDs are full of
        them, so a raw ``run_<uuid>`` group_id causes
        ``Syntax error at offset N near run_xxx`` on every
        search. Replace separators with underscores and strip the
        rest so the namespace stays unique but RediSearch-clean."""
        out: list[str] = []
        for ch in s:
            if ch.isalnum() or ch == "_":
                out.append(ch)
            else:
                out.append("_")
        return "".join(out) or "x"

    def _group_id(self, run_id: str, goal_id: str | None = None) -> str:
        """Namespace episodes by run + optional goal, RediSearch-safe."""
        run = self._sanitize(run_id)
        if goal_id:
            return f"run_{run}__goal_{self._sanitize(goal_id)}"
        return f"run_{run}"

    async def add_action_fire_and_forget(
        self,
        run_id: str,
        goal_id: str | None,
        episode_text: str,
        episode_name: str = "action",
    ) -> None:
        """Schedule an episode write; return immediately.

        Graphiti's add_episode triggers ~3-6 extraction LLM calls
        (~15s on Qwen3-8B) and we cannot block the agent on it.
        We spawn a task and log on completion / failure but never
        propagate the error.
        """
        async def _bg() -> None:
            g = await self._ensure_init()
            if g is None:
                return
            try:
                from graphiti_core.nodes import EpisodeType
                await g.add_episode(
                    name=episode_name,
                    episode_body=episode_text,
                    source=EpisodeType.text,
                    source_description="agent_action",
                    reference_time=datetime.now(timezone.utc),
                    group_id=self._group_id(run_id, goal_id),
                )
            except Exception as exc:
                # PER-206: full traceback once, then terse debug — a dead
                # extraction endpoint must not dump a traceback on every
                # dispatched action (observed: hundreds of identical
                # tracebacks in one run).
                if not self._add_warn_logged:
                    self._add_warn_logged = True
                    logger.warning(
                        "Episodic memory add_episode failed (run=%s, goal=%s): "
                        "%s: %s — suppressing further tracebacks this process.",
                        run_id, goal_id, type(exc).__name__, exc,
                    )
                else:
                    logger.debug(
                        "Episodic memory add_episode failed again (run=%s): %s",
                        run_id, exc,
                    )
        task = asyncio.create_task(_bg())
        self._inflight_tasks.add(task)
        # Drop the strong ref once the task finishes so the set
        # doesn't grow unboundedly across thousands of actions.
        task.add_done_callback(self._inflight_tasks.discard)

    async def recall(
        self,
        run_id: str,
        goal_id: str | None,
        query: str,
        num_results: int = 5,
    ) -> list[str]:
        """Recall recent episodes for this run + goal in chronological order.

        We use ``retrieve_episodes`` (time-ordered scan over the goal's
        EpisodicNode'ы) rather than ``Graphiti.search`` because:

        * Our episodes are short factual lines ("Step N: tap on X — OK"),
          not narrative text. The extraction LLM doesn't infer
          Entity↔Entity ``EntityEdge.fact`` strings between them, so
          ``g.search`` (which only returns EntityEdges) returns 0 even
          when the graph has 11 episodes and 3 entities for this goal —
          confirmed empirically in PER-169 smoke #1.
        * For loop-prevention the agent doesn't need semantic recall:
          it needs the recent action trail in chronological order so
          it can see "PIN digit 8 was already tapped, 1/4 done". A
          top-N reverse-chronological list of episodes is exactly that.
        * ``retrieve_episodes`` is a single Cypher MATCH + ORDER BY +
          LIMIT — sub-millisecond, no LLM, no embedding round-trip.
          Eliminates the 3-second timeout cliff we worried about.

        ``query`` is kept in the signature for API stability — callers
        in scenario_runner still pass a goal-shaped string — but is
        unused. Future work (PER-170?) may layer semantic ranking on
        top once we have edges worth searching.
        """
        g = await self._ensure_init()
        if g is None:
            return []
        group_id = self._group_id(run_id, goal_id)
        # FalkorDB quirk (verified against graphiti_core 0.29.1):
        # ``add_episode`` mutates ``self.driver = self.driver.clone(database=group_id)``
        # to route writes into a per-group graph (each group_id maps to
        # its own FalkorDB graph). ``retrieve_episodes`` does NOT do
        # the same auto-clone, and the @handle_multiple_group_ids
        # decorator only triggers for len(group_ids) > 1 — so passing
        # group_ids=[GROUP] (one element) reads from the *current*
        # driver database, which is whichever group_id was add_episode'd
        # last across the whole process. We saw the empirical
        # consequence in PER-169 smoke #1: 11 episodes persisted into
        # graph ``run_d7b8..._goal_n_9occ35`` (confirmed via
        # ``GRAPH.QUERY``), retrieve_episodes returned 0.
        # Fix: explicitly pass a cloned driver scoped to *this* graph.
        scoped_driver = g.driver.clone(database=group_id)
        try:
            episodes = await asyncio.wait_for(
                g.retrieve_episodes(
                    reference_time=datetime.now(timezone.utc),
                    last_n=num_results,
                    group_ids=[group_id],
                    driver=scoped_driver,
                ),
                timeout=self._config.recall_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Episodic memory recall timed out after %.1fs (run=%s, goal=%s)",
                self._config.recall_timeout_sec, run_id, goal_id,
            )
            return []
        except Exception:
            logger.exception(
                "Episodic memory recall failed (run=%s, goal=%s)", run_id, goal_id,
            )
            return []
        # graphiti_core 0.29.1 retrieve_episodes already returns
        # episodes in chronological order (oldest-first): it queries
        # ``ORDER BY valid_at DESC LIMIT N`` to pick the latest N, then
        # ``list(reversed(...))`` to hand them back oldest-first. So
        # we iterate as-is — the chat-LLM will read Step 0, 1, 2…
        facts: list[str] = []
        for ep in episodes or []:
            content = getattr(ep, "content", None) or getattr(ep, "name", None) or ""
            if isinstance(content, str) and content.strip():
                facts.append(content.strip())
        return facts

    async def summary_for_prompt(
        self,
        run_id: str,
        goal_id: str | None,
        query: str,
    ) -> str:
        """Build a single block of text to inject into the LLM prompt.

        Empty string when there is no memory (no graphiti, no
        episodes for this goal yet, recall failed). Capped at the
        configured ``memory_block_max_chars`` so it doesn't
        balloon the prompt.
        """
        facts = await self.recall(run_id, goal_id, query, num_results=8)
        if not facts:
            return ""
        budget = self._config.memory_block_max_chars
        out_lines: list[str] = []
        used = 0
        for f in facts:
            line = f"- {f}"
            if used + len(line) + 1 > budget:
                break
            out_lines.append(line)
            used += len(line) + 1
        if not out_lines:
            return ""
        return "\n".join(out_lines)

    async def close(self) -> None:
        """Close the FalkorDB connection. Safe to call multiple times.

        Best-effort: waits up to 5s for in-flight add_episode tasks to
        finish so we don't drop a write right before shutdown. Tasks
        that haven't completed get cancelled — Graphiti's add_episode
        is idempotent on retry by uuid, so a half-written episode
        re-replays cleanly next run.
        """
        if self._inflight_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._inflight_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Episodic memory close: %d add_episode tasks still in flight after 5s — cancelling",
                    len(self._inflight_tasks),
                )
                for t in list(self._inflight_tasks):
                    t.cancel()
        g = self._graphiti
        if g is None:
            return
        try:
            await g.close()
        except Exception:
            logger.exception("Episodic memory close failed")
        finally:
            self._graphiti = None
