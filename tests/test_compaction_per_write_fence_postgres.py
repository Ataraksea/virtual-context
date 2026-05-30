"""Compaction-fence Phase 3 PostgreSQL smoke tests.

Gated on ``DATABASE_URL`` so a developer machine without a Postgres
instance can run the full test suite. CI / prod pipelines export
``DATABASE_URL`` to a throwaway database and run these alongside the
SQLite parity tests in ``test_compaction_per_write_fence.py``.

Covers fencing plan §5.7 Phase 3 PG smoke scenarios:

* T3.22: end-to-end Postgres fence behavior across all five methods
  (store_chunk_embeddings, link_segment_tool_output, store_fact_links,
  set_fact_superseded, update_fact_fields) -- matching guard succeeds,
  mismatched guard raises CompactionLeaseLost.
* T3.23: real Postgres guarded ``store_facts`` /
  ``replace_facts_for_segment`` stamp ``facts.operation_id`` on insert
  and conflict-update.
* T3.24: cross-tenant probe -- caller supplies endpoint facts from
  tenant A while the active op is on tenant B; rowcount=0 (raises
  CompactionLeaseLost via the both-endpoint validation).
"""

from __future__ import annotations

import os
import uuid

import pytest

from virtual_context.types import (
    ChunkEmbedding,
    CompactionLeaseLost,
    Fact,
    FactLink,
)


_pg_required = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping PG fence smoke tests",
)


_TS = "2026-05-29T00:00:00+00:00"


@pytest.fixture(scope="module")
def store():
    from virtual_context.core.compaction_fence import CompactionFenceMode
    from virtual_context.storage.postgres import PostgresStore
    s = PostgresStore(
        os.environ["DATABASE_URL"],
        compaction_fence_mode=CompactionFenceMode.ACTIVE,
    )
    yield s


def _seed_running_op(store, conv_id: str, *, op_id: str, worker_id: str,
                     epoch: int = 1, tenant_id: str = "t-default") -> None:
    """Seed conversation + lifecycle + running compaction_operation in PG."""
    with store.pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                """INSERT INTO conversation_lifecycle
                   (conversation_id, generation, deleted, updated_at)
                   VALUES (%s, 0, FALSE, %s)
                   ON CONFLICT (conversation_id) DO NOTHING""",
                (conv_id, _TS),
            )
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, tenant_id, phase, lifecycle_epoch,
                    created_at, updated_at)
                   VALUES (%s, %s, 'compacting', %s, %s, %s)
                   ON CONFLICT (conversation_id) DO UPDATE SET
                       tenant_id = EXCLUDED.tenant_id,
                       phase = EXCLUDED.phase,
                       lifecycle_epoch = EXCLUDED.lifecycle_epoch""",
                (conv_id, tenant_id, epoch, _TS, _TS),
            )
            conn.execute(
                """INSERT INTO compaction_operation
                   (operation_id, conversation_id, lifecycle_epoch,
                    phase_index, phase_count, phase_name, status,
                    started_at, heartbeat_ts, owner_worker_id, created_at)
                   VALUES (%s, %s, %s, 0, 7, 'starting', 'running',
                           %s, %s, %s, %s)""",
                (op_id, conv_id, epoch, _TS, _TS, worker_id, _TS),
            )


def _seed_segment(store, ref: str, conv_id: str) -> None:
    with store.pool.connection() as conn:
        conn.execute(
            """INSERT INTO segments
               (ref, conversation_id, primary_tag, created_at,
                start_timestamp, end_timestamp)
               VALUES (%s, %s, '_general', %s, %s, %s)
               ON CONFLICT (ref) DO NOTHING""",
            (ref, conv_id, _TS, _TS, _TS),
        )


def _seed_fact(store, fact_id: str, conv_id: str) -> None:
    with store.pool.connection() as conn:
        conn.execute(
            """INSERT INTO facts
               (id, subject, verb, object, status, what, who, when_date,
                "where", why, fact_type, tags_json, segment_ref,
                conversation_id, turn_numbers_json, mentioned_at,
                session_date)
               VALUES (%s, 'alice', 'likes', 'tea', 'active', '', '',
                       '', '', '', 'personal', '[]', '', %s, '[]',
                       %s, '')
               ON CONFLICT (id) DO NOTHING""",
            (fact_id, conv_id, _TS),
        )


def _fact_operation_id(store, fact_id: str) -> str | None:
    with store.pool.connection() as conn:
        row = conn.execute(
            "SELECT operation_id FROM facts WHERE id = %s", (fact_id,),
        ).fetchone()
        return row["operation_id"] if row else None


@_pg_required
class TestT322_PGAllFiveMethodsEndToEnd:
    """T3.22: end-to-end fence behavior across all five methods."""

    def test_store_chunk_embeddings_matching_guard_succeeds(self, store):
        conv = f"pg-t322-chunks-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        seg = f"seg-{uuid.uuid4().hex[:8]}"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_segment(store, seg, conv)
        store.store_chunk_embeddings(
            seg,
            [ChunkEmbedding(segment_ref=seg, chunk_index=0,
                            text="x", embedding=[0.1])],
            operation_id=op, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )

    def test_store_chunk_embeddings_mismatched_op_raises(self, store):
        conv = f"pg-t322-chunks-miss-{uuid.uuid4().hex[:8]}"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        seg = f"seg-{uuid.uuid4().hex[:8]}"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        _seed_segment(store, seg, conv)
        with pytest.raises(CompactionLeaseLost):
            store.store_chunk_embeddings(
                seg,
                [ChunkEmbedding(segment_ref=seg, chunk_index=0,
                                text="x", embedding=[0.1])],
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=conv,
            )

    def test_link_segment_tool_output_matching_guard_succeeds(self, store):
        conv = f"pg-t322-lsto-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        store.link_segment_tool_output(
            conv, f"seg-{uuid.uuid4().hex[:8]}",
            f"tool-{uuid.uuid4().hex[:8]}",
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )

    def test_link_segment_tool_output_mismatched_op_raises(self, store):
        conv = f"pg-t322-lsto-miss-{uuid.uuid4().hex[:8]}"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        _seed_running_op(store, conv, op_id=op_real, worker_id="w-1")
        with pytest.raises(CompactionLeaseLost):
            store.link_segment_tool_output(
                conv, f"seg-{uuid.uuid4().hex[:8]}",
                f"tool-{uuid.uuid4().hex[:8]}",
                operation_id=op_loser, owner_worker_id="w-1",
                lifecycle_epoch=1,
            )

    def test_store_fact_links_matching_guard_succeeds(self, store):
        conv = f"pg-t322-fl-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        src = f"fact-{uuid.uuid4().hex[:8]}"
        tgt = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, src, conv)
        _seed_fact(store, tgt, conv)
        link = FactLink(source_fact_id=src, target_fact_id=tgt,
                        relation_type="r", confidence=1.0, context="c",
                        created_by="compaction")
        n = store.store_fact_links(
            [link],
            operation_id=op, owner_worker_id=worker,
            lifecycle_epoch=1, conversation_id=conv,
        )
        assert n == 1

    def test_store_fact_links_mismatched_op_raises(self, store):
        conv = f"pg-t322-fl-miss-{uuid.uuid4().hex[:8]}"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        src = f"fact-{uuid.uuid4().hex[:8]}"
        tgt = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, src, conv)
        _seed_fact(store, tgt, conv)
        link = FactLink(source_fact_id=src, target_fact_id=tgt,
                        relation_type="r", confidence=1.0, context="c",
                        created_by="compaction")
        with pytest.raises(CompactionLeaseLost):
            store.store_fact_links(
                [link],
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=conv,
            )

    def test_set_fact_superseded_matching_guard_succeeds(self, store):
        conv = f"pg-t322-sfs-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        old = f"fact-{uuid.uuid4().hex[:8]}"
        new = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, old, conv)
        _seed_fact(store, new, conv)
        store.set_fact_superseded(
            old, new,
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )

    def test_set_fact_superseded_mismatched_op_raises(self, store):
        conv = f"pg-t322-sfs-miss-{uuid.uuid4().hex[:8]}"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        old = f"fact-{uuid.uuid4().hex[:8]}"
        new = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, old, conv)
        _seed_fact(store, new, conv)
        with pytest.raises(CompactionLeaseLost):
            store.set_fact_superseded(
                old, new,
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1,
            )

    def test_update_fact_fields_matching_guard_succeeds(self, store):
        conv = f"pg-t322-uff-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        fid = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, fid, conv)
        store.update_fact_fields(
            fid, "knows", "python", "completed", "learned",
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )

    def test_update_fact_fields_mismatched_op_raises(self, store):
        conv = f"pg-t322-uff-miss-{uuid.uuid4().hex[:8]}"
        op_real = uuid.uuid4().hex
        op_loser = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op_real, worker_id=worker)
        fid = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, fid, conv)
        with pytest.raises(CompactionLeaseLost):
            store.update_fact_fields(
                fid, "v", "o", "active", "w",
                operation_id=op_loser, owner_worker_id=worker,
                lifecycle_epoch=1,
            )


@_pg_required
class TestT323_PGFactsStampingEndToEnd:
    """T3.23: store_facts + replace_facts_for_segment stamp
    facts.operation_id on insert and conflict-update."""

    def test_store_facts_stamps_operation_id_on_insert(self, store):
        conv = f"pg-t323-store-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        fid = f"fact-{uuid.uuid4().hex[:8]}"
        fact = Fact(id=fid, subject="s", verb="v", object="o",
                    conversation_id=conv)
        store.store_facts(
            [fact],
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        assert _fact_operation_id(store, fid) == op

    def test_replace_facts_for_segment_stamps_operation_id(self, store):
        conv = f"pg-t323-repl-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        seg = f"seg-{uuid.uuid4().hex[:8]}"
        _seed_running_op(store, conv, op_id=op, worker_id=worker)
        _seed_segment(store, seg, conv)
        fid = f"fact-{uuid.uuid4().hex[:8]}"
        replacement = Fact(id=fid, subject="s", verb="v", object="o",
                           segment_ref=seg, conversation_id=conv)
        store.replace_facts_for_segment(
            conv, seg, [replacement],
            operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
        )
        assert _fact_operation_id(store, fid) == op


@_pg_required
class TestT324_PGCrossTenantProbe:
    """T3.24: cross-tenant probe. Caller supplies endpoint facts
    from tenant A while the active op is on tenant B. The fence
    must reject via the both-endpoint cross-conversation validation
    because the active op's conversation does not match either
    endpoint's conversation.
    """

    def test_cross_tenant_endpoints_raise_lease_lost_on_fact_links(self, store):
        conv_a = f"pg-t324-a-{uuid.uuid4().hex[:8]}"
        conv_b = f"pg-t324-b-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv_b, op_id=op, worker_id=worker,
                         tenant_id="t-B")
        src = f"fact-{uuid.uuid4().hex[:8]}"
        tgt = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, src, conv_a)
        _seed_fact(store, tgt, conv_a)
        link = FactLink(
            source_fact_id=src, target_fact_id=tgt,
            relation_type="r", confidence=1.0, context="c",
            created_by="compaction",
        )
        with pytest.raises(CompactionLeaseLost):
            # Caller-supplied conversation_id matches the active op
            # (conv_b), but the endpoint facts live in conv_a. The
            # f_src.conversation_id = conversation_id predicate fails.
            store.store_fact_links(
                [link],
                operation_id=op, owner_worker_id=worker,
                lifecycle_epoch=1, conversation_id=conv_b,
            )

    def test_cross_tenant_endpoints_raise_lease_lost_on_supersede(self, store):
        conv_a = f"pg-t324-supA-{uuid.uuid4().hex[:8]}"
        conv_b = f"pg-t324-supB-{uuid.uuid4().hex[:8]}"
        op = uuid.uuid4().hex
        worker = "w-1"
        _seed_running_op(store, conv_b, op_id=op, worker_id=worker,
                         tenant_id="t-B")
        old = f"fact-{uuid.uuid4().hex[:8]}"
        new = f"fact-{uuid.uuid4().hex[:8]}"
        _seed_fact(store, old, conv_a)
        _seed_fact(store, new, conv_a)
        with pytest.raises(CompactionLeaseLost):
            # Active op is on conv_b but the facts live in conv_a;
            # the co.conversation_id = f_old.conversation_id predicate
            # fails.
            store.set_fact_superseded(
                old, new,
                operation_id=op, owner_worker_id=worker, lifecycle_epoch=1,
            )
