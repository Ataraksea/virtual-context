"""Compaction-fence Phase 7 tests for the CompactionFenceMode
parser and the store-construction holder discipline.

Per fencing plan §9.0 + §9.6:

* T7.4. Invalid ``VC_COMPACTION_FENCE_MODE`` fails at store
  construction with a clear ``ValueError``, never silently
  downgrading to a weaker mode.
* T7.5. ``CompactionFenceMode.from_env()`` parses the documented
  values (default + accepted + whitespace/case normalization) under
  env isolation so no mode state leaks across cases.
* T7.6. The mode resolved at store construction is pinned for the
  lifetime of the store. Flipping the env var after construction
  does NOT affect existing stores; only fresh construction observes
  the new value. Cleanup and write paths read the same holder.

The behavioral tier gates (T7.1 / T7.2 / T7.3) -- runtime branching
on ``self._compaction_fence_mode`` inside the per-write fence and
cleanup paths -- are scheduled for a follow-up commit so the
construction holder and the env-parse contract land independently
and so the behavioral wiring can be reviewed against a tighter
patch.

Tests use ``monkeypatch`` to scope every ``VC_COMPACTION_FENCE_MODE``
manipulation to a single test case, restoring the prior value at
teardown.
"""

from __future__ import annotations

import pytest

from virtual_context.core.compaction_fence import CompactionFenceMode
from virtual_context.storage.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# T7.5: parser tests for CompactionFenceMode.from_env.
# ---------------------------------------------------------------------------


class TestT75_FromEnvParser:
    """T7.5: env-isolated parser tests covering default, accepted
    values, normalization, and unset behavior.
    """

    def test_default_is_off_when_var_unset(self, monkeypatch):
        monkeypatch.delenv("VC_COMPACTION_FENCE_MODE", raising=False)
        assert CompactionFenceMode.from_env() is CompactionFenceMode.OFF

    @pytest.mark.parametrize("raw,expected", [
        ("off", CompactionFenceMode.OFF),
        ("observe", CompactionFenceMode.OBSERVE),
        ("active", CompactionFenceMode.ACTIVE),
    ])
    def test_accepted_values(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        assert CompactionFenceMode.from_env() is expected

    @pytest.mark.parametrize("raw", ["OFF", "OBSERVE", "ACTIVE", "Off",
                                     "Observe", "Active"])
    def test_case_insensitive(self, monkeypatch, raw):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        assert CompactionFenceMode.from_env().value == raw.lower()

    @pytest.mark.parametrize("raw,expected", [
        ("  off  ", CompactionFenceMode.OFF),
        ("\tobserve\n", CompactionFenceMode.OBSERVE),
        (" ACTIVE ", CompactionFenceMode.ACTIVE),
    ])
    def test_whitespace_stripped(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        assert CompactionFenceMode.from_env() is expected

    def test_explicit_environ_overrides_os_environ(self, monkeypatch):
        # Set os.environ to one value and pass a different mapping; the
        # passed mapping must win so callers can scope a from_env() call
        # to a captured environment snapshot.
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "off")
        assert (
            CompactionFenceMode.from_env(
                environ={"VC_COMPACTION_FENCE_MODE": "active"},
            )
            is CompactionFenceMode.ACTIVE
        )


# ---------------------------------------------------------------------------
# T7.4: invalid env values fail loud at construction time.
# ---------------------------------------------------------------------------


class TestT74_InvalidValueFailsLoud:
    @pytest.mark.parametrize("raw", [
        "enforce", "ENFORCED", "1", "true", "off,observe",
        " on ", "act ive",
    ])
    def test_invalid_value_raises_value_error(self, monkeypatch, raw):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", raw)
        with pytest.raises(ValueError, match="VC_COMPACTION_FENCE_MODE"):
            CompactionFenceMode.from_env()

    def test_invalid_env_blocks_sqlite_store_construction(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "ENFORCED")
        with pytest.raises(ValueError, match="VC_COMPACTION_FENCE_MODE"):
            SQLiteStore(tmp_path / "bad.db")

    def test_explicit_kwarg_bypasses_invalid_env(self, monkeypatch, tmp_path):
        """When the caller passes an explicit ``CompactionFenceMode``,
        the env var is not consulted -- a bad env var must not break
        construction in that case.
        """
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "garbage")
        store = SQLiteStore(
            tmp_path / "ok.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        assert store._compaction_fence_mode is CompactionFenceMode.OBSERVE


# ---------------------------------------------------------------------------
# T7.6: store-lifetime invariants.
# ---------------------------------------------------------------------------


class TestT76_StoreLifetimePin:
    """T7.6: mode resolved at construction is pinned for the
    store's lifetime. Flipping the env afterward must not affect
    the existing store; a new store sees the new value.
    """

    def test_existing_store_unchanged_when_env_flips(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "active")
        store = SQLiteStore(tmp_path / "pin.db")
        assert store._compaction_fence_mode is CompactionFenceMode.ACTIVE

        # Flip the env mid-lifetime. The store keeps its pinned holder.
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "off")
        assert store._compaction_fence_mode is CompactionFenceMode.ACTIVE

    def test_new_store_observes_new_env_after_flip(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "active")
        store_a = SQLiteStore(tmp_path / "a.db")
        assert store_a._compaction_fence_mode is CompactionFenceMode.ACTIVE

        monkeypatch.setenv("VC_COMPACTION_FENCE_MODE", "observe")
        store_b = SQLiteStore(tmp_path / "b.db")
        # store_a still ACTIVE; store_b OBSERVE.
        assert store_a._compaction_fence_mode is CompactionFenceMode.ACTIVE
        assert store_b._compaction_fence_mode is CompactionFenceMode.OBSERVE


# ---------------------------------------------------------------------------
# Holder property surface: enforces / stamps_operation_id flags map
# the tier semantics consistently. The behavioral wiring that READS
# these will land in the follow-up commit; verifying the holder's
# contract now keeps that follow-up small.
# ---------------------------------------------------------------------------


class TestModePropertySurface:
    def test_off_holder_flags(self):
        m = CompactionFenceMode.OFF
        assert m.is_off and not m.is_observe and not m.is_active
        assert not m.enforces
        assert not m.stamps_operation_id

    def test_observe_holder_flags(self):
        m = CompactionFenceMode.OBSERVE
        assert m.is_observe and not m.is_off and not m.is_active
        assert not m.enforces
        assert m.stamps_operation_id

    def test_active_holder_flags(self):
        m = CompactionFenceMode.ACTIVE
        assert m.is_active and not m.is_off and not m.is_observe
        assert m.enforces
        assert m.stamps_operation_id


# ---------------------------------------------------------------------------
# CompositeStore mode-mismatch guard: when a delegate exposes a
# different mode holder than the composite's, construction fails so
# cleanup and write paths cannot run on divergent sources.
# ---------------------------------------------------------------------------


class TestCompositeStoreModeMismatch:
    def test_matching_modes_pass(self, tmp_path):
        from virtual_context.core.composite_store import CompositeStore
        store = SQLiteStore(
            tmp_path / "comp-match.db",
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        composite = CompositeStore(
            segments=store, facts=store, fact_links=store,
            state=store, search=store,
            compaction_fence_mode=CompactionFenceMode.OBSERVE,
        )
        assert composite._compaction_fence_mode is CompactionFenceMode.OBSERVE

    def test_mismatched_modes_raise(self, tmp_path):
        from virtual_context.core.composite_store import CompositeStore
        active_store = SQLiteStore(
            tmp_path / "comp-mix.db",
            compaction_fence_mode=CompactionFenceMode.ACTIVE,
        )
        with pytest.raises(ValueError, match="compaction_fence_mode mismatch"):
            CompositeStore(
                segments=active_store, facts=active_store,
                fact_links=active_store, state=active_store,
                search=active_store,
                compaction_fence_mode=CompactionFenceMode.OBSERVE,
            )
