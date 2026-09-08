import pytest

from tests.test_episode_runtime import NoEogBackend
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Hold
from trajectory_editor.episode_cli import (
    _create_episode, _fork_engine, _model_continuation, _restore_engine, _rewind_episode,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore


def start(store, allowance=10):
    engine = EpisodeEngine(NoEogBackend(), initial_token_ids=[7],
                           sampling=SamplingConfig(temperature=0), max_tokens=allowance)
    identifier = _create_episode(store, engine, backend_provenance={})
    return identifier, engine


def advance(store, identifier, engine, count):
    store.record_action(identifier, store.next_action_ordinal(identifier), engine.apply(Hold(count)))
    store.update_episode(identifier, visible_text=engine.text, max_tokens=engine.max_tokens)


def change(store, identifier, engine, allowance="keep"):
    engine.resume(max_tokens=allowance)
    store.record_budget(identifier, engine.boundary, engine.max_tokens, engine.checkpoint_boundary)


@pytest.mark.parametrize("target,remaining", [(0, 10), (3, 7), (9, 1), (10, 10), (12, 8)])
def test_rewind_and_fork_restore_budget_before_and_after_renewal(tmp_path, target, remaining):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store)
        advance(store, identifier, engine, 10)
        change(store, identifier, engine)
        advance(store, identifier, engine, 3)
        child = _fork_engine(store, identifier, engine, target, backend=NoEogBackend(), max_tokens=None)
        assert (child.max_tokens, child.remaining) == (10, remaining)
        assert engine.remaining == 7
        child_id = _create_episode(store, child, backend_provenance={})
        restored = _restore_engine(store, child_id, NoEogBackend(), max_tokens=None, sampling_override=None)
        assert (restored.max_tokens, restored.remaining) == (10, remaining)
        _rewind_episode(store, identifier, engine, target)
        assert (engine.max_tokens, engine.remaining) == (10, remaining)
        restored = _restore_engine(store, identifier, NoEogBackend(), max_tokens=None, sampling_override=None)
        assert restored.remaining == remaining
        assert store.budget_at(identifier, target)["checkpoint_boundary"] == target + remaining


@pytest.mark.parametrize("target,allowance,remaining", [(1, 10, 9), (2, None, None), (4, 5, 5), (5, 5, 4)])
def test_budget_edits_restore_at_exact_boundary(tmp_path, target, allowance, remaining):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store)
        advance(store, identifier, engine, 2)
        change(store, identifier, engine, None)
        advance(store, identifier, engine, 2)
        change(store, identifier, engine, 5)
        advance(store, identifier, engine, 1)
        _rewind_episode(store, identifier, engine, target)
        assert (engine.max_tokens, engine.remaining) == (allowance, remaining)


def test_exhausted_fork_stays_exhausted_until_user_continues(tmp_path):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store, 2)
        advance(store, identifier, engine, 2)
        child = _fork_engine(store, identifier, engine, 2, backend=NoEogBackend(), max_tokens=None)
        assert child.checkpointed
        child_id = _create_episode(store, child, backend_provenance={})
        child = _restore_engine(store, child_id, NoEogBackend(), max_tokens=None, sampling_override=None)
        assert child.checkpointed
        change(store, child_id, child)
        assert child.remaining == 2


def test_pause_does_not_renew_but_explicit_resume_override_does(tmp_path):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store)
        advance(store, identifier, engine, 3)
        change(store, identifier, engine)
        assert engine.remaining == 7
        assert store.connection.execute("SELECT COUNT(*) FROM budget_segments").fetchone()[0] == 1
        restored = _restore_engine(store, identifier, NoEogBackend(), max_tokens=4, sampling_override=None)
        assert (restored.max_tokens, restored.remaining) == (4, 4)


def test_model_continuation_preserves_remaining(tmp_path):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store)
        advance(store, identifier, engine, 3)
        child, child_id = _model_continuation(store, identifier, NoEogBackend(), {})
        assert (child.boundary, child.max_tokens, child.remaining) == (0, 10, 7)
        assert store.budget_at(child_id, 0)["checkpoint_boundary"] == 7


@pytest.mark.parametrize("operation", ["rewind", "fork", "resume"])
def test_missing_history_uses_unlimited_without_guessing(tmp_path, capsys, operation):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store)
        advance(store, identifier, engine, 3)
        with store.transaction() as db:
            db.execute("DELETE FROM budget_segments")
        if operation == "rewind":
            _rewind_episode(store, identifier, engine, 1)
        elif operation == "fork":
            engine = _fork_engine(store, identifier, engine, 1, backend=NoEogBackend(), max_tokens=None)
        else:
            engine = _restore_engine(store, identifier, NoEogBackend(), max_tokens=None, sampling_override=None)
        assert engine.max_tokens is None and engine.remaining is None
        assert "history is missing" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["rewind", "fork", "resume"])
def test_unlimited_without_history_is_quiet(tmp_path, capsys, operation):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store, None)
        advance(store, identifier, engine, 3)
        with store.transaction() as db:
            db.execute("DELETE FROM budget_segments")
        if operation == "rewind":
            _rewind_episode(store, identifier, engine, 1)
        elif operation == "fork":
            engine = _fork_engine(store, identifier, engine, 1, backend=NoEogBackend(), max_tokens=None)
        else:
            engine = _restore_engine(store, identifier, NoEogBackend(), max_tokens=None, sampling_override=None)
        assert engine.remaining is None
        assert capsys.readouterr().out == ""


def test_incomplete_finite_history_warns_and_uses_unlimited(tmp_path, capsys):
    with EpisodeStore(tmp_path / "ledger") as store:
        identifier, engine = start(store)
        with store.transaction() as db:
            db.execute("UPDATE budget_segments SET checkpoint_boundary = NULL")
        restored = _restore_engine(store, identifier, NoEogBackend(), max_tokens=None, sampling_override=None)
        assert restored.remaining is None
        assert "history is missing" in capsys.readouterr().out
