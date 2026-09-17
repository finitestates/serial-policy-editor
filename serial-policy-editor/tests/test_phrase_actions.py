import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.controller_pipeline import ControllerPipeline
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Phrase, Write
from trajectory_editor.episode_engine import EpisodeEngine, InstructionRejected
from trajectory_editor.episode_policy import EpisodeRunner
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.tui import CommandKind, parse_command


class PhraseBackend(ConformingFakeBackend):
    def tokenize(self, text, *, add_bos=False, special=False):
        if not add_bos and text == "C!":
            return [3, 5]
        return super().tokenize(text, add_bos=add_bos, special=special)


def _engine(backend=None, *, trace=False):
    return EpisodeEngine(
        backend or PhraseBackend(),
        initial_token_ids=[7],
        sampling=SamplingConfig(),
        controller_pipeline=ControllerPipeline(capture_trace=trace) if trace else None,
    )


def test_phrase_commands_mirror_continuation_and_exact_write_modes():
    common = dict(menu_size=12, default_hold_tokens=10, vocabulary_size=8)
    check = parse_command("check the phrase", **common)
    checkx = parse_command("checkx the phrase", **common)
    force = parse_command("force the phrase", **common)
    forcex = parse_command("forcex the phrase", **common)
    spaced = parse_command("check   the phrase  ", **common)

    assert (check.kind, check.phrase_text, check.phrase_mode, check.phrase_force) == (
        CommandKind.PHRASE, "the phrase", "continuation", False
    )
    assert (checkx.phrase_text, checkx.phrase_mode, checkx.phrase_force) == (
        "the phrase", "exact", False
    )
    assert (force.phrase_text, force.phrase_mode, force.phrase_force) == (
        "the phrase", "continuation", True
    )
    assert (forcex.phrase_text, forcex.phrase_mode, forcex.phrase_force) == (
        "the phrase", "exact", True
    )
    assert spaced.phrase_text == "  the phrase  "


@pytest.mark.parametrize("mode, expected", [("continuation", [1, 2]), ("exact", [6])])
def test_phrase_action_preserves_mode(mode, expected):
    engine = _engine()
    action = Phrase("A B", mode=mode)
    planned, _ = engine._write_tokens(Write(action.text, action.mode))
    assert planned == expected


def test_check_phrase_commits_only_when_every_token_is_within_bound():
    engine = _engine()
    outcome = engine.apply(Phrase("A B", mode="continuation", max_shift=0.0))

    assert outcome.resolved_token_ids == (1, 2)
    assert engine.boundary == 2
    assert engine.sampling == SamplingConfig()
    assert [row["token_id"] for row in outcome.diagnostics["tokens"]] == [1, 2]
    assert all(row["required_policy_shift"] == 0.0 for row in outcome.diagnostics["tokens"])
    assert all(row["applied_policy_shift"] == 0.0 for row in outcome.diagnostics["tokens"])


def test_check_phrase_rolls_back_without_partial_state():
    engine = _engine()
    original_sampling = engine.sampling

    with pytest.raises(InstructionRejected, match="check phrase rejected"):
        engine.apply(Phrase("C!", mode="exact", max_shift=0.5))

    assert engine.boundary == 0
    assert engine.token_ids == [7]
    assert engine.backend.tokens == [7]
    assert engine.sampling == original_sampling
    assert engine._ephemeral_logit_biases == {}


def test_force_phrase_uses_temporary_bias_and_clears_it():
    engine = _engine(trace=True)
    outcome = engine.apply(Phrase("C!", mode="exact", force=True, max_shift=0.5))

    assert outcome.resolved_token_ids == (3, 5)
    assert [row["applied_policy_shift"] for row in outcome.diagnostics["tokens"]] == pytest.approx(
        [3.000001, 10.000001]
    )
    assert [item.policy_rank for item in outcome.evidence] == [1, 1]
    assert engine._ephemeral_logit_biases == {}
    final = engine.observe()
    assert final.statistics.ephemeral_logit_biases == {}


def test_phrase_action_round_trips_through_replay_with_diagnostics(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    source_engine = _engine()
    with EpisodeStore(path) as store:
        episode_id = store.create_episode(
            episode_id="source",
            initial_text=source_engine.initial_text,
            initial_token_ids=list(source_engine.initial_token_ids),
            sampling=source_engine.sampling,
            stream_fingerprint=source_engine.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend=source_engine.backend.provenance(),
        )
        outcome = source_engine.apply(Phrase("C!", mode="exact", force=True))
        store.record_action(episode_id, 0, outcome)
        row = store.actions(episode_id)[0]
        assert row["arguments"]["diagnostics"]["operation"] == "force-phrase"
        step = store.replay_procedure(episode_id)[0]
        assert isinstance(step["action"], Phrase)

        replay_engine = _engine()
        replay_outcome = replay_engine.apply(
            step["action"], expectation=step["expectation"], replay=True
        )
        assert replay_outcome.status == "completed"
        assert replay_engine.token_ids == source_engine.token_ids


def test_phrase_action_rejects_invalid_limits():
    with pytest.raises(EditorError, match="max_tokens"):
        Phrase("A", max_tokens=0)
    with pytest.raises(EditorError, match="max_shift"):
        Phrase("A", max_shift=float("nan"))


def test_live_check_rejection_stays_at_the_edge_without_an_action(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    engine = _engine()

    class Policy:
        def __init__(self):
            self.rejections = []

        def choose(self, _engine, _observation):
            return Phrase("C!", mode="exact", max_shift=0.5)

        def action_rejected(self, action, reason):
            self.rejections.append((action.kind, reason))

    policy = Policy()
    with EpisodeStore(path) as store:
        episode_id = store.create_episode(
            episode_id="live",
            initial_text=engine.initial_text,
            initial_token_ids=list(engine.initial_token_ids),
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend=engine.backend.provenance(),
        )
        result = EpisodeRunner(engine, store, episode_id).run(
            live_policy=policy, max_live_actions=1
        )
        assert result.outcomes == ()
        assert engine.boundary == 0
        assert store.actions(episode_id) == []
        assert store.interactions(episode_id)[-1]["kind"] == "phrase-rejected"
    assert policy.rejections[0][0] == "check-phrase"
