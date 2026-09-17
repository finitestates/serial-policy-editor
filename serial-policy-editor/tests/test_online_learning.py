import pytest
import numpy as np

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Accept, SelectRawRank
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import build_parser
from trajectory_editor.online_learning import OnlineLearner


class TwoStageBackend(ConformingFakeBackend):
    """Expose one learned group dimension again on a different future token."""

    def last_logits(self):
        if self.tokens[-1] == 7:
            logits = np.full(self.vocabulary_size(), -10.0, dtype=np.float64)
            logits[1], logits[2], logits[3], logits[0] = 10.0, 8.0, 7.0, -9.0
            return logits
        if self.tokens[-1] == 3:
            logits = np.full(self.vocabulary_size(), -10.0, dtype=np.float64)
            logits[1], logits[2], logits[4], logits[5], logits[0] = (
                10.0, 7.02, 7.0, 6.0, -9.0
            )
            return logits
        return super().last_logits()


def _group(name: str, route: tuple[int, ...]) -> BiasGroup:
    return BiasGroup(name, (BiasRule(routes=(route,), bias=0.0),))


def _engine(sampling: SamplingConfig) -> EpisodeEngine:
    return EpisodeEngine(
        ConformingFakeBackend(), initial_token_ids=[7], sampling=sampling
    )


def _sampling(*groups: BiasGroup) -> SamplingConfig:
    return SamplingConfig(
        temperature=0.8,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
        bias_groups=groups,
    )


def test_online_learner_moves_the_group_that_makes_choice_more_probable():
    sampling = _sampling(_group("concrete", (3,)))
    runtime = _engine(sampling)
    observation = runtime.observe()

    result = OnlineLearner(
        enabled=True, learning_rate=0.5, epsilon=0.05, max_step=0.25
    ).update(observation, 3, sampling)

    assert result.old_policy_rank == 3
    assert result.old_policy_probability == pytest.approx(
        observation.statistics.policy_probabilities[3]
    )
    assert result.severity > 0.0
    assert result.gradients["concrete"] < 0.0
    assert result.group_deltas["concrete"] > 0.0
    assert result.new_group_weights["concrete"] > 0.0
    assert result.update_norm == pytest.approx(result.group_deltas["concrete"])
    assert result.sampling.bias_groups[0].rules == sampling.bias_groups[0].rules


def test_online_learner_does_not_change_unselected_or_inactive_groups():
    sampling = _sampling(
        _group("concrete", (3,)),
        _group("inactive", (4, 3)),
        _group("fixed", (2,)),
    )
    runtime = _engine(sampling)
    result = OnlineLearner(
        enabled=True,
        learning_rate=1.0,
        learnable_groups=("concrete", "inactive"),
    ).update(runtime.observe(), 3, sampling)

    assert result.group_deltas["concrete"] > 0.0
    assert result.group_deltas["inactive"] == pytest.approx(0.0)
    assert result.group_deltas["fixed"] == pytest.approx(0.0)
    assert result.gradients["fixed"] == pytest.approx(0.0)
    assert {
        group.name: group.bias for group in result.sampling.bias_groups
    }["fixed"] == 0.0


def test_online_learner_is_bounded_and_off_by_default():
    group = _group("concrete", (3,))
    sampling = _sampling(group)
    runtime = _engine(sampling)
    disabled = OnlineLearner().update(runtime.observe(), 3, sampling)
    assert disabled.sampling == sampling
    assert disabled.update_norm == 0.0
    assert disabled.enabled is False

    bounded = OnlineLearner(
        enabled=True,
        learning_rate=100.0,
        max_step=0.01,
        min_bias=-0.02,
        max_bias=0.02,
    ).update(runtime.observe(), 3, sampling)
    assert bounded.group_deltas["concrete"] <= 0.01
    assert -0.02 <= bounded.new_group_weights["concrete"] <= 0.02


def test_online_learner_rejects_unknown_group_selection():
    sampling = _sampling(_group("concrete", (3,)))
    runtime = _engine(sampling)
    with pytest.raises(EditorError, match="not present"):
        OnlineLearner(
            enabled=True, learnable_groups=("missing",)
        ).update(runtime.observe(), 3, sampling)


def test_a_later_never_selected_token_benefits_from_the_group_update():
    group = BiasGroup(
        "concrete",
        (
            BiasRule(routes=((3,),), bias=0.0, triggers=((7,),), until=3),
            BiasRule(routes=((4,),), bias=0.0, triggers=((3,),), until=0),
        ),
    )
    sampling = _sampling(group)

    learned = EpisodeEngine(
        TwoStageBackend(), initial_token_ids=[7], sampling=sampling
    )
    before = learned.observe()
    update = OnlineLearner(enabled=True, learning_rate=0.5).update(
        before, 3, sampling
    )
    learned.apply(SelectRawRank(3))
    learned.sampling = update.sampling
    later = learned.observe()

    baseline = EpisodeEngine(
        TwoStageBackend(), initial_token_ids=[7], sampling=sampling
    )
    baseline.apply(SelectRawRank(3))
    baseline_later = baseline.observe()

    assert 4 not in learned.visible_token_ids
    assert later.statistics.policy_rank(4) < baseline_later.statistics.policy_rank(4)


def test_runner_learns_only_after_live_select_raw_rank_and_persists_boundary(tmp_path):
    sampling = _sampling(_group("concrete", (3,)))
    runtime = _engine(sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="live-learning",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        result = EpisodeRunner(
            runtime,
            store,
            episode_id,
            learner=OnlineLearner(enabled=True, learning_rate=0.5),
        ).run(
            live_policy=type("Once", (), {"choose": lambda self, _e, _o: SelectRawRank(3)})(),
            max_live_actions=1,
        )

        assert result.outcomes[0].evidence[0].token_id == 3
        assert runtime.sampling.bias_groups[0].bias > 0.0
        segment = store.sampling_segment(episode_id, 1)
        saved = SamplingConfig.from_record(segment["sampling"])
        assert saved.bias_groups[0].bias == runtime.sampling.bias_groups[0].bias
        interaction = store.interactions(episode_id)[-1]
        assert interaction["kind"] == "online-learning-update"
        assert interaction["boundary"] == 1
        assert interaction["payload"]["observation_boundary"] == 0
        assert interaction["payload"]["chosen_token_id"] == 3
        assert interaction["payload"]["new_group_weights"]["concrete"] > 0.0


def test_runner_learns_after_live_accept_as_an_explicit_teacher_selection(tmp_path):
    sampling = _sampling(_group("concrete", (3,)))
    runtime = _engine(sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="live-accept-learning",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        result = EpisodeRunner(
            runtime,
            store,
            episode_id,
            learner=OnlineLearner(enabled=True, learning_rate=0.5),
        ).run(
            live_policy=type("Once", (), {"choose": lambda self, _e, _o: Accept()})(),
            max_live_actions=1,
        )

        assert result.outcomes[0].evidence[0].token_id == 3
        assert runtime.sampling.bias_groups[0].bias > 0.0
        interaction = store.interactions(episode_id)[-1]
        assert interaction["kind"] == "online-learning-update"
        assert interaction["payload"]["chosen_token_id"] == 3


def test_runner_does_not_learn_during_replay(tmp_path):
    sampling = _sampling(_group("concrete", (3,)))
    runtime = _engine(sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="replay-no-learning",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        result = EpisodeRunner(
            runtime, store, episode_id,
            learner=OnlineLearner(enabled=True),
        ).run(
            tape=[TapeStep(SelectRawRank(3), ReplayExpectation((3,)))],
        )

        assert result.replayed_actions == 1
        assert runtime.sampling == sampling
        assert store.interactions(episode_id) == []
        assert store.sampling_segment(episode_id, 1)["start_boundary"] == 0


def test_online_learning_cli_controls_are_conservative_and_explicit():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.online_learning is False
    args = parser.parse_args([
        "--online-learning",
        "--learning-rate", "0.1",
        "--epsilon", "0.02",
        "--max-step", "0.03",
        "--min-bias", "-1",
        "--max-bias", "1",
        "--learnable-groups", "technical", "unusual",
    ])
    assert args.online_learning is True
    assert args.learning_rate == 0.1
    assert args.learning_epsilon == 0.02
    assert args.learning_max_step == 0.03
    assert args.learning_min_bias == -1
    assert args.learning_max_bias == 1
    assert args.learnable_groups == ["technical", "unusual"]
