import numpy as np

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import build_parser
from trajectory_editor.token_preference import TokenPreferenceLearner
from trajectory_editor.online_learning import OnlineLearner


FEATURES = np.asarray(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.95, 0.05],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.2, 0.0],
        [0.0, 0.0],
    ],
    dtype=np.float32,
)


class WriteBackend(ConformingFakeBackend):
    def token_preference_features(self, *, feature_dimension: int, projection_seed: int):
        del projection_seed
        assert feature_dimension == FEATURES.shape[1]
        return FEATURES

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        if not add_bos and text == "C!":
            return [3, 5]
        return super().tokenize(text, add_bos=add_bos, special=special)


def _group() -> BiasGroup:
    return BiasGroup(
        "concrete",
        (
            BiasRule(routes=((3,),), bias=0.0),
            BiasRule(routes=((5,),), bias=0.0),
        ),
    )


def _run_write(
    tmp_path, *, learner=None, token_preference_learner=None, replay=False, learn_from_write=True
):
    sampling = SamplingConfig(
        bias_groups=(_group(),) if learner is not None else (),
    )
    runtime = EpisodeEngine(
        WriteBackend(), initial_token_ids=[7], sampling=sampling
    )
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="write-learning",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        runner = EpisodeRunner(
            runtime,
            store,
            episode_id,
            learner=learner,
            token_preference_learner=token_preference_learner,
            learn_from_write=learn_from_write,
        )
        result = runner.run(
            tape=(
                [TapeStep(Write("C!", "exact"), ReplayExpectation((3, 5)))]
                if replay
                else None
            ),
            live_policy=(
                None
                if replay
                else type(
                    "Once",
                    (),
                    {"choose": lambda self, _e, _o: Write("C!", "exact")},
                )()
            ),
            max_live_actions=1,
        )
        interactions = store.interactions(episode_id)
        segment = store.sampling_segment(episode_id, runtime.boundary)
        return runtime, interactions, segment, result


def test_write_learning_averages_tokens_and_persists_one_boundary(tmp_path):
    runtime, interactions, segment, result = _run_write(
        tmp_path,
        learner=OnlineLearner(enabled=True, learning_rate=0.5),
    )

    assert result.outcomes[0].resolved_token_ids == (3, 5)
    assert runtime.boundary == 2
    assert runtime.sampling.bias_groups[0].bias > 0.0
    assert segment["start_boundary"] == 2
    interaction = interactions[-1]
    assert interaction["kind"] == "write-learning-update"
    assert interaction["boundary"] == 2
    payload = interaction["payload"]
    assert payload["token_count"] == 2
    assert [item["chosen_token_id"] for item in payload["tokens"]] == [3, 5]
    assert [item["observation_boundary"] for item in payload["tokens"]] == [0, 1]
    assert [item["old_policy_rank"] for item in payload["tokens"]] == [3, 2]
    assert all(item["old_policy_probability"] > 0.0 for item in payload["tokens"])
    assert all(item["loss"] > 0.0 for item in payload["tokens"])


def test_write_learning_combines_named_and_token_preference_updates(tmp_path):
    runtime, interactions, _segment, _ = _run_write(
        tmp_path,
        learner=OnlineLearner(enabled=True, learning_rate=0.5),
        token_preference_learner=TokenPreferenceLearner(
            FEATURES,
            enabled=True,
            dimension=2,
            learning_rate=0.5,
        ),
    )

    assert runtime.sampling.bias_groups[0].bias > 0.0
    assert runtime.sampling.token_preference_vector
    payload = interactions[-1]["payload"]
    assert "group_update" in payload
    assert "token_preference_update" in payload


def test_write_learning_does_not_run_during_replay(tmp_path):
    runtime, interactions, _segment, result = _run_write(
        tmp_path,
        learner=OnlineLearner(enabled=True),
        token_preference_learner=TokenPreferenceLearner(
            FEATURES,
            enabled=True,
            dimension=2,
        ),
        replay=True,
    )

    assert result.replayed_actions == 1
    assert runtime.sampling.token_preference_vector == ()
    assert runtime.sampling.bias_groups[0].bias == 0.0
    assert interactions == []


def test_write_learning_is_separately_opt_in_at_the_cli(tmp_path):
    parser = build_parser()
    assert parser.parse_args([]).learn_from_write is False
    args = parser.parse_args(["--online-learning", "--learn-from-write"])
    assert args.online_learning is True
    assert args.learn_from_write is True


def test_existing_learners_ignore_write_without_the_new_opt_in(tmp_path):
    runtime, interactions, _segment, _result = _run_write(
        tmp_path,
        learner=OnlineLearner(enabled=True),
        learn_from_write=False,
    )

    assert runtime.sampling.bias_groups[0].bias == 0.0
    assert interactions == []
