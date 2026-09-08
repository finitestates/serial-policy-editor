from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import pytest

from tests.fakes import (
    BranchingFakeBackend,
    ChangedProposalBackend,
    ConformingFakeBackend,
    ScriptedIO,
)
from trajectory_editor.domain import MAX_SEED, MIN_SEED, SamplingConfig
from trajectory_editor.episode_actions import Accept, EndGeneration, Finish, Hold, Write
from trajectory_editor.episode_cli import (
    _fork_engine,
    _random_seed,
    _sampling_from_args,
    build_parser,
    main as episode_main,
)
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_policy import (
    EpisodeRunner,
    SeamlessRewindRequested,
    TapeStep,
)
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
try:
    from trajectory_editor.live_tui import (
        _candidate_command_cycle,
        action_preview,
        read_live_choice,
    )
except ModuleNotFoundError:  # Live UI dependency is optional in minimal test runs.
    _candidate_command_cycle = None
    action_preview = None
    read_live_choice = None


class LiveScriptedIO(ScriptedIO):
    def __init__(self, responses: list[str | None]) -> None:
        super().__init__(responses)
        self.initial_commands: list[str | None] = []
        self.remaining_tokens: list[int] = []

    @property
    def supports_live_choices(self) -> bool:
        return True

    def read_choice(self, choice, **kwargs):
        del choice
        self.initial_commands.append(kwargs["initial_command"])
        self.remaining_tokens.append(kwargs["remaining_tokens"])
        if not self.responses:
            raise AssertionError("unexpected live input request")
        return self.responses.pop(0)


class NoEogBackend(ConformingFakeBackend):
    def last_logits(self):
        logits = super().last_logits()
        logits[0] = -100.0
        return logits


def engine(backend=None, *, max_tokens: int = 2) -> EpisodeEngine:
    return EpisodeEngine(
        backend or ConformingFakeBackend(),
        sampling=SamplingConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        max_tokens=max_tokens,
        initial_text="P",
        initial_token_ids=[7],
    )


def test_budget_is_checkpoint_not_termination() -> None:
    runtime = engine(max_tokens=1)
    runtime.apply(Accept())
    assert runtime.checkpointed
    assert not runtime.ended
    assert runtime.terminal_reason is None
    runtime.resume(max_tokens=2)
    assert not runtime.checkpointed
    assert runtime.remaining == 2


def test_finish_delegates_to_checkpoint() -> None:
    runtime = engine(max_tokens=1)
    outcome = runtime.apply(Finish())
    assert outcome.stop_reason == 'checkpoint'
    assert runtime.checkpointed
    assert not runtime.ended


def test_model_eog_during_hold_is_terminal() -> None:
    runtime = engine(max_tokens=5)
    runtime.apply(Accept())
    runtime.apply(Accept())
    outcome = runtime.apply(Hold(3))
    assert outcome.stop_reason == 'eog'
    assert runtime.ended
    assert runtime.terminal_reason == 'model-eog'


def test_teacher_eog_is_terminal() -> None:
    runtime = engine(max_tokens=5)
    outcome = runtime.apply(EndGeneration())
    assert runtime.ended
    assert runtime.terminal_reason == 'teacher-eog'
    assert outcome.stop_reason == 'eog'


def test_menu_end_can_terminate_without_eog() -> None:
    runtime = engine(max_tokens=1)
    runtime.apply(Accept())
    runtime.terminate("menu-end")
    assert runtime.ended
    assert runtime.terminal_reason == 'menu-end'


def test_rewind_shortens_a_hold_and_keeps_the_episode_id(tmp_path) -> None:
    runtime = engine(NoEogBackend(), max_tokens=20)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="seamless-hold",
            initial_text="P",
            initial_token_ids=runtime.initial_token_ids,
            sampling=runtime.sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=runtime.max_tokens,
            backend={"backend": "fake"},
        )
        outcome = runtime.apply(Hold(10))
        store.record_action(episode_id, 0, outcome)

        runtime.rewind_to(7)
        details = store.rewind_to(
            episode_id,
            7,
            visible_text=runtime.backend.render(runtime.visible_token_ids),
            max_tokens=runtime.max_tokens,
        )

        assert details['trimmed_action']['new_boundary_after'] == 7
        action = store.actions(episode_id)[0]
        assert action['arguments']['limit'] == 7
        assert action['boundary_after'] == 7
        assert len(store.tokens(episode_id)) == 7
        assert store.get_episode(episode_id)['status'] == 'open'
        assert runtime.boundary == 7
        assert not runtime.ended

        next_outcome = runtime.apply(Accept())
        store.record_action(
            episode_id, store.next_action_ordinal(episode_id), next_outcome
        )
        assert [row['kind'] for row in store.actions(episode_id)] == ['hold', 'accept']


def test_interactive_fork_reuses_a_positioned_backend(tmp_path) -> None:
    runtime_backend = BranchingFakeBackend()
    runtime = engine(runtime_backend, max_tokens=10)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        parent_id = store.create_episode(
            episode_id="fork-parent",
            initial_text="P",
            initial_token_ids=runtime.initial_token_ids,
            sampling=runtime.sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=runtime.max_tokens,
            backend={"backend": "fake"},
        )
        outcome = runtime.apply(Write(" A B", "exact"))
        store.record_action(parent_id, 0, outcome)
        reset_calls = runtime_backend.reset_calls

        child = _fork_engine(
            store,
            parent_id,
            runtime,
            1,
            backend=runtime_backend,
            max_tokens=runtime.max_tokens,
        )

        assert runtime_backend.reset_calls == reset_calls
        assert runtime_backend.branch_prefixes == [[7, 1]]
        assert child.initial_token_ids == (7, 1)
        assert child.backend is runtime_backend


def test_seamless_targets_include_the_middle_of_a_write(tmp_path) -> None:
    runtime = engine(NoEogBackend(), max_tokens=10)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="seamless-write",
            initial_text="P",
            initial_token_ids=runtime.initial_token_ids,
            sampling=runtime.sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=runtime.max_tokens,
            backend={"backend": "fake"},
        )
        outcome = runtime.apply(Write(" A B", "exact"))
        store.record_action(episode_id, 0, outcome)
        policy = InteractivePolicy(
            io=ScriptedIO([]),
            menu_size=2,
            store=store,
            episode_id=episode_id,
            seamless=True,
        )

        assert policy._seamless_targets(runtime, runtime.boundary) == (0, 1, 2)


def test_enter_on_seamless_review_requests_reactivation(tmp_path) -> None:
    from trajectory_editor.tui import SEAMLESS_REACTIVATE

    runtime = engine(NoEogBackend(), max_tokens=20)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="seamless-review",
            initial_text="P",
            initial_token_ids=runtime.initial_token_ids,
            sampling=runtime.sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=runtime.max_tokens,
            backend={"backend": "fake"},
        )
        outcome = runtime.apply(Hold(10))
        store.record_action(episode_id, 0, outcome)
        policy = InteractivePolicy(
            io=LiveScriptedIO(["[", SEAMLESS_REACTIVATE]),
            menu_size=2,
            store=store,
            episode_id=episode_id,
            seamless=True,
        )

        with pytest.raises(SeamlessRewindRequested) as raised:
            policy.choose(runtime, runtime.observe())

        assert raised.value.boundary == 9


def test_token_evidence_keeps_editorial_stats() -> None:
    runtime = engine(max_tokens=2)
    outcome = runtime.apply(Write("C", "exact"))
    evidence = outcome.evidence[0]
    assert evidence.raw_rank >= 1
    assert evidence.raw_model_nll >= 0.0
    assert isinstance(evidence.proposal_agreement, bool)
    assert evidence.proposal_token_id == 1


def test_full_vocab_search_is_nonmutating() -> None:
    runtime = engine(max_tokens=3)
    before = list(runtime.backend.tokens)
    io = ScriptedIO(["/P", "8"])
    policy = InteractivePolicy(io=io, menu_size=1, search_radius=1)
    action = policy.choose(runtime, runtime.observe())
    assert action.kind == 'select-raw-rank'
    assert runtime.backend.tokens == before
    assert 'absolute raw rank=8' in ''.join(io.output)


def test_e_confirms_but_e_bang_does_not() -> None:
    runtime = engine(max_tokens=3)
    io = ScriptedIO(["e", "x", "e!"])
    policy = InteractivePolicy(io=io, menu_size=2)
    action = policy.choose(runtime, runtime.observe())
    assert action.kind == 'end-generation'
    assert any(('confirm EOG' in item for item in io.output))


def test_proposal_rank_is_the_first_candidate_tab_command() -> None:
    if _candidate_command_cycle is None or action_preview is None:
        pytest.skip("prompt-toolkit is not installed")
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    from trajectory_editor.episode_ui import _choice_from_observation

    choice = _choice_from_observation(
        runtime, observation, candidates, context_characters=320, serial=1
    )
    commands = _candidate_command_cycle(choice, candidates)
    assert commands[0] == str(observation.proposal_raw_rank)
    assert 'a' not in commands

    preview = action_preview(choice, commands[0], (), lambda text, mode: text)
    assert preview.label == 'sampled proposal'
    assert preview.token_id == observation.proposal_token_id


def test_typing_proposal_rank_returns_rank_action() -> None:
    runtime = engine(max_tokens=3)
    proposal_rank = runtime.observe().proposal_raw_rank
    policy = InteractivePolicy(io=ScriptedIO([str(proposal_rank)]), menu_size=2)
    action = policy.choose(runtime, runtime.observe())
    assert action.kind == 'select-raw-rank'


def test_default_prefills_proposal_rank_without_auto_committing() -> None:
    runtime = engine(max_tokens=3)
    proposal_rank = runtime.observe().proposal_raw_rank
    io = LiveScriptedIO([str(proposal_rank)])
    policy = InteractivePolicy(io=io, menu_size=2)
    action = policy.choose(runtime, runtime.observe())
    assert io.initial_commands == [str(proposal_rank)]
    assert action.kind == 'select-raw-rank'


def test_resumed_budget_reaches_the_live_decision_surface() -> None:
    runtime = engine(max_tokens=1)
    runtime.apply(Accept())
    runtime.resume(max_tokens=3)
    io = LiveScriptedIO(["accept"])
    policy = InteractivePolicy(io=io, menu_size=2)

    action = policy.choose(runtime, runtime.observe())

    assert io.remaining_tokens == [3]
    assert action.kind == 'select-raw-rank'


@pytest.mark.parametrize('command', ('t', 'x', 'p', '/', 'h', 'q', 'e', 'n', '?'))
def test_first_typed_command_replaces_live_proposal_prefill(command) -> None:
    if read_live_choice is None:
        pytest.skip('prompt-toolkit is not installed')
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from trajectory_editor.episode_ui import _choice_from_observation
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(runtime, observation, candidates, context_characters=320, serial=1)
    with create_pipe_input() as pipe:
        pipe.send_text(command + '\n')
        result = read_live_choice(
            choice,
            remaining_tokens=runtime.remaining,
            candidates=candidates,
            resolve_insertion=lambda text, mode: text,
            initial_command=str(observation.proposal_raw_rank),
            input_device=pipe,
            output_device=DummyOutput(),
        )
    assert result == command


def test_tab_navigation_ends_proposal_replacement_mode() -> None:
    if read_live_choice is None or _candidate_command_cycle is None:
        pytest.skip("prompt-toolkit is not installed")
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from trajectory_editor.episode_ui import _choice_from_observation

    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(
        runtime, observation, candidates, context_characters=320, serial=1
    )
    navigation = _candidate_command_cycle(choice, candidates)
    assert len(navigation) >= 2

    with create_pipe_input() as pipe:
        pipe.send_text("\tt\n")
        result = read_live_choice(
            choice,
            remaining_tokens=runtime.remaining,
            candidates=candidates,
            resolve_insertion=lambda text, mode: text,
            initial_command=str(observation.proposal_raw_rank),
            input_device=pipe,
            output_device=DummyOutput(),
        )

    assert result == navigation[1] + 't'


def test_seamless_review_enter_returns_reactivation_signal() -> None:
    if read_live_choice is None:
        pytest.skip("prompt-toolkit is not installed")
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from trajectory_editor.episode_ui import _choice_from_observation
    from trajectory_editor.tui import BoundaryReview, SEAMLESS_REACTIVATE

    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(
        runtime, observation, candidates, context_characters=320, serial=1
    )
    review = BoundaryReview(
        aligned_step=0,
        active_aligned_step=observation.boundary,
        context_text_tail="P",
        context_token_sha256=choice.context_token_sha256,
        position={"kind": "token-boundary"},
    )

    with create_pipe_input() as pipe:
        pipe.send_text("\n")
        result = read_live_choice(
            choice,
            remaining_tokens=runtime.remaining,
            candidates=candidates,
            resolve_insertion=lambda text, mode: text,
            review=review,
            seamless=True,
            reactivate_on_review_enter=True,
            input_device=pipe,
            output_device=DummyOutput(),
        )

    assert result == SEAMLESS_REACTIVATE


def test_manual_acceptance_leaves_live_command_blank() -> None:
    runtime = engine(max_tokens=3)
    proposal_rank = runtime.observe().proposal_raw_rank
    io = LiveScriptedIO([str(proposal_rank)])
    policy = InteractivePolicy(io=io, menu_size=2, manual_acceptance=True)
    action = policy.choose(runtime, runtime.observe())
    assert io.initial_commands == [None]
    assert action.kind == 'select-raw-rank'


def test_default_plain_prompt_accepts_blank_enter() -> None:
    runtime = engine(max_tokens=3)
    policy = InteractivePolicy(io=ScriptedIO([""]), menu_size=2)
    action = policy.choose(runtime, runtime.observe())
    assert action.kind == 'select-raw-rank'


def test_manual_acceptance_resolves_blank_enter_to_proposal_rank() -> None:
    runtime = engine(max_tokens=3)
    proposal_rank = runtime.observe().proposal_raw_rank
    io = ScriptedIO(["", str(proposal_rank)])
    policy = InteractivePolicy(io=io, menu_size=2, manual_acceptance=True)
    action = policy.choose(runtime, runtime.observe())
    assert action.kind == 'select-raw-rank'
    assert not any(('invalid command' in item for item in io.output))


def test_manual_acceptance_is_the_only_acceptance_mode_flag() -> None:
    parser = build_parser()
    assert not parser.parse_args([]).manual_acceptance
    assert parser.parse_args(['--manual-acceptance']).manual_acceptance
    assert parser.parse_args(['--seamless']).seamless
    assert parser.parse_args([]).cache == 'auto'
    assert parser.parse_args(['--cache', 'off']).cache == 'off'
    assert parser.parse_args(['--no-cache']).cache == 'off'
    assert parser.parse_args(['--random-seed']).random_seed
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "7", "--random-seed"])
    assert parser.parse_args(['--list', '--lineage', 'root']).lineage == 'root'
    with pytest.raises(SystemExit):
        parser.parse_args(["--default-acceptance"])


def test_replay_divergence_hands_off_without_committing_changed_token(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source_runtime = engine(max_tokens=3)
        source_id = store.create_episode(
            episode_id="source",
            initial_text="P",
            initial_token_ids=source_runtime.initial_token_ids,
            sampling=source_runtime.sampling,
            stream_fingerprint=source_runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=3,
            backend={"backend": "fake"},
        )
        outcome = source_runtime.apply(Accept())
        store.record_action(source_id, 0, outcome)
        store.finish_episode(
            source_id,
            visible_text=source_runtime.backend.render(source_runtime.visible_token_ids),
            terminal_token_id=None,
            terminal_reason="menu-end",
        )

        target = engine(ChangedProposalBackend(), max_tokens=3)
        target_id = store.create_episode(
            episode_id="target",
            initial_text="P",
            initial_token_ids=target.initial_token_ids,
            sampling=target.sampling,
            stream_fingerprint=target.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=3,
            backend={"backend": "fake"},
        )
        action, expectation = store.replay_tape(source_id)[0]
        result = EpisodeRunner(target, store, target_id).run(
            tape=[TapeStep(action, expectation)]
        )
        assert result.handed_off
        assert target.visible_token_ids == []
        assert not target.ended


def test_ballistic_replay_records_counterfactual_and_yields_live(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = engine(max_tokens=3)
        source_id = store.create_episode(
            episode_id="source",
            initial_text="P",
            initial_token_ids=source.initial_token_ids,
            sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=3,
            backend={"backend": "fake"},
        )
        first = source.apply(Accept())
        store.record_action(source_id, 0, first)
        store.finish_episode(
            source_id,
            visible_text=source.backend.render(source.visible_token_ids),
            terminal_token_id=None,
            terminal_reason="menu-end",
        )
        target = engine(ChangedProposalBackend(), max_tokens=3)
        target_id = store.create_episode(
            episode_id="target",
            initial_text="P",
            initial_token_ids=target.initial_token_ids,
            sampling=target.sampling,
            stream_fingerprint=target.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=3,
            backend={"backend": "fake"},
        )
        action, expectation = store.replay_tape(source_id)[0]
        result = EpisodeRunner(
            target, store, target_id, divergence_policy="ballistic"
        ).run(tape=[TapeStep(action, expectation)])
        assert result.replay_exhausted
        assert target.visible_token_ids == [2]
        assert not target.ended
        assert store.actions(target_id)[0]['status'] == 'completed-with-divergence'


def test_explicit_sampler_args_preserve_unspecified_source_values() -> None:
    source = SamplingConfig(
        temperature=0.25,
        top_k=3,
        top_p=0.7,
        min_p=0.01,
        repeat_penalty=1.5,
        repeat_last_n=11,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        seed=77,
    )
    args = build_parser().parse_args(["--seed", "100"])

    merged = _sampling_from_args(args, source)

    assert merged == replace(source, seed=100)


def test_random_seed_spans_the_supported_signed_64_bit_range() -> None:
    with patch(
        "trajectory_editor.episode_cli.secrets.randbelow", return_value=0
    ) as randbelow:
        assert _random_seed() == MIN_SEED
        randbelow.assert_called_once_with(MAX_SEED - MIN_SEED + 1)

    with patch(
        "trajectory_editor.episode_cli.secrets.randbelow",
        return_value=MAX_SEED - MIN_SEED,
    ):
        assert _random_seed() == MAX_SEED


def test_random_seed_is_printed_and_recorded_in_the_sampler_segment(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["h 3"])
    output = StringIO()
    with (
        patch(
            "trajectory_editor.episode_cli.secrets.randbelow",
            return_value=123,
        ),
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=ConformingFakeBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
        redirect_stdout(output),
    ):
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--new-prompt",
                "P",
                "--episode-id",
                "randomized",
                "--random-seed",
                "--max-tokens",
                "3",
                "--plain-ui",
            ]
        )

    with EpisodeStore(workspace) as store:
        segment = store.sampling_segment("randomized", 0)

    selected = MIN_SEED + 123
    assert status == 0
    assert f'Random seed: {selected}' in output.getvalue()
    assert segment['sampling']['seed'] == selected


def test_edge_random_seed_creates_a_sampler_transition(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["accept", "s random-seed", "e"])
    with (
        patch(
            "trajectory_editor.episode_cli.secrets.randbelow",
            return_value=456,
        ),
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=ConformingFakeBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--new-prompt",
                "P",
                "--episode-id",
                "edge-randomized",
                "--max-tokens",
                "1",
                "--plain-ui",
            ]
        )

    with EpisodeStore(workspace) as store:
        segment = store.sampling_segment("edge-randomized", 1)

    assert status == 0
    assert segment['sampling']['seed'] == MIN_SEED + 456
    assert any(('seed=' + str(MIN_SEED + 456) in item for item in io.output))


def test_seamless_cli_rewinds_and_reuses_the_same_episode(tmp_path) -> None:
    from trajectory_editor.tui import SEAMLESS_REACTIVATE

    workspace = tmp_path / "episodes.sqlite3"
    io = LiveScriptedIO(
        ["h 10", "[", SEAMLESS_REACTIVATE, "accept", "e!"]
    )
    with (
        patch("trajectory_editor.episode_cli.sys.stdin") as stdin,
        patch("trajectory_editor.episode_cli.sys.stdout") as stdout,
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=NoEogBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        stdin.isatty.return_value = True
        stdout.isatty.return_value = True
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--new-prompt",
                "P",
                "--episode-id",
                "seamless",
                "--max-tokens",
                "20",
                "--seamless",
            ]
        )

    with EpisodeStore(workspace) as store:
        episode = store.get_episode("seamless")
        actions = store.actions("seamless")
        interactions = store.interactions("seamless")

    assert status == 0
    assert episode['status'] == 'completed'
    assert [action['kind'] for action in actions] == ['hold', 'select-raw-rank', 'end-generation']
    assert actions[0]['arguments']['limit'] == 9
    assert [item['kind'] for item in interactions] == ['seamless-rewind']


def test_seamless_checkpoint_rewinds_without_reopening_menu(tmp_path) -> None:
    from trajectory_editor.tui import SEAMLESS_REACTIVATE

    workspace = tmp_path / "episodes.sqlite3"
    io = LiveScriptedIO(
        [
            "h 1",
            "accept",
            "c",
            "accept",
            "[",
            SEAMLESS_REACTIVATE,
            "accept",
            "e!",
        ]
    )
    with (
        patch("trajectory_editor.episode_cli.sys.stdin") as stdin,
        patch("trajectory_editor.episode_cli.sys.stdout") as stdout,
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=NoEogBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        stdin.isatty.return_value = True
        stdout.isatty.return_value = True
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--new-prompt",
                "P",
                "--episode-id",
                "edge-reopen",
                "--max-tokens",
                "2",
                "--seamless",
            ]
        )

    with EpisodeStore(workspace) as store:
        episode = store.get_episode("edge-reopen")
        actions = store.actions("edge-reopen")
        interactions = store.interactions("edge-reopen")
        episode_count = len(store.list_episodes())

    assert status == 0
    assert episode['status'] == 'completed'
    assert [action['kind'] for action in actions] == ['hold', 'select-raw-rank', 'select-raw-rank', 'end-generation']
    assert [item['kind'] for item in interactions] == ['seamless-rewind']
    assert episode_count == 1


def test_no_source_on_noninteractive_streams_reports_an_error(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    with (
        patch("trajectory_editor.episode_cli.sys.stdin") as stdin,
        patch("trajectory_editor.episode_cli.sys.stdout") as stdout,
    ):
        stdin.isatty.return_value = False
        stdout.isatty.return_value = False
        status = episode_main(["--workspace", str(workspace)])
    assert status == 2


def test_interactive_no_source_reads_an_initial_prompt(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["accept", "q"])
    with (
        patch("trajectory_editor.episode_cli.sys.stdin") as stdin,
        patch("trajectory_editor.episode_cli.sys.stdout") as stdout,
        patch(
            "trajectory_editor.episode_cli._read_initial_prompt",
            return_value="P",
        ) as read_initial_prompt,
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=ConformingFakeBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        stdin.isatty.return_value = True
        stdout.isatty.return_value = True
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--episode-id",
                "prompted",
                "--max-tokens",
                "1",
                "--plain-ui",
            ]
        )
    assert status == 0
    read_initial_prompt.assert_called_once_with()
    with EpisodeStore(workspace) as store:
        assert store.get_episode('prompted')['initial_text'] == 'P'


def test_one_episode_crosses_checkpoint_then_terminates(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    output = tmp_path / "final.txt"
    io = ScriptedIO(["accept", "n 2", "h 2"])
    with (
        patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main(
            [
                "--workspace", str(workspace),
                "--model", "fake.gguf",
                "--new-prompt", "P",
                "--episode-id", "one",
                "--max-tokens", "1",
                "--plain-ui",
                "--output", str(output),
            ]
        )
    assert status == 0
    assert output.exists()
    with EpisodeStore(workspace) as store:
        rows = store.list_episodes()
        assert len(rows) == 1
        episode = store.get_episode("one")
        assert episode['status'] == 'completed'
        assert episode['terminal_reason'] == 'model-eog'
        assert [a['kind'] for a in store.actions('one')] == ['select-raw-rank', 'hold']


def test_quit_leaves_episode_resumeable(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["accept", "q"])
    with (
        patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main([
            "--workspace", str(workspace), "--model", "fake.gguf",
            "--new-prompt", "P", "--episode-id", "open",
            "--max-tokens", "1", "--plain-ui",
        ])
    assert status == 0
    with EpisodeStore(workspace) as store:
        assert store.get_episode('open')['status'] == 'open'


def test_sampler_change_is_a_boundary_transition_not_a_new_episode(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["accept", "s top_k=1 temperature=0", "c", "accept", "e"])
    with (
        patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main([
            "--workspace", str(workspace), "--model", "fake.gguf",
            "--new-prompt", "P", "--episode-id", "sampler",
            "--max-tokens", "1", "--plain-ui",
            "--temperature", "0", "--top-k", "8", "--top-p", "1", "--min-p", "0",
        ])
    assert status == 0
    with EpisodeStore(workspace) as store:
        assert len(store.list_episodes()) == 1
        segment = store.sampling_segment("sampler", 1)
        assert segment['sampling']['top_k'] == 1
        assert segment['sampling']['temperature'] == 0.0
        assert store.get_episode('sampler')['terminal_reason'] == 'menu-end'


def test_unsealed_episode_resumes_under_same_id(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    first_io = ScriptedIO(["accept", "q"])
    with (
        patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=first_io),
    ):
        assert episode_main(
            [
                '--workspace',
                str(workspace),
                '--model',
                'fake.gguf',
                '--new-prompt',
                'P',
                '--episode-id',
                'resume-me',
                '--max-tokens',
                '1',
                '--plain-ui',
                '--temperature',
                '0.25',
                '--top-k',
                '3',
                '--top-p',
                '0.7',
                '--min-p',
                '0.01',
                '--repeat-penalty',
                '1.5',
                '--repeat-last-n',
                '11',
            ]
        ) == 0
    second_io = ScriptedIO(["h 2"])
    with (
        patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=second_io),
    ):
        assert episode_main(
            [
                '--workspace',
                str(workspace),
                '--model',
                'fake.gguf',
                '--resume',
                'resume-me',
                '--max-tokens',
                '2',
                '--plain-ui',
                '--seed',
                '100',
            ]
        ) == 0
    with EpisodeStore(workspace) as store:
        assert len(store.list_episodes()) == 1
        assert [a['ordinal'] for a in store.actions('resume-me')] == [0, 1]
        assert store.get_episode('resume-me')['terminal_reason'] == 'model-eog'
        segment = store.sampling_segment("resume-me", 1)
        resumed_sampling = SamplingConfig.from_mapping(segment["sampling"])
        assert segment['start_boundary'] == 1
        assert resumed_sampling.temperature == 0.25
        assert resumed_sampling.top_k == 3
        assert resumed_sampling.repeat_penalty == 1.5
        assert resumed_sampling.seed == 100


def test_cli_spr_replays_then_yields_to_live_edge(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        source = engine(max_tokens=3)
        source_id = store.create_episode(
            episode_id="spr-source", initial_text="P",
            initial_token_ids=source.initial_token_ids, sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint, coordinate_offset=0,
            max_tokens=3, backend={"backend": "fake"},
        )
        outcome = source.apply(Write("C", "exact"))
        store.record_action(source_id, 0, outcome)
        store.finish_episode(
            source_id, visible_text=source.backend.render(source.visible_token_ids),
            terminal_token_id=None, terminal_reason="menu-end",
        )
    io = ScriptedIO(["e"])
    with (
        patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main([
            "--workspace", str(workspace), "--model", "fake.gguf",
            "--replay", "spr-source", "--episode-id", "spr-target",
            "--plain-ui",
        ])
    assert status == 0
    with EpisodeStore(workspace) as store:
        assert store.actions('spr-target')[0]['kind'] == 'write'
        assert store.get_episode('spr-target')['terminal_reason'] == 'menu-end'
        assert any(('SPR route exhausted' in item for item in io.output))


def test_default_cli_replay_follows_source_sampler_transitions(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    source_sampling = SamplingConfig(
        temperature=0.0,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
        repeat_penalty=1.5,
        repeat_last_n=12,
        seed=41,
    )
    changed_sampling = SamplingConfig(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        min_p=0.0,
        repeat_penalty=1.0,
        repeat_last_n=0,
        seed=99,
    )
    with EpisodeStore(workspace) as store:
        source = EpisodeEngine(
            ConformingFakeBackend(),
            sampling=source_sampling,
            max_tokens=10,
            initial_text="P",
        )
        source_id = store.create_episode(
            episode_id="segmented-source",
            initial_text=source.initial_text,
            initial_token_ids=source.initial_token_ids,
            sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=source.coordinate_offset,
            max_tokens=source.max_tokens,
            backend={"backend": "fake"},
        )
        first = source.apply(Accept())
        store.record_action(source_id, 0, first)
        source.sampling = changed_sampling
        store.record_sampling_segment(
            source_id,
            start_boundary=source.boundary,
            sampling=changed_sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=source.coordinate_offset,
        )
        second = source.apply(Accept())
        store.record_action(source_id, 1, second)
        store.finish_episode(
            source_id,
            visible_text=source.backend.render(source.visible_token_ids),
            terminal_token_id=None,
            terminal_reason="menu-end",
        )

    io = ScriptedIO(["e"])
    with (
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=ConformingFakeBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--replay",
                source_id,
                "--episode-id",
                "segmented-target",
                "--plain-ui",
            ]
        )

    with EpisodeStore(workspace) as store:
        target = store.get_episode("segmented-target")
        target_segment = store.sampling_segment("segmented-target", 1)
        target_actions = store.actions("segmented-target")

    assert status == 0
    assert target['status'] == 'completed'
    assert [row['boundary_after'] for row in target_actions] == [1, 2]
    assert target_segment['start_boundary'] == 1
    assert SamplingConfig.from_mapping(target_segment['sampling']) == changed_sampling


def test_explicit_replay_sampler_is_fixed_after_source_merge(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    source_sampling = SamplingConfig(
        temperature=0.0, top_k=8, top_p=1.0, min_p=0.0, seed=41
    )
    changed_sampling = SamplingConfig(
        temperature=0.0, top_k=1, top_p=1.0, min_p=0.0, seed=99
    )
    with EpisodeStore(workspace) as store:
        source = EpisodeEngine(
            ConformingFakeBackend(),
            sampling=source_sampling,
            max_tokens=10,
            initial_text="P",
        )
        source_id = store.create_episode(
            episode_id="fixed-source",
            initial_text=source.initial_text,
            initial_token_ids=source.initial_token_ids,
            sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=source.coordinate_offset,
            max_tokens=source.max_tokens,
            backend={"backend": "fake"},
        )
        first = source.apply(Accept())
        store.record_action(source_id, 0, first)
        source.sampling = changed_sampling
        store.record_sampling_segment(
            source_id,
            start_boundary=source.boundary,
            sampling=changed_sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=source.coordinate_offset,
        )
        second = source.apply(Accept())
        store.record_action(source_id, 1, second)
        store.finish_episode(
            source_id,
            visible_text=source.backend.render(source.visible_token_ids),
            terminal_token_id=None,
            terminal_reason="menu-end",
        )

    io = ScriptedIO(["e"])
    with (
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=ConformingFakeBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--replay",
                source_id,
                "--episode-id",
                "fixed-target",
                "--seed",
                "100",
                "--fixed-config",
                "--plain-ui",
            ]
        )

    with EpisodeStore(workspace) as store:
        target = store.get_episode("fixed-target")
        target_segment = store.sampling_segment("fixed-target", 1)

    assert status == 0
    assert target['status'] == 'completed'
    assert target_segment['start_boundary'] == 0
    fixed = SamplingConfig.from_mapping(target_segment["sampling"])
    assert fixed.seed == 100
    assert fixed.top_k == source_sampling.top_k
