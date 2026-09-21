from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.core.runtime_helpers import NoEogBackend
from tests.fakes import ScriptedIO
from trajectory_editor.core.actions import Hold, Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import _live_edge_menu, main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_projector import project_fork_map
from trajectory_editor.episode_replay_source import final_sampling
from trajectory_editor.episode_store import EpisodeStore


@pytest.fixture
def source_workspace(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    with EpisodeStore(path) as store:
        source = EpisodeEngine(
            NoEogBackend(),
            sampling=SamplerConfig(temperature=0, seed=999),
            initial_text="P",
            initial_token_ids=[7],
            coordinate_offset=40,
        )
        store.create_episode(
            episode_id="source",
            initial_text=source.initial_text,
            initial_token_ids=source.initial_token_ids,
            sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=40,
            max_tokens=None,
            backend={},
        )
        for ordinal, action in enumerate([Write(" A B", "exact"), Hold(2)]):
            store.record_action("source", ordinal, source.apply(action))
        store.update_episode(
            "source",
            visible_text=source.backend.render(source.visible_token_ids),
            max_tokens=None,
        )
    return path


def run_cli(path, commands, *flags):
    io = ScriptedIO(commands)
    with patch("trajectory_editor.episode_cli._backend", return_value=NoEogBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        status = main(["--workspace", str(path), "--model", "fake", "--plain-ui", *flags])
    assert status == 0
    return io


def start(path, commands, *flags):
    return run_cli(
        path,
        commands,
        "--new-prompt", "P",
        "--episode-id", "destination",
        "--seed", "77",
        "--temperature", "0",
        "--top-k", "4",
        *flags,
    )


@pytest.mark.parametrize(
    "until, expected",
    [(0, ""), (1, " A"), (2, " A B"), (3, " A B!"), (4, " A B!!")],
)
def test_cli_replay_until_returns_to_the_selected_live_edge(
    source_workspace, until, expected
):
    run_cli(
        source_workspace,
        ["quit"],
        "--replay", "#1",
        "--until", str(until),
        "--episode-id", "destination",
    )

    with EpisodeStore(source_workspace) as store:
        destination = store.get_episode("destination")
        assert destination["initial_text"] == "P"
        assert destination["visible_text"] == expected
        assert len(store.tokens("destination")) == until
        assert not any(row["mismatch"] for row in store.actions("destination"))


@pytest.mark.parametrize(
    "commands",
    [["spr #1 m", "", "quit"], ["spr #1 --until 5", "quit"], ["spr #1 --until -1", "quit"], ["spr #1 --until nope", "quit"]],
)
def test_cli_replay_until_invalid_or_cancelled_selection_does_not_insert(
    source_workspace, commands
):
    start(source_workspace, ["q", *commands])
    with EpisodeStore(source_workspace) as store:
        assert store.tokens("destination") == []


@pytest.mark.parametrize("until, seed", [(0, 999), (1, 999), (2, 123), (3, 123)])
def test_cli_replay_until_uses_sampler_state_at_the_selected_boundary(
    source_workspace, until, seed
):
    with EpisodeStore(source_workspace) as store:
        segment = store.sampling_segment("source", 0)
        for boundary, next_seed in [(2, 123), (4, 456)]:
            store.record_sampling_segment(
                "source",
                start_boundary=boundary,
                sampling=replace(SamplerConfig.from_record(segment["sampling"]), seed=next_seed),
                stream_fingerprint=segment["stream_fingerprint"],
                coordinate_offset=40,
            )

    run_cli(
        source_workspace,
        ["quit"],
        "--replay", "#1",
        "--until", str(until),
        "--episode-id", "destination",
    )
    with EpisodeStore(source_workspace) as store:
        assert final_sampling(store, "destination").seed == seed


def test_cli_edge_replay_appends_live_text_without_mutating_the_source(source_workspace):
    with EpisodeStore(source_workspace) as store:
        source_tokens = store.tokens("source")

    start(source_workspace, ["t hello", "q", "spr #1", "quit"])

    with EpisodeStore(source_workspace) as store:
        destination = store.get_episode("destination")
        assert len(store.list_episodes()) == 2
        assert destination["initial_text"] == "P"
        assert destination["parent_episode_id"] is None
        assert destination["visible_text"] == " helloP A B!!"
        assert store.tokens("source") == source_tokens
        assert project_fork_map(store, "destination") == "P|0| hello|1|P|2| A|3| B|4|!|5|!|6|"


def test_cli_fork_from_persists_inherited_history_in_root_coordinates(source_workspace):
    run_cli(
        source_workspace,
        ["q", "quit"],
        "--fork-from", "#1",
        "--at", "1",
        "--episode-id", "child",
    )

    with EpisodeStore(source_workspace) as store:
        child = store.get_episode("child")
        assert child["initial_text"] == "P"
        assert child["initial_token_ids"] == [7]
        assert child["visible_text"] == " A"
        assert [row["boundary"] for row in store.tokens("child") if row["realized_visible"]] == [0]
        assert project_fork_map(store, "child") == "P|0| A|1|"

    run_cli(source_workspace, ["q", "rewind 0", "quit"], "--resume", "child")

    with EpisodeStore(source_workspace) as store:
        assert store.actions("child") == []
        assert project_fork_map(store, "child") == "P|0|"


def test_cli_prompt_replay_uses_the_destination_tokenizer_and_remains_rewindable(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplerConfig(), initial_token_ids=[7])
        store.create_episode(
            episode_id="source",
            initial_text=" A B",
            initial_token_ids=[7],
            sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )

    backend = NoEogBackend()
    with patch("trajectory_editor.episode_cli._backend", return_value=backend), patch(
        "trajectory_editor.episode_cli.TerminalIO",
        return_value=ScriptedIO(["t hello", "q", "spr #1", "rewind 2", "quit"]),
    ):
        assert main([
            "--workspace", str(path), "--model", "fake", "--plain-ui",
            "--new-prompt", "P", "--episode-id", "destination",
        ]) == 0

    with EpisodeStore(path) as store:
        destination = store.get_episode("destination")
        assert destination["initial_token_ids"] == [7]
        assert destination["visible_text"] == " hello A"
        assert store.actions("destination")[1]["arguments"]["mode"] == "exact"


def test_durable_edge_bare_sampler_opens_the_existing_override_prompt(source_workspace):
    with EpisodeStore(source_workspace) as store:
        engine = EpisodeEngine(
            NoEogBackend(),
            sampling=SamplerConfig(temperature=0.0),
            initial_text="P",
            initial_token_ids=[7],
        )
        action, value = _live_edge_menu(
            ScriptedIO(["s", "temperature=0.7", "q"]),
            store,
            "source",
            engine,
        )

    assert (action, value) == ("quit", None)
    assert engine.sampling.temperature == 0.7
