from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.episode_cli import build_parser, main
from trajectory_editor.tui import CommandKind, parse_command
from trajectory_editor.tui import display_candidates


def runtime():
    return EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )


def parse(raw: str):
    return parse_command(raw, menu_size=12, default_hold_tokens=24, vocabulary_size=1000)


def test_m01_full_vocabulary_search_is_nonmutating():
    episode = runtime()
    before = list(episode.backend.tokens)
    policy = InteractivePolicy(io=ScriptedIO(["/P", "8"]), menu_size=1, search_radius=1)

    action = policy.choose(episode, episode.observe())

    assert action.kind == "select-raw-rank"
    assert episode.backend.tokens == before
    assert "absolute raw rank=8" in "".join(policy.io.output)


@pytest.mark.parametrize(
    "responses, expected_rank",
    [
        (["ms 7", "ms - 2", "ms", "7"], 7),
        (["999", "8"], 8),
    ],
)
def test_m02_rank_navigation_resolves_absolute_raw_rank_without_commit(
    responses, expected_rank
):
    episode = runtime()
    before = list(episode.backend.tokens)
    policy = InteractivePolicy(io=ScriptedIO(responses), menu_size=1)

    action = policy.choose(episode, episode.observe())

    assert action.rank == expected_rank
    assert episode.boundary == 0
    assert episode.backend.tokens == before


@pytest.mark.parametrize(
    "raw, kind, action_kind",
    [
        ("1", CommandKind.EDIT, "select"),
        ("t hello", CommandKind.EDIT, "insert"),
        ("x hello", CommandKind.EDIT, "insert"),
        ("h 2", CommandKind.HOLD, None),
        ("b group -> {anchor}", CommandKind.BIAS, None),
        ("m 10", CommandKind.MENU_EXPAND, None),
        ("[", CommandKind.REVIEW_BACK, None),
        ("q", CommandKind.FINISH, None),
    ],
)
def test_m03_menu_commands_distinguish_editorial_moves_from_token_actions(
    raw, kind, action_kind
):
    command = parse(raw)
    assert command.kind == kind
    if action_kind is not None:
        assert command.action.kind.value == action_kind


def test_m04_logit_views_are_sticky_and_cycle_raw_model_gap_modes():
    episode = runtime()
    candidates = episode.candidates(episode.observe(), count=3)

    io = ScriptedIO([])
    display_candidates(io, candidates, heading=True, logit_view="none")
    assert "model-logit" not in io.output[0]

    io = ScriptedIO([])
    display_candidates(io, candidates, heading=True, logit_view="raw")
    assert "model-logit" in io.output[0]
    assert "model-gap" not in io.output[0]

    io = ScriptedIO(["l", "1", "1"])
    policy = InteractivePolicy(io=io)
    action = policy.choose(episode, episode.observe())
    episode.apply(action)
    policy.choose(episode, episode.observe())
    assert policy.view_preferences.logit_view == "raw"
    assert sum("model-logit" in line for line in io.output) >= 2

    io = ScriptedIO(["L", "1"])
    episode = runtime()
    InteractivePolicy(io=io).choose(episode, episode.observe())
    assert any("model-logit" in line and "model-gap" in line for line in io.output)

    io = ScriptedIO(["L", "L", "1"])
    policy = InteractivePolicy(io=io)
    episode = runtime()
    policy.choose(episode, episode.observe())
    headers = [line for line in io.output if line.startswith("\n  rank")]
    assert "model-logit" in headers[1]
    assert "model-logit" not in headers[-1]


def test_m05_core_help_exposes_only_core_flags():
    parser = build_parser(include_vector=False)
    with pytest.raises(SystemExit):
        parser.parse_args(["--online-learning"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--steering-vector", "vector.json"])

    args = build_parser().parse_args([])
    with patch("sys.argv", ["policy-editor", "--help"]):
        with pytest.raises(SystemExit) as outcome:
            main()
    assert outcome.value.code == 0
