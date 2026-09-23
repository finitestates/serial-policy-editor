from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.episode_cli import build_parser, main
from trajectory_editor.live_tui import _render_review
from trajectory_editor.plain_tui import display_candidates
from trajectory_editor.teacher_commands import (
    CommandKind, ForkAddress, ForkAddressKind, parse_command, parse_fork_address,
)
from trajectory_editor.terminal_contracts import BoundaryReview
from trajectory_editor.core.ui import InsertMode

pytestmark = pytest.mark.current_workflow

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


def test_review_labels_an_interior_action_boundary_as_inside():
    review = BoundaryReview(
        active_aligned_step=2,
        aligned_step=1,
        context_text_tail="P",
        context_token_sha256="0" * 64,
        position={"kind": "action-boundary", "action_kind": "write", "side": "inside"},
    )
    rendered = "".join(text for _, text in _render_review(review, terminal_size=(80, 24)))

    assert "WRITE inside" in rendered


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


@pytest.mark.parametrize("raw", ["h /", "h / 2", "h/", "h/2"])
def test_legacy_newline_hold_spellings_are_rejected(raw):
    with pytest.raises(EditorError):
        parse(raw)


@pytest.mark.parametrize(
    "raw, tokens, boundary",
    [
        ("h", 24, None),
        ("hold", 24, None),
        ("h5", 5, None),
        ("hold 5", 5, None),
        ("h.5", 5, "sentence"),
        ("hold . 5", 5, "sentence"),
        ("h|5", 5, "newline"),
        ("hold | 5", 5, "newline"),
    ],
)
def test_m03_hold_aliases_and_compact_forms(raw, tokens, boundary):
    command = parse(raw)
    assert command.kind == CommandKind.HOLD
    assert (command.hold_tokens, command.hold_boundary) == (tokens, boundary)


def test_m03_text_commands_keep_exact_payload_whitespace():
    exact = parse("x  leading  ")
    continuation = parse("t  leading  ")
    phrase = parse("checkx  exact  ")
    assert exact.action.supplied_text == " leading  "
    assert exact.action.insert_mode == InsertMode.EXACT
    assert continuation.action.supplied_text == " leading  "
    assert continuation.action.insert_mode == InsertMode.CONTINUATION
    assert (phrase.phrase_text, phrase.phrase_mode) == (" exact  ", "exact")


def test_m03_bias_group_scope_and_rank_prefix_syntax():
    scoped = parse('b {wings, " claws"} +0.5 after {dragon, " wyvern"} until "."')
    assert scoped.kind == CommandKind.BIAS
    assert (scoped.bias_targets, scoped.bias_target_bare) == (
        (" wings", " claws"), (True, False),
    )
    assert (scoped.bias_triggers, scoped.bias_trigger_bare) == (
        (" dragon", " wyvern"), (True, False),
    )
    assert (scoped.bias_operator, scoped.bias_amount, scoped.bias_stop_text) == (
        "+", .5, ".",
    )

    group = parse('b nautical -> {anchor, " steamship"}')
    assert (group.bias_group_name, group.bias_group_members) == (
        "nautical", (" anchor", " steamship"),
    )
    ranked = parse('1+0.5 ... " P"')
    assert (ranked.search_rank, ranked.bias_prefix, ranked.bias_amount) == (1, " P", .5)


@pytest.mark.parametrize(
    "raw, address",
    [
        ("f", ForkAddress(ForkAddressKind.CURRENT)),
        ("fork", ForkAddress(ForkAddressKind.CURRENT)),
        ("f 4", ForkAddress(ForkAddressKind.ABSOLUTE, 4)),
        ("f - 2", ForkAddress(ForkAddressKind.RELATIVE_BACKWARD, 2)),
        ("f-2", ForkAddress(ForkAddressKind.RELATIVE_BACKWARD, 2)),
    ],
)
def test_m03_fork_addresses_keep_absolute_and_backward_meanings(raw, address):
    assert parse_fork_address(raw) == address
    command = parse(raw)
    assert command.kind == CommandKind.FORK
    assert command.fork_address == address


@pytest.mark.parametrize("raw", ["f +2", "f - 0", "fork nope"])
def test_m03_invalid_fork_addresses_remain_errors(raw):
    with pytest.raises(EditorError):
        parse(raw)


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
    with pytest.raises(SystemExit):
        parser.parse_args(["--setup-menu"])

    args = build_parser().parse_args([])
    with patch("sys.argv", ["policy-editor", "--help"]):
        with pytest.raises(SystemExit) as outcome:
            main()
    assert outcome.value.code == 0
