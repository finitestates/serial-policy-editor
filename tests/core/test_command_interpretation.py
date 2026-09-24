"""One command meaning for live drafts and submitted episode input."""

from __future__ import annotations

import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.episode_hash import token_prefix_sha256
from trajectory_editor.live_tui import LIVE_STYLES, PreviewPending, _render_choice, action_preview
from trajectory_editor.teacher_commands import (
    CommandKind, CommandState, interpret_command, parse_command,
)

pytestmark = pytest.mark.current_workflow


def _decision():
    engine = EpisodeEngine(
        ConformingFakeBackend(), initial_text="P", initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    observation = engine.observe()
    candidates = engine.candidates(observation, count=4)
    choice = _choice_from_observation(
        engine, observation, candidates,
        context_text_tail=observation.context_text[-1:],
        context_token_sha256=token_prefix_sha256(list(observation.prefix_token_ids)),
        serial=1,
    )
    return engine, choice, candidates


def _interpret(raw, choice):
    return interpret_command(
        raw, menu_size=len(choice.candidates), default_hold_tokens=24,
        vocabulary_size=choice.vocabulary_size, default_search_radius=2,
    )


def _preview(raw, choice, candidates, **kwargs):
    return action_preview(
        choice, raw, candidates, lambda text, mode: text,
        default_hold_tokens=24, default_search_radius=2, **kwargs,
    )


@pytest.mark.parametrize("raw", [
    "", "accept", "8", "t hello", "x  exact", "check hello", "checkx hello",
    "force hello", "forcex hello", "h", "hold", "h5", "h.5", "h|5",
    "m", "more", "m10", "ms", "ms 8", "ms+2", "ms-2", "/A", '/"\\n"',
    "context", "c 200", "c all", "c900", "f", "f-1", "[", "]", "b",
    "b A +", "bl 1 +", "1+", "v", "V", "l", "L", "%", "c", "C",
    "overlay pct", "n note", "p note", "q", "e", "e!", "?",
    "chord 1 2", "chord\t1 2",
])
def test_ready_preview_carries_the_submitted_command_meaning(raw):
    _, choice, candidates = _decision()
    interpretation = _interpret(raw, choice)
    preview = _preview(raw, choice, candidates)
    assert interpretation.state == CommandState.READY
    assert preview.state == "ready"
    assert preview.command == interpretation.command
    if raw:
        assert preview.command == parse_command(
            raw, menu_size=len(candidates), default_hold_tokens=24,
            vocabulary_size=choice.vocabulary_size, default_search_radius=2,
        )
    else:
        assert preview.command.kind == CommandKind.EDIT
        assert preview.command.action.kind.value == "accept"


@pytest.mark.parametrize("raw", [
    "t", "t ", "x", "x ", "/", "check", "checkx", "force", "forcex",
    "chord", "chord 1", "chord\t1", "overlay",
])
def test_recognized_drafts_remain_incomplete(raw):
    _, choice, candidates = _decision()
    assert _interpret(raw, choice).state == CommandState.INCOMPLETE
    preview = _preview(raw, choice, candidates)
    assert preview.state == "incomplete" and not preview.valid


@pytest.mark.parametrize("raw", [
    "h /", "h / 2", "m nope", "ms nope", "context nope", "c nope",
    "overlay unknown", '/"\\x"', "9", "chord 1 1", "chord 1 9", "f +2",
])
def test_malformed_commands_never_look_ready(raw):
    _, choice, candidates = _decision()
    interpretation = _interpret(raw, choice)
    preview = _preview(raw, choice, candidates)
    assert interpretation.state == CommandState.INVALID
    assert preview.state == "invalid" and not preview.valid
    assert preview.detail == interpretation.message


def test_invalid_and_incomplete_cues_are_static_and_textual():
    _, choice, candidates = _decision()
    for raw, marker, style in (
        ("h /", "INVALID ·", "class:invalid"),
        ("t", "INCOMPLETE ·", "class:hint"),
    ):
        fragments = _render_choice(
            choice, candidates, raw, None, lambda text, mode: text, None,
            terminal_size=(80, 30), default_hold_tokens=24,
        )
        assert any(marker in text and fragment_style == style
                   for fragment_style, text in fragments)


    for theme in LIVE_STYLES.values():
        invalid_style = theme.get_attrs_for_style_str("class:invalid")
        assert not invalid_style.blink
        assert "red" not in invalid_style.color


def test_submit_reinterprets_the_actual_buffer_and_blank_accepts_proposal():
    engine, choice, candidates = _decision()
    preview = _preview("h 2", choice, candidates)
    assert preview.command.kind == CommandKind.HOLD

    io = ScriptedIO(["x changed"])
    submitted = InteractivePolicy(
        io=io, default_hold_tokens=24, search_radius=2,
    ).choose(engine, engine.observe())
    assert io.choice_requests[0].default_hold_tokens == 24
    assert io.choice_requests[0].default_search_radius == 2
    assert submitted.kind == "write"
    assert submitted.text == "changed"

    blank_engine, _, _ = _decision()
    observation = blank_engine.observe()
    accepted = InteractivePolicy(io=ScriptedIO([""])).choose(blank_engine, observation)
    assert accepted.rank == observation.proposal_raw_rank
    assert _interpret("", choice).command.action.kind.value == "accept"
    assert interpret_command(
        "", menu_size=4, default_hold_tokens=24, vocabulary_size=8,
        implicit_accept=False,
    ).state == CommandState.INVALID


def test_syntax_ready_runtime_rejection_keeps_choice_editable():
    engine, choice, candidates = _decision()
    preview = _preview("bl 5 +", choice, candidates)
    assert preview.state == "ready"
    io = ScriptedIO(["bl 5 +", "8"])
    action = InteractivePolicy(io=io).choose(engine, engine.observe())
    assert action.rank == 8
    assert "INVALID BIAS" in "".join(io.output)
