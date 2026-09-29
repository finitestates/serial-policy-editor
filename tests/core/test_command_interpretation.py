"""One command meaning for live drafts and submitted episode input."""

from __future__ import annotations

import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.actions import Reroll
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.sampling import draw_token
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.episode_hash import token_prefix_sha256
from trajectory_editor.tui_render import PreviewPending, _render_choice, action_preview
from trajectory_editor.teacher_commands import (
    CommandKind, CommandState, interpret_command, parse_command,
)
from trajectory_editor.ui_themes import (
    LIVE_THEME_NAMES, is_dark_terminal, resolve_live_theme, semantic_style,
    theme_palette, theme_stylesheet,
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
        context_text_tail=engine.backend.render(
            list(observation.prefix_token_ids), special=True
        )[-1:],
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


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4
                  for value in channels]
        return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]

    first, second = luminance(foreground), luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + .05) / (darker + .05)


@pytest.mark.parametrize("raw", [
    "", "accept", "8", "t hello", "x  exact", "check hello", "checkx hello",
    "force hello", "forcex hello", "h", "hold", "h5", "h.5", "h|5",
    "m", "more", "m10", "ms", "ms 8", "ms+2", "ms-2", "/A", '/"\\n"',
    "context", "c 200", "c all", "c900", "f", "f-1", "[", "]", "b",
    "s", "s temperature=0.7", "reroll", "reroll 42", "draw 1",
    "groups", "groups ships", 'b ships -> {steamship, "my favorite couch"}',
    'b ships remove {steamship}', "b ships +0.5", "b {ships, sky} +",
    "b token #4", "b token #4 +0.5", "b A +", "1+", "1-", "1=0.5",
    "v", "V", "l", "L", "%", "c", "C",
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
    "chord", "chord 1", "chord\t1", "overlay", "draw",
])
def test_recognized_drafts_remain_incomplete(raw):
    _, choice, candidates = _decision()
    assert _interpret(raw, choice).state == CommandState.INCOMPLETE
    preview = _preview(raw, choice, candidates)
    assert preview.state == "incomplete" and not preview.valid


@pytest.mark.parametrize("raw", [
    "h /", "h / 2", "m nope", "ms nope", "context nope", "c nope",
    "overlay unknown", '/"\\x"', "9", "chord 1 1", "chord 1 9", "f +2",
    "draw 0", "draw -1", "draw nope", "draw 9",
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
    for raw, marker in (
        ("h /", "INVALID ·"),
        ("t", "INCOMPLETE ·"),
    ):
        rendered = _render_choice(
            choice, candidates, raw, None, lambda text, mode: text, None,
            default_hold_tokens=24,
        )
        assert marker in rendered.plain
        assert rendered.spans


    for theme in LIVE_THEME_NAMES:
        stylesheet = theme_stylesheet(theme)
        assert "blink" not in stylesheet.lower()
        assert "blink" not in semantic_style("invalid", theme).lower()
    monochrome_style = semantic_style("invalid", "monochrome")
    assert "underline" in monochrome_style and "red" not in monochrome_style
    high_contrast = theme_palette("high-contrast", environment={"COLORTERM": "truecolor"})
    for foreground in (
        high_contrast.foreground, high_contrast.primary, high_contrast.secondary,
        high_contrast.accent, high_contrast.error, high_contrast.muted,
    ):
        assert _contrast_ratio(foreground, high_contrast.background) >= 4.5


def test_theme_selection_respects_color_environment_and_ansi_fallback():
    assert resolve_live_theme(None, environment={}) == "amber-cyan"
    assert resolve_live_theme(None, environment={"NO_COLOR": ""}) == "monochrome"
    assert is_dark_terminal({})
    assert not is_dark_terminal({"COLORFGBG": "0;15"})
    assert theme_palette("amber-cyan", environment={}).primary == "ansi_yellow"
    assert theme_palette(
        "amber-cyan", environment={"COLORTERM": "truecolor", "COLORFGBG": "0;15"}
    ).background == "#FFFDF7"


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
    preview = _preview("b ships +", choice, candidates)
    assert preview.state == "ready"
    io = ScriptedIO(["b ships +", "8"])
    action = InteractivePolicy(io=io).choose(engine, engine.observe())
    assert action.rank == 8
    assert "INVALID BIAS" in "".join(io.output)


def test_draw_preview_names_the_targeted_reroll():
    _, choice, candidates = _decision()

    preview = _preview("draw 1", choice, candidates)

    assert preview.kind == "effect"
    assert preview.label == "targeted draw"
    assert "draws the token at raw rank 1" in preview.detail
    assert "recorded as a reroll" in preview.detail


def test_draw_command_returns_a_seed_recordable_as_a_normal_reroll(monkeypatch):
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.7, top_k=3, seed=12345),
    )
    observation = engine.observe()
    active_ids = set(int(token_id) for token_id in observation.distribution.ids)
    ordered_raw_ids = observation.policy_calculations.top_raw_ids(len(observation.logits))
    target_rank = next(
        rank
        for rank, token_id in enumerate(ordered_raw_ids, start=1)
        if token_id in active_ids and token_id != observation.proposal_token_id
    )
    target_token_id = ordered_raw_ids[target_rank - 1]
    matching_seed = next(
        seed
        for seed in range(1000)
        if seed != engine.sampling.seed
        and draw_token(
            observation.distribution,
            seed=seed,
            stream_fingerprint=engine.stream_fingerprint,
            aligned_step=observation.sampling_boundary,
            kernel=engine.sampling.draw_kernel,
        ) == target_token_id
    )
    monkeypatch.setattr("trajectory_editor.episode_ui.random_seed", lambda: matching_seed)

    action = InteractivePolicy(io=ScriptedIO([f"draw {target_rank}"])).choose(
        engine, observation
    )

    assert action == Reroll(matching_seed)
    outcome = engine.apply(action)
    assert outcome.action == Reroll(matching_seed)
    assert engine.observe().proposal_token_id == target_token_id


def test_draw_command_rejects_a_rank_whose_token_is_outside_the_truncated_set():
    engine, _, _ = _decision()
    io = ScriptedIO(["draw 2", "accept"])

    action = InteractivePolicy(io=io).choose(engine, engine.observe())

    assert action.kind == "select-raw-rank"
    assert "DRAW UNAVAILABLE" in "".join(io.output)
    assert "outside the active truncated candidate set" in "".join(io.output)


def test_draw_command_cancellation_records_no_reroll(monkeypatch):
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.7, top_k=3, seed=12345),
    )
    observation = engine.observe()
    active_ids = set(int(token_id) for token_id in observation.distribution.ids)
    ordered_raw_ids = observation.policy_calculations.top_raw_ids(len(observation.logits))
    target_rank = next(
        rank
        for rank, token_id in enumerate(ordered_raw_ids, start=1)
        if token_id in active_ids and token_id != observation.proposal_token_id
    )

    def cancel_search():
        raise KeyboardInterrupt

    monkeypatch.setattr("trajectory_editor.episode_ui.random_seed", cancel_search)
    io = ScriptedIO([f"draw {target_rank}", "accept"])
    action = InteractivePolicy(io=io).choose(engine, observation)

    assert action.kind == "select-raw-rank"
    assert "DRAW SEARCH CANCELLED" in "".join(io.output)
    assert "No sampler action was recorded." in "".join(io.output)
