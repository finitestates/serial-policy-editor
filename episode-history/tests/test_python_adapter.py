from __future__ import annotations

import pytest

from rust_episode_history import parse_action, sampler_after_action, truncate, visible
from trajectory_editor.core.actions import (
    Accept,
    Hold,
    Reroll,
    SetSampler,
    UnsupportedPolicyActionKind,
    Write,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import ActionOutcome, TokenEvidence
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_history import EpisodeHistory, RecordedAttempt


def _evidence(boundary: int, token_id: int, text: str) -> TokenEvidence:
    return TokenEvidence(
        boundary=boundary,
        sampling_boundary=boundary,
        token_id=token_id,
        text=text,
        proposal_token_id=token_id,
        raw_model_nll=0.0,
        raw_rank=1,
        policy_rank=1,
        decoder_probability=1.0,
        proposal_agreement=True,
        is_eog=False,
        realized_visible=True,
    )


def _outcome(action, before: int, token_ids: tuple[int, ...], texts: tuple[str, ...]):
    evidence = tuple(
        _evidence(before + index, token_id, text)
        for index, (token_id, text) in enumerate(zip(token_ids, texts))
    )
    return ActionOutcome(
        action=action,
        boundary_before=before,
        boundary_after=before + len(token_ids),
        resolved_text="".join(texts),
        resolved_token_ids=token_ids,
        visible_token_ids=token_ids,
        terminal_token_id=None,
        stop_reason="completed",
        evidence=evidence,
        diagnostics={"source": "adapter-test"},
    )


def _history() -> EpisodeHistory:
    action = Write("ab", mode="exact")
    return EpisodeHistory((
        RecordedAttempt(0, action, _outcome(action, 0, (4, 5), ("a", "b"))),
        RecordedAttempt(1, Accept(), _outcome(Accept(), 2, (), ())),
    ))


def test_adapter_projects_and_truncates_typed_history_in_one_call():
    history = _history()
    projection = visible(history)
    assert projection["visible_token_ids"] == (4, 5)
    assert projection["visible_text"] == "ab"

    result = truncate(history, 1)
    partial = result.retained.attempts[-1]
    assert partial.action == Write("a", mode="exact")
    assert partial.outcome.diagnostics is None
    assert result.partial == history.attempts[0]
    assert result.discarded == (history.attempts[1],)
    assert len(truncate(history, 2).retained.attempts) == 1
    assert len(truncate(history, 2, include_boundary_events=1).retained.attempts) == 2


def test_adapter_preserves_exact_integer_and_unknown_action_distinctions():
    with pytest.raises(ValueError, match="nonnegative integer"):
        truncate(_history(), True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        truncate(_history(), 1.5)
    with pytest.raises(EditorError, match="no valid raw rank"):
        parse_action({"kind": "select", "rank": True})
    with pytest.raises(UnsupportedPolicyActionKind, match="unsupported policy action kind"):
        parse_action({"kind": "future-action"})


def test_sampler_actions_keep_opaque_fields_or_replace_the_whole_config():
    original = SamplerConfig(
        temperature=0.4,
        top_k=19,
        presence_penalty=0.25,
        repeat_last_n=27,
        seed=7,
    )
    rerolled = sampler_after_action(original, Reroll(-(1 << 63)))
    assert rerolled.seed == -(1 << 63)
    assert rerolled.to_dict() == {**original.to_dict(), "seed": -(1 << 63)}

    replacement = SamplerConfig(
        temperature=0.0,
        top_k=3,
        cfg_unconditional_prompt="new context",
        cfg_scale=1.5,
        seed=88,
    )
    changed = sampler_after_action(original, SetSampler(replacement))
    assert changed.to_dict() == replacement.to_dict()


def test_adapter_parses_canonical_action_forms_and_aliases():
    assert parse_action({"kind": "insert", "supplied_text": "x"}) == Write("x")
    assert parse_action({"kind": "teacher-eog"}) == parse_action({"kind": "end-generation"})
    assert parse_action({"kind": "hold", "limit": 2, "boundary": None}) == Hold(2)
    with pytest.raises(EditorError, match="saved sampler settings are missing fields"):
        parse_action({"kind": "set-sampler", "sampling": {}})
