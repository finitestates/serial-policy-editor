from __future__ import annotations

from dataclasses import replace

import pytest

from trajectory_editor.core.actions import Accept, Hold, Phrase, Write
from trajectory_editor.core.results import (
    ActionOutcome,
    Divergence,
    ReplayExpectation,
    TokenEvidence,
)
from trajectory_editor.episode_history import EpisodeHistory, RecordedAttempt

pytestmark = pytest.mark.invariant

def evidence(boundary: int, token_id: int, text: str) -> TokenEvidence:
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


def outcome(
    action,
    before: int,
    token_ids: tuple[int, ...] = (),
    texts: tuple[str, ...] = (),
    *,
    status: str = "completed",
    terminal_token_id: int | None = None,
    divergence: Divergence | None = None,
    replay_eog_token_id: int | None = None,
) -> ActionOutcome:
    items = tuple(
        evidence(before + index, token_id, text)
        for index, (token_id, text) in enumerate(zip(token_ids, texts))
    )
    return ActionOutcome(
        action=action,
        boundary_before=before,
        boundary_after=before + len(token_ids),
        resolved_text="".join(texts),
        resolved_token_ids=token_ids,
        visible_token_ids=token_ids,
        terminal_token_id=terminal_token_id,
        stop_reason="completed",
        evidence=items,
        status=status,
        divergence=divergence,
        replay_eog_token_id=replay_eog_token_id,
    )


def attempt(ordinal: int, action, result: ActionOutcome, expectation=None):
    return RecordedAttempt(ordinal, action, result, expectation)


def test_valid_contiguous_and_zero_width_histories_are_immutable_values():
    first = outcome(Hold(2), 0, (10, 11), ("A", "B"))
    zero = outcome(Accept(), 2)
    last = outcome(Hold(1), 2, (12,), ("C",))
    history = EpisodeHistory(
        (
            attempt(0, Hold(2), first),
            attempt(1, Accept(), zero),
            attempt(2, Hold(1), last),
        )
    )

    assert history.current_boundary == 3
    assert history.visible_token_ids == (10, 11, 12)
    assert history.visible_text == "ABC"
    with pytest.raises(AttributeError):
        history.attempts = ()


def test_history_rejects_unordered_ordinals_and_noncontiguous_boundaries():
    first = outcome(Hold(1), 0, (1,), ("A",))
    second = outcome(Hold(1), 1, (2,), ("B",))

    with pytest.raises(ValueError, match="ordinals"):
        EpisodeHistory((attempt(2, Hold(1), first), attempt(1, Hold(1), second)))

    with pytest.raises(ValueError, match="contiguous"):
        EpisodeHistory((attempt(0, Hold(1), first), attempt(1, Hold(1), outcome(Hold(1), 2, (2,), ("B",)))))

    with pytest.raises(ValueError, match="root-visible boundary zero"):
        EpisodeHistory((attempt(0, Hold(1), outcome(Hold(1), 1, (2,), ("B",))),))


def test_history_validates_evidence_types_and_exact_visible_boundaries():
    action = Hold(2)
    original = outcome(action, 0, (1, 2), ("A", "B"))

    with pytest.raises(TypeError, match="TokenEvidence"):
        EpisodeHistory((attempt(0, action, replace(original, evidence=(object(),))),))

    duplicate = replace(
        original,
        evidence=(evidence(0, 1, "A"), evidence(0, 2, "B")),
    )
    with pytest.raises(ValueError, match="contiguous and ordered"):
        EpisodeHistory((attempt(0, action, duplicate),))


def test_visible_projection_uses_only_realized_token_evidence():
    first = outcome(Hold(2), 0, (7, 8), ("hello", " world"))
    terminal = replace(
        evidence(2, 99, "<eog>"), is_eog=True, realized_visible=False
    )
    second = replace(
        outcome(Write("!", mode="exact"), 2, (9,), ("!",)),
        evidence=(*outcome(Write("!", mode="exact"), 2, (9,), ("!",)).evidence, terminal),
        resolved_token_ids=(9, 99),
        terminal_token_id=99,
        stop_reason="eog",
    )
    history = EpisodeHistory((attempt(0, Hold(2), first), attempt(1, Write("!", mode="exact"), second)))

    assert tuple(item.token_id for item in history.visible_token_evidence) == (7, 8, 9)
    assert history.visible_text == "hello world!"
    assert history.current_boundary == 3


def test_cutting_at_zero_discards_even_zero_width_attempts_at_the_root():
    zero = outcome(Phrase("check", mode="exact"), 0, status="handed-off")
    visible = outcome(Hold(1), 0, (4,), ("A",))
    history = EpisodeHistory((attempt(0, Phrase("check", mode="exact"), zero), attempt(1, Hold(1), visible)))

    result = history.truncate(0)

    assert result.retained.attempts == ()
    assert tuple(item.ordinal for item in result.discarded) == (0, 1)


@pytest.mark.parametrize("boundary", [True, 1.5])
def test_truncate_rejects_boolean_and_fractional_boundaries(boundary):
    history = EpisodeHistory(
        (attempt(0, Hold(2), outcome(Hold(2), 0, (1, 2), ("A", "B"))),)
    )

    with pytest.raises(ValueError, match="nonnegative integer"):
        history.truncate(boundary)


def test_cutting_at_an_action_boundary_discards_the_future_without_rebasing():
    first = outcome(Hold(2), 0, (1, 2), ("A", "B"))
    second = outcome(Write(" C", mode="exact"), 2, (3,), (" C",))
    history = EpisodeHistory((attempt(0, Hold(2), first), attempt(1, Write(" C", mode="exact"), second)))

    retained = history.retain_through(2)

    assert retained.current_boundary == 2
    assert retained.visible_token_ids == (1, 2)
    assert retained.attempts[0].outcome.boundary_before == 0
    assert retained.attempts[0].outcome.boundary_after == 2


def test_cut_inside_hold_makes_a_finite_hold_and_updates_expectation():
    original = outcome(Hold(3, boundary="sentence"), 0, (1, 2, 3), ("A", "B", "C"))
    history = EpisodeHistory((attempt(4, Hold(3, boundary="sentence"), original),))

    retained = history.retain_through(2)
    saved = retained.attempts[0]

    assert saved.action == Hold(2)
    assert saved.outcome.action == Hold(2)
    assert saved.outcome.boundary_before == 0
    assert saved.outcome.boundary_after == 2
    assert saved.outcome.stop_reason == "requested-length"
    assert saved.expectation == ReplayExpectation((1, 2), None, "requested-length")


def test_cut_inside_phrase_produces_exact_write_from_retained_evidence_text():
    phrase = Phrase("original phrase", mode="continuation")
    prefix = outcome(Hold(2), 0, (1, 2), ("prefix", " "))
    original = outcome(phrase, 2, (5, 6, 7), ("one", " two", " three"))
    history = EpisodeHistory(
        (attempt(0, Hold(2), prefix), attempt(8, phrase, original))
    )

    saved = history.retain_through(4).attempts[1]

    assert saved.action == Write("one two", mode="exact")
    assert saved.outcome.resolved_text == "one two"
    assert saved.outcome.visible_token_ids == (5, 6)
    assert saved.outcome.evidence == original.evidence[:2]
    assert saved.expectation == ReplayExpectation((5, 6), None, "completed")


def test_partial_retained_action_clears_terminal_divergence_and_replay_eog_state():
    phrase = Phrase("three tokens", mode="exact")
    divergence = Divergence(1, phrase.kind, "changed", 2, 9)
    original = outcome(
        phrase,
        0,
        (1, 2, 3),
        ("A", "B", "C"),
        status="completed-with-divergence",
        terminal_token_id=99,
        divergence=divergence,
        replay_eog_token_id=99,
    )
    history = EpisodeHistory((attempt(0, phrase, original),))

    saved = history.retain_through(2).attempts[0].outcome

    assert saved.status == "completed"
    assert saved.terminal_token_id is None
    assert saved.divergence is None
    assert saved.replay_eog_token_id is None


def test_zero_width_handoff_remains_in_raw_history():
    handed_off = Phrase("check", mode="exact")
    zero = outcome(handed_off, 0, status="handed-off")
    forced = Phrase("forced", mode="continuation", force=True)
    forced_result = outcome(forced, 0, (3,), ("X",))
    history = EpisodeHistory(
        (
            attempt(0, handed_off, zero),
            attempt(1, forced, forced_result),
        )
    )

    retained = history.retain_through(1)
    assert retained.visible_token_ids == (3,)
    assert len(retained.attempts) == 2
    assert retained.attempts[0].outcome.status == "handed-off"
