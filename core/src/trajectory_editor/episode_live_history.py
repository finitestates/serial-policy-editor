"""Adapt aligned live tape and outcomes into a root-relative history prefix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core.errors import EditorError
from .core.results import ActionOutcome
from .episode_history import EpisodeHistory, RecordedAttempt
from .episode_runner import TapeStep


@dataclass(frozen=True)
class LiveHistoryTruncation:
    """A live branch's retained prefix and original discarded suffix."""

    boundary: int
    retained_tape: tuple[TapeStep, ...]
    retained_outcomes: tuple[ActionOutcome, ...]
    discarded_tape: tuple[TapeStep, ...]
    discarded_outcomes: tuple[ActionOutcome, ...]


def history_from_live(
    tape: Sequence[TapeStep],
    outcomes: Sequence[ActionOutcome],
) -> EpisodeHistory:
    """Transform aligned live execution records into typed history attempts."""

    if len(tape) != len(outcomes):
        raise EditorError("live history tape and outcomes must align")
    try:
        return EpisodeHistory(
            tuple(
                RecordedAttempt(index, step.action, outcome, step.expectation)
                for index, (step, outcome) in enumerate(zip(tape, outcomes))
            )
        )
    except (TypeError, ValueError) as exc:
        raise EditorError(f"live history is invalid: {exc}") from exc


def truncate_live_history(
    tape: Sequence[TapeStep],
    outcomes: Sequence[ActionOutcome],
    boundary: int,
) -> LiveHistoryTruncation:
    """Retain one live root-relative prefix without rebasing boundaries."""

    history = history_from_live(tape, outcomes)
    try:
        truncation = history.truncate(boundary)
    except ValueError as exc:
        raise EditorError(str(exc)) from exc

    partial_ordinal = (
        truncation.partial.ordinal if truncation.partial is not None else None
    )
    retained_tape = tuple(
        TapeStep(attempt.action, attempt.expectation)
        if attempt.ordinal == partial_ordinal
        else tape[attempt.ordinal]
        for attempt in truncation.retained
    )
    retained_outcomes = tuple(
        attempt.outcome
        if attempt.ordinal == partial_ordinal
        else outcomes[attempt.ordinal]
        for attempt in truncation.retained
    )
    discarded = (
        (() if truncation.partial is None else (truncation.partial,))
        + truncation.discarded
    )
    discarded_tape = tuple(tape[attempt.ordinal] for attempt in discarded)
    discarded_outcomes = tuple(outcomes[attempt.ordinal] for attempt in discarded)
    return LiveHistoryTruncation(
        boundary=boundary,
        retained_tape=retained_tape,
        retained_outcomes=retained_outcomes,
        discarded_tape=discarded_tape,
        discarded_outcomes=discarded_outcomes,
    )


__all__ = [
    "LiveHistoryTruncation",
    "history_from_live",
    "truncate_live_history",
]
