"""A storage-neutral, root-relative history of episode action attempts.

``EpisodeHistory`` is the semantic seam between execution and persistence.
Adapters can turn rows or live outcomes into :class:`RecordedAttempt` values,
then use this module for boundary validation and prefix/rewind projection.
The history deliberately has no tokenizer or backend: visible text comes from
the recorded :class:`~trajectory_editor.core.results.TokenEvidence`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

from .core.actions import Hold, Phrase, PolicyAction, Write
from .core.results import ActionOutcome, ReplayExpectation, TokenEvidence
from .surviving_procedure import (
    ProcedureRecord,
    SurvivingProcedure,
    project_surviving_procedure,
)


def _visible_evidence(outcome: ActionOutcome) -> tuple[TokenEvidence, ...]:
    return tuple(item for item in outcome.evidence if item.realized_visible)


@dataclass(frozen=True, slots=True)
class RecordedAttempt:
    """One root-relative action attempt and the evidence it produced.

    ``action`` is the requested policy action.  It normally equals
    ``outcome.action``; keeping it as a first-class field makes the adapter
    contract explicit and lets validation reject mixed records.  An explicit
    ``expectation`` is used when the attempt came from an earlier replay; for
    ordinary live attempts, the outcome's expectation is used when exporting
    to a surviving procedure.
    """

    ordinal: int
    action: PolicyAction
    outcome: ActionOutcome
    expectation: ReplayExpectation | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("attempt ordinal must be a nonnegative integer")
        if not isinstance(self.outcome, ActionOutcome):
            raise TypeError("attempt outcome must be an ActionOutcome")
        if self.action != self.outcome.action:
            raise ValueError("attempt action must match outcome action")
        if self.expectation is not None and not isinstance(
            self.expectation, ReplayExpectation
        ):
            raise TypeError("attempt expectation must be a ReplayExpectation or None")

    @property
    def requested_action(self) -> PolicyAction:
        """Alias spelling for adapters that name the action explicitly."""

        return self.action

    @classmethod
    def from_outcome(
        cls,
        ordinal: int,
        outcome: ActionOutcome,
        *,
        expectation: ReplayExpectation | None = None,
    ) -> "RecordedAttempt":
        return cls(ordinal, outcome.action, outcome, expectation)


@dataclass(frozen=True, slots=True)
class HistoryTruncation:
    """The result of retaining a history prefix.

    ``discarded`` contains complete attempts that begin at or after the
    requested boundary.  If the boundary cuts an attempt, that original
    attempt is represented by ``partial`` and its retained form is in
    ``retained``; it is not duplicated in ``discarded``.
    """

    requested_boundary: int
    retained: "EpisodeHistory"
    discarded: tuple[RecordedAttempt, ...]
    partial: RecordedAttempt | None = None

    @property
    def retained_history(self) -> "EpisodeHistory":
        return self.retained

    @property
    def discarded_attempts(self) -> tuple[RecordedAttempt, ...]:
        return self.discarded


@dataclass(frozen=True, slots=True)
class EpisodeHistory:
    """Ordered action/evidence attempts on one root-relative visible stream."""

    attempts: tuple[RecordedAttempt, ...] = ()

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        object.__setattr__(self, "attempts", attempts)

        previous_ordinal: int | None = None
        previous_boundary = 0
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, RecordedAttempt):
                raise TypeError("history attempts must be RecordedAttempt values")
            if previous_ordinal is not None and attempt.ordinal <= previous_ordinal:
                raise ValueError("attempt ordinals must be strictly increasing")

            outcome = attempt.outcome
            before = outcome.boundary_before
            after = outcome.boundary_after
            if type(before) is not int or type(after) is not int:
                raise ValueError("attempt boundaries must be integers")
            if before < 0 or after < before:
                raise ValueError("attempt boundaries must be nonnegative and ordered")
            if index == 0 and before != 0:
                raise ValueError("history must begin at root-visible boundary zero")
            if before != previous_boundary:
                raise ValueError("attempt boundaries must be contiguous")
            if after - before != len(outcome.visible_token_ids):
                raise ValueError(
                    "boundary width must equal the number of visible token ids"
                )

            evidence = tuple(outcome.evidence)
            for item in evidence:
                if not isinstance(item, TokenEvidence):
                    raise TypeError("outcome evidence must contain TokenEvidence values")

            for item in evidence:
                if type(item.boundary) is not int or item.boundary < before:
                    raise ValueError("evidence boundaries must be root-relative")
                if item.realized_visible:
                    if item.boundary >= after:
                        raise ValueError(
                            "visible evidence must lie inside its visible boundary span"
                        )
                elif item.boundary > after:
                    raise ValueError(
                        "non-visible evidence must not pass the outcome boundary"
                    )

            visible = tuple(item for item in evidence if item.realized_visible)
            if tuple(item.token_id for item in visible) != tuple(
                outcome.visible_token_ids
            ):
                raise ValueError(
                    "visible evidence token ids must match the outcome visible ids"
                )
            if tuple(item.boundary for item in visible) != tuple(
                range(before, after)
            ):
                raise ValueError(
                    "visible evidence boundaries must be contiguous and ordered"
                )

            previous_ordinal = attempt.ordinal
            previous_boundary = after

    @classmethod
    def from_attempts(cls, attempts: Iterable[RecordedAttempt]) -> "EpisodeHistory":
        return cls(tuple(attempts))

    def __len__(self) -> int:
        return len(self.attempts)

    def __iter__(self) -> Iterator[RecordedAttempt]:
        return iter(self.attempts)

    @property
    def current_boundary(self) -> int:
        """The current root-relative visible boundary."""

        return self.attempts[-1].outcome.boundary_after if self.attempts else 0

    @property
    def visible_boundary(self) -> int:
        return self.current_boundary

    @property
    def visible_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for attempt in self.attempts
            for token_id in attempt.outcome.visible_token_ids
        )

    @property
    def visible_token_evidence(self) -> tuple[TokenEvidence, ...]:
        return tuple(
            item
            for attempt in self.attempts
            for item in _visible_evidence(attempt.outcome)
        )

    @property
    def visible_evidence(self) -> tuple[TokenEvidence, ...]:
        return self.visible_token_evidence

    @property
    def visible_text(self) -> str:
        return "".join(item.text for item in self.visible_token_evidence)

    def _partial_attempt(
        self, attempt: RecordedAttempt, boundary: int
    ) -> RecordedAttempt | None:
        outcome = attempt.outcome
        count = boundary - outcome.boundary_before
        retained_evidence = _visible_evidence(outcome)[:count]
        visible_ids = tuple(item.token_id for item in retained_evidence)
        if not visible_ids:
            return None

        retained_text = "".join(item.text for item in retained_evidence)
        if isinstance(attempt.action, (Write, Phrase)):
            action: PolicyAction = Write(retained_text, mode="exact")
            stop_reason = "completed"
        else:
            action = Hold(len(visible_ids))
            stop_reason = "requested-length"

        retained_outcome = replace(
            outcome,
            action=action,
            boundary_after=boundary,
            resolved_text=retained_text,
            resolved_token_ids=visible_ids,
            visible_token_ids=visible_ids,
            terminal_token_id=None,
            stop_reason=stop_reason,
            evidence=retained_evidence,
            status="completed",
            divergence=None,
            replay_eog_token_id=None,
            diagnostics=None,
        )
        expectation = ReplayExpectation(visible_ids, None, stop_reason)
        return RecordedAttempt(
            ordinal=attempt.ordinal,
            action=action,
            outcome=retained_outcome,
            expectation=expectation,
        )

    def truncate(self, boundary: int) -> HistoryTruncation:
        """Retain the history through ``boundary`` without rebasing it.

        Raw zero-width handed-off attempts before the requested boundary are
        retained as attempted actions.  The surviving-procedure projector is
        responsible for omitting them from replay.  A cut strictly inside an
        attempt creates one transformed retained attempt; all later complete
        attempts are returned as the discarded suffix.
        """

        if type(boundary) is not int or boundary < 0:
            raise ValueError("retained boundary must be a nonnegative integer")
        if boundary > self.current_boundary:
            raise ValueError(
                f"retained boundary must be between 0 and {self.current_boundary}"
            )

        retained: list[RecordedAttempt] = []
        discarded: list[RecordedAttempt] = []
        partial: RecordedAttempt | None = None

        for attempt in self.attempts:
            outcome = attempt.outcome
            before = outcome.boundary_before
            after = outcome.boundary_after

            if after < boundary or (after == boundary and before < boundary):
                retained.append(attempt)
                continue

            if before < boundary < after:
                partial = attempt
                transformed = self._partial_attempt(attempt, boundary)
                if transformed is not None:
                    retained.append(transformed)
                else:
                    discarded.append(attempt)
                continue

            discarded.append(attempt)

        return HistoryTruncation(
            requested_boundary=boundary,
            retained=EpisodeHistory(tuple(retained)),
            discarded=tuple(discarded),
            partial=partial,
        )

    def retain_through(self, boundary: int) -> "EpisodeHistory":
        """Return only the root-relative history retained through ``boundary``."""

        return self.truncate(boundary).retained

    def rewind(self, boundary: int) -> HistoryTruncation:
        """Alias for :meth:`truncate` using episode terminology."""

        return self.truncate(boundary)

    def rewind_to(self, boundary: int) -> "EpisodeHistory":
        return self.retain_through(boundary)

    def to_procedure_records(self) -> tuple[ProcedureRecord, ...]:
        """Translate attempts for the shared surviving-procedure projector."""

        return tuple(
            ProcedureRecord(
                action=attempt.action,
                expectation=(
                    attempt.expectation
                    if attempt.expectation is not None
                    else attempt.outcome.expectation()
                ),
                status=attempt.outcome.status,
                visible_token_ids=tuple(attempt.outcome.visible_token_ids),
                visible_text="".join(
                    item.text for item in _visible_evidence(attempt.outcome)
                ),
                boundary_before=attempt.outcome.boundary_before,
            )
            for attempt in self.attempts
        )

    def project_surviving_procedure(
        self, *, normalize_for_replay: bool = True
    ) -> SurvivingProcedure:
        """Delegate semantic replay filtering to the canonical projector."""

        return project_surviving_procedure(
            self.to_procedure_records(),
            normalize_for_replay=normalize_for_replay,
        )

    def surviving_procedure(
        self, *, normalize_for_replay: bool = True
    ) -> SurvivingProcedure:
        return self.project_surviving_procedure(
            normalize_for_replay=normalize_for_replay
        )


__all__ = [
    "EpisodeHistory",
    "HistoryTruncation",
    "RecordedAttempt",
]
