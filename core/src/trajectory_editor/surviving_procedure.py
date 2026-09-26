"""Project execution attempts into a replayable surviving procedure.

The runtime records what it tried, including replay handoffs and partially
realized actions.  A replay procedure is a different view: it contains only
the actions that should be executed again.

This module deliberately knows nothing about SQLite, JSON, or UI state.  A
persistence adapter can translate its records into :class:`ProcedureRecord`
values, and an in-memory session can do the same from its action outcomes.
The projection rules then remain identical for both paths.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .core.actions import Hold, Phrase, PolicyAction, Write
from .core.results import ActionOutcome, ReplayExpectation
from .core.sampler_config import SamplerConfig
from .run_loop import TapeStep


HANDED_OFF = "handed-off"


@dataclass(frozen=True)
class ProcedureRecord:
    """One recorded action attempt before procedure projection.

    ``expectation`` is the source-side replay evidence, when available.  It
    is intentionally separate from the outcome that caused this record to be
    retained: a replayed action may hand off while its original expectation
    remains the evidence needed by a future replay.

    ``visible_text`` is used only when a partially realized non-``Hold`` span
    must be represented as an exact finite write.  Adapters that do not have
    text available can leave it empty; the projection will use a finite
    ``Hold`` instead.
    """

    action: PolicyAction
    expectation: ReplayExpectation | None = None
    status: str = "completed"
    visible_token_ids: tuple[int, ...] = ()
    visible_text: str = ""
    boundary_before: int = 0
    sampling: SamplerConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("procedure record status must be a nonempty string")
        if type(self.boundary_before) is not int or self.boundary_before < 0:
            raise ValueError("procedure record boundary must be nonnegative")
        if any(type(token_id) is not int for token_id in self.visible_token_ids):
            raise ValueError("procedure record visible token ids must be integers")
        if not isinstance(self.visible_text, str):
            raise ValueError("procedure record visible text must be a string")

    @classmethod
    def from_outcome(
        cls,
        outcome: ActionOutcome,
        *,
        expectation: ReplayExpectation | None = None,
        sampling: SamplerConfig | None = None,
    ) -> "ProcedureRecord":
        """Build a neutral record from an in-memory runtime outcome.

        Replay callers should pass the source ``expectation`` explicitly.
        Without it, the outcome's own expectation is used, which is correct
        for ordinary live actions.
        """

        return cls(
            action=outcome.action,
            expectation=(
                expectation if expectation is not None else outcome.expectation()
            ),
            status=outcome.status,
            visible_token_ids=tuple(outcome.visible_token_ids),
            visible_text=outcome.resolved_text,
            boundary_before=outcome.boundary_before,
            sampling=sampling,
        )


@dataclass(frozen=True)
class ProcedureStep:
    """One surviving executable step plus source projection metadata."""

    tape_step: TapeStep
    boundary: int
    source_index: int
    sampling: SamplerConfig | None = None
    partial: bool = False

    @property
    def action(self) -> PolicyAction:
        return self.tape_step.action

    @property
    def expectation(self) -> ReplayExpectation | None:
        return self.tape_step.expectation


@dataclass(frozen=True)
class SurvivingProcedure:
    """The canonical replayable projection of recorded action attempts."""

    steps: tuple[ProcedureStep, ...] = ()
    skipped_source_indices: tuple[int, ...] = ()
    partial_source_indices: tuple[int, ...] = ()

    @property
    def tape(self) -> tuple[TapeStep, ...]:
        """Return the executable portion without projection metadata."""

        return tuple(step.tape_step for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


def _partial_step(record: ProcedureRecord) -> TapeStep:
    """Represent the visible portion of a handed-off action as finite work."""

    visible = tuple(record.visible_token_ids)
    if isinstance(record.action, Hold) or not record.visible_text:
        action: PolicyAction = Hold(len(visible))
    else:
        action = Write(record.visible_text, mode="exact")
    return TapeStep(
        action,
        ReplayExpectation(visible, None, "requested-length"),
    )


def _surviving_action(action: PolicyAction) -> PolicyAction:
    """Remove implementation-only force steering from surviving history."""

    if isinstance(action, Phrase) and action.force:
        # ``force`` and ``forcex`` use the same text resolution as Write; the
        # only extra behavior is temporary per-token policy steering. Preserve
        # the original mode so force remains continuation and forcex remains
        # exact.
        return Write(action.text, mode=action.mode)
    return action


def project_surviving_procedure(
    records: Iterable[ProcedureRecord],
    *,
    normalize_for_replay: bool = True,
) -> SurvivingProcedure:
    """Project recorded attempts into the procedure that should survive.

    A zero-width ``handed-off`` record is an attempted action, not replayable
    history, so it is omitted.  If the handoff occurred after visible tokens
    were realized, those tokens become a finite action.  All other statuses
    are retained as executable steps; in particular, a successful source
    check remains a check whose expectation can cause a future replay to
    hand off again if the target diverges.  Successful forced phrases are
    normalized to ordinary writes while retaining their continuation/exact
    mode when ``normalize_for_replay`` is true. Persistence adapters that
    preserve historical action kinds can disable that compatibility policy.
    """

    steps: list[ProcedureStep] = []
    skipped: list[int] = []
    partial: list[int] = []

    for source_index, record in enumerate(records):
        if record.status == HANDED_OFF:
            if not record.visible_token_ids:
                skipped.append(source_index)
                continue
            tape_step = _partial_step(record)
            partial.append(source_index)
            steps.append(
                ProcedureStep(
                    tape_step=tape_step,
                    boundary=record.boundary_before,
                    source_index=source_index,
                    sampling=record.sampling,
                    partial=True,
                )
            )
            continue

        action = (
            _surviving_action(record.action)
            if normalize_for_replay
            else record.action
        )
        steps.append(
            ProcedureStep(
                tape_step=TapeStep(action, record.expectation),
                boundary=record.boundary_before,
                source_index=source_index,
                sampling=record.sampling,
            )
        )

    return SurvivingProcedure(
        steps=tuple(steps),
        skipped_source_indices=tuple(skipped),
        partial_source_indices=tuple(partial),
    )


__all__ = [
    "HANDED_OFF",
    "ProcedureRecord",
    "ProcedureStep",
    "SurvivingProcedure",
    "project_surviving_procedure",
]
