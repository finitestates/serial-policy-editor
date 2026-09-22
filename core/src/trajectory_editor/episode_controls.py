"""Storage-neutral control state for a root-relative episode timeline.

This module describes the controls that are effective between visible-token
boundaries.  It deliberately contains no episode execution or persistence
knowledge: adapters can translate their own control-point records into these
values and back again.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .episode_hash import validate_coordinate, validate_fingerprint


@dataclass(frozen=True, init=False)
class SamplerState:
    """Sampler and root-relative sampling-coordinate state.

    ``stream_fingerprint`` is nullable so an in-memory adapter can represent
    a coordinate that has not yet been assigned a stream identity.  When it
    is present, it is always a lowercase SHA-256 digest.
    """

    sampling: SamplerConfig
    stream_fingerprint: str | None
    coordinate_offset: int

    def __init__(
        self,
        sampling: SamplerConfig | None = None,
        stream_fingerprint: str | None = None,
        coordinate_offset: int = 0,
        *,
        sampler: SamplerConfig | None = None,
    ) -> None:
        if sampling is not None and sampler is not None:
            raise EditorError("sampler state cannot set sampling and sampler twice")
        selected = sampling if sampling is not None else sampler
        if not isinstance(selected, SamplerConfig):
            raise EditorError("sampler state requires a SamplerConfig")
        if stream_fingerprint is not None:
            validate_fingerprint(stream_fingerprint)
        validate_coordinate(coordinate_offset, "coordinate_offset")
        object.__setattr__(self, "sampling", selected)
        object.__setattr__(self, "stream_fingerprint", stream_fingerprint)
        object.__setattr__(self, "coordinate_offset", coordinate_offset)

    @property
    def sampler(self) -> SamplerConfig:
        """Alias matching the conceptual sampler/coordinate state name."""

        return self.sampling


@dataclass(frozen=True, init=False)
class BudgetState:
    """Allowance and the root-visible boundary at which it checkpoints.

    ``None`` for both fields means unlimited.  A finite allowance is a
    positive integer; its checkpoint is validated against the transition
    boundary by :class:`ControlTransition`.
    """

    allowance: int | None
    checkpoint_boundary: int | None

    def __init__(
        self,
        allowance: int | None = None,
        checkpoint_boundary: int | None = None,
        *,
        max_tokens: int | None = None,
    ) -> None:
        if allowance is not None and max_tokens is not None:
            raise EditorError("budget state cannot set allowance and max_tokens twice")
        selected = allowance if allowance is not None else max_tokens
        if selected is not None and (type(selected) is not int or selected < 1):
            raise EditorError("allowance must be a positive integer or null")
        if checkpoint_boundary is not None:
            validate_coordinate(checkpoint_boundary, "checkpoint_boundary")
        if (selected is None) != (checkpoint_boundary is None):
            raise EditorError("allowance and checkpoint boundary must both be set or null")
        object.__setattr__(self, "allowance", selected)
        object.__setattr__(self, "checkpoint_boundary", checkpoint_boundary)

    @property
    def max_tokens(self) -> int | None:
        """Compatibility spelling used by the runtime-facing adapters."""

        return self.allowance

    @property
    def unlimited(self) -> bool:
        return self.allowance is None

    def valid_at(self, boundary: int) -> "BudgetState":
        """Validate this budget as the state beginning at ``boundary``."""

        validate_coordinate(boundary, "boundary")
        if (
            self.checkpoint_boundary is not None
            and self.checkpoint_boundary < boundary
        ):
            raise EditorError(
                "checkpoint boundary must not precede its control boundary"
            )
        return self


@dataclass(frozen=True)
class ControlState:
    """Complete control state effective from one timeline transition."""

    sampler: SamplerState
    budget: BudgetState

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, SamplerState):
            raise EditorError("control state requires a SamplerState")
        if not isinstance(self.budget, BudgetState):
            raise EditorError("control state requires a BudgetState")

    @classmethod
    def from_parts(
        cls,
        sampling: SamplerConfig,
        stream_fingerprint: str | None,
        coordinate_offset: int,
        allowance: int | None,
        checkpoint_boundary: int | None,
    ) -> "ControlState":
        """Build a state directly from adapter-friendly scalar fields."""

        return cls(
            SamplerState(sampling, stream_fingerprint, coordinate_offset),
            BudgetState(allowance, checkpoint_boundary),
        )

    @property
    def sampling(self) -> SamplerConfig:
        return self.sampler.sampling

    @property
    def stream_fingerprint(self) -> str | None:
        return self.sampler.stream_fingerprint

    @property
    def coordinate_offset(self) -> int:
        return self.sampler.coordinate_offset

    @property
    def allowance(self) -> int | None:
        return self.budget.allowance

    @property
    def max_tokens(self) -> int | None:
        return self.budget.allowance

    @property
    def checkpoint_boundary(self) -> int | None:
        return self.budget.checkpoint_boundary

    def with_sampler(self, sampler: SamplerState) -> "ControlState":
        """Return this state with only its sampler-coordinate part changed."""

        if not isinstance(sampler, SamplerState):
            raise EditorError("control state requires a SamplerState")
        return ControlState(sampler, self.budget)

    def with_budget(self, budget: BudgetState) -> "ControlState":
        """Return this state with only its budget part changed."""

        if not isinstance(budget, BudgetState):
            raise EditorError("control state requires a BudgetState")
        return ControlState(self.sampler, budget)


@dataclass(frozen=True)
class ControlTransition:
    """A state transition effective at its own ``start_boundary``."""

    start_boundary: int
    state: ControlState

    def __post_init__(self) -> None:
        validate_coordinate(self.start_boundary, "start_boundary")
        if not isinstance(self.state, ControlState):
            raise EditorError("control transition requires a ControlState")
        self.state.budget.valid_at(self.start_boundary)

    @property
    def boundary(self) -> int:
        """Adapter-friendly alias for ``start_boundary``."""

        return self.start_boundary

    @property
    def control(self) -> ControlState:
        return self.state


@dataclass(frozen=True)
class ControlTimeline:
    """Ordered, immutable control transitions in root-relative coordinates."""

    transitions: tuple[ControlTransition, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if not transitions:
            raise EditorError("control timeline requires a root transition")
        if transitions[0].start_boundary != 0:
            raise EditorError("control timeline must begin at boundary zero")
        previous = -1
        for transition in transitions:
            if not isinstance(transition, ControlTransition):
                raise EditorError("control timeline entries must be transitions")
            if transition.start_boundary <= previous:
                raise EditorError(
                    "control transition boundaries must be strictly increasing"
                )
            previous = transition.start_boundary
        object.__setattr__(self, "transitions", transitions)

    @classmethod
    def from_state(cls, state: ControlState) -> "ControlTimeline":
        """Create a root timeline whose initial state begins at boundary zero."""

        return cls((ControlTransition(0, state),))

    @property
    def segments(self) -> tuple[ControlTransition, ...]:
        """Alias emphasizing that transitions describe constant-state spans."""

        return self.transitions

    def effective_at(self, boundary: int) -> ControlState:
        """Return the state effective at ``boundary``.

        A transition is effective at its own start boundary, so an exact
        boundary match selects that transition rather than the prior one.
        """

        validate_coordinate(boundary, "boundary")
        effective: ControlState | None = None
        for transition in self.transitions:
            if transition.start_boundary > boundary:
                break
            effective = transition.state
        if effective is None:
            raise EditorError("no control state is available at this boundary")
        return effective

    def append_transition(
        self,
        start_boundary: int,
        state: ControlState,
    ) -> "ControlTimeline":
        """Append a changed state, returning a new canonical timeline.

        Repeating the effective state is a no-op.  A changed state recorded at
        the last transition boundary replaces that transition, which keeps a
        same-boundary update canonical without changing any root coordinate.
        Earlier boundaries must be truncated explicitly before new future
        history can be appended.
        """

        transition = ControlTransition(start_boundary, state)
        if not self.transitions:
            return ControlTimeline((transition,))
        last = self.transitions[-1]
        if start_boundary < last.start_boundary:
            raise EditorError(
                "new control transition must not precede the latest transition"
            )
        if state == last.state:
            return self
        if start_boundary == last.start_boundary:
            return ControlTimeline((*self.transitions[:-1], transition))
        return ControlTimeline((*self.transitions, transition))

    append = append_transition

    def append_sampler_transition(
        self,
        start_boundary: int,
        sampler: SamplerState,
    ) -> "ControlTimeline":
        """Append a sampler-coordinate change, carrying forward the budget."""

        if not self.transitions:
            raise EditorError("a sampler transition needs an initial control state")
        return self.append_transition(
            start_boundary,
            self.effective_at(start_boundary).with_sampler(sampler),
        )

    def append_budget_transition(
        self,
        start_boundary: int,
        budget: BudgetState,
    ) -> "ControlTimeline":
        """Append a budget change, carrying forward sampler coordinates."""

        if not self.transitions:
            raise EditorError("a budget transition needs an initial control state")
        return self.append_transition(
            start_boundary,
            self.effective_at(start_boundary).with_budget(budget),
        )

    append_sampler = append_sampler_transition
    append_budget = append_budget_transition

    def truncate_after(self, retained_boundary: int) -> "ControlTimeline":
        """Drop transitions after a retained root-visible boundary.

        The retained transition and all state values, especially coordinate
        offsets, are copied unchanged.  No local or branch-relative rebasing
        is performed.
        """

        validate_coordinate(retained_boundary, "retained_boundary")
        kept = tuple(
            transition
            for transition in self.transitions
            if transition.start_boundary <= retained_boundary
        )
        return self if len(kept) == len(self.transitions) else ControlTimeline(kept)

    truncate = truncate_after


# Names that make adapter code read naturally without creating a second model.
SamplerCoordinateState = SamplerState
ControlSegment = ControlTransition
EpisodeControlState = ControlState


__all__ = [
    "BudgetState",
    "ControlSegment",
    "ControlState",
    "ControlTimeline",
    "ControlTransition",
    "EpisodeControlState",
    "SamplerCoordinateState",
    "SamplerState",
]
