"""Storage-neutral execution of replay plans and live teacher actions.

The runner owns control flow only. Durable episodes and persistence-free live
sessions provide small targets for state changes and recording; neither target
is represented as a fake version of the other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .core.actions import Phrase, PolicyAction
from .core.results import ActionOutcome, ReplayExpectation
from .core.sampler_config import SamplerConfig
from .episode_engine import InstructionRejected, Observation


class EdgeRequested(Exception):
    """Open the live edge without changing token state."""


class ForkRequested(Exception):
    """UI control-flow request to branch from an earlier token boundary."""

    def __init__(self, boundary: int) -> None:
        super().__init__(boundary)
        self.boundary = int(boundary)


class SeamlessRewindRequested(Exception):
    """UI control-flow request to retry an earlier point in this episode."""

    def __init__(self, boundary: int) -> None:
        super().__init__(boundary)
        self.boundary = int(boundary)


class SeamlessEdgeRequested(Exception):
    """UI control-flow request to reopen the latest live-edge menu."""

    def __init__(self, boundary: int) -> None:
        super().__init__(boundary)
        self.boundary = int(boundary)


class LivePolicy(Protocol):
    def choose(self, engine, observation: Observation) -> PolicyAction: ...


@dataclass(frozen=True)
class TapeStep:
    """One generative replay instruction and its optional handoff result."""

    action: PolicyAction
    expectation: ReplayExpectation | None


@dataclass(frozen=True)
class ReplayOrigin:
    """Optional source label for a replay-plan item."""

    source_episode_id: str | None = None
    source_boundary: int | None = None
    source_part: str | None = None


@dataclass(frozen=True)
class ReplayContext:
    """Auxiliary execution context aligned with a replay plan's steps."""

    sampling: tuple[SamplerConfig | None, ...] = ()
    origins: tuple[ReplayOrigin | None, ...] = ()

    def sampling_at(self, index: int) -> SamplerConfig | None:
        if index >= len(self.sampling):
            return None
        return self.sampling[index]

    def origin_at(self, index: int) -> ReplayOrigin | None:
        if index >= len(self.origins):
            return None
        return self.origins[index]


@dataclass(frozen=True)
class ReplayPlan(Sequence[TapeStep]):
    """A finite procedure whose authority ends when the runner yields."""

    steps: tuple[TapeStep, ...] = ()
    follow_source_sampling: bool = True
    final_sampling: SamplerConfig | None = None
    context: ReplayContext = ReplayContext()

    def __post_init__(self) -> None:
        for label, values in (
            ("sampling", self.context.sampling),
            ("origins", self.context.origins),
        ):
            if values and len(values) != len(self.steps):
                raise ValueError(
                    f"replay {label} context must align with replay steps"
                )

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]


@dataclass(frozen=True)
class RunResult:
    episode_id: str
    outcomes: tuple[ActionOutcome, ...]
    replayed_actions: int
    handed_off: bool
    replay_exhausted: bool = False
    handoff_reason: str | None = None


class RunTarget(Protocol):
    """The state and recording surface required by :func:`run_plan`."""

    @property
    def identifier(self) -> str: ...

    @property
    def engine(self): ...

    def begin(self) -> int: ...

    def set_sampler(self, sampling: SamplerConfig) -> None: ...

    def observe(self) -> Observation: ...

    def apply(
        self,
        action: PolicyAction,
        *,
        expectation: ReplayExpectation | None = None,
        divergence_policy: str,
        replay: bool,
    ) -> ActionOutcome: ...

    def record_replay(
        self,
        ordinal: int,
        index: int,
        step: TapeStep,
        outcome: ActionOutcome,
        origin: ReplayOrigin | None,
    ) -> None: ...

    def record_live(self, ordinal: int, outcome: ActionOutcome) -> None: ...

    def record_phrase_rejected(self, action: Phrase, reason: str) -> None: ...

    def record_instruction_rejected(
        self, action: PolicyAction | None, reason: str, replay: bool
    ) -> None: ...

    def record_control_flow(self) -> None: ...

    def record_failure(self) -> None: ...

    def complete(self, had_tape: bool) -> None: ...


def run_plan(
    target: RunTarget,
    *,
    divergence_policy: str,
    tape: Sequence[TapeStep] | ReplayPlan | None = None,
    live_policy: LivePolicy | None = None,
    stop_after_tape: bool = True,
    max_live_actions: int | None = None,
) -> RunResult:
    """Run one replay/live turn against either kind of episode target."""

    outcomes: list[ActionOutcome] = []
    replayed = 0
    handed_off = False
    handoff_reason: str | None = None
    tape_input = tape
    had_tape = tape is not None
    plan = tape if isinstance(tape, ReplayPlan) else ReplayPlan(tuple(tape or ()))
    replay_exhausted = False
    ordinal = target.begin()
    active_action: PolicyAction | None = None
    executing_replay = False

    try:
        for index, step in enumerate(plan.steps):
            if target.engine.ended or target.engine.checkpointed:
                break
            sampling = plan.context.sampling_at(index)
            if (
                plan.follow_source_sampling
                and sampling is not None
                and target.engine.sampling != sampling
            ):
                target.set_sampler(sampling)
            active_action = step.action
            executing_replay = True
            outcome = target.apply(
                step.action,
                expectation=step.expectation,
                divergence_policy=divergence_policy,
                replay=True,
            )
            target.record_replay(
                ordinal, index, step, outcome, plan.context.origin_at(index)
            )
            ordinal += 1
            outcomes.append(outcome)
            if outcome.status == "handed-off":
                handed_off = True
                break
            replayed += 1

        if (
            had_tape
            and not handed_off
            and not target.engine.ended
            and not target.engine.checkpointed
        ):
            replay_exhausted = replayed == len(plan.steps)
        if (
            replay_exhausted
            and plan.follow_source_sampling
            and plan.final_sampling is not None
            and target.engine.sampling != plan.final_sampling
        ):
            target.set_sampler(plan.final_sampling)

        # Explicit plans return through the edge. Legacy callers may ask for
        # live continuation after a successfully exhausted plain tape.
        should_run_live = not had_tape or (
            replay_exhausted
            and not stop_after_tape
            and not isinstance(tape_input, ReplayPlan)
        )
        if outcomes and outcomes[-1].stop_reason == "replay-eog":
            should_run_live = False
        live_actions = 0
        while (
            (max_live_actions is None or live_actions < max_live_actions)
            and should_run_live
            and not target.engine.ended
            and not target.engine.checkpointed
            and live_policy is not None
        ):
            observation = target.observe()
            action = live_policy.choose(target.engine, observation)
            # A policy may edit steering while choosing. Capture the actual
            # surface that the action will use.
            target.observe()
            active_action = action
            executing_replay = False
            try:
                outcome = target.apply(
                    action,
                    divergence_policy=divergence_policy,
                    replay=False,
                )
            except InstructionRejected as exc:
                if not isinstance(action, Phrase):
                    raise
                target.record_phrase_rejected(action, str(exc))
                rejected = getattr(live_policy, "action_rejected", None)
                if callable(rejected):
                    rejected(action, str(exc))
                live_actions += 1
                continue
            target.record_live(ordinal, outcome)
            ordinal += 1
            outcomes.append(outcome)
            live_actions += 1
    except InstructionRejected as exc:
        handoff_reason = str(exc)
        target.record_instruction_rejected(active_action, str(exc), executing_replay)
        handed_off = True
    except (
        EdgeRequested,
        ForkRequested,
        SeamlessRewindRequested,
        SeamlessEdgeRequested,
    ):
        target.record_control_flow()
        raise
    except BaseException:
        target.record_failure()
        raise

    target.complete(had_tape)
    return RunResult(
        target.identifier,
        tuple(outcomes),
        replayed,
        handed_off,
        replay_exhausted,
        handoff_reason,
    )


__all__ = [
    "EdgeRequested",
    "ForkRequested",
    "LivePolicy",
    "ReplayContext",
    "ReplayOrigin",
    "ReplayPlan",
    "RunResult",
    "RunTarget",
    "SeamlessEdgeRequested",
    "SeamlessRewindRequested",
    "TapeStep",
    "run_plan",
]
