"""Storage-neutral execution of replay plans and live teacher actions."""

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


class LivePolicy(Protocol):
    def choose(self, engine, observation: Observation) -> PolicyAction: ...


@dataclass(frozen=True)
class TapeStep:
    """One generative replay instruction and its optional handoff result."""

    action: PolicyAction
    expectation: ReplayExpectation | None


@dataclass(frozen=True)
class ReplayContext:
    """Auxiliary execution context aligned with a replay plan's steps."""

    sampling: tuple[SamplerConfig | None, ...] = ()

    def sampling_at(self, index: int) -> SamplerConfig | None:
        if index >= len(self.sampling):
            return None
        return self.sampling[index]


@dataclass(frozen=True)
class ReplayPlan(Sequence[TapeStep]):
    """A finite procedure whose authority ends when the runner yields."""

    steps: tuple[TapeStep, ...] = ()
    follow_source_sampling: bool = True
    final_sampling: SamplerConfig | None = None
    context: ReplayContext = ReplayContext()
    incomplete_handoff_reason: str | None = None

    def __post_init__(self) -> None:
        if self.context.sampling and len(self.context.sampling) != len(self.steps):
            raise ValueError("replay sampling context must align with replay steps")

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]


@dataclass(frozen=True)
class RunResult:
    outcomes: tuple[ActionOutcome, ...]
    replayed_actions: int
    handed_off: bool
    replay_exhausted: bool = False
    handoff_reason: str | None = None


class RunTarget(Protocol):
    """The in-memory state surface required by :func:`run_plan`."""

    @property
    def engine(self): ...

    def set_sampler(self, sampling: SamplerConfig) -> None: ...

    def generate(
        self,
        action: PolicyAction,
        *,
        expectation: ReplayExpectation | None = None,
        divergence_policy: str,
        replay: bool,
    ) -> ActionOutcome: ...


def run_plan(
    target: RunTarget,
    *,
    divergence_policy: str,
    tape: Sequence[TapeStep] | ReplayPlan | None = None,
    live_policy: LivePolicy | None = None,
    max_live_actions: int | None = None,
) -> RunResult:
    """Apply replay/live work to an in-memory execution target."""

    outcomes: list[ActionOutcome] = []
    replayed = 0
    handed_off = False
    handoff_reason: str | None = None
    had_tape = tape is not None
    plan = tape if isinstance(tape, ReplayPlan) else ReplayPlan(tuple(tape or ()))
    replay_exhausted = False
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
            outcome = target.generate(
                step.action,
                expectation=step.expectation,
                divergence_policy=divergence_policy,
                replay=True,
            )
            outcomes.append(outcome)
            if outcome.status == "handed-off":
                handed_off = True
                break
            replayed += 1

        if had_tape and not handed_off and not target.engine.ended:
            if (
                replayed == len(plan.steps)
                and plan.incomplete_handoff_reason is not None
            ):
                handed_off = True
                handoff_reason = plan.incomplete_handoff_reason
            elif not target.engine.checkpointed:
                replay_exhausted = replayed == len(plan.steps)
        if (
            replay_exhausted
            and plan.follow_source_sampling
            and plan.final_sampling is not None
            and target.engine.sampling != plan.final_sampling
        ):
            target.set_sampler(plan.final_sampling)

        live_actions = 0
        while (
            (max_live_actions is None or live_actions < max_live_actions)
            and not had_tape
            and not target.engine.ended
            and not target.engine.checkpointed
            and live_policy is not None
        ):
            observation = target.engine.observe()
            action = live_policy.choose(target.engine, observation)
            # A policy may edit steering while choosing. Capture the actual
            # surface that the action will use.
            target.engine.observe()
            try:
                outcome = target.generate(
                    action,
                    divergence_policy=divergence_policy,
                    replay=False,
                )
            except InstructionRejected as exc:
                if not isinstance(action, Phrase):
                    raise
                rejected = getattr(live_policy, "action_rejected", None)
                if callable(rejected):
                    rejected(action, str(exc))
                live_actions += 1
                continue
            outcomes.append(outcome)
            live_actions += 1
    except InstructionRejected as exc:
        handoff_reason = str(exc)
        handed_off = True
    except (
        EdgeRequested,
        ForkRequested,
        SeamlessRewindRequested,
    ):
        raise
    return RunResult(
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
    "ReplayPlan",
    "RunResult",
    "RunTarget",
    "SeamlessRewindRequested",
    "TapeStep",
    "run_plan",
]
