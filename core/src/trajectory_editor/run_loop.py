"""Storage-neutral execution of replay plans and live teacher actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .core.actions import Phrase, PolicyAction
from .core.episode_observation import EpisodeObservation
from .core.results import ActionOutcome, ReplayExpectation
from .episode_engine import InstructionRejected


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
    def choose(self, engine, observation: EpisodeObservation) -> PolicyAction: ...


@dataclass(frozen=True)
class TapeStep:
    """One generative replay instruction and its optional handoff result."""

    action: PolicyAction
    expectation: ReplayExpectation | None


@dataclass(frozen=True)
class ReplayPlan(Sequence[TapeStep]):
    """A finite procedure whose authority ends when the runner yields."""

    steps: tuple[TapeStep, ...] = ()
    incomplete_handoff_reason: str | None = None

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
            if target.engine.ended:
                break
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
            else:
                replay_exhausted = replayed == len(plan.steps)

        live_actions = 0
        while (
            (max_live_actions is None or live_actions < max_live_actions)
            and not had_tape
            and not target.engine.ended
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
    "ReplayPlan",
    "RunResult",
    "RunTarget",
    "SeamlessRewindRequested",
    "TapeStep",
    "run_plan",
]
