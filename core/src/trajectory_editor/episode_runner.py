"""Core episode runner for replay, live actions, and persistence.

This module deliberately knows only the core action/result contract. Optional
extensions can decorate it outside the core runner without changing the
authoritative replay path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .core.actions import Phrase, PolicyAction
from .core.results import ActionOutcome, ReplayExpectation
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine, InstructionRejected, Observation

if TYPE_CHECKING:
    from .episode_store import EpisodeStore


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
    def choose(
        self, engine: EpisodeEngine, observation: Observation
    ) -> PolicyAction: ...


@dataclass(frozen=True)
class TapeStep:
    """One generative replay instruction and its optional handoff result.

    The sequence position (and persisted action ordinal) supplies step-N. This
    is intentionally the small replay contract. Sampler transitions and
    source lineage belong to :class:`ReplayContext`, not to the tape step.
    Editorial operations such as forks, rewinds, and searches never become
    tape steps.
    """

    action: PolicyAction
    expectation: ReplayExpectation | None


@dataclass(frozen=True)
class ReplayOrigin:
    """Optional source label for a replay-plan item.

    Origins are persistence/projector metadata.  They do not affect whether
    an action is replayable.
    """

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


class EpisodeRunner:
    """Execute the core replay/live loop and persist its durable results."""

    def __init__(
        self,
        engine: EpisodeEngine,
        store: "EpisodeStore",
        episode_id: str,
        *,
        divergence_policy: str = "handoff",
    ) -> None:
        self.engine = engine
        self.store = store
        self.episode_id = episode_id
        self.divergence_policy = divergence_policy

    def run(
        self,
        *,
        tape: Sequence[TapeStep] | ReplayPlan | None = None,
        live_policy: LivePolicy | None = None,
        stop_after_tape: bool = True,
        max_live_actions: int | None = None,
    ) -> RunResult:
        outcomes: list[ActionOutcome] = []
        replayed = 0
        handed_off = False
        handoff_reason: str | None = None
        tape_input = tape
        had_tape = tape is not None
        plan = tape if isinstance(tape, ReplayPlan) else ReplayPlan(tuple(tape or ()))
        tape = plan.steps
        replay_exhausted = False
        self.store.record_budget(
            self.episode_id,
            self.engine.boundary,
            self.engine.max_tokens,
            self.engine.checkpoint_boundary,
        )
        active_action: PolicyAction | None = None
        executing_replay = False
        ordinal = self.store.next_action_ordinal(self.episode_id)
        try:
            for index, step in enumerate(tape):
                if self.engine.ended or self.engine.checkpointed:
                    break
                sampling = plan.context.sampling_at(index)
                if plan.follow_source_sampling and sampling is not None:
                    if self.engine.sampling != sampling:
                        self.engine.sampling = sampling
                        self.store.record_sampling_segment(
                            self.episode_id,
                            start_boundary=self.engine.boundary,
                            sampling=self.engine.sampling,
                            stream_fingerprint=self.engine.stream_fingerprint,
                            coordinate_offset=self.engine.coordinate_offset,
                        )
                active_action = step.action
                executing_replay = True
                outcome = self.engine.apply(
                    step.action,
                    expectation=step.expectation,
                    divergence_policy=self.divergence_policy,
                    replay=True,
                )
                source = plan.context.origin_at(index)
                origin = None
                if source is not None and source.source_episode_id is not None:
                    origin = {
                        "episode_id": source.source_episode_id,
                        "boundary": source.source_boundary,
                    }
                    if source.source_part is not None:
                        origin["part"] = source.source_part
                self.store.record_action(
                    self.episode_id, ordinal, outcome, replay_origin=origin
                )
                if outcome.replay_eog_token_id is not None:
                    self.store.record_interaction(
                        self.episode_id,
                        self.engine.boundary,
                        "replay-eog",
                        {
                            "token_id": outcome.replay_eog_token_id,
                            "action_kind": step.action.kind,
                            "matched_expectation": outcome.divergence is None,
                        },
                    )
                outcomes.append(outcome)
                ordinal += 1
                if outcome.status == "handed-off":
                    handed_off = True
                    break
                replayed += 1
            if (
                had_tape
                and not handed_off
                and not self.engine.ended
                and not self.engine.checkpointed
            ):
                replay_exhausted = replayed == len(tape)
            if replay_exhausted and plan.follow_source_sampling and plan.final_sampling is not None:
                if self.engine.sampling != plan.final_sampling:
                    self.engine.sampling = plan.final_sampling
                    self.store.record_sampling_segment(
                        self.episode_id,
                        start_boundary=self.engine.boundary,
                        sampling=self.engine.sampling,
                        stream_fingerprint=self.engine.stream_fingerprint,
                        coordinate_offset=self.engine.coordinate_offset,
                    )
            # Explicit plans return through the edge. Legacy callers may ask
            # for live continuation after a successfully exhausted plain tape.
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
                and not self.engine.ended
                and not self.engine.checkpointed
                and live_policy is not None
            ):
                observation = self.engine.observe()
                action = live_policy.choose(self.engine, observation)
                # The policy may edit steering while choosing. Capture the
                # actual surface that the action will use.
                observation = self.engine.observe()
                active_action = action
                executing_replay = False
                try:
                    outcome = self.engine.apply(action)
                except InstructionRejected as exc:
                    if not isinstance(action, Phrase):
                        raise
                    self.store.record_interaction(
                        self.episode_id,
                        self.engine.boundary,
                        "phrase-rejected",
                        {"action": action.to_dict(), "reason": str(exc)},
                    )
                    rejected = getattr(live_policy, "action_rejected", None)
                    if callable(rejected):
                        rejected(action, str(exc))
                    live_actions += 1
                    continue
                live_actions += 1
                self.store.record_action(self.episode_id, ordinal, outcome)
                outcomes.append(outcome)
                ordinal += 1
        except InstructionRejected as exc:
            handoff_reason = str(exc)
            self.store.record_interaction(
                self.episode_id,
                self.engine.boundary,
                "instruction-rejected",
                {
                    "action": active_action.to_dict() if active_action else None,
                    "reason": str(exc),
                    "replay": executing_replay,
                },
            )
            handed_off = True
        except (
            EdgeRequested,
            ForkRequested,
            SeamlessRewindRequested,
            SeamlessEdgeRequested,
        ):
            self.store.update_episode(
                self.episode_id,
                visible_text=self.engine.backend.render(self.engine.visible_token_ids),
                max_tokens=self.engine.max_tokens,
                status="open",
            )
            raise
        except BaseException:
            self.store.finish_episode(
                self.episode_id,
                visible_text=self.engine.backend.render(self.engine.visible_token_ids),
                terminal_token_id=self.engine.terminal_token_id,
                terminal_reason=self.engine.terminal_reason or "error",
                status="failed",
            )
            raise
        visible_text = self.engine.backend.render(self.engine.visible_token_ids)
        if self.engine.ended:
            self.store.finish_episode(
                self.episode_id,
                visible_text=visible_text,
                terminal_token_id=self.engine.terminal_token_id,
                terminal_reason=self.engine.terminal_reason,
                status="completed",
            )
        else:
            self.store.update_episode(
                self.episode_id,
                visible_text=visible_text,
                max_tokens=self.engine.max_tokens,
                status=(
                    "checkpoint"
                    if self.engine.checkpointed
                    else "replay-edge"
                    if had_tape
                    else "open"
                ),
            )
        return RunResult(
            self.episode_id,
            tuple(outcomes),
            replayed,
            handed_off,
            replay_exhausted,
            handoff_reason,
        )


class LiveSessionRunner:
    """Run a persistence-free :class:`LiveEpisode` at its current branch.

    This is deliberately the sibling of :class:`EpisodeRunner`, rather than a
    fake in-memory ``EpisodeStore``.  A live session owns its tape and outcomes
    itself; its caller can later export or save a selected branch explicitly.
    """

    def __init__(self, session, *, divergence_policy: str = "handoff") -> None:
        self.session = session
        self.engine = session.engine
        self.divergence_policy = divergence_policy

    def run(
        self,
        *,
        tape: Sequence[TapeStep] | ReplayPlan | None = None,
        live_policy: LivePolicy | None = None,
        stop_after_tape: bool = True,
        max_live_actions: int | None = None,
    ) -> RunResult:
        outcomes: list[ActionOutcome] = []
        replayed = 0
        handed_off = False
        handoff_reason: str | None = None
        tape_input = tape
        had_tape = tape is not None
        plan = tape if isinstance(tape, ReplayPlan) else ReplayPlan(tuple(tape or ()))
        replay_exhausted = False
        active_action: PolicyAction | None = None
        executing_replay = False
        try:
            for index, step in enumerate(plan.steps):
                if self.engine.ended or self.engine.checkpointed:
                    break
                sampling = plan.context.sampling_at(index)
                if plan.follow_source_sampling and sampling is not None:
                    self.session.set_sampler(sampling)
                active_action = step.action
                executing_replay = True
                outcome = self.session.generate(
                    step.action,
                    expectation=step.expectation,
                    divergence_policy=self.divergence_policy,
                    replay=True,
                )
                outcomes.append(outcome)
                if outcome.status == "handed-off":
                    handed_off = True
                    break
                replayed += 1
            if had_tape and not handed_off and not self.engine.ended and not self.engine.checkpointed:
                replay_exhausted = replayed == len(plan.steps)
            if replay_exhausted and plan.follow_source_sampling and plan.final_sampling is not None:
                self.session.set_sampler(plan.final_sampling)
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
                and not self.engine.ended
                and not self.engine.checkpointed
                and live_policy is not None
            ):
                observation = self.engine.observe()
                action = live_policy.choose(self.engine, observation)
                active_action = action
                executing_replay = False
                try:
                    outcome = self.session.generate(action)
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
        return RunResult(
            self.session.branch.branch_id,
            tuple(outcomes),
            replayed,
            handed_off,
            replay_exhausted,
            handoff_reason,
        )


__all__ = [
    "EdgeRequested",
    "EpisodeRunner",
    "ForkRequested",
    "LiveSessionRunner",
    "LivePolicy",
    "ReplayPlan",
    "RunResult",
    "SeamlessEdgeRequested",
    "SeamlessRewindRequested",
    "TapeStep",
]
