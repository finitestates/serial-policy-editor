"""Policy providers and the common episode runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .domain import SamplingConfig
from .episode_actions import PolicyAction
from .episode_engine import ActionOutcome, EpisodeEngine, Observation, ReplayExpectation, InstructionRejected
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
    action: PolicyAction
    expectation: ReplayExpectation | None
    # A source replay step carries the sampler active at its source boundary.
    # ``None`` intentionally means "keep the current sampler", which is used
    # for manually constructed tapes and counterfactual live-edge SPR.
    sampling: SamplingConfig | None = None
    # Immediate source location; destination boundaries remain engine-owned.
    source_episode_id: str | None = None
    source_boundary: int | None = None
    source_part: str | None = None


@dataclass(frozen=True)
class ReplayPlan(Sequence[TapeStep]):
    """A finite procedure whose authority ends when the runner yields."""

    steps: tuple[TapeStep, ...] = ()
    follow_source_sampling: bool = True
    final_sampling: SamplingConfig | None = None

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
    """Feed replay and live policies into the same action interpreter."""

    def __init__(
        self,
        engine: EpisodeEngine,
        store: EpisodeStore,
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
        self.store.record_budget(self.episode_id, self.engine.boundary,
                                 self.engine.max_tokens, self.engine.checkpoint_boundary)
        active_action: PolicyAction | None = None
        executing_replay = False
        ordinal = self.store.next_action_ordinal(self.episode_id)
        try:
            for step in tape:
                if self.engine.ended or self.engine.checkpointed:
                    break
                if plan.follow_source_sampling and step.sampling is not None:
                    if self.engine.sampling != step.sampling:
                        self.engine.sampling = step.sampling
                        self.store.record_sampling_segment(
                            self.episode_id,
                            start_boundary=self.engine.boundary,
                            sampling=step.sampling,
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
                origin = (
                    {"episode_id": step.source_episode_id, "boundary": step.source_boundary}
                    if step.source_episode_id is not None else None
                )
                if origin is not None and step.source_part is not None:
                    origin["part"] = step.source_part
                self.store.record_action(self.episode_id, ordinal, outcome, replay_origin=origin)
                if outcome.replay_eog_token_id is not None:
                    self.store.record_interaction(
                        self.episode_id, self.engine.boundary, "replay-eog",
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
            if had_tape and not handed_off and not self.engine.ended and not self.engine.checkpointed:
                replay_exhausted = replayed == len(tape)
            if replay_exhausted and plan.follow_source_sampling and plan.final_sampling is not None:
                if self.engine.sampling != plan.final_sampling:
                    self.engine.sampling = plan.final_sampling
                    self.store.record_sampling_segment(
                        self.episode_id,
                        start_boundary=self.engine.boundary,
                        sampling=plan.final_sampling,
                        stream_fingerprint=self.engine.stream_fingerprint,
                        coordinate_offset=self.engine.coordinate_offset,
                    )
            # Explicit plans always return through the edge. Legacy callers may
            # request live continuation after a successfully exhausted tape.
            should_run_live = not had_tape or (
                replay_exhausted and not stop_after_tape and not isinstance(tape_input, ReplayPlan)
            )
            if outcomes and outcomes[-1].stop_reason == "replay-eog":
                should_run_live = False
            while (
                should_run_live
                and not self.engine.ended
                and not self.engine.checkpointed
                and live_policy is not None
            ):
                observation = self.engine.observe()
                action = live_policy.choose(self.engine, observation)
                active_action = action
                executing_replay = False
                outcome = self.engine.apply(action)
                self.store.record_action(self.episode_id, ordinal, outcome)
                outcomes.append(outcome)
                ordinal += 1
        except InstructionRejected as exc:
            handoff_reason = str(exc)
            self.store.record_interaction(
                self.episode_id, self.engine.boundary, "instruction-rejected",
                {
                    "action": active_action.to_dict() if active_action else None,
                    "reason": str(exc),
                    "replay": executing_replay,
                },
            )
            # Rejected moves contribute no action or token rows to future tapes.
            handed_off = True
        except (EdgeRequested, ForkRequested, SeamlessRewindRequested, SeamlessEdgeRequested):
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
