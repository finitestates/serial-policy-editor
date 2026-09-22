"""Adapters that run the shared replay/live loop for each episode medium."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .core.actions import Phrase, PolicyAction
from .core.results import ActionOutcome, ReplayExpectation
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine, Observation
from .run_loop import (
    EdgeRequested,
    ForkRequested,
    LivePolicy,
    ReplayContext,
    ReplayOrigin,
    ReplayPlan,
    RunResult,
    SeamlessRewindRequested,
    TapeStep,
    run_plan,
)

if TYPE_CHECKING:
    from .episode_store import EpisodeStore


class _RunnerTarget:
    """Common adapter shell; recording hooks are optional for live sessions."""

    def run(
        self,
        *,
        tape: Sequence[TapeStep] | ReplayPlan | None = None,
        live_policy: LivePolicy | None = None,
        max_live_actions: int | None = None,
    ) -> RunResult:
        return run_plan(
            self,
            divergence_policy=self.divergence_policy,
            tape=tape,
            live_policy=live_policy,
            max_live_actions=max_live_actions,
        )

    def record_replay(
        self,
        ordinal: int,
        index: int,
        step: TapeStep,
        outcome: ActionOutcome,
        origin: ReplayOrigin | None,
    ) -> None:
        del ordinal, index, step, outcome, origin

    def record_live(self, ordinal: int, outcome: ActionOutcome) -> None:
        del ordinal, outcome

    def record_phrase_rejected(self, action: Phrase, reason: str) -> None:
        del action, reason

    def record_instruction_rejected(
        self, action: PolicyAction | None, reason: str, replay: bool
    ) -> None:
        del action, reason, replay

    def record_control_flow(self) -> None:
        return None

    def complete(self, had_tape: bool) -> None:
        del had_tape


class EpisodeRunner(_RunnerTarget):
    """Run the shared loop while recording durable results in SQLite."""

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
        self._ordinal = 0

    @property
    def identifier(self) -> str:
        return self.episode_id

    def begin(self) -> int:
        self.store.record_budget(
            self.episode_id,
            self.engine.boundary,
            self.engine.max_tokens,
            self.engine.checkpoint_boundary,
        )
        self._ordinal = self.store.next_action_ordinal(self.episode_id)
        return self._ordinal

    def set_sampler(self, sampling: SamplerConfig) -> None:
        self.engine.sampling = sampling
        self.store.record_sampling_segment(
            self.episode_id,
            start_boundary=self.engine.boundary,
            sampling=self.engine.sampling,
            stream_fingerprint=self.engine.stream_fingerprint,
            coordinate_offset=self.engine.coordinate_offset,
        )

    def observe(self) -> Observation:
        return self.engine.observe()

    def apply(
        self,
        action: PolicyAction,
        *,
        expectation: ReplayExpectation | None = None,
        divergence_policy: str,
        replay: bool,
    ) -> ActionOutcome:
        return self.engine.apply(
            action,
            expectation=expectation,
            divergence_policy=divergence_policy,
            replay=replay,
        )

    def record_replay(
        self,
        ordinal: int,
        index: int,
        step: TapeStep,
        outcome: ActionOutcome,
        origin: ReplayOrigin | None,
    ) -> None:
        del index
        payload = None
        if origin is not None and origin.source_episode_id is not None:
            payload = {
                "episode_id": origin.source_episode_id,
                "boundary": origin.source_boundary,
            }
            if origin.source_part is not None:
                payload["part"] = origin.source_part
        self.store.record_action(
            self.episode_id, ordinal, outcome, replay_origin=payload
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
        self._ordinal = ordinal + 1

    def record_live(self, ordinal: int, outcome: ActionOutcome) -> None:
        self.store.record_action(self.episode_id, ordinal, outcome)
        self._ordinal = ordinal + 1

    def record_phrase_rejected(self, action: Phrase, reason: str) -> None:
        self.store.record_interaction(
            self.episode_id,
            self.engine.boundary,
            "phrase-rejected",
            {"action": action.to_dict(), "reason": reason},
        )

    def record_instruction_rejected(
        self, action: PolicyAction | None, reason: str, replay: bool
    ) -> None:
        self.store.record_interaction(
            self.episode_id,
            self.engine.boundary,
            "instruction-rejected",
            {
                "action": action.to_dict() if action else None,
                "reason": reason,
                "replay": replay,
            },
        )

    def record_control_flow(self) -> None:
        self.store.update_episode(
            self.episode_id,
            visible_text=self.engine.backend.render(self.engine.visible_token_ids),
            max_tokens=self.engine.max_tokens,
            status="open",
        )

    def complete(self, had_tape: bool) -> None:
        visible_text = self.engine.backend.render(self.engine.visible_token_ids)
        if self.engine.ended:
            self.store.finish_episode(
                self.episode_id,
                visible_text=visible_text,
                terminal_token_id=self.engine.terminal_token_id,
                terminal_reason=self.engine.terminal_reason,
            )
            return
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


class LiveSessionRunner(_RunnerTarget):
    """Run the shared loop against a persistence-free live session."""

    def __init__(self, session, *, divergence_policy: str = "handoff") -> None:
        self.session = session
        self.engine = session.engine
        self.divergence_policy = divergence_policy

    @property
    def identifier(self) -> str:
        return self.session.branch.branch_id

    def begin(self) -> int:
        return len(self.session.history_tape)

    def set_sampler(self, sampling: SamplerConfig) -> None:
        self.session.set_sampler(sampling)

    def observe(self) -> Observation:
        return self.engine.observe()

    def apply(
        self,
        action: PolicyAction,
        *,
        expectation: ReplayExpectation | None = None,
        divergence_policy: str,
        replay: bool,
    ) -> ActionOutcome:
        return self.session.generate(
            action,
            expectation=expectation,
            divergence_policy=divergence_policy,
            replay=replay,
        )


__all__ = [
    "EdgeRequested",
    "EpisodeRunner",
    "ForkRequested",
    "LivePolicy",
    "LiveSessionRunner",
    "ReplayContext",
    "ReplayOrigin",
    "ReplayPlan",
    "RunResult",
    "SeamlessRewindRequested",
    "TapeStep",
]
