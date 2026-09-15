"""Policy providers and the common episode runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
from typing import Protocol

from .domain import SamplingConfig
from .episode_actions import PolicyAction, SelectRawRank, Write
from .episode_engine import ActionOutcome, EpisodeEngine, Observation, ReplayExpectation, InstructionRejected
from .episode_store import EpisodeStore
from .latent_preference import LatentPreferenceLearner, LatentPreferenceResult
from .online_learning import LearningResult, OnlineLearner


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


@dataclass(frozen=True)
class WriteTokenLearning:
    """Small diagnostic for one token in a live teacher-written sequence."""

    observation_boundary: int
    token_id: int
    policy_rank: int
    policy_probability: float
    severity: float
    loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_boundary": self.observation_boundary,
            "chosen_token_id": self.token_id,
            "old_policy_rank": self.policy_rank,
            "old_policy_probability": self.policy_probability,
            "severity": self.severity,
            "loss": self.loss,
        }


@dataclass(frozen=True)
class WriteLearningResult:
    """One aggregate update produced by a live multi-token ``Write``."""

    sampling: SamplingConfig
    boundary_before: int
    boundary_after: int
    tokens: tuple[WriteTokenLearning, ...]
    group_result: LearningResult | None
    latent_result: LatentPreferenceResult | None
    latent_token_results: tuple[LatentPreferenceResult, ...] = ()

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "boundary": self.boundary_after,
            "boundary_before": self.boundary_before,
            "token_count": self.token_count,
            "tokens": [token.to_dict() for token in self.tokens],
        }
        if self.group_result is not None:
            payload["group_update"] = self.group_result.to_dict()
        if self.latent_result is not None:
            payload["latent_update"] = self.latent_result.to_dict()
            payload["latent_token_observations"] = [
                {key: getattr(result, key) for key in (
                    "observation_boundary", "chosen_token_id", "proposal_token_id",
                    "proposal_rejected", "severity", "rejection_strength")}
                for result in self.latent_token_results
            ]
        return payload


class _WriteLearningAccumulator:
    """Compute conservative per-token updates while a Write remains atomic."""

    def __init__(
        self,
        backend,
        sampling: SamplingConfig,
        learner: OnlineLearner | None,
        latent_learner: LatentPreferenceLearner | None,
    ) -> None:
        self.backend = backend
        self.sampling = sampling
        self.learner = learner if learner is not None and learner.enabled else None
        self.latent_learner = (
            latent_learner
            if latent_learner is not None and latent_learner.enabled
            else None
        )
        self.tokens: list[WriteTokenLearning] = []
        self.group_results: list[LearningResult] = []
        self.latent_results: list[LatentPreferenceResult] = []

    @property
    def enabled(self) -> bool:
        return self.learner is not None or self.latent_learner is not None

    def add(self, observation: Observation, token_id: int) -> None:
        # Terminal selection is not a preference-bearing token in either v0
        # learner, matching the SelectRawRank path.
        if self.backend.is_eog(token_id):
            return
        group_result = (
            self.learner.update(observation, token_id, self.sampling)
            if self.learner is not None
            else None
        )
        latent_result = (
            self.latent_learner.update(observation, token_id, self.sampling)
            if self.latent_learner is not None
            else None
        )
        source = group_result or latent_result
        if source is None:
            return
        self.tokens.append(
            WriteTokenLearning(
                observation_boundary=observation.boundary,
                token_id=token_id,
                policy_rank=source.old_policy_rank,
                policy_probability=source.old_policy_probability,
                severity=source.severity,
                loss=source.loss,
            )
        )
        if group_result is not None:
            self.group_results.append(group_result)
        if latent_result is not None:
            self.latent_results.append(latent_result)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _mean_int(values: Sequence[int]) -> int:
        return max(1, int(round(sum(values) / len(values))))

    def _aggregate_groups(self) -> LearningResult | None:
        if not self.group_results:
            return None
        result = self.learner.aggregate(self.group_results, self.sampling)
        return replace(result,
                       old_policy_rank=self._mean_int([r.old_policy_rank for r in self.group_results]),
                       old_policy_probability=self._mean([r.old_policy_probability for r in self.group_results]),
                       severity=self._mean([r.severity for r in self.group_results]),
                       loss=self._mean([r.loss for r in self.group_results]))

    def _aggregate_latent(self) -> LatentPreferenceResult | None:
        if not self.latent_results:
            return None
        latent_learner = self.latent_learner
        assert latent_learner is not None
        aggregate = latent_learner.aggregate(self.latent_results, self.sampling)
        return replace(
            aggregate,
            old_policy_rank=self._mean_int([r.old_policy_rank for r in self.latent_results]),
            old_policy_probability=self._mean([r.old_policy_probability for r in self.latent_results]),
            severity=self._mean([r.severity for r in self.latent_results]),
            loss=self._mean([r.loss for r in self.latent_results]),
        )

    def finish(self, boundary_after: int) -> WriteLearningResult | None:
        if not self.tokens:
            return None
        group_result = self._aggregate_groups()
        latent_result = self._aggregate_latent()
        updated = self.sampling
        if group_result is not None:
            updated = group_result.sampling
        if latent_result is not None:
            updated = replace(
                updated,
                latent_preference_z=latent_result.sampling.latent_preference_z,
                latent_strength=latent_result.sampling.latent_strength,
                latent_preference_fast_z=latent_result.sampling.latent_preference_fast_z,
                latent_fast_strength=latent_result.sampling.latent_fast_strength,
                latent_projection_seed=latent_result.sampling.latent_projection_seed,
            )
        return WriteLearningResult(
            sampling=updated,
            boundary_before=self.tokens[0].observation_boundary,
            boundary_after=boundary_after,
            tokens=tuple(self.tokens),
            group_result=group_result,
            latent_result=latent_result,
            latent_token_results=tuple(self.latent_results),
        )


class EpisodeRunner:
    """Feed replay and live policies into the same action interpreter."""

    def __init__(
        self,
        engine: EpisodeEngine,
        store: EpisodeStore,
        episode_id: str,
        *,
        divergence_policy: str = "handoff",
        learner: OnlineLearner | None = None,
        on_learning_update: Callable[[LearningResult], None] | None = None,
        latent_learner: LatentPreferenceLearner | None = None,
        on_latent_learning_update: Callable[[LatentPreferenceResult], None] | None = None,
        learn_from_write: bool = False,
        on_write_learning_update: Callable[[WriteLearningResult], None] | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.episode_id = episode_id
        self.divergence_policy = divergence_policy
        self.learner = learner
        self.on_learning_update = on_learning_update
        self.latent_learner = latent_learner
        self.on_latent_learning_update = on_latent_learning_update
        self.learn_from_write = bool(learn_from_write)
        self.on_write_learning_update = on_write_learning_update

    def _learn_live_selection(
        self,
        observation: Observation,
        action: PolicyAction,
        outcome: ActionOutcome,
    ) -> LearningResult | LatentPreferenceResult | None:
        """Learn after a committed live raw-rank selection only."""
        if (
            not isinstance(action, SelectRawRank)
            or outcome.status != "completed"
            or len(outcome.evidence) != 1
            or outcome.evidence[0].is_eog
        ):
            return None
        evidence = outcome.evidence[0]
        old_sampling = self.engine.sampling
        group_result = None
        if self.learner is not None and self.learner.enabled:
            group_result = self.learner.update(
                observation, evidence.token_id, old_sampling
            )
        latent_result = None
        if self.latent_learner is not None and self.latent_learner.enabled:
            latent_result = self.latent_learner.update(
                observation, evidence.token_id, old_sampling
            )

        updated_sampling = old_sampling
        if group_result is not None:
            updated_sampling = group_result.sampling
        if latent_result is not None:
            updated_sampling = replace(
                updated_sampling,
                latent_preference_z=latent_result.sampling.latent_preference_z,
                latent_strength=latent_result.sampling.latent_strength,
                latent_preference_fast_z=latent_result.sampling.latent_preference_fast_z,
                latent_fast_strength=latent_result.sampling.latent_fast_strength,
                latent_projection_seed=latent_result.sampling.latent_projection_seed,
            )
        if updated_sampling != old_sampling:
            self.engine.sampling = updated_sampling
            self.store.record_sampling_segment(
                self.episode_id,
                start_boundary=self.engine.boundary,
                sampling=updated_sampling,
                stream_fingerprint=self.engine.stream_fingerprint,
                coordinate_offset=self.engine.coordinate_offset,
            )
        if group_result is not None:
            payload = group_result.to_dict()
            payload["boundary"] = self.engine.boundary
            payload["observation_boundary"] = group_result.observation_boundary
            self.store.record_interaction(
                self.episode_id,
                self.engine.boundary,
                "online-learning-update",
                payload,
            )
            if self.on_learning_update is not None:
                self.on_learning_update(group_result)
        if latent_result is not None:
            payload = latent_result.to_dict()
            payload["boundary"] = self.engine.boundary
            payload["observation_boundary"] = latent_result.observation_boundary
            self.store.record_interaction(
                self.episode_id,
                self.engine.boundary,
                "latent-preference-update",
                payload,
            )
            if self.on_latent_learning_update is not None:
                self.on_latent_learning_update(latent_result)
        return latent_result or group_result

    def _learn_live_write(
        self,
        accumulator: _WriteLearningAccumulator,
        outcome: ActionOutcome,
    ) -> WriteLearningResult | None:
        """Apply one aggregate update after an atomic live Write."""
        if outcome.status != "completed" or any(
            evidence.is_eog for evidence in outcome.evidence
        ):
            return None
        result = accumulator.finish(outcome.boundary_after)
        if result is None:
            return None
        if result.sampling != self.engine.sampling:
            self.engine.sampling = result.sampling
            self.store.record_sampling_segment(
                self.episode_id,
                start_boundary=outcome.boundary_after,
                sampling=result.sampling,
                stream_fingerprint=self.engine.stream_fingerprint,
                coordinate_offset=self.engine.coordinate_offset,
            )
        self.store.record_interaction(
            self.episode_id,
            outcome.boundary_after,
            "write-learning-update",
            result.to_dict(),
        )
        if self.on_write_learning_update is not None:
            self.on_write_learning_update(result)
        return result

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
                        sampling=self.engine.sampling,
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
                # A policy UI can edit steering before returning an action.
                # Learning must use exactly the surface that will be committed.
                observation = self.engine.observe()
                active_action = action
                executing_replay = False
                write_accumulator = (
                    _WriteLearningAccumulator(
                        self.engine.backend,
                        self.engine.sampling,
                        self.learner,
                        self.latent_learner,
                    )
                    if self.learn_from_write and isinstance(action, Write)
                    else None
                )
                outcome = self.engine.apply(
                    action,
                    on_precommit_observation=(
                        write_accumulator.add
                        if write_accumulator is not None and write_accumulator.enabled
                        else None
                    ),
                )
                live_actions += 1
                if write_accumulator is not None:
                    self._learn_live_write(write_accumulator, outcome)
                else:
                    self._learn_live_selection(observation, action, outcome)
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
