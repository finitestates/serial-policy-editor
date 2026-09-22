"""Research learning adapter for the core episode runner.

The durable replay/live loop lives in :mod:`trajectory_editor.episode_runner`.
This compatibility module adds the historical learner and instrumentation
hooks without making them part of the core runner's dependency graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .core.actions import Accept, Phrase, PolicyAction, SelectRawRank, Write
from .core.results import ActionOutcome
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine, Observation
from .episode_runner import (
    EdgeRequested,
    EpisodeRunner as CoreEpisodeRunner,
    ForkRequested,
    LivePolicy,
    ReplayPlan,
    ReplayContext,
    ReplayOrigin,
    RunResult,
    SeamlessRewindRequested,
    TapeStep,
)


def _compile_learning_observation(observation: Observation) -> Any:
    """Load the optional research observation compiler only when learning runs."""

    from .learning_observation import CompiledLearningObservation

    return CompiledLearningObservation.from_observation(observation)


@dataclass(frozen=True)
class WriteTokenLearning:
    """Small diagnostic for one token in a live teacher-written sequence."""

    observation_boundary: int
    token_id: int
    policy_rank: int
    policy_probability: float
    severity: float
    loss: float
    sampler_eligible: bool | None = None
    sampler_probability: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_boundary": self.observation_boundary,
            "chosen_token_id": self.token_id,
            "old_policy_rank": self.policy_rank,
            "old_policy_probability": self.policy_probability,
            "severity": self.severity,
            "loss": self.loss,
            "sampler_eligible": self.sampler_eligible,
            "sampler_probability": self.sampler_probability,
        }


@dataclass(frozen=True)
class WriteLearningResult:
    """Per-token research learning events for one atomic live write."""

    sampling: SamplerConfig
    boundary_before: int
    boundary_after: int
    tokens: tuple[WriteTokenLearning, ...]
    group_result: Any | None
    token_preference_result: Any | None
    token_preference_token_results: tuple[Any, ...] = ()
    group_token_results: tuple[Any, ...] = ()

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
            payload["group_token_updates"] = [
                result.to_dict() for result in self.group_token_results
            ]
        if self.token_preference_result is not None:
            payload["token_preference_update"] = self.token_preference_result.to_dict()
            payload["token_preference_token_updates"] = [
                result.to_dict() for result in self.token_preference_token_results
            ]
            payload["token_preference_token_observations"] = [
                {
                    key: getattr(result, key)
                    for key in (
                        "observation_boundary",
                        "chosen_token_id",
                        "proposal_token_id",
                        "proposal_rejected",
                        "severity",
                        "rejection_strength",
                        "learning_gate",
                        "sampler_eligible",
                        "sampler_probability",
                        "decay_on",
                        "effective_decay",
                        "effective_fast_decay",
                        "rejection_target",
                    )
                }
                for result in self.token_preference_token_results
            ]
        return payload


class _WriteLearningAccumulator:
    """Apply bounded research updates at each committed write token."""

    def __init__(
        self,
        backend,
        sampling: SamplerConfig,
        learner: Any | None,
        token_preference_learner: Any | None,
    ) -> None:
        self.backend = backend
        self.sampling = sampling
        self.learner = learner if learner is not None and learner.enabled else None
        self.token_preference_learner = (
            token_preference_learner
            if token_preference_learner is not None and token_preference_learner.enabled
            else None
        )
        self.tokens: list[WriteTokenLearning] = []
        self.group_results: list[Any] = []
        self.token_preference_results: list[Any] = []

    @property
    def enabled(self) -> bool:
        return self.learner is not None or self.token_preference_learner is not None

    def add(self, observation: Observation, token_id: int) -> SamplerConfig | None:
        if self.backend.is_eog(token_id):
            return None
        old_sampling = self.sampling
        compiled = _compile_learning_observation(observation)
        group_result = (
            self.learner.update(observation, token_id, old_sampling, compiled=compiled)
            if self.learner is not None
            else None
        )
        token_preference_result = (
            self.token_preference_learner.update(
                observation, token_id, old_sampling, compiled=compiled
            )
            if self.token_preference_learner is not None
            else None
        )
        source = group_result or token_preference_result
        if source is None:
            return self.sampling
        updated = group_result.sampling if group_result is not None else old_sampling
        if token_preference_result is not None:
            updated = replace(
                updated,
                token_preference_vector=token_preference_result.sampling.token_preference_vector,
                token_preference_strength=token_preference_result.sampling.token_preference_strength,
                token_preference_fast_vector=token_preference_result.sampling.token_preference_fast_vector,
                token_preference_fast_strength=token_preference_result.sampling.token_preference_fast_strength,
                token_preference_projection_seed=token_preference_result.sampling.token_preference_projection_seed,
                token_preference_learning_scheme=token_preference_result.sampling.token_preference_learning_scheme,
                token_preference_coordinate_identity=token_preference_result.sampling.token_preference_coordinate_identity,
            )
        self.sampling = updated
        self.tokens.append(
            WriteTokenLearning(
                observation_boundary=observation.boundary,
                token_id=token_id,
                policy_rank=source.old_policy_rank,
                policy_probability=source.old_policy_probability,
                severity=source.severity,
                loss=source.loss,
                sampler_eligible=source.sampler_eligible,
                sampler_probability=source.sampler_probability,
            )
        )
        if group_result is not None:
            self.group_results.append(group_result)
        if token_preference_result is not None:
            self.token_preference_results.append(token_preference_result)
        return self.sampling

    def finish(self, boundary_after: int) -> WriteLearningResult | None:
        if not self.tokens:
            return None
        return WriteLearningResult(
            sampling=self.sampling,
            boundary_before=self.tokens[0].observation_boundary,
            boundary_after=boundary_after,
            tokens=tuple(self.tokens),
            group_result=self.group_results[-1] if self.group_results else None,
            token_preference_result=(
                self.token_preference_results[-1]
                if self.token_preference_results
                else None
            ),
            token_preference_token_results=tuple(self.token_preference_results),
            group_token_results=tuple(self.group_results),
        )


class EpisodeRunner(CoreEpisodeRunner):
    """Compatibility runner with optional research learning hooks."""

    def __init__(
        self,
        engine: EpisodeEngine,
        store,
        episode_id: str,
        *,
        divergence_policy: str = "handoff",
        learner: Any | None = None,
        on_learning_update: Callable[[Any], None] | None = None,
        token_preference_learner: Any | None = None,
        on_token_preference_learning_update: Callable[[Any], None] | None = None,
        learn_from_write: bool = True,
        on_write_learning_update: Callable[[WriteLearningResult], None] | None = None,
    ) -> None:
        super().__init__(
            engine,
            store,
            episode_id,
            divergence_policy=divergence_policy,
        )
        self.learner = learner
        self.on_learning_update = on_learning_update
        self.token_preference_learner = token_preference_learner
        self.on_token_preference_learning_update = on_token_preference_learning_update
        self.learn_from_write = bool(learn_from_write)
        self.on_write_learning_update = on_write_learning_update

    def _live_action_hooks(
        self, observation: Observation, action: PolicyAction
    ) -> tuple[Any, Callable[[Observation, int], None] | None]:
        accumulator = (
            _WriteLearningAccumulator(
                self.engine.backend,
                self.engine.sampling,
                self.learner,
                self.token_preference_learner,
            )
            if ((self.learn_from_write and isinstance(action, Write)) or isinstance(action, Phrase))
            else None
        )
        if accumulator is None or not accumulator.enabled:
            return accumulator, None
        return accumulator, lambda observed, token_id: self._learn_live_write_token(
            accumulator, observed, token_id
        )

    def _after_live_action(
        self,
        context: Any,
        observation: Observation,
        action: PolicyAction,
        outcome: ActionOutcome,
    ) -> None:
        if context is not None:
            self._learn_live_write(
                context,
                outcome,
                interaction_kind=(
                    "phrase-learning-update"
                    if isinstance(action, Phrase)
                    else "write-learning-update"
                ),
            )
        else:
            self._learn_live_selection(observation, action, outcome)

    def _learn_live_selection(
        self,
        observation: Observation,
        action: PolicyAction,
        outcome: ActionOutcome,
    ) -> Any | None:
        if (
            not isinstance(action, (Accept, SelectRawRank))
            or outcome.status != "completed"
            or len(outcome.evidence) != 1
            or outcome.evidence[0].is_eog
        ):
            return None
        evidence = outcome.evidence[0]
        old_sampling = self.engine.sampling
        compiled = _compile_learning_observation(observation)
        group_result = None
        if self.learner is not None and self.learner.enabled:
            group_result = self.learner.update(
                observation, evidence.token_id, old_sampling, compiled=compiled
            )
        token_preference_result = None
        if self.token_preference_learner is not None and self.token_preference_learner.enabled:
            token_preference_result = self.token_preference_learner.update(
                observation, evidence.token_id, old_sampling, compiled=compiled
            )
        updated_sampling = group_result.sampling if group_result is not None else old_sampling
        if token_preference_result is not None:
            updated_sampling = replace(
                updated_sampling,
                token_preference_vector=token_preference_result.sampling.token_preference_vector,
                token_preference_strength=token_preference_result.sampling.token_preference_strength,
                token_preference_fast_vector=token_preference_result.sampling.token_preference_fast_vector,
                token_preference_fast_strength=token_preference_result.sampling.token_preference_fast_strength,
                token_preference_projection_seed=token_preference_result.sampling.token_preference_projection_seed,
                token_preference_learning_scheme=token_preference_result.sampling.token_preference_learning_scheme,
                token_preference_coordinate_identity=token_preference_result.sampling.token_preference_coordinate_identity,
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
        if token_preference_result is not None:
            payload = token_preference_result.to_dict()
            payload["boundary"] = self.engine.boundary
            payload["observation_boundary"] = token_preference_result.observation_boundary
            self.store.record_interaction(
                self.episode_id,
                self.engine.boundary,
                "token-preference-update",
                payload,
            )
            if self.on_token_preference_learning_update is not None:
                self.on_token_preference_learning_update(token_preference_result)
        return token_preference_result or group_result

    def _learn_live_write(
        self,
        accumulator: _WriteLearningAccumulator,
        outcome: ActionOutcome,
        *,
        interaction_kind: str = "write-learning-update",
    ) -> WriteLearningResult | None:
        if outcome.status != "completed":
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
            interaction_kind,
            result.to_dict(),
        )
        if self.on_write_learning_update is not None:
            self.on_write_learning_update(result)
        return result

    def _learn_live_write_token(
        self,
        accumulator: _WriteLearningAccumulator,
        observation: Observation,
        token_id: int,
    ) -> None:
        old_sampling = self.engine.sampling
        accumulator.add(observation, token_id)
        if accumulator.sampling == old_sampling:
            return
        self.engine.sampling = accumulator.sampling
        self.store.record_sampling_segment(
            self.episode_id,
            start_boundary=self.engine.boundary,
            sampling=self.engine.sampling,
            stream_fingerprint=self.engine.stream_fingerprint,
            coordinate_offset=self.engine.coordinate_offset,
        )


__all__ = [
    "EdgeRequested",
    "EpisodeRunner",
    "ForkRequested",
    "LivePolicy",
    "ReplayContext",
    "ReplayOrigin",
    "ReplayPlan",
    "RunResult",
    "SeamlessRewindRequested",
    "TapeStep",
    "WriteLearningResult",
    "_WriteLearningAccumulator",
]
