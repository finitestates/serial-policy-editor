"""Optional teacher fitting of explicitly learnable manual group strengths.

Appearance objectives live in group_control and never consume teacher labels.
This compatibility fitter uses sparse group features on a frozen policy surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .domain import EditorError, SamplingConfig
from .sampling import ObservationStatistics
from .learning_controls import decay_applies, validate_controls, write_scale


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise EditorError(f"{name} must be a finite number")
    value = float(value)
    if nonnegative and value < 0.0:
        raise EditorError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class OnlineLearningConfig:
    """Conservative controls for the optional live learner."""

    enabled: bool = False
    learning_rate: float = 0.05
    epsilon: float = 0.05
    max_step: float = 0.25
    min_bias: float = -4.0
    max_bias: float = 4.0
    learnable_groups: tuple[str, ...] | None = None
    severity_cap: int = 1000
    dead_zone_rank: int = 1
    no_severity_attenuation: bool = False
    rejection_strength: float = 0.0
    decay: float = 0.0
    learning_gate: str = "rank"
    decay_on: str = "update"
    write_reduction: str = "sum"
    rejection_target: str = "proposal"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EditorError("online learning enabled must be a boolean")
        if self.learning_gate not in ("rank", "sampler"):
            raise EditorError("learning gate must be rank or sampler")
        validate_controls(self.decay_on, self.write_reduction, self.rejection_target)
        learning_rate = _finite_number(
            self.learning_rate, "learning_rate", nonnegative=True
        )
        epsilon = _finite_number(self.epsilon, "epsilon")
        max_step = _finite_number(self.max_step, "max_step", nonnegative=True)
        min_bias = _finite_number(self.min_bias, "min_bias")
        max_bias = _finite_number(self.max_bias, "max_bias")
        if epsilon <= 0.0:
            raise EditorError("epsilon must be positive")
        if min_bias > max_bias:
            raise EditorError("min_bias must not exceed max_bias")
        for name in ("severity_cap", "dead_zone_rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise EditorError(f"learning {name} must be a positive integer")
        if type(self.no_severity_attenuation) is not bool:
            raise EditorError("learning no_severity_attenuation must be a boolean")
        _finite_number(self.rejection_strength, "rejection_strength", nonnegative=True)
        decay = _finite_number(self.decay, "decay", nonnegative=True)
        if decay > 1:
            raise EditorError("learning decay must be between 0 and 1")
        if self.learnable_groups is None:
            groups = None
        else:
            if isinstance(self.learnable_groups, str) or not hasattr(
                self.learnable_groups, "__iter__"
            ):
                raise EditorError("learnable_groups must be a list of names")
            groups = []
            for name in self.learnable_groups:
                if not isinstance(name, str):
                    raise EditorError("learnable_groups must contain names")
                groups.extend(name_part.strip() for name_part in name.split(","))
            if any(not name for name in groups):
                raise EditorError("learnable_groups must contain nonempty names")
            groups = tuple(dict.fromkeys(groups))
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "max_step", max_step)
        object.__setattr__(self, "min_bias", min_bias)
        object.__setattr__(self, "max_bias", max_bias)
        object.__setattr__(self, "learnable_groups", groups)


@dataclass(frozen=True)
class LearningResult:
    """One attempted live correction and the resulting sampler settings."""

    sampling: SamplingConfig
    observation_boundary: int
    chosen_token_id: int
    old_policy_rank: int
    old_policy_probability: float
    severity: float
    loss: float
    old_group_weights: dict[str, float]
    new_group_weights: dict[str, float]
    group_deltas: dict[str, float]
    gradients: dict[str, float]
    update_norm: float
    enabled: bool
    severity_cap: int = 1000
    dead_zone_rank: int = 1
    no_severity_attenuation: bool = False
    rejection_strength: float = 0.0
    proposal_token_id: int | None = None
    proposal_rejected: bool = False
    decay: float = 0.0
    evidence: dict[str, float] | None = None
    skipped: dict[str, str] | None = None
    learning_gate: str = "rank"
    sampler_eligible: bool | None = None
    sampler_probability: float | None = None
    decay_on: str = "update"
    effective_decay: float = 0.0
    write_reduction: str = "sum"
    write_evidence_scale: float = 1.0
    write_evidence_tokens: int | None = None
    rejection_target: str = "proposal"

    @property
    def updated_sampling(self) -> SamplingConfig:
        """Readable alias for callers that prefer an explicit name."""
        return self.sampling

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.observation_boundary,
            "chosen_token_id": self.chosen_token_id,
            "old_policy_rank": self.old_policy_rank,
            "old_policy_probability": self.old_policy_probability,
            "severity": self.severity,
            "loss": self.loss,
            "old_group_weights": dict(self.old_group_weights),
            "new_group_weights": dict(self.new_group_weights),
            "group_deltas": dict(self.group_deltas),
            "gradients": dict(self.gradients),
            "update_norm": self.update_norm,
            "enabled": self.enabled,
            "severity_cap": self.severity_cap, "dead_zone_rank": self.dead_zone_rank,
            "no_severity_attenuation": self.no_severity_attenuation,
            "rejection_strength": self.rejection_strength,
            "proposal_token_id": self.proposal_token_id, "proposal_rejected": self.proposal_rejected,
            "decay": self.decay, "evidence": self.evidence or {}, "skipped": self.skipped or {},
            "learning_gate": self.learning_gate, "sampler_eligible": self.sampler_eligible,
            "sampler_probability": self.sampler_probability,
            "decay_on": self.decay_on, "effective_decay": self.effective_decay,
            "write_reduction": self.write_reduction, "write_evidence_scale": self.write_evidence_scale,
            "write_evidence_tokens": self.write_evidence_tokens, "rejection_target": self.rejection_target,
        }


class OnlineLearner:
    """Teacher learner for fixed named manual bias groups.

    This object is deliberately stateless.  Learned state lives only in the
    returned ``SamplingConfig`` and therefore remains replayable through the
    existing sampler-segment records.
    """

    def __init__(
        self,
        config: OnlineLearningConfig | None = None,
        **settings: Any,
    ) -> None:
        if config is not None and settings:
            raise TypeError("pass either config or online-learning settings")
        self.config = config or OnlineLearningConfig(**settings)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _severity(self, policy_rank: int) -> float:
        if policy_rank <= self.config.dead_zone_rank:
            return 0.0
        if self.config.no_severity_attenuation:
            return 1.0
        return min(1.0, math.log1p(policy_rank - self.config.dead_zone_rank) / math.log1p(self.config.severity_cap))

    @staticmethod
    def _loss(statistics: ObservationStatistics, token_id: int) -> float:
        probability = float(statistics.policy_probabilities[token_id])
        # Keep diagnostic payloads JSON-safe even for an extreme logit spread
        # that underflows a probability to zero.
        return -math.log(max(probability, float.fromhex("0x1.0p-1022")))

    @staticmethod
    def _with_group_bias(
        sampling: SamplingConfig, group_name: str, bias: float
    ) -> SamplingConfig:
        groups = tuple(
            replace(group, bias=bias) if group.name == group_name else group
            for group in sampling.bias_groups
        )
        return replace(sampling, bias_groups=groups)

    def _counterfactual(
        self,
        observation,
        sampling: SamplingConfig,
    ) -> ObservationStatistics:
        return ObservationStatistics(
            observation.logits,
            sampling,
            observation.prefix_token_ids,
            observation.statistics.boundaries,
            token_preference_features=getattr(observation.statistics, "token_preference_features", None),
            render_tokens=getattr(observation.statistics, "render_tokens", None),
        )

    def update(
        self,
        observation,
        chosen_token_id: int,
        sampling: SamplingConfig,
    ) -> LearningResult:
        """Apply one bounded correction using the pre-action observation."""
        if type(chosen_token_id) is not int or not 0 <= chosen_token_id < len(
            observation.logits
        ):
            raise EditorError("chosen token is outside the observation vocabulary")

        statistics = observation.statistics
        old_policy_rank = statistics.policy_rank(chosen_token_id)
        old_policy_probability = float(statistics.policy_probabilities[chosen_token_id])
        # Membership, rather than probability > 0, also handles numerical
        # underflow for a token that survived the actual decoder filters.
        sampler_eligible = bool(chosen_token_id in statistics.distribution.ids)
        sampler_probability = statistics.distribution.probability(chosen_token_id)
        severity = (float(not sampler_eligible) if self.config.learning_gate == "sampler"
                    else self._severity(old_policy_rank))
        loss = self._loss(statistics, chosen_token_id)
        rejected = observation.proposal_token_id != chosen_token_id
        effective_decay = self.config.decay if self.enabled and decay_applies(
            self.config.decay_on, rejected=rejected, severity=severity) else 0.
        groups = tuple(sampling.bias_groups)
        old_weights = {group.name: float(group.bias) for group in groups}
        selected_names = (
            set(old_weights)
            if self.config.learnable_groups is None
            else set(self.config.learnable_groups)
        )
        unknown = selected_names - set(old_weights)
        if unknown:
            raise EditorError(
                "learnable group not present in sampling configuration: "
                + ", ".join(sorted(unknown))
            )

        gradients = {group.name: 0.0 for group in groups}
        new_weights = dict(old_weights)
        evidence = {}
        skipped = {}
        if self.enabled:
            from .bias_rules import BiasMatcher
            for group in groups:
                if group.name not in selected_names or not group.enabled or not group.learnable:
                    skipped[group.name] = "disabled or frozen"
                    continue
                if any(c.group == group.name for c in sampling.group_controls):
                    skipped[group.name] = "controlled by appearance objective"
                    continue
                if not self.config.min_bias <= group.bias <= self.config.max_bias:
                    skipped[group.name] = "manual amount outside learning bounds"
                    continue
                # A group's deduplicated edge scales are sparse fixed features.
                # Standalone lexical priors do not depend on group magnitude.
                if not sampling.group_controls and not (sampling.reference_prior_active and sampling.reference_prior_scope == "active"):
                    scales = BiasMatcher(replace(group, bias=1.).effective_rules()).active_biases(
                        observation.prefix_token_ids, statistics.boundaries)
                    mean = sum(float(statistics.policy_probabilities[t]) * v for t, v in scales.items())
                    gradient = mean - scales.get(chosen_token_id, 0.)
                    if observation.proposal_token_id != chosen_token_id:
                        negative = scales.get(observation.proposal_token_id, 0.)
                        if self.config.rejection_target == "sampler" and self.config.rejection_strength:
                            negative = sum(float(p) * scales.get(int(t), 0.) for t, p in zip(
                                statistics.distribution.ids, statistics.distribution.probabilities))
                        gradient += self.config.rejection_strength * (negative - mean)
                else:
                    # Compatibility with nonlinear legacy policies. The normal
                    # lexical/group-objective workflow never needs this path.
                    plus = self._counterfactual(observation, self._with_group_bias(sampling, group.name, group.bias + self.config.epsilon))
                    minus = self._counterfactual(observation, self._with_group_bias(sampling, group.name, group.bias - self.config.epsilon))
                    gradient = (self._loss(plus, chosen_token_id) - self._loss(minus, chosen_token_id)) / (2 * self.config.epsilon)
                    if observation.proposal_token_id != chosen_token_id and self.config.rejection_strength:
                        proposal = observation.proposal_token_id
                        if self.config.rejection_target == "sampler":
                            # Freeze the original target distribution across the
                            # two counterfactual policies, just as for a proposal.
                            difference = sum(float(p) * (self._loss(plus, int(t)) - self._loss(minus, int(t)))
                                             for t, p in zip(statistics.distribution.ids,
                                                             statistics.distribution.probabilities))
                        else:
                            difference = self._loss(plus, proposal) - self._loss(minus, proposal)
                        gradient -= self.config.rejection_strength * difference / (2 * self.config.epsilon)
                gradients[group.name] = float(gradient) if math.isfinite(float(gradient)) else 0.
                evidence[group.name] = -self.config.learning_rate * severity * gradients[group.name]
                new_weights[group.name] = self._apply_evidence(group.bias, evidence[group.name], effective_decay)

        deltas = {
            name: float(new_weights[name] - old_weights[name])
            for name in old_weights
        }
        update_norm = math.sqrt(sum(delta * delta for delta in deltas.values()))
        updated_groups = tuple(
            replace(group, bias=new_weights[group.name]) for group in groups
        )
        return LearningResult(
            sampling=replace(sampling, bias_groups=updated_groups),
            observation_boundary=observation.boundary,
            chosen_token_id=chosen_token_id,
            old_policy_rank=old_policy_rank,
            old_policy_probability=old_policy_probability,
            severity=severity,
            loss=loss,
            old_group_weights=old_weights,
            new_group_weights=new_weights,
            group_deltas=deltas,
            gradients=gradients,
            update_norm=update_norm,
            enabled=self.enabled,
            severity_cap=self.config.severity_cap, dead_zone_rank=self.config.dead_zone_rank,
            no_severity_attenuation=self.config.no_severity_attenuation,
            rejection_strength=self.config.rejection_strength,
            proposal_token_id=observation.proposal_token_id,
            proposal_rejected=observation.proposal_token_id != chosen_token_id,
            decay=self.config.decay, evidence=evidence, skipped=skipped,
            learning_gate=self.config.learning_gate, sampler_eligible=sampler_eligible,
            sampler_probability=sampler_probability,
            decay_on=self.config.decay_on, effective_decay=effective_decay,
            write_reduction=self.config.write_reduction, rejection_target=self.config.rejection_target,
        )

    def _apply_evidence(self, old, evidence, decay=None):
        decay = self.config.decay if decay is None else decay
        step = max(-self.config.max_step, min(self.config.max_step, evidence))
        return max(self.config.min_bias, min(self.config.max_bias, (1 - decay) * old + step))

    def aggregate(self, results, sampling):
        """Reduce a typed span's evidence; clip and conditionally decay once."""
        first = results[0]
        scale, evidence_tokens = write_scale(self.config.write_reduction, results)
        effective_decay = self.config.decay if self.enabled and any(decay_applies(
            self.config.decay_on, rejected=r.proposal_rejected, severity=r.severity) for r in results) else 0.
        names = {name for r in results for name in (r.evidence or {})}
        evidence = {name: scale * sum((r.evidence or {}).get(name, 0.) for r in results) for name in names}
        weights = dict(first.old_group_weights)
        for name in names:
            weights[name] = self._apply_evidence(weights[name], evidence[name], effective_decay)
        deltas = {name: weights[name] - old for name, old in first.old_group_weights.items()}
        return replace(first, sampling=replace(sampling, bias_groups=tuple(replace(g, bias=weights[g.name]) for g in sampling.bias_groups)),
                       new_group_weights=weights, group_deltas=deltas, evidence=evidence,
                       gradients={name: sum(r.gradients[name] for r in results) / len(results) for name in weights},
                       update_norm=math.sqrt(sum(d*d for d in deltas.values())),
                       sampler_eligible=None, sampler_probability=None,
                       proposal_rejected=any(r.proposal_rejected for r in results),
                       effective_decay=effective_decay, write_evidence_scale=scale,
                       write_evidence_tokens=evidence_tokens)
