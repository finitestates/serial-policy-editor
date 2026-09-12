"""Single interpreter for interactive and replayed policy actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from .boundaries import token_boundaries
from .domain import Candidate, EditorError, SamplingConfig
from .episode_actions import (
    Accept,
    EndGeneration,
    Finish,
    Hold,
    PolicyAction,
    SelectRawRank,
    Write,
)
from .episode_backend import EpisodeBackend, require_episode_backend
from .episode_hash import token_prefix_sha256, validate_fingerprint
from .sampling import (
    SparseDistribution,
    ObservationStatistics,
    draw_token,
)


class InstructionRejected(EditorError):
    """A recognized move cannot execute in the current target state."""


class TokenBudgetExceeded(InstructionRejected):
    """An atomic action cannot fit within the current allowance."""


@dataclass(frozen=True)
class ReplayExpectation:
    """Recorded result used to test whether an action still means the same thing."""

    token_ids: tuple[int, ...]
    terminal_token_id: int | None = None
    stop_reason: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ReplayExpectation:
        values = raw.get("token_ids") or ()
        if not isinstance(values, (list, tuple)) or any(
            type(v) is not int for v in values
        ):
            raise EditorError("replay expectation token ids are malformed")
        terminal = raw.get("terminal_token_id")
        if terminal is not None and type(terminal) is not int:
            raise EditorError("replay expectation terminal token is malformed")
        stop = raw.get("stop_reason")
        if stop is not None and not isinstance(stop, str):
            raise EditorError("replay expectation stop reason is malformed")
        return cls(tuple(int(v) for v in values), terminal, stop)

    @property
    def resolved_token_ids(self) -> tuple[int, ...]:
        if self.terminal_token_id is None:
            return self.token_ids
        return (*self.token_ids, self.terminal_token_id)


@dataclass(frozen=True)
class Divergence:
    boundary: int
    action_kind: str
    reason: str
    expected_token_id: int | None
    actual_token_id: int | None
    expected_stop_reason: str | None = None
    actual_stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "action_kind": self.action_kind,
            "reason": self.reason,
            "expected_token_id": self.expected_token_id,
            "actual_token_id": self.actual_token_id,
            "expected_stop_reason": self.expected_stop_reason,
            "actual_stop_reason": self.actual_stop_reason,
        }


@dataclass(frozen=True)
class Observation:
    boundary: int
    sampling_coordinate: int
    prefix_token_ids: tuple[int, ...]
    _render_context: Callable[..., str] = field(repr=False, compare=False)
    logits: np.ndarray = field(repr=False, compare=False)
    distribution: SparseDistribution = field(repr=False, compare=False)
    proposal_token_id: int
    proposal_text: str
    proposal_raw_rank: int
    proposal_raw_probability: float
    proposal_decoder_probability: float
    proposal_policy_rank: int
    statistics: ObservationStatistics = field(repr=False, compare=False)

    @cached_property
    def context_text(self) -> str:
        """Render this captured boundary once, only when display needs it."""
        return self._render_context(list(self.prefix_token_ids), special=True)


@dataclass(frozen=True)
class TokenEvidence:
    boundary: int
    sampling_coordinate: int
    token_id: int
    text: str
    proposal_token_id: int
    raw_model_nll: float
    raw_rank: int
    policy_rank: int
    decoder_probability: float
    proposal_agreement: bool
    is_eog: bool
    realized_visible: bool


@dataclass(frozen=True)
class ActionOutcome:
    action: PolicyAction
    boundary_before: int
    boundary_after: int
    resolved_text: str
    resolved_token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...]
    terminal_token_id: int | None
    stop_reason: str
    evidence: tuple[TokenEvidence, ...]
    status: str = "completed"
    divergence: Divergence | None = None
    replay_eog_token_id: int | None = None

    def expectation(self) -> ReplayExpectation:
        return ReplayExpectation(
            self.visible_token_ids,
            self.terminal_token_id,
            self.stop_reason,
        )


class EpisodeEngine:
    """Own token state, sampler coordinates, and all action resolution.

    The backend owns its private evaluation state. The engine's complete
    semantic state is the token ledger plus the sampler configuration and
    coordinate.
    """

    def __init__(
        self,
        backend: EpisodeBackend,
        *,
        sampling: SamplingConfig,
        max_tokens: int | None = None,
        initial_text: str | None = None,
        initial_token_ids: Sequence[int] | None = None,
        add_bos: bool = True,
        special: bool = True,
        stream_fingerprint: str | None = None,
        coordinate_offset: int = 0,
        backend_positioned: bool = False,
    ) -> None:
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise EditorError("max_tokens must be a positive integer")
        if type(coordinate_offset) is not int or coordinate_offset < 0:
            raise EditorError("coordinate_offset must be a nonnegative integer")
        require_episode_backend(backend)
        if initial_token_ids is None:
            if not isinstance(initial_text, str) or not initial_text:
                raise EditorError("an initial write or token ledger is required")
            tokens = backend.tokenize(initial_text, add_bos=add_bos, special=special)
        else:
            tokens = list(initial_token_ids)
            token_prefix_sha256(tokens)  # Reject malformed IDs instead of coercing them.
        if not tokens:
            raise EditorError("the initial write produced no tokens")
        if any(value < 0 or value >= backend.vocabulary_size() for value in tokens):
            raise EditorError("the initial token ledger contains an invalid token id")
        if backend.is_eog(tokens[-1]):
            raise EditorError("the initial write ends with an EOG token")
        fingerprint = token_prefix_sha256(tokens) if stream_fingerprint is None else validate_fingerprint(stream_fingerprint)
        if not backend_positioned:
            backend.reset(list(tokens))
        self.backend = backend
        self.sampling = sampling
        self.max_tokens = max_tokens
        self.checkpoint_boundary = max_tokens
        self.initial_text = (
            initial_text
            if isinstance(initial_text, str)
            else backend.render(list(tokens), special=True)
        )
        self.initial_token_ids = tuple(tokens)
        self.visible_token_ids: list[int] = []
        self.terminal_token_id: int | None = None
        self.terminal_reason: str | None = None
        self.coordinate_offset = coordinate_offset
        self.stream_fingerprint = fingerprint
        self._observation: Observation | None = None
        self._observation_key: tuple | None = None
        # This engine owns one backend/tokenizer. Sampler changes and rewinds
        # do not change token spellings, so their classifications remain valid.
        self._token_boundaries: dict[int, frozenset[str]] = {}

    @property
    def sampling(self) -> SamplingConfig:
        return self._sampling

    @sampling.setter
    def sampling(self, value: SamplingConfig) -> None:
        bias_tokens = [token for token, _ in value.logit_bias]
        bias_tokens.extend(token for tokens, _ in value.sequence_bias for token in tokens)
        for rule in value.scoped_bias:
            bias_tokens.extend(rule.target)
            bias_tokens.extend(token for trigger in rule.triggers for token in trigger)
        for rule in value.bias_rules:
            bias_tokens.extend(token for route in rule.routes for token in route)
            bias_tokens.extend(token for trigger in rule.triggers for token in trigger)
            if type(rule.until) is int:
                bias_tokens.append(rule.until)
        if any(token >= self.backend.vocabulary_size() for token in bias_tokens):
            raise EditorError("bias token id is outside the model vocabulary")
        self._sampling = value
        self._invalidate_observation()

    @property
    def boundary(self) -> int:
        return len(self.visible_token_ids)

    @property
    def remaining(self) -> int | None:
        """Visible tokens remaining until the next checkpoint."""
        return (None if self.checkpoint_boundary is None
                else max(0, self.checkpoint_boundary - self.boundary))

    @property
    def checkpointed(self) -> bool:
        """Whether the live episode has yielded at its current checkpoint."""
        return self.terminal_reason is None and self.remaining == 0

    @property
    def ended(self) -> bool:
        """True only after a genuine terminal event."""
        return self.terminal_reason is not None

    def resume(
        self,
        *,
        max_tokens: int | None | str = "keep",
        sampling: SamplingConfig | None = None,
    ) -> None:
        """Continue with the remaining allowance, renewing only when exhausted.

        An explicit integer starts a fresh allowance; None removes the budget.
        Sampling changes begin at the current token boundary.
        """
        if self.ended:
            raise EditorError("cannot resume a terminated episode")
        budget = self.max_tokens if max_tokens == "keep" else max_tokens
        if budget is not None and (type(budget) is not int or budget < 1):
            raise EditorError("max_tokens must be a positive integer")
        self.max_tokens = budget
        if max_tokens != "keep" or self.checkpointed:
            self.checkpoint_boundary = None if budget is None else self.boundary + budget
        if sampling is not None:
            self.sampling = sampling

    def rewind_to(self, boundary: int) -> None:
        """Discard visible state after a token boundary and reposition the backend.

        The checkpoint boundary is kept intact. The caller restores historical
        sampler settings and stream coordinates from the episode store; this
        method only repositions token/backend state and clears cached evidence.
        """
        if type(boundary) is not int or boundary < 0 or boundary > self.boundary:
            raise EditorError(
                f"rewind boundary must be between 0 and {self.boundary}"
            )
        retained = list(self.visible_token_ids[:boundary])
        self._invalidate_observation()
        prefix = [*self.initial_token_ids, *retained]
        branch = getattr(self.backend, "branch_to_prefix", None)
        if callable(branch):
            branch(prefix)
        else:
            self.backend.reset(prefix)
        self.visible_token_ids = retained
        self.terminal_token_id = None
        self.terminal_reason = None

    def terminate(self, reason: str = "menu-end") -> None:
        """Seal a live episode without manufacturing an EOG token."""
        if self.ended:
            return
        if not isinstance(reason, str) or not reason:
            raise EditorError("termination reason must be nonempty")
        self._invalidate_observation()
        self.terminal_reason = reason

    @property
    def token_ids(self) -> list[int]:
        return [*self.initial_token_ids, *self.visible_token_ids]

    @property
    def text(self) -> str:
        return self.backend.render(self.token_ids, special=True)

    def _invalidate_observation(self) -> None:
        self._observation = None
        self._observation_key = None

    def _decision_key(self) -> tuple:
        return (
            tuple(self.token_ids), self.sampling, self.coordinate_offset,
            self.stream_fingerprint,
        )

    def _validate_observation(self, observation: Observation) -> None:
        if (
            self.ended or self.checkpointed
            or observation is not self._observation
            or self._observation_key != self._decision_key()
        ):
            raise EditorError("request refers to a stale observation")

    def _classify_token_boundary(self, token_id):
        if token_id not in self._token_boundaries:
            self._token_boundaries[token_id] = token_boundaries(self.backend.token_text(token_id))
        return self._token_boundaries[token_id]

    def observe(self) -> Observation:
        if self.ended or self.checkpointed:
            raise EditorError("the episode has no live decision boundary")
        key = self._decision_key()
        if self._observation is not None and self._observation_key == key:
            return self._observation
        logits = np.asarray(self.backend.last_logits(), dtype=np.float64)
        if logits.ndim != 1 or len(logits) != self.backend.vocabulary_size():
            raise RuntimeError("backend logits do not match its vocabulary")
        statistics = ObservationStatistics(logits, self.sampling, key[0], self._classify_token_boundary)
        logits = statistics.logits
        distribution = statistics.distribution
        coordinate = self.coordinate_offset + self.boundary
        proposal = draw_token(
            distribution,
            seed=self.sampling.seed,
            stream_fingerprint=self.stream_fingerprint,
            aligned_step=coordinate,
        )
        observation = Observation(
            boundary=self.boundary,
            sampling_coordinate=coordinate,
            prefix_token_ids=tuple(self.token_ids),
            _render_context=self.backend.render,
            logits=logits,
            distribution=distribution,
            proposal_token_id=proposal,
            proposal_text=self.backend.token_text(proposal),
            proposal_raw_rank=statistics.raw_rank(proposal),
            proposal_raw_probability=float(statistics.raw_probabilities([proposal])[0]),
            proposal_decoder_probability=distribution.probability(proposal),
            proposal_policy_rank=statistics.policy_rank(proposal),
            statistics=statistics,
        )
        self._observation = observation
        self._observation_key = key
        return observation

    def candidates(
        self, observation: Observation, *, start_rank: int = 1, count: int = 12
    ) -> tuple[Candidate, ...]:
        if start_rank < 1 or count < 1:
            raise EditorError("candidate rank and count must be positive")
        self._validate_observation(observation)
        end = min(len(observation.logits), start_rank + count - 1)
        if start_rank > end:
            return ()
        statistics = observation.statistics
        ordered = statistics.top_raw_ids(end)[start_rank - 1 : end]
        probabilities = statistics.raw_probabilities(ordered)
        biases = observation.statistics.active_biases
        return tuple(
            Candidate(
                rank=rank,
                token_id=int(token_id),
                text=self.backend.token_text(int(token_id)),
                raw_probability=float(probability),
                decoder_probability=observation.distribution.probability(int(token_id)),
                is_eog=self.backend.is_eog(int(token_id)),
                logit_bias=biases.get(int(token_id), 0.0),
                policy_rank=statistics.policy_rank(int(token_id)),
                policy_probability=float(statistics.policy_probabilities[token_id]),
                policy_logit_adjustment=float(statistics.adjusted[token_id] - statistics.logits[token_id]),
            )
            for rank, (token_id, probability) in enumerate(
                zip(ordered, probabilities), start_rank
            )
        )

    def _write_tokens(self, action: Write) -> tuple[list[int], str]:
        text = action.text
        if action.mode == "continuation":
            prior = self.backend.render(self.token_ids[-8:], special=True)[-1:]
            if (
                text
                and text[0].isalnum()
                and not text[0].isspace()
                and prior
                and not prior.isspace()
                and prior not in "([{'\"\u201c\u2018"
            ):
                text = " " + text
        tokens = self.backend.tokenize(text, add_bos=False, special=False)
        if not tokens:
            raise InstructionRejected("the write produced no tokens")
        return [int(value) for value in tokens], text

    def _resolve_once(
        self, action: Accept | SelectRawRank | Write | EndGeneration
    ) -> tuple[list[int], str]:
        observation = self.observe()
        if isinstance(action, Accept):
            return [observation.proposal_token_id], observation.proposal_text
        if isinstance(action, SelectRawRank):
            if action.rank > len(observation.logits):
                raise InstructionRejected("raw rank is outside the current vocabulary")
            token_id = int(observation.statistics.top_raw_ids(action.rank)[-1])
            return [token_id], self.backend.token_text(token_id)
        if isinstance(action, Write):
            return self._write_tokens(action)
        eog_ids = [
            int(token_id)
            for token_id in self.backend.eog_token_ids()
            if 0 <= int(token_id) < len(observation.logits)
            and self.backend.is_eog(int(token_id))
        ]
        if not eog_ids:
            raise InstructionRejected("the backend exposes no selectable EOG token")
        token_id = min(eog_ids, key=lambda value: observation.statistics.raw_rank(value))
        return [token_id], self.backend.token_text(token_id)

    @staticmethod
    def _token_mismatch(
        action: PolicyAction,
        boundary: int,
        expected: int | None,
        actual: int | None,
    ) -> Divergence:
        return Divergence(
            boundary=boundary,
            action_kind=action.kind,
            reason="action-resolution-changed",
            expected_token_id=expected,
            actual_token_id=actual,
        )

    def _evidence(self, observation: Observation, token_id: int) -> TokenEvidence:
        statistics = observation.statistics
        is_eog = self.backend.is_eog(token_id)
        return TokenEvidence(
            boundary=self.boundary,
            sampling_coordinate=observation.sampling_coordinate,
            token_id=token_id,
            text=self.backend.token_text(token_id),
            proposal_token_id=observation.proposal_token_id,
            raw_model_nll=statistics.raw_nll(token_id),
            raw_rank=statistics.raw_rank(token_id),
            policy_rank=statistics.policy_rank(token_id),
            decoder_probability=observation.distribution.probability(token_id),
            proposal_agreement=token_id == observation.proposal_token_id,
            is_eog=is_eog,
            realized_visible=not is_eog,
        )

    def _commit_token(self, observation: Observation, token_id: int) -> TokenEvidence:
        self._validate_observation(observation)
        if not 0 <= token_id < self.backend.vocabulary_size():
            raise EditorError("action resolved outside the vocabulary")
        evidence = self._evidence(observation, token_id)
        self._invalidate_observation()
        if evidence.is_eog:
            self.terminal_token_id = token_id
        else:
            self.backend.eval([token_id])
            self.visible_token_ids.append(token_id)
        return evidence

    def apply(
        self,
        action: PolicyAction,
        *,
        expectation: ReplayExpectation | None = None,
        divergence_policy: str = "handoff",
        replay: bool = False,
    ) -> ActionOutcome:
        """Resolve and apply one action.

        Under ``handoff``, a changed action is stopped before the first changed
        token is committed.  ``ballistic`` records the first mismatch and uses
        the action's current meaning. In replay mode, resolved EOG yields a
        live edge without committing the terminal token in either mode.
        """
        if divergence_policy not in {"handoff", "ballistic"}:
            raise EditorError("divergence policy must be handoff or ballistic")
        if self.ended:
            raise EditorError("cannot apply an action after the episode ended")
        if self.checkpointed:
            raise EditorError("cannot apply an action until the checkpoint is resumed")
        if (
            isinstance(action, Hold)
            and self.remaining is not None
            and action.limit > self.remaining
        ):
            raise TokenBudgetExceeded(
                f"hold requests {action.limit} tokens but only {self.remaining} remain"
            )
        before = self.boundary
        evidence: list[TokenEvidence] = []
        visible: list[int] = []
        resolved: list[int] = []
        divergence: Divergence | None = None
        resolved_text = ""
        stop_reason = "completed"
        expected_ids = expectation.resolved_token_ids if expectation else ()

        def check(actual: int, index: int) -> bool:
            nonlocal divergence
            if expectation is None:
                return True
            expected = expected_ids[index] if index < len(expected_ids) else None
            if expected == actual:
                return True
            if divergence is None:
                divergence = self._token_mismatch(
                    action, self.boundary, expected, actual
                )
            return divergence_policy == "ballistic"

        def eog_handoff(token_id: int) -> ActionOutcome:
            # Compare the attempted terminal resolution, but keep it out of the
            # committed ledger. A matching terminal is still a live replay edge.
            nonlocal divergence
            check(token_id, len(resolved))
            attempted = (*resolved, token_id)
            if expectation is not None and divergence is None:
                if attempted != expected_ids:
                    index = min(len(attempted), len(expected_ids))
                    divergence = self._token_mismatch(
                        action, self.boundary,
                        expected_ids[index] if index < len(expected_ids) else None,
                        attempted[index] if index < len(attempted) else None,
                    )
                elif expectation.stop_reason not in {None, "eog"}:
                    divergence = Divergence(
                        self.boundary, action.kind, "stop-condition-changed",
                        None, None, expectation.stop_reason, "eog",
                    )
            return ActionOutcome(
                action=action, boundary_before=before, boundary_after=self.boundary,
                resolved_text=self.backend.render(visible),
                resolved_token_ids=tuple(resolved), visible_token_ids=tuple(visible),
                terminal_token_id=None, stop_reason="replay-eog",
                evidence=tuple(evidence), status="handed-off",
                divergence=divergence, replay_eog_token_id=token_id,
            )

        if isinstance(action, (Accept, SelectRawRank, Write, EndGeneration)):
            planned, resolved_text = self._resolve_once(action)
            if self.remaining is not None and len(planned) > self.remaining + sum(
                self.backend.is_eog(token_id) for token_id in planned
            ):
                raise TokenBudgetExceeded("action exceeds the remaining token budget")
            # Write is atomic at the policy boundary: its tokenization is
            # checked before any wedge is committed.
            if isinstance(action, Write) and expectation is not None:
                actual = tuple(planned)
                if actual != expected_ids:
                    mismatch_index = next(
                        (
                            index
                            for index in range(max(len(actual), len(expected_ids)))
                            if (actual[index] if index < len(actual) else None)
                            != (
                                expected_ids[index]
                                if index < len(expected_ids)
                                else None
                            )
                        ),
                        0,
                    )
                    divergence = self._token_mismatch(
                        action,
                        self.boundary,
                        expected_ids[mismatch_index]
                        if mismatch_index < len(expected_ids)
                        else None,
                        actual[mismatch_index]
                        if mismatch_index < len(actual)
                        else None,
                    )
                    if divergence_policy == "handoff":
                        return ActionOutcome(
                            action,
                            before,
                            before,
                            resolved_text,
                            actual,
                            (),
                            None,
                            "divergence",
                            (),
                            "handed-off",
                            divergence,
                        )
            for index, token_id in enumerate(planned):
                if replay and self.backend.is_eog(token_id):
                    return eog_handoff(token_id)
                if not isinstance(action, Write) and not check(token_id, index):
                    return ActionOutcome(
                        action,
                        before,
                        self.boundary,
                        resolved_text,
                        tuple(resolved),
                        tuple(visible),
                        self.terminal_token_id,
                        "divergence",
                        tuple(evidence),
                        "handed-off",
                        divergence,
                    )
                observation = self.observe()
                item = self._commit_token(observation, token_id)
                evidence.append(item)
                resolved.append(token_id)
                if item.realized_visible:
                    visible.append(token_id)
                else:
                    self.terminal_reason = "teacher-eog"
                    stop_reason = "eog"
                    break
        else:
            assert isinstance(action, (Hold, Finish))
            limit = action.limit if isinstance(action, Hold) else self.remaining
            if limit is None:
                raise InstructionRejected("legacy finish requires a token budget; use hold")
            boundary_kind = action.boundary if isinstance(action, Hold) else None
            while (
                len(visible) < limit
                and not self.ended
                and not self.checkpointed
            ):
                observation = self.observe()
                token_id = observation.proposal_token_id
                if replay and self.backend.is_eog(token_id):
                    return eog_handoff(token_id)
                if not check(token_id, len(resolved)):
                    return ActionOutcome(
                        action,
                        before,
                        self.boundary,
                        self.backend.render(visible),
                        tuple(resolved),
                        tuple(visible),
                        self.terminal_token_id,
                        "divergence",
                        tuple(evidence),
                        "handed-off",
                        divergence,
                    )
                item = self._commit_token(observation, token_id)
                evidence.append(item)
                resolved.append(token_id)
                if not item.realized_visible:
                    self.terminal_reason = "model-eog"
                    stop_reason = "eog"
                    break
                visible.append(token_id)
                if boundary_kind is not None:
                    boundaries = self._token_boundaries.get(token_id)
                    if boundaries is None:
                        # Evidence already contains the decoded token text;
                        # checking a hold never needs an extra decode.
                        boundaries = token_boundaries(item.text)
                        self._token_boundaries[token_id] = boundaries
                    if boundary_kind in boundaries:
                        stop_reason = f"{boundary_kind}-boundary"
                        break
            else:
                if self.checkpointed:
                    stop_reason = "checkpoint"
                elif isinstance(action, Finish):
                    stop_reason = "checkpoint"
                else:
                    stop_reason = "requested-length"
            resolved_text = self.backend.render(visible)

        if expectation is not None and divergence is None:
            actual = tuple(resolved)
            if actual != expected_ids:
                index = min(len(actual), len(expected_ids))
                divergence = self._token_mismatch(
                    action,
                    self.boundary,
                    expected_ids[index] if index < len(expected_ids) else None,
                    actual[index] if index < len(actual) else None,
                )
            elif (
                expectation.stop_reason is not None
                and expectation.stop_reason != stop_reason
            ):
                divergence = Divergence(
                    boundary=self.boundary,
                    action_kind=action.kind,
                    reason="stop-condition-changed",
                    expected_token_id=None,
                    actual_token_id=None,
                    expected_stop_reason=expectation.stop_reason,
                    actual_stop_reason=stop_reason,
                )
        status = (
            "handed-off"
            if divergence is not None and divergence_policy == "handoff"
            else "completed-with-divergence"
            if divergence is not None
            else "completed"
        )
        return ActionOutcome(
            action=action,
            boundary_before=before,
            boundary_after=self.boundary,
            resolved_text=resolved_text,
            resolved_token_ids=tuple(resolved),
            visible_token_ids=tuple(visible),
            terminal_token_id=self.terminal_token_id
            if evidence and evidence[-1].is_eog
            else None,
            stop_reason=stop_reason,
            evidence=tuple(evidence),
            status=status,
            divergence=divergence,
        )
