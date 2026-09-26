"""Single interpreter for interactive and replayed policy actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np

from .boundaries import token_boundaries
from .core.actions import (
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    PolicyAction,
    SelectRawRank,
    Write,
)
from .core.backend import (
    InferenceBackend,
    require_inference_backend,
)
from .core.candidates import Candidate
from .candidate_columns import CandidateViewPlan
from .core.errors import EditorError
from .core.results import ActionOutcome, Divergence, ReplayExpectation, TokenEvidence
from .core.sampler_config import SamplerConfig
from .core.trajectory import TrajectoryState
from .episode_hash import (
    token_prefix_sha256,
    validate_fingerprint,
)
from .core.observation import ObservationStatistics
from .core.sampling import (
    SparseDistribution,
    draw_token,
)


class InstructionRejected(EditorError):
    """A recognized move cannot execute in the current target state."""


class TokenBudgetExceeded(InstructionRejected):
    """An atomic action cannot fit within the current allowance."""


@dataclass(frozen=True, eq=False, slots=True)
class TokenPrefixSnapshot(Sequence[int]):
    """Immutable token prefix assembled from shared append-only chunks."""

    parent: "TokenPrefixSnapshot | None"
    values: tuple[int, ...]
    length: int

    @classmethod
    def root(cls, values: Sequence[int]) -> "TokenPrefixSnapshot":
        tokens = tuple(values)
        return cls(None, tokens, len(tokens))

    def append(self, values: Sequence[int]) -> "TokenPrefixSnapshot":
        tokens = tuple(values)
        if not tokens:
            return self
        return TokenPrefixSnapshot(self, tokens, self.length + len(tokens))

    def chunks_since(
        self, ancestor: "TokenPrefixSnapshot | None"
    ) -> tuple[tuple[int, ...], ...] | None:
        chunks: list[tuple[int, ...]] = []
        current: TokenPrefixSnapshot | None = self
        while current is not ancestor and current is not None:
            if current.values:
                chunks.append(current.values)
            current = current.parent
        if current is not ancestor:
            return None
        return tuple(reversed(chunks))

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        chunks: list[TokenPrefixSnapshot] = []
        current: TokenPrefixSnapshot | None = self
        while current is not None:
            if current.values:
                chunks.append(current)
            current = current.parent
        for chunk in reversed(chunks):
            yield from chunk.values

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            if step != 1:
                return tuple(self)[index]
            if stop <= start:
                return ()
            pieces: list[tuple[int, ...]] = []
            current: TokenPrefixSnapshot | None = self
            while current is not None and current.length > start:
                chunk_start = current.length - len(current.values)
                lower = max(start, chunk_start) - chunk_start
                upper = min(stop, current.length) - chunk_start
                if upper > lower:
                    pieces.append(current.values[lower:upper])
                current = current.parent
            return tuple(token for piece in reversed(pieces) for token in piece)
        resolved = index + self.length if index < 0 else index
        if resolved < 0 or resolved >= self.length:
            raise IndexError("token prefix index out of range")
        current: TokenPrefixSnapshot | None = self
        while current is not None:
            chunk_start = current.length - len(current.values)
            if resolved >= chunk_start:
                return current.values[resolved - chunk_start]
            current = current.parent
        raise IndexError("token prefix index out of range")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        return len(other) == self.length and all(
            left == right for left, right in zip(self, other)
        )


@dataclass(frozen=True)
class Observation:
    boundary: int
    sampling_boundary: int
    prefix_token_ids: Sequence[int] = field(repr=False)
    _render_context: Callable[..., str] = field(repr=False, compare=False)
    logits: np.ndarray = field(repr=False, compare=False)
    distribution: SparseDistribution = field(repr=False, compare=False)
    proposal_token_id: int
    proposal_text: str
    proposal_raw_rank: int
    proposal_decoder_probability: float
    statistics: ObservationStatistics = field(repr=False, compare=False)

    @cached_property
    def context_text(self) -> str:
        """Render this captured boundary once, only when display needs it."""
        return self._render_context(list(self.prefix_token_ids), special=True)

    @cached_property
    def proposal_raw_probability(self) -> float:
        """Model soft-max mass for the proposal; computed on first read."""
        return float(self.statistics.raw_probabilities([self.proposal_token_id])[0])

    @cached_property
    def proposal_policy_rank(self) -> int:
        return self.statistics.policy_rank(self.proposal_token_id)


@dataclass(frozen=True)
class _PreparedAccept:
    observation: Observation = field(repr=False, compare=False)
    raw_rank: int
    token_id: int
    generation: int
    prefix_token_ids: tuple[int, ...]


class EpisodeEngine:
    """Own token state, sampling controls, and all action resolution.

    The backend owns its private evaluation state. The engine's complete
    semantic state is the token ledger plus the sampler configuration,
    stream identity, and current boundary.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        sampling: SamplerConfig,
        max_tokens: int | None = None,
        initial_text: str | None = None,
        initial_token_ids: Sequence[int] | None = None,
        add_bos: bool = True,
        special: bool = True,
        stream_fingerprint: str | None = None,
        backend_positioned: bool = False,
        guidance_backend: InferenceBackend | None = None,
    ) -> None:
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise EditorError("max_tokens must be a positive integer")
        require_inference_backend(backend)
        if guidance_backend is not None:
            require_inference_backend(guidance_backend)
            if guidance_backend.vocabulary_size() != backend.vocabulary_size():
                raise EditorError("CFG guidance backend vocabulary does not match the primary backend")
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
        fingerprint = (
            token_prefix_sha256(tokens)
            if stream_fingerprint is None
            else validate_fingerprint(stream_fingerprint)
        )
        if not backend_positioned:
            backend.reset(list(tokens))
        self.backend = backend
        self._backend_positioned = True
        self.sampling = sampling
        initial_text_value = (
            initial_text
            if isinstance(initial_text, str)
            else backend.render(list(tokens), special=True)
        )
        self.trajectory = TrajectoryState(
            initial_token_ids=tuple(tokens),
            initial_text=initial_text_value,
            max_tokens=max_tokens,
            checkpoint_boundary=max_tokens,
            stream_fingerprint=fingerprint,
        )
        self._prefix_snapshot = TokenPrefixSnapshot.root(tokens)
        self._prefix_snapshot_boundary = 0
        self._prefix_snapshot_dirty = False
        self.guidance_backend = guidance_backend
        self._guidance_owner = object()
        self._guidance_prompt_key: tuple | None = None
        self._guidance_prompt_ids: tuple[int, ...] = ()
        self._guidance_evaluated_prefix: tuple[int, ...] | None = None
        if sampling.cfg_unconditional_prompt is not None:
            self._guidance_prompt_tokens()
        self._observation: Observation | None = None
        self._observation_key: tuple | None = None
        self._latest_speculation_generation = -1
        self._speculative_accept_prefix: tuple[int, ...] | None = None
        self._ephemeral_logit_biases: dict[int, float] = {}
        self._activation_runtime_key: tuple | None = None
        # This engine owns one backend/tokenizer. Sampler changes and rewinds
        # do not change token spellings, so their classifications remain valid.
        self._token_boundaries: dict[int, frozenset[str]] = {}
        self._metric_sink: Callable[[Observation, int], None] | None = None

    @property
    def initial_token_ids(self) -> tuple[int, ...]:
        return self.trajectory.initial_token_ids

    @property
    def initial_text(self) -> str:
        return self.trajectory.initial_text

    @property
    def visible_token_ids(self) -> list[int]:
        return self.trajectory.visible_token_ids

    @visible_token_ids.setter
    def visible_token_ids(self, value: Sequence[int]) -> None:
        self.trajectory.visible_token_ids = list(value)
        self._prefix_snapshot_dirty = True

    @property
    def terminal_token_id(self) -> int | None:
        return self.trajectory.terminal_token_id

    @terminal_token_id.setter
    def terminal_token_id(self, value: int | None) -> None:
        self.trajectory.terminal_token_id = value

    @property
    def terminal_reason(self) -> str | None:
        return self.trajectory.terminal_reason

    @terminal_reason.setter
    def terminal_reason(self, value: str | None) -> None:
        self.trajectory.terminal_reason = value

    @property
    def stream_fingerprint(self) -> str | None:
        return self.trajectory.stream_fingerprint

    @stream_fingerprint.setter
    def stream_fingerprint(self, value: str | None) -> None:
        self.trajectory.stream_fingerprint = value

    @property
    def max_tokens(self) -> int | None:
        return self.trajectory.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 1):
            raise EditorError("max_tokens must be a positive integer")
        self.trajectory.max_tokens = value

    @property
    def checkpoint_boundary(self) -> int | None:
        return self.trajectory.checkpoint_boundary

    @checkpoint_boundary.setter
    def checkpoint_boundary(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise EditorError("checkpoint boundary must be nonnegative")
        self.trajectory.checkpoint_boundary = value

    @property
    def sampling(self) -> SamplerConfig:
        return self._sampling

    @sampling.setter
    def sampling(self, value: SamplerConfig) -> None:
        previous = getattr(self, "_sampling", None)
        previous_activation_key = (
            self._activation_backend_key_for(previous)
            if previous is not None else None
        )
        bias_tokens = []
        for rule in (*value.bias_rules,
                     *(rule for group in value.bias_groups for rule in group.rules)):
            bias_tokens.extend(token for route in rule.routes for token in route)
            bias_tokens.extend(token for trigger in rule.triggers for token in trigger)
            if type(rule.until) is int:
                bias_tokens.append(rule.until)
        if any(token >= self.backend.vocabulary_size() for token in bias_tokens):
            raise EditorError("bias token id is outside the model vocabulary")
        self._sampling = value
        if hasattr(self, "_activation_runtime_key") and (
            previous_activation_key != self._activation_backend_key_for(value)
        ):
            self._activation_runtime_key = None
        self._invalidate_observation()

    @property
    def boundary(self) -> int:
        return self.trajectory.boundary

    @property
    def remaining(self) -> int | None:
        """Visible tokens remaining until the next checkpoint."""
        return self.trajectory.remaining

    @property
    def checkpointed(self) -> bool:
        """Whether the live episode has yielded at its current checkpoint."""
        return self.trajectory.checkpointed

    @property
    def ended(self) -> bool:
        """True only after a genuine terminal event."""
        return self.trajectory.ended

    def resume(
        self,
        *,
        max_tokens: int | None | str = "keep",
        sampling: SamplerConfig | None = None,
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

    def rewind_to(self, boundary: int, *, _defer_backend_positioning: bool = False) -> None:
        """Discard visible state after a token boundary and reposition the backend.

        The checkpoint boundary is kept intact. The caller restores historical
        sampler settings and stream identity from the episode store; the retained
        visible-token boundary determines the next sampling boundary.
        This method only repositions token/backend state and clears cached evidence.
        """
        if type(boundary) is not int or boundary < 0 or boundary > self.boundary:
            raise EditorError(
                f"rewind boundary must be between 0 and {self.boundary}"
            )
        retained = list(self.visible_token_ids[:boundary])
        self._rollback_speculative_accept()
        self._invalidate_observation()
        self._prefix_snapshot_dirty = True
        self._ephemeral_logit_biases = {}
        if _defer_backend_positioning:
            self._backend_positioned = False
        else:
            prefix = [*self.initial_token_ids, *retained]
            branch = getattr(self.backend, "branch_to_prefix", None)
            if callable(branch):
                branch(prefix)
            else:
                self.backend.reset(prefix)
            self._backend_positioned = True
        self.trajectory.rewind_to(boundary)

    def terminate(self, reason: str = "menu-end") -> None:
        """Seal a live episode without manufacturing an EOG token."""
        if self.ended:
            return
        if not isinstance(reason, str) or not reason:
            raise EditorError("termination reason must be nonempty")
        self._invalidate_observation()
        self.trajectory.terminate(reason)

    @property
    def token_ids(self) -> list[int]:
        return self.trajectory.token_ids

    @property
    def text(self) -> str:
        return self.backend.render(self.token_ids, special=True)

    def _observation_prefix_snapshot(self) -> TokenPrefixSnapshot:
        boundary = self.boundary
        if self._prefix_snapshot_dirty or boundary < self._prefix_snapshot_boundary:
            self._prefix_snapshot = TokenPrefixSnapshot.root(self.initial_token_ids)
            self._prefix_snapshot_boundary = 0
            self._prefix_snapshot_dirty = False
        if boundary > self._prefix_snapshot_boundary:
            appended = self.visible_token_ids[self._prefix_snapshot_boundary:boundary]
            self._prefix_snapshot = self._prefix_snapshot.append(appended)
            self._prefix_snapshot_boundary = boundary
        return self._prefix_snapshot

    def _invalidate_observation(self) -> None:
        self._rollback_speculative_accept()
        self._observation = None
        self._observation_key = None
        self._prepared_accept = None

    def _ensure_backend_positioned(self) -> None:
        """Backfill a deferred preview cache before the next model mutation."""
        if self._speculative_accept_prefix is not None:
            self._rollback_speculative_accept()
            return
        if self._backend_positioned:
            return
        prefix = list(self.token_ids)
        branch = getattr(self.backend, "branch_to_prefix", None)
        if callable(branch):
            branch(prefix)
        else:
            self.backend.reset(prefix)
        self._backend_positioned = True

    def discard_speculative_accept(self) -> None:
        """Drop a prepared continuation and restore the committed backend prefix."""
        self._rollback_speculative_accept()

    def _rollback_speculative_accept(self) -> None:
        if getattr(self, "_speculative_accept_prefix", None) is None:
            self._prepared_accept = None
            return
        self.backend.rollback_speculation()
        self._speculative_accept_prefix = None
        self._prepared_accept = None
        self._backend_positioned = True

    def has_prepared_accept(
        self, observation: Observation, raw_rank: int, token_id: int
    ) -> bool:
        """Return whether an exact warm is ready for this decision and target."""
        prepared = self._prepared_accept
        return bool(
            prepared is not None
            and prepared.observation is observation
            and prepared.raw_rank == raw_rank
            and prepared.token_id == token_id
            and prepared.prefix_token_ids == tuple(self.token_ids)
            and self._speculative_accept_prefix == prepared.prefix_token_ids
        )

    def speculate_accept(
        self,
        observation: Observation,
        *,
        raw_rank: int | None = None,
        token_id: int | None = None,
        generation: int = 0,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Warm one selected token in place, then commit or roll it back later."""
        is_cancelled = cancelled or (lambda: False)
        if (
            type(generation) is not int or generation < 0
            or generation < self._latest_speculation_generation
            or is_cancelled()
        ):
            return False
        if self.ended or self.checkpointed:
            return False
        self._validate_observation(observation)
        if raw_rank is None and token_id is None:
            raw_rank = observation.proposal_raw_rank
            token_id = observation.proposal_token_id
        elif raw_rank is None:
            if type(token_id) is not int or not 0 <= token_id < len(observation.logits):
                return False
            raw_rank = observation.statistics.raw_rank(token_id)
        if type(raw_rank) is not int or not 1 <= raw_rank <= len(observation.logits):
            return False
        selected_token_id = int(observation.statistics.top_raw_ids(raw_rank)[-1])
        if token_id is not None and (
            type(token_id) is not int or token_id != selected_token_id
        ):
            return False
        token_id = selected_token_id
        self._latest_speculation_generation = generation
        prepared = self._prepared_accept
        if (
            prepared is not None
            and prepared.observation is observation
            and prepared.raw_rank == raw_rank
            and prepared.token_id == token_id
            and prepared.prefix_token_ids == tuple(self.token_ids)
            and self._speculative_accept_prefix == prepared.prefix_token_ids
        ):
            return True
        self.discard_speculative_accept()
        if (
            self.backend.is_eog(token_id)
            or (self.remaining is not None and self.remaining <= 1)
            or self._cfg_active()
            or not callable(getattr(self.backend, "speculate", None))
            or not callable(getattr(self.backend, "commit_speculation", None))
            or not callable(getattr(self.backend, "rollback_speculation", None))
        ):
            return False

        if not self._backend_positioned:
            self._ensure_backend_positioned()
        prefix = tuple(self.token_ids)
        self._speculative_accept_prefix = prefix
        self._backend_positioned = False
        if not self.backend.speculate(token_id):
            self._speculative_accept_prefix = None
            self._backend_positioned = True
            return False
        if is_cancelled():
            self._rollback_speculative_accept()
            return False
        if self._observation is not observation or self._decision_key() != self._observation_key:
            self._rollback_speculative_accept()
            return False
        self._prepared_accept = _PreparedAccept(
            observation, raw_rank, token_id, generation, prefix
        )
        return True

    @staticmethod
    def _activation_backend_key_for(sampling: SamplerConfig | None) -> tuple:
        """Return the backend-state identity, including vector contents."""
        if sampling is None or not (
            sampling.activation_vector_layer == "control-vector"
            and sampling.activation_vector
            and sampling.activation_vector_strength != 0.0
        ):
            return ("plain",)
        from .activation_vectors import steering_vector_digest_for

        content_digest = steering_vector_digest_for(
            sampling.activation_vector,
            layer=sampling.activation_vector_layer,
            position=sampling.activation_vector_position,
            strength=sampling.activation_vector_strength,
            layer_start=sampling.activation_vector_layer_start,
            layer_end=sampling.activation_vector_layer_end,
        )
        return (
            "control-vector",
            content_digest,
            sampling.activation_vector_digest,
            sampling.activation_vector_layer_start,
            sampling.activation_vector_layer_end,
            sampling.activation_vector_strength,
        )

    def _prepare_activation_runtime(self) -> None:
        """Reconcile backend model-state controls before reading logits."""
        current_key = self._activation_backend_key_for(self.sampling)
        if current_key == self._activation_runtime_key:
            return
        is_control = current_key[0] == "control-vector"
        if not is_control:
            clear = getattr(self.backend, "clear_activation_control_vector", None)
            if callable(clear):
                try:
                    clear()
                    # Clearing an adapter does not retroactively change logits
                    # already present in a reused KV cache.
                    self.backend.reset(self.token_ids)
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise EditorError(f"could not clear hidden-state control vector: {exc}") from exc
        if is_control:
            setter = getattr(self.backend, "set_activation_control_vector", None)
            if not callable(setter):
                raise EditorError(
                    "the loaded backend does not expose llama.cpp control-vector runtime support"
                )
            try:
                setter(
                    self.sampling.activation_vector,
                    layer_start=self.sampling.activation_vector_layer_start,
                    layer_end=self.sampling.activation_vector_layer_end,
                    strength=self.sampling.activation_vector_strength,
                )
                # Existing KV state was evaluated without the newly selected
                # control vector; rebuild the current prefix under the adapter.
                self.backend.reset(self.token_ids)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise EditorError(f"could not apply activation control vector: {exc}") from exc
        self._activation_runtime_key = current_key

    def _decision_key(self) -> tuple:
        return (
            tuple(self.token_ids), self.sampling,
            self.stream_fingerprint,
            tuple(sorted(self._ephemeral_logit_biases.items())),
        )

    def _validate_observation(self, observation: Observation) -> None:
        if (
            self.ended or self.checkpointed
            or observation is not self._observation
            or self._observation_key != self._decision_key()
        ):
            raise EditorError("request refers to a stale observation")

    def observe(self) -> Observation:
        if self.ended or self.checkpointed:
            raise EditorError("the episode has no live decision boundary")
        key = self._decision_key()
        if self._cfg_active() and self.guidance_backend is not None and (
            getattr(self.guidance_backend, "_spe_cfg_owner", None) is not self._guidance_owner
        ):
            self._invalidate_guidance()
        if self._observation is not None and self._observation_key == key:
            return self._observation
        self._ensure_backend_positioned()
        self._prepare_activation_runtime()
        # Make one owned float64 snapshot here. ObservationStatistics validates
        # and freezes this same array instead of copying the full vocabulary again.
        logits = np.array(self.backend.last_logits(), dtype=np.float64, copy=True)
        if logits.ndim != 1 or len(logits) != self.backend.vocabulary_size():
            raise RuntimeError("backend logits do not match its vocabulary")
        if self._cfg_active():
            self._position_guidance()
            unconditional = np.asarray(self.guidance_backend.last_logits(), dtype=np.float64)
            if unconditional.shape != logits.shape or not np.all(np.isfinite(unconditional)):
                raise RuntimeError("CFG unconditional logits do not match the primary backend")
            logits = unconditional + float(self.sampling.cfg_scale) * (logits - unconditional)
            if not np.all(np.isfinite(logits)):
                raise RuntimeError("CFG guidance produced non-finite logits")
        activation_logit_adjustments = None
        if (
            self.sampling.activation_vector_layer == "output"
            and self.sampling.activation_vector
            and self.sampling.activation_vector_strength != 0.0
        ):
            provider = getattr(self.backend, "activation_logit_adjustments", None)
            if not callable(provider):
                raise EditorError(
                    "the loaded backend does not expose output-head steering runtime support"
                )
            try:
                activation_logit_adjustments = provider(
                    self.sampling.activation_vector,
                    layer=self.sampling.activation_vector_layer,
                    position=self.sampling.activation_vector_position,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise EditorError(f"could not apply output-head steering vector: {exc}") from exc
            activation_logit_adjustments = np.asarray(
                activation_logit_adjustments, dtype=np.float64
            )
            if activation_logit_adjustments.shape != logits.shape:
                raise RuntimeError(
                    "output-head steering adjustments do not match the backend vocabulary"
                )
            if not np.all(np.isfinite(activation_logit_adjustments)):
                raise RuntimeError("output-head steering adjustments are not finite")
        statistics = ObservationStatistics(
            logits,
            self.sampling,
            key[0],
            activation_logit_adjustments=activation_logit_adjustments,
            ephemeral_logit_biases=self._ephemeral_logit_biases,
            take_logits_ownership=True,
        )
        logits = statistics.logits
        distribution = statistics.distribution
        sampling_boundary = self.boundary
        proposal = draw_token(
            distribution,
            seed=self.sampling.seed,
            stream_fingerprint=self.stream_fingerprint,
            aligned_step=sampling_boundary,
            kernel=self.sampling.draw_kernel,
        )
        observation = Observation(
            boundary=self.boundary,
            sampling_boundary=sampling_boundary,
            prefix_token_ids=self._observation_prefix_snapshot(),
            _render_context=self.backend.render,
            logits=logits,
            distribution=distribution,
            proposal_token_id=proposal,
            proposal_text=self.backend.token_text(proposal),
            proposal_raw_rank=statistics.raw_rank(proposal),
            proposal_decoder_probability=distribution.probability(proposal),
            statistics=statistics,
        )
        self._observation = observation
        self._observation_key = key
        return observation

    def _invalidate_guidance(self) -> None:
        """Release knowledge of shared guidance state before backend reuse."""
        self._guidance_evaluated_prefix = None
        self._invalidate_observation()

    def adopt_preview_state(
        self, preview: "EpisodeEngine", *, backend_positioned: bool = True
    ) -> None:
        """Adopt a speculative engine that has already advanced this trajectory.

        Chord previews use the same backend instances and the same starting
        token ledger. Once one preview is selected, its semantic engine state
        can become the live engine directly; replaying its actions would repeat
        decoding and evidence work. The preview's mutable trajectory and
        incremental snapshots are transferred by reference. Keep the live
        metric sink, which belongs to the durable/runtime owner rather than to
        speculative work.
        """
        if not isinstance(preview, EpisodeEngine):
            raise TypeError("preview must be an EpisodeEngine")
        if preview is self:
            return
        if (
            self.backend is not preview.backend
            or self.guidance_backend is not preview.guidance_backend
            or self.initial_token_ids != preview.initial_token_ids
            or self.initial_text != preview.initial_text
        ):
            raise EditorError("preview engine does not share this episode's model and root")
        metric_sink = self._metric_sink
        self.__dict__.update(preview.__dict__)
        self._metric_sink = metric_sink
        self._backend_positioned = backend_positioned
        if not backend_positioned:
            # The selected observation is still exact, but the shared primary
            # and CFG caches belong to another preview until the next action.
            self._guidance_evaluated_prefix = None

    def _guidance_prompt_tokens(self) -> tuple[int, ...]:
        backend = self.guidance_backend
        if backend is None:
            raise EditorError("CFG is configured but no unconditional guidance backend was provided")
        if backend is self.backend or backend.vocabulary_size() != self.backend.vocabulary_size():
            raise EditorError("CFG requires a separate matching guidance backend")
        prompt = self.sampling.cfg_unconditional_prompt
        key = (backend, prompt)
        if self._guidance_prompt_key != key:
            # Guidance always enters as a standalone prompt. Primary text/IDs
            # and primary-only tokenization options cannot change this policy.
            tokens = tuple(backend.tokenize(prompt, add_bos=True, special=True))
            if not tokens:
                raise EditorError("CFG unconditional prompt produced no tokens")
            if any(value < 0 or value >= backend.vocabulary_size() for value in tokens):
                raise EditorError("CFG unconditional prompt contains an invalid token id")
            self._guidance_prompt_key = key
            self._guidance_prompt_ids = tokens
            self._guidance_evaluated_prefix = None
        return self._guidance_prompt_ids

    def _position_guidance(self) -> None:
        """Synchronize U + V lazily; append-only decisions reuse evaluation."""
        desired = (*self._guidance_prompt_tokens(), *self.visible_token_ids)
        evaluated = self._guidance_evaluated_prefix
        if getattr(self.guidance_backend, "_spe_cfg_owner", None) is not self._guidance_owner:
            evaluated = None
        # Clear our claim before calling the backend: failed evaluation must
        # not leave a prefix marked as successfully positioned.
        self._guidance_evaluated_prefix = None
        self.guidance_backend._spe_cfg_owner = None
        if evaluated is None or desired[:len(evaluated)] != evaluated:
            self.guidance_backend.reset(list(desired))
        elif len(desired) > len(evaluated):
            self.guidance_backend.eval(list(desired[len(evaluated):]))
        self._guidance_evaluated_prefix = desired
        # A token, not an engine reference: shared adapters do not keep old
        # sessions alive. Every guidance evaluation is owned by this helper.
        self.guidance_backend._spe_cfg_owner = self._guidance_owner

    def _cfg_active(self) -> bool:
        prefix_tokens = self.sampling.cfg_prefix_tokens
        return bool(
        self.sampling.cfg_unconditional_prompt is not None
        and (
            prefix_tokens == 0
            or len(self.visible_token_ids) < prefix_tokens
            )
        )

    def candidates(
        self,
        observation: Observation,
        *,
        start_rank: int = 1,
        count: int = 12,
        view: CandidateViewPlan | None = None,
    ) -> tuple[Candidate, ...]:
        if start_rank < 1 or count < 1:
            raise EditorError("candidate rank and count must be positive")
        self._validate_observation(observation)
        end = min(len(observation.logits), start_rank + count - 1)
        if start_rank > end:
            return ()
        statistics = observation.statistics
        ordered = statistics.top_raw_ids(end)[start_rank - 1 : end]
        return self._candidates_for_tokens(observation, ordered, view=view)

    def policy_candidates(
        self,
        observation: Observation,
        *,
        count: int = 12,
        view: CandidateViewPlan | None = None,
    ) -> tuple[Candidate, ...]:
        """Full-vocabulary policy top-N; selections still use absolute raw rank."""
        if count < 1:
            raise EditorError("candidate count must be positive")
        self._validate_observation(observation)
        ordered = observation.statistics.top_policy_ids(min(count, len(observation.logits)))
        return self._candidates_for_tokens(
            observation, ordered,
            view=(view or CandidateViewPlan((), frozenset())).policy_ordered(),
        )

    def _candidates_for_tokens(
        self,
        observation: Observation,
        ordered: list[int],
        *,
        view: CandidateViewPlan | None = None,
    ) -> tuple[Candidate, ...]:
        statistics = observation.statistics
        metrics = view.metrics if view is not None else frozenset()
        probabilities = (
            statistics.raw_probabilities(ordered)
            if "raw_probability" in metrics else [None] * len(ordered)
        )
        policy_probabilities = (
            statistics.policy_probabilities_at(ordered)
            if "policy_probability" in metrics else [None] * len(ordered)
        )
        logits = [float(statistics.logits[token_id]) for token_id in ordered]
        if "neighbor_margin" in metrics and ordered:
            # Uniform consecutive margin over the ordered menu list:
            # logit[i] - logit[i+1] (advantage over next-worse). Last row: None.
            margins: list[float | None] = [
                logits[i] - logits[i + 1] for i in range(len(logits) - 1)
            ] + [None]
        else:
            margins = [None] * len(ordered)
        if "logit_z" in metrics and ordered:
            # Full-vocab logit z-score: (logit - mean) / std (population ddof=0).
            # O(V) mean/std once; does not wake soft-max / logsumexp.
            zs = statistics.logit_z_scores(ordered)
        else:
            zs = [None] * len(ordered)
        return tuple(
            Candidate(
                rank=statistics.raw_rank(int(token_id)),
                token_id=int(token_id),
                text=self.backend.token_text(int(token_id)),
                raw_probability=(
                    None if probability is None else float(probability)
                ),
                decoder_probability=(
                    observation.distribution.probability(int(token_id))
                    if "decoder_probability" in metrics else None
                ),
                is_eog=self.backend.is_eog(int(token_id)),
                bias=statistics.active_biases.get(int(token_id), 0.0),
                policy_rank=(statistics.policy_rank(int(token_id))
                             if "policy_rank" in metrics else None),
                policy_probability=(
                    None
                    if policy_probability is None
                    else float(policy_probability)
                ),
                raw_logit=(logit if metrics.intersection({"raw_logit", "top_raw_logit"}) else None),
                neighbor_margin=margin,
                logit_z=z_val,
            )
            for token_id, probability, policy_probability, logit, margin, z_val in zip(
                ordered, probabilities, policy_probabilities, logits, margins, zs
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
        is_eog = self.backend.is_eog(token_id)
        if self._metric_sink is not None:
            self._metric_sink(observation, token_id)
        return TokenEvidence(
            boundary=self.boundary,
            sampling_boundary=observation.sampling_boundary,
            token_id=token_id,
            text=self.backend.token_text(token_id),
            proposal_token_id=observation.proposal_token_id,
            raw_model_nll=None,
            raw_rank=None,
            policy_rank=None,
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
        prepared = self._prepared_accept
        promoted = bool(
            prepared is not None
            and prepared.observation is observation
            and prepared.token_id == token_id
            and prepared.prefix_token_ids == tuple(self.token_ids)
            and self._speculative_accept_prefix == prepared.prefix_token_ids
        )
        if promoted:
            self.backend.commit_speculation()
            self._speculative_accept_prefix = None
            self._backend_positioned = True
        else:
            self.discard_speculative_accept()
        self._invalidate_observation()
        if evidence.is_eog:
            if not promoted:
                self._ensure_backend_positioned()
            self.terminal_token_id = token_id
        else:
            if not promoted:
                self._ensure_backend_positioned()
                self.backend.eval([token_id])
                self._backend_positioned = True
            self.visible_token_ids.append(token_id)
        return evidence

    @staticmethod
    def _phrase_required_shift(observation: Observation, token_id: int) -> float:
        """Return the additive policy shift needed to make ``token_id`` rank 1."""
        values = np.asarray(observation.statistics.policy_logits, dtype=np.float64)
        target = float(values[token_id])
        if len(values) <= 1:
            return 0.0
        competitors = np.concatenate((values[:token_id], values[token_id + 1:]))
        return max(0.0, float(np.max(competitors)) - target + 1e-6)

    def _phrase_step_diagnostic(
        self, observation: Observation, token_id: int, required_shift: float,
        *, applied_shift: float = 0.0,
    ) -> dict[str, Any]:
        statistics = observation.statistics
        return {
            "boundary": observation.boundary,
            "sampling_boundary": observation.sampling_boundary,
            "token_id": int(token_id),
            "text": self.backend.token_text(token_id),
            "model_rank": int(statistics.model_rank(token_id)),
            "policy_rank": int(statistics.policy_rank(token_id)),
            "model_probability": float(statistics.model_probabilities([token_id])[0]),
            "policy_probability": float(statistics.policy_probabilities_at([token_id])[0]),
            "sampler_eligible": bool(np.any(observation.distribution.ids == int(token_id))),
            "sampler_probability": float(observation.distribution.probability(token_id)),
            "required_policy_shift": float(required_shift),
            "applied_policy_shift": float(applied_shift),
            "within_bound": bool(required_shift <= 0.0),
        }

    def _apply_phrase(
        self,
        action: Phrase,
        *,
        expectation: ReplayExpectation | None,
        divergence_policy: str,
    ) -> ActionOutcome:
        """Apply a phrase as sequential teacher selections."""
        before = self.boundary
        planned, resolved_text = self._write_tokens(Write(action.text, action.mode))
        if len(planned) > action.max_tokens:
            raise InstructionRejected(
                f"{action.kind} has {len(planned)} tokens; max is {action.max_tokens}"
            )
        if self.remaining is not None and len(planned) > self.remaining:
            raise TokenBudgetExceeded("phrase exceeds the remaining token budget")
        if any(self.backend.is_eog(token_id) for token_id in planned):
            raise InstructionRejected("phrase text resolves to an EOG token")

        expected_ids = expectation.resolved_token_ids if expectation else ()
        divergence = None
        if expectation is not None and tuple(planned) != expected_ids:
            mismatch_index = next(
                (
                    index
                    for index in range(max(len(planned), len(expected_ids)))
                    if (planned[index] if index < len(planned) else None)
                    != (expected_ids[index] if index < len(expected_ids) else None)
                ),
                0,
            )
            divergence = self._token_mismatch(
                action,
                before,
                expected_ids[mismatch_index] if mismatch_index < len(expected_ids) else None,
                planned[mismatch_index] if mismatch_index < len(planned) else None,
            )
            if divergence_policy == "handoff":
                return ActionOutcome(
                    action=action,
                    boundary_before=before,
                    boundary_after=before,
                    resolved_text=resolved_text,
                    resolved_token_ids=tuple(planned),
                    visible_token_ids=(),
                    terminal_token_id=None,
                    stop_reason="divergence",
                    evidence=(),
                    status="handed-off",
                    divergence=divergence,
                )

        evidence: list[TokenEvidence] = []
        visible: list[int] = []
        resolved: list[int] = []
        details: list[dict[str, Any]] = []
        try:
            for token_id in planned:
                natural = self.observe()
                required = self._phrase_required_shift(natural, token_id)
                if not action.force and required > float(action.max_shift):
                    text = self.backend.token_text(token_id)
                    raise InstructionRejected(
                        f"check phrase rejected at token {len(details) + 1} {text!r}: "
                        f"requires policy shift +{required:.4g}, "
                        f"bound is +{float(action.max_shift):.4g}"
                    )
                applied = required if action.force else 0.0
                detail = self._phrase_step_diagnostic(
                    natural, token_id, required, applied_shift=applied
                )
                detail["within_bound"] = bool(required <= float(action.max_shift))
                if action.force:
                    self._ephemeral_logit_biases = {int(token_id): float(required)}
                    self._invalidate_observation()
                    forced = self.observe()
                    item = self._commit_token(forced, token_id)
                    self._ephemeral_logit_biases = {}
                    self._invalidate_observation()
                else:
                    item = self._commit_token(natural, token_id)
                evidence.append(item)
                resolved.append(token_id)
                if item.realized_visible:
                    visible.append(token_id)
                details.append(detail)
        except InstructionRejected:
            if not action.force:
                # Validate successive prefixes in one pass while keeping the
                # checked phrase atomic to its caller.
                self.rewind_to(before)
            raise
        finally:
            self._ephemeral_logit_biases = {}
            self._invalidate_observation()

        diagnostics = {
            "operation": action.kind,
            "supplied_text": action.text,
            "resolved_text": resolved_text,
            "mode": action.mode,
            "max_tokens": action.max_tokens,
            "max_shift": float(action.max_shift),
            "shift_metric": "policy-rank-1",
            "tokens": details,
        }
        return ActionOutcome(
            action=action,
            boundary_before=before,
            boundary_after=self.boundary,
            resolved_text=resolved_text,
            resolved_token_ids=tuple(resolved),
            visible_token_ids=tuple(visible),
            terminal_token_id=None,
            stop_reason="completed",
            evidence=tuple(evidence),
            status="completed-with-divergence" if divergence is not None else "completed",
            divergence=divergence,
            diagnostics=diagnostics,
        )

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
        prepared = self._prepared_accept
        if replay or not isinstance(action, (Accept, SelectRawRank)):
            self.discard_speculative_accept()
        elif prepared is not None and (
            (
                isinstance(action, Accept)
                and prepared.token_id != prepared.observation.proposal_token_id
            )
            or (
                isinstance(action, SelectRawRank)
                and action.rank != prepared.raw_rank
            )
        ):
            self.discard_speculative_accept()
        if (
            isinstance(action, Hold)
            and self.remaining is not None
            and action.limit > self.remaining
        ):
            raise TokenBudgetExceeded(
                f"hold requests {action.limit} tokens but only {self.remaining} remain"
            )
        if isinstance(action, Phrase):
            return self._apply_phrase(
                action,
                expectation=expectation,
                divergence_policy=divergence_policy,
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
                if not item.realized_visible:
                    break
        else:
            assert isinstance(action, Hold)
            limit = action.limit
            boundary_kind = action.boundary
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
