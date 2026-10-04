"""Interactive deterministic beams over policy scores."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .chord import _position, _recent_context
from .core.actions import PolicyAction, SelectRawRank
from .core.backend import BatchedInferenceSession
from .core.episode_observation import EpisodeObservation
from .core.errors import EditorError
from .core.results import ActionOutcome
from .episode_engine import EpisodeEngine
from .terminal_contracts import BeamViewRow, BeamViewState


class BeamRequested(Exception):
    def __init__(
        self,
        width: int,
        *,
        skip_root_rank_ranges: tuple[tuple[int, int], ...] = (),
        add_root_model_ranks: tuple[int, ...] = (),
    ) -> None:
        super().__init__(width, skip_root_rank_ranges, add_root_model_ranks)
        self.width = width
        self.skip_root_rank_ranges = skip_root_rank_ranges
        self.add_root_model_ranks = add_root_model_ranks


@dataclass(frozen=True)
class BeamNode:
    parent: "BeamNode | None"
    action: PolicyAction
    outcome: ActionOutcome
    token_id: int
    model_rank: int
    step_log_probability: float
    is_eog: bool = False


@dataclass
class BeamPath:
    label: str
    engine: EpisodeEngine
    node: BeamNode | None
    score: float
    model_log_probability: float
    search_log_probability: float
    model_rank: int | None
    step_log_probability: float | None
    lane_id: int | None
    pre_terminal_engine: EpisodeEngine | None = None

    @property
    def state(self) -> str:
        return "finished" if self.engine.ended else "live"


@dataclass(frozen=True)
class _Candidate:
    parent: BeamPath
    observation: EpisodeObservation
    token_id: int
    model_rank: int
    log_probability: float
    model_log_probability: float
    search_log_probability: float
    score: float
    is_eog: bool
    parent_order: int

    @property
    def ordering(self) -> tuple[float, int, int, int]:
        return (-self.score, self.parent_order, self.model_rank, self.token_id)


@dataclass(frozen=True)
class _Checkpoint:
    active: tuple[BeamPath, ...]
    finished: tuple[BeamPath, ...]
    protected_prefixes: tuple[tuple[int, ...], ...]


class BeamSearch:
    """Maintain a temporary deterministic or Gumbel-Top-k frontier."""

    def __init__(
        self,
        engine: EpisodeEngine,
        width: int = 5,
        *,
        skip_root_rank_ranges: tuple[tuple[int, int], ...] = (),
        add_root_model_ranks: tuple[int, ...] = (),
    ) -> None:
        if type(width) is not int or not 1 <= width <= 100:
            raise EditorError("beam width must be between 1 and 100")
        if engine.ended:
            raise EditorError("beam search requires a live decision boundary")
        if engine._speculative_accept_prefix is not None:
            engine.discard_speculative_accept()
            engine._ensure_backend_positioned()
        self.original = engine
        self.width = width
        self.skip_root_rank_ranges = tuple(skip_root_rank_ranges)
        self.add_root_model_ranks = tuple(add_root_model_ranks)
        if len(self.add_root_model_ranks) > self.width:
            raise EditorError("beam add cannot reserve more roots than the beam width")
        if (
            any(type(rank) is not int or rank < 1 for rank in self.add_root_model_ranks)
            or len(set(self.add_root_model_ranks)) != len(self.add_root_model_ranks)
        ):
            raise EditorError("beam add ranks must be distinct positive model ranks")
        for rank_range in self.skip_root_rank_ranges:
            if (
                not isinstance(rank_range, tuple)
                or len(rank_range) != 2
                or type(rank_range[0]) is not int
                or type(rank_range[1]) is not int
                or rank_range[0] < 1
                or rank_range[1] < rank_range[0]
            ):
                raise EditorError("beam root skip ranks must be positive ascending ranges")
        self.base_visible = list(engine.visible_token_ids)
        self.base_prefix = list(engine.token_ids)
        self.shared_context = engine.backend.render(self.base_prefix, special=True)
        self.selected_outcomes: tuple[ActionOutcome, ...] = ()
        self.active: list[BeamPath] = []
        self.finished: list[BeamPath] = []
        self._history: list[_Checkpoint] = []
        self._killed_paths: set[tuple[int, ...]] = set()
        self._protected_prefixes: set[tuple[int, ...]] = set()
        self.show_family_metadata = False
        self._next_label = 0
        self._primary_batch: BatchedInferenceSession | None = None
        self._guidance_batch: BatchedInferenceSession | None = None
        self._context_tail = _recent_context(self.shared_context)
        self.selected_label: str | None = None
        self.closed = False
        self._root_skip_token_mask: np.ndarray | None = None
        self._root_add_token_ids: set[int] = set()

        shared_observation = engine.observe()
        vocabulary_size = len(shared_observation.policy_calculations.adjusted)
        if any(last > vocabulary_size for _, last in self.skip_root_rank_ranges):
            raise EditorError(
                f"beam root skip ranks must be between 1 and {vocabulary_size}"
            )
        if any(rank > vocabulary_size for rank in self.add_root_model_ranks):
            raise EditorError(
                f"beam add ranks must be between 1 and {vocabulary_size}"
            )
        skipped_ranks = {
            rank
            for first, last in self.skip_root_rank_ranges
            for rank in range(first, last + 1)
        }
        if skipped_ranks.intersection(self.add_root_model_ranks):
            raise EditorError("a beam model rank cannot be both skipped and added")
        requested_ranks = [
            *(last for _, last in self.skip_root_rank_ranges),
            *self.add_root_model_ranks,
        ]
        if requested_ranks:
            highest_rank = max(requested_ranks)
            top_ids = shared_observation.policy_calculations.top_raw_ids(
                highest_rank
            )
            rank_to_id = {
                rank: int(token_id)
                for rank, token_id in enumerate(top_ids, start=1)
            }
        else:
            rank_to_id = {}
        if self.skip_root_rank_ranges:
            skipped = np.zeros(vocabulary_size, dtype=bool)
            for first, last in self.skip_root_rank_ranges:
                skipped[np.asarray(
                    [rank_to_id[rank] for rank in range(first, last + 1)],
                    dtype=np.int64,
                )] = True
            if np.all(skipped):
                raise EditorError("beam root skip ranks cannot exclude the whole vocabulary")
            self._root_skip_token_mask = skipped
        if self.add_root_model_ranks:
            self._root_add_token_ids = {
                rank_to_id[rank] for rank in self.add_root_model_ranks
            }
            eog_ids = set(engine.backend.eog_token_ids())
            if self._root_add_token_ids.intersection(eog_ids):
                raise EditorError("beam add cannot force an EOG root that has no continuation")
            self._protected_prefixes = {
                (token_id,) for token_id in self._root_add_token_ids
            }
        try:
            primary_factory = getattr(engine.backend, "create_batch", None)
            guidance_active = engine._cfg_active()
            guidance_factory = (
                getattr(engine.guidance_backend, "create_batch", None)
                if guidance_active and engine.guidance_backend is not None
                else None
            )
            if callable(primary_factory) and (
                not guidance_active or callable(guidance_factory)
            ):
                self._primary_batch = primary_factory(
                    [self.base_prefix for _ in range(width)]
                )
                if guidance_active:
                    guidance_prefix = [
                        *engine._guidance_prompt_tokens(), *self.base_visible
                    ]
                    try:
                        self._guidance_batch = guidance_factory(
                            [guidance_prefix for _ in range(width)]
                        )
                    except BaseException:
                        self._primary_batch.close()
                        self._primary_batch = None
                        raise
            root_engine = self._copy_engine(
                engine,
                backend=(
                    self._primary_batch.lane(0)
                    if self._primary_batch is not None else engine.backend
                ),
                guidance_backend=(
                    self._guidance_batch.lane(0)
                    if self._guidance_batch is not None else engine.guidance_backend
                ),
                observation=shared_observation,
            )
            self.active = [BeamPath(
                "root", root_engine, None, 0.0, 0.0, 0.0, None, None,
                0 if self._primary_batch else None,
            )]
            self.expand()
            self._retain_selection()
        except BaseException:
            self.discard()
            raise

    @staticmethod
    def _copy_engine(
        source: EpisodeEngine,
        *,
        backend,
        guidance_backend,
        observation: EpisodeObservation | None,
    ) -> EpisodeEngine:
        preview = copy.copy(source)
        preview.backend = backend
        preview.guidance_backend = guidance_backend
        preview.trajectory = copy.copy(source.trajectory)
        preview.trajectory.visible_token_ids = list(source.visible_token_ids)
        preview._prefix_snapshot = source._observation_prefix_snapshot()
        preview._prefix_snapshot_boundary = source.boundary
        preview._prefix_snapshot_dirty = False
        preview._ephemeral_logit_biases = dict(source._ephemeral_logit_biases)
        preview._token_boundaries = dict(source._token_boundaries)
        preview._observation = observation
        preview._observation_key = preview._decision_key() if observation is not None else None
        preview._prepared_accept = None
        preview._speculative_accept_prefix = None
        preview._metric_sink = None
        return preview

    @staticmethod
    def _branch_label(index: int) -> str:
        """Return a stable ID distinct from leaderboard and model ranks."""
        return f"b{index + 1}"

    def _new_label(self) -> str:
        label = self._branch_label(self._next_label)
        self._next_label += 1
        return label

    @staticmethod
    def _path_token_ids(path: BeamPath) -> tuple[int, ...]:
        tokens: list[int] = []
        node = path.node
        while node is not None:
            tokens.append(node.token_id)
            node = node.parent
        return tuple(reversed(tokens))

    def _candidates(self) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for parent_order, path in enumerate(self.active):
            observation = path.engine.observe()
            adjusted = np.asarray(
                observation.policy_calculations.adjusted, dtype=np.float64
            )
            if adjusted.ndim != 1 or not np.all(np.isfinite(adjusted)):
                raise RuntimeError("beam policy logits are invalid")
            log_normalizer = float(np.logaddexp.reduce(adjusted))
            model_log_probabilities = adjusted - log_normalizer
            ids = np.arange(len(adjusted), dtype=np.int64)
            allowed_ids = ids
            is_root = path.node is None
            if is_root and self._root_skip_token_mask is not None:
                # This local support mask is a one-step -infinity bias; it does
                # not alter the observation or the vocabulary at descendants.
                allowed_ids = ids[~self._root_skip_token_mask]
                if not len(allowed_ids):
                    raise EditorError("beam root skip ranks leave no first-token candidates")
            eog_ids = set(path.engine.backend.eog_token_ids())
            path_tokens = self._path_token_ids(path)
            is_eog = np.isin(ids, tuple(eog_ids)) if eog_ids else np.zeros(ids.shape, dtype=bool)

            order = allowed_ids[np.lexsort((
                allowed_ids, -model_log_probabilities[allowed_ids]
            ))]

            live_count = 0
            live_candidate_ids: set[int] = set()
            for token_id in order:
                token_id = int(token_id)
                log_probability = float(model_log_probabilities[token_id])
                if (
                    is_eog[token_id]
                    or (*path_tokens, token_id) in self._killed_paths
                ):
                    continue
                candidates.append(self._candidate(
                    path,
                    observation,
                    token_id,
                    log_probability,
                    False,
                    parent_order,
                    search_step_log_probability=float(
                        search_log_probabilities[
                            int(np.searchsorted(allowed_ids, token_id))
                        ]
                    ),
                ))
                live_candidate_ids.add(token_id)
                live_count += 1
                if live_count == self.width:
                    break

            if is_root and self._root_add_token_ids:
                for token_id in sorted(self._root_add_token_ids - live_candidate_ids):
                    if (*path_tokens, token_id) in self._killed_paths:
                        continue
                    candidates.append(self._candidate(
                        path,
                        observation,
                        token_id,
                        float(model_log_probabilities[token_id]),
                        False,
                        parent_order,
                        search_step_log_probability=float(
                            search_log_probabilities[
                                int(np.searchsorted(allowed_ids, token_id))
                            ]
                        ),
                    ))

            # Keep terminal candidates available even when their log-probability
            # is below this parent's top-width live continuations.
            for token_id in allowed_ids[is_eog[allowed_ids]]:
                token_id = int(token_id)
                log_probability = float(model_log_probabilities[token_id])
                if (*path_tokens, token_id) in self._killed_paths:
                    continue
                candidates.append(self._candidate(
                    path,
                    observation,
                    token_id,
                    log_probability,
                    True,
                    parent_order,
                    search_step_log_probability=float(
                        search_log_probabilities[
                            int(np.searchsorted(allowed_ids, token_id))
                        ]
                    ),
                ))
        return candidates

    def _candidate(
        self,
        parent: BeamPath,
        observation: EpisodeObservation,
        token_id: int,
        log_probability: float,
        is_eog: bool,
        parent_order: int,
        *,
        search_step_log_probability: float | None = None,
        score: float | None = None,
    ) -> _Candidate:
        model_rank = observation.policy_calculations.raw_rank(token_id)
        model_log_probability = parent.model_log_probability + log_probability
        search_step_log_probability = (
            log_probability
            if search_step_log_probability is None
            else search_step_log_probability
        )
        cumulative_search_log_probability = (
            parent.search_log_probability + search_step_log_probability
        )
        return _Candidate(
            parent=parent,
            observation=observation,
            token_id=token_id,
            model_rank=model_rank,
            log_probability=log_probability,
            model_log_probability=model_log_probability,
            search_log_probability=cumulative_search_log_probability,
            score=(parent.score + log_probability if score is None else score),
            is_eog=is_eog,
            parent_order=parent_order,
        )

    def expand(self) -> bool:
        """Advance every retained live path by one token and prune globally."""
        if self.closed:
            raise EditorError("beam search is already closed")
        if not self.active:
            return False
        self._history.append(_Checkpoint(
            tuple(self.active), tuple(self.finished),
            tuple(sorted(self._protected_prefixes)),
        ))
        return self._expand_one()

    def advance(self, steps: int) -> bool:
        """Expand up to ``steps`` times, recording one rewind checkpoint."""
        if self.closed:
            raise EditorError("beam search is already closed")
        if type(steps) is not int or not 1 <= steps <= 256:
            raise EditorError("beam advance must be between 1 and 256 steps")
        if not self.active:
            return False
        self._history.append(_Checkpoint(
            tuple(self.active), tuple(self.finished),
            tuple(sorted(self._protected_prefixes)),
        ))
        for _ in range(steps):
            if not self._expand_one():
                break
        return bool(self.active)

    def _expand_one(self) -> bool:
        if not self.active:
            return False
        previous_active = tuple(self.active)
        candidates = self._candidates()
        ordered_live_candidates = sorted(
            (candidate for candidate in candidates if not candidate.is_eog),
            key=lambda candidate: candidate.ordering,
        )
        live_candidates = self._reserve_protected_candidates(ordered_live_candidates)

        finished_pool: list[tuple[tuple, BeamPath | _Candidate]] = []
        for path in self.finished:
            finished_pool.append(((-path.score, 0, path.label), path))
        for candidate in candidates:
            if candidate.is_eog:
                finished_pool.append((
                    (-candidate.score, 1, *candidate.ordering[1:]), candidate
                ))
        finished_pool.sort(key=lambda entry: entry[0])
        kept_finished = [entry[1] for entry in finished_pool[:self.width]]
        new_finished_candidates = [
            item for item in kept_finished if isinstance(item, _Candidate)
        ]

        self._expand_finished(new_finished_candidates)
        self._expand_live(live_candidates)
        kept_existing = [
            item for item in kept_finished if isinstance(item, BeamPath)
        ]
        self.finished = [*kept_existing, *self._new_finished]
        self.finished.sort(key=lambda path: (-path.score, path.label))

        # Old frontier observations own full-vocabulary arrays. They are no
        # longer needed after child outcomes have been resolved.
        for path in previous_active:
            path.engine._invalidate_observation()
        self._retain_selection()
        return bool(self.active)

    def _reserve_protected_candidates(
        self, ordered_candidates: list[_Candidate]
    ) -> list[_Candidate]:
        """Keep one best live child for each pinned prefix, then fill normally."""
        if not self._protected_prefixes:
            return ordered_candidates[:self.width]

        reserved: list[_Candidate] = []
        selected_paths: set[tuple[int, ...]] = set()
        for prefix in sorted(self._protected_prefixes, key=lambda item: (len(item), item)):
            candidate = next((
                item for item in ordered_candidates
                if self._candidate_token_ids(item)[:len(prefix)] == prefix
            ), None)
            if candidate is None:
                continue
            token_path = self._candidate_token_ids(candidate)
            if token_path not in selected_paths:
                reserved.append(candidate)
                selected_paths.add(token_path)

        remaining = self.width - len(reserved)
        if remaining < 0:
            raise EditorError("protected beam families exceed the beam width")
        for candidate in ordered_candidates:
            if len(reserved) >= self.width:
                break
            token_path = self._candidate_token_ids(candidate)
            if token_path in selected_paths:
                continue
            reserved.append(candidate)
            selected_paths.add(token_path)
        return sorted(reserved, key=lambda candidate: candidate.ordering)

    def _candidate_token_ids(self, candidate: _Candidate) -> tuple[int, ...]:
        return (*self._path_token_ids(candidate.parent), candidate.token_id)

    def _expand_finished(self, candidates: list[_Candidate]) -> None:
        self._new_finished: list[BeamPath] = []
        for candidate in candidates:
            if self._primary_batch is None:
                self._position_path(candidate.parent)
                primary = self.original.backend
                guidance = self.original.guidance_backend
            else:
                primary = candidate.parent.engine.backend
                guidance = candidate.parent.engine.guidance_backend
            engine = self._copy_engine(
                candidate.parent.engine,
                backend=primary,
                guidance_backend=guidance,
                observation=candidate.observation,
            )
            pre_terminal_engine = (
                self._copy_engine(
                    candidate.parent.engine,
                    backend=primary,
                    guidance_backend=guidance,
                    observation=candidate.observation,
                )
                if candidate.is_eog else None
            )
            action = SelectRawRank(candidate.model_rank)
            outcome = engine.apply(action)
            node = self._new_node(candidate, action, outcome)
            self._new_finished.append(BeamPath(
                self._new_label(), engine, node, candidate.score,
                candidate.model_log_probability,
                candidate.search_log_probability,
                candidate.model_rank, candidate.log_probability, None,
                pre_terminal_engine,
            ))

    def _expand_live(self, candidates: list[_Candidate]) -> None:
        if not candidates:
            if self._primary_batch is not None:
                self._primary_batch.retire(tuple(range(self.width)))
                if self._guidance_batch is not None:
                    self._guidance_batch.retire(tuple(range(self.width)))
            self.active = []
            return

        if self._primary_batch is not None:
            source_by_destination = {
                lane_id: int(candidate.parent.lane_id)
                for lane_id, candidate in enumerate(candidates)
            }
            self._primary_batch.fork(source_by_destination)
            if self._guidance_batch is not None:
                self._guidance_batch.fork(source_by_destination)

        next_active: list[BeamPath] = []
        for lane_id, candidate in enumerate(candidates):
            if self._primary_batch is None:
                self._position_path(candidate.parent)
                primary = self.original.backend
                guidance = self.original.guidance_backend
            else:
                primary = self._primary_batch.lane(lane_id)
                guidance = (
                    self._guidance_batch.lane(lane_id)
                    if self._guidance_batch is not None else candidate.parent.engine.guidance_backend
                )
            engine = self._copy_engine(
                candidate.parent.engine,
                backend=primary,
                guidance_backend=guidance,
                observation=candidate.observation,
            )
            action = SelectRawRank(candidate.model_rank)
            outcome = engine.apply(action)
            if outcome.terminal_token_id is not None or engine.ended:
                # EOS candidates are handled separately; each full-vocabulary
                # model rank must resolve against this parent's same snapshot.
                raise RuntimeError("live beam candidate unexpectedly resolved to EOG")
            if self._guidance_batch is not None and engine._cfg_active():
                self._guidance_batch.lane(lane_id).eval([candidate.token_id])
            node = self._new_node(candidate, action, outcome)
            next_active.append(BeamPath(
                self._new_label(), engine, node, candidate.score,
                candidate.model_log_probability,
                candidate.search_log_probability,
                candidate.model_rank, candidate.log_probability,
                lane_id if self._primary_batch is not None else None,
            ))

        if self._primary_batch is not None:
            lane_ids = tuple(range(len(next_active)))
            self._primary_batch.flush(lane_ids)
            if self._guidance_batch is not None:
                guided = tuple(
                    path.lane_id for path in next_active
                    if path.engine._cfg_active() and path.lane_id is not None
                )
                self._guidance_batch.flush(guided)
        self.active = next_active

    def _new_node(
        self,
        candidate: _Candidate,
        action: PolicyAction,
        outcome: ActionOutcome,
    ) -> BeamNode:
        node = BeamNode(
            candidate.parent.node,
            action,
            outcome,
            candidate.token_id,
            candidate.model_rank,
            candidate.log_probability,
            candidate.is_eog,
        )
        return node

    def _position_path(self, path: BeamPath) -> None:
        suffix = list(path.engine.visible_token_ids[len(self.base_visible):])
        _position(self.original.backend, self.base_prefix, suffix)
        if path.engine._cfg_active() and self.original.guidance_backend is not None:
            prompt = list(path.engine._guidance_prompt_tokens())
            _position(
                self.original.guidance_backend,
                [*prompt, *self.base_visible], suffix,
            )

    def rewind(self) -> bool:
        if not self._history:
            return False
        checkpoint = self._history.pop()
        self.active = [
            path for path in checkpoint.active
            if self._path_token_ids(path) not in self._killed_paths
        ]
        self.finished = [
            path for path in checkpoint.finished
            if self._path_token_ids(path) not in self._killed_paths
        ]
        self._protected_prefixes = {
            prefix for prefix in checkpoint.protected_prefixes
            if not any(
                len(killed) <= len(prefix)
                and prefix[:len(killed)] == killed
                for killed in self._killed_paths
            )
        }
        if self._primary_batch is not None:
            self._rebuild_batches()
        else:
            for path in self.active:
                path.engine._invalidate_observation()
        self._retain_selection()
        return True

    def _rebuild_batches(self) -> None:
        if self._primary_batch is None:
            return
        primary_ids: list[int] = []
        guidance_ids: list[int] = []
        for lane_id, path in enumerate(self.active):
            path.lane_id = lane_id
            path.engine.backend = self._primary_batch.lane(lane_id)
            primary_prefix = [*path.engine.initial_token_ids, *path.engine.visible_token_ids]
            self._primary_batch.lane(lane_id).reset(primary_prefix)
            primary_ids.append(lane_id)
            if self._guidance_batch is not None:
                path.engine.guidance_backend = self._guidance_batch.lane(lane_id)
                guidance_prefix = [
                    *path.engine._guidance_prompt_tokens(),
                    *path.engine.visible_token_ids,
                ]
                self._guidance_batch.lane(lane_id).reset(guidance_prefix)
                if path.engine._cfg_active():
                    guidance_ids.append(lane_id)
            path.engine._invalidate_observation()
        self._primary_batch.rebuild(primary_ids)
        if self._guidance_batch is not None:
            self._guidance_batch.rebuild(guidance_ids)

    def _find_path(self, label: str) -> BeamPath:
        key = label.strip().lower()
        for path in (*self.active, *self.finished):
            if key == path.label:
                return path
        raise EditorError("select a beam branch by its stable ID")

    def ordered_paths(self) -> list[BeamPath]:
        return sorted(
            (*self.active, *self.finished),
            key=lambda path: (-path.score, path.label),
        )

    def set_selection(self, label: str | None) -> str | None:
        if label is not None and any(
            path.label == label for path in (*self.active, *self.finished)
        ):
            self.selected_label = label
        else:
            self._retain_selection()
        return self.selected_label

    def _retain_selection(self) -> None:
        paths = self.ordered_paths()
        if not any(path.label == self.selected_label for path in paths):
            self.selected_label = paths[0].label if paths else None

    @staticmethod
    def _safe_text(text: str) -> str:
        return "".join(
            char if char == "\n" or char.isprintable()
            else "    " if char == "\t"
            else f"\\x{ord(char):02x}"
            for char in text
        )

    def view_state(self, *, notice: str = "", at_edge: bool = False) -> BeamViewState:
        self._retain_selection()
        paths = self.ordered_paths()
        depth = max(
            (len(self._path_actions(path)) for path in paths),
            default=0,
        )
        continuations: dict[str, str] = {}
        roots: dict[str, str] = {}
        root_token_ids: dict[str, int | None] = {}
        for path in paths:
            generated = path.engine.visible_token_ids[len(self.base_visible):]
            continuations[path.label] = self._safe_text(
                path.engine.backend.render(generated)
            )
            root_node = path.node
            while root_node is not None and root_node.parent is not None:
                root_node = root_node.parent
            if root_node is None:
                roots[path.label] = "—"
                root_token_ids[path.label] = None
            elif root_node.is_eog:
                roots[path.label] = "EOG"
                root_token_ids[path.label] = root_node.token_id
            else:
                root_text = self._safe_text(
                    path.engine.backend.render([root_node.token_id])
                ).replace("\n", " ").strip()
                root_text = root_text or f"token {root_node.token_id}"
                roots[path.label] = (
                    f"r{root_node.model_rank} “{root_text[:16]}”"
                )
                root_token_ids[path.label] = root_node.token_id

        family_metadata: dict[str, str] = {}
        if self.show_family_metadata:
            for path in paths:
                peers = [
                    peer for peer in paths
                    if peer.label != path.label
                    and root_token_ids[peer.label] == root_token_ids[path.label]
                ]
                if not peers:
                    family_metadata[path.label] = f"root {roots[path.label]} · alone"
                    continue
                text = continuations[path.label]
                peer = max(
                    peers,
                    key=lambda candidate: (
                        self._common_prefix_length(
                            text, continuations[candidate.label]
                        )
                    ),
                )
                peer_text = continuations[peer.label]
                shared = self._common_prefix_length(text, peer_text)
                if shared == len(text) == len(peer_text):
                    relation = f"same visible text as {peer.label}"
                elif shared == min(len(text), len(peer_text)):
                    relation = f"shares prefix with {peer.label}"
                elif shared:
                    snippet = " ".join(text[:shared].split())[-24:]
                    relation = f"{peer.label} splits after “…{snippet}”"
                else:
                    relation = f"diverges at start from {peer.label}"
                family_metadata[path.label] = (
                    f"root {roots[path.label]} · {relation}"
                )

        rows: list[BeamViewRow] = []
        for path in paths:
            continuation = continuations[path.label]
            if path.engine.ended:
                continuation = (
                    f"{continuation} [EOG {path.engine.terminal_token_id}]"
                ).strip()
            nodes: list[BeamNode] = []
            node = path.node
            while node is not None:
                nodes.append(node)
                node = node.parent
            recent: list[str] = []
            for node in reversed(nodes[:5]):
                token = (
                    "EOG" if node.is_eog else
                    self._safe_text(path.engine.backend.render([node.token_id]))
                )
                probability = math.exp(node.step_log_probability)
                recent.append(
                    f"“{token}” · model rank {node.model_rank} · "
                    f"model-p {probability:.2f}"
                )
            rows.append(BeamViewRow(
                label=path.label,
                continuation=continuation or "(no visible continuation)",
                score=f"{path.score:.6f}",
                state="EOS" if path.state == "finished" else "LIVE",
                recent_steps=tuple(recent),
                model_rank=path.model_rank,
                step_log_probability=path.step_log_probability,
                model_log_probability=path.model_log_probability,
                protected=(
                    not path.engine.ended
                    and any(
                        self._path_token_ids(path)[:len(prefix)] == prefix
                        for prefix in self._protected_prefixes
                    )
                ),
                family_metadata=family_metadata.get(path.label, ""),
            ))

        title = (
            f"BEAM   width {self.width} · depth {depth} · "
            "score: cumulative model log-p"
        )
        if self.skip_root_rank_ranges:
            skipped = " ".join(
                str(first) if first == last else f"{first}-{last}"
                for first, last in self.skip_root_rank_ranges
            )
            title += f" · root skip model ranks {skipped}"
        if self.add_root_model_ranks:
            added = " ".join(str(rank) for rank in self.add_root_model_ranks)
            title += f" · root add ranks {added}"
        if self._protected_prefixes:
            title += f" · protected {len(self._protected_prefixes)}"
        return BeamViewState(
            title=title,
            shared_context=self._context_tail,
            rows=tuple(rows),
            selected_label=self.selected_label,
            notice=notice,
            at_edge=at_edge,
            show_family_metadata=self.show_family_metadata,
        )

    @staticmethod
    def _common_prefix_length(left: str, right: str) -> int:
        index = 0
        limit = min(len(left), len(right))
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def kill(self, label: str) -> bool:
        """Remove a branch like SIGKILL, without recording an episode action."""
        path = self._find_path(label)
        token_path = self._path_token_ids(path)
        self._protected_prefixes = {
            prefix for prefix in self._protected_prefixes
            if not (
                len(prefix) >= len(token_path)
                and prefix[:len(token_path)] == token_path
            )
        }
        active_index = next(
            (index for index, active in enumerate(self.active) if active is path),
            None,
        )
        if active_index is not None:
            self.active.pop(active_index)
            if path.node is not None:
                self._killed_paths.add(self._path_token_ids(path))
            if self._primary_batch is not None and path.lane_id is not None:
                self._primary_batch.retire((path.lane_id,))
                if self._guidance_batch is not None:
                    self._guidance_batch.retire((path.lane_id,))
        else:
            self.finished.remove(path)
            if path.node is not None:
                self._killed_paths.add(self._path_token_ids(path))
        self._retain_selection()
        if not self.active and not self.finished:
            self.discard()
            return False
        return True

    def toggle_protection(self, label: str | None) -> str:
        """Toggle a one-slot reservation for the selected live lineage."""
        if label is None:
            raise EditorError("select a live branch before protecting it")
        path = self._find_path(label)
        if path.engine.ended:
            raise EditorError("only live branches can be protected")
        token_path = self._path_token_ids(path)
        if not token_path:
            raise EditorError("select a generated live branch before protecting it")

        covering = sorted(
            (
                prefix for prefix in self._protected_prefixes
                if token_path[:len(prefix)] == prefix
            ),
            key=lambda prefix: (len(prefix), prefix),
            reverse=True,
        )
        if covering:
            self._protected_prefixes.remove(covering[0])
            return f"Removed protection from lineage at {label}."

        descendants = {
            prefix for prefix in self._protected_prefixes
            if prefix[:len(token_path)] == token_path
        }
        if len(self._protected_prefixes - descendants) + 1 > self.width:
            raise EditorError("protected beam families cannot exceed the beam width")
        self._protected_prefixes.difference_update(descendants)
        self._protected_prefixes.add(token_path)
        return f"Protected {label}'s lineage with one beam slot."

    def toggle_family_metadata(self) -> str:
        self.show_family_metadata = not self.show_family_metadata
        return (
            "Family metadata shown."
            if self.show_family_metadata else
            "Family metadata hidden."
        )

    def promote(self, label: str) -> tuple[PolicyAction, ...]:
        if self.closed:
            raise EditorError("beam search is already closed")
        path = self._find_path(label)
        terminal = path.engine.ended
        commit_engine = (
            path.pre_terminal_engine if terminal else path.engine
        )
        if commit_engine is None:
            raise EditorError("finished beam is missing its pre-EOG state")
        boundary = len(self.base_visible)
        nodes: list[BeamNode] = []
        node = path.node
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        committed_nodes = nodes[:-1] if terminal else nodes
        if terminal and (not nodes or not nodes[-1].is_eog):
            raise EditorError("finished beam does not end with an EOG action")
        for item in committed_nodes:
            if item.outcome.boundary_before != boundary:
                raise EditorError("beam outcomes do not continue the shared prefix")
            boundary = item.outcome.boundary_after
        if (
            boundary != commit_engine.boundary
            or tuple(commit_engine.visible_token_ids[:len(self.base_visible)])
            != tuple(self.base_visible)
        ):
            raise EditorError("beam outcomes do not match the selected branch boundary")

        suffix = list(commit_engine.visible_token_ids[len(self.base_visible):])
        _position(self.original.backend, self.base_prefix, suffix)
        if self.original._cfg_active() and self.original.guidance_backend is not None:
            prompt = list(self.original._guidance_prompt_tokens())
            _position(
                self.original.guidance_backend,
                [*prompt, *self.base_visible], suffix,
            )
        commit_engine.backend = self.original.backend
        commit_engine.guidance_backend = self.original.guidance_backend
        commit_engine._invalidate_observation()
        commit_engine._invalidate_guidance()
        if not commit_engine.ended:
            commit_engine.observe()
        self._close_batches()
        self.original.adopt_preview_state(commit_engine)
        self.selected_outcomes = tuple(item.outcome for item in committed_nodes)
        self.closed = True
        return tuple(item.action for item in committed_nodes)

    def select(self, label: str, *, promote: bool = False) -> tuple[PolicyAction, ...]:
        if promote:
            return self.promote(label)
        path = self._find_path(label)
        actions = self._path_actions(path)
        self.discard()
        return actions

    @staticmethod
    def _path_actions(path: BeamPath) -> tuple[PolicyAction, ...]:
        nodes: list[BeamNode] = []
        node = path.node
        while node is not None:
            nodes.append(node)
            node = node.parent
        ordered = list(reversed(nodes))
        if ordered and ordered[-1].is_eog:
            ordered.pop()
        return tuple(item.action for item in ordered)

    def discard(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._primary_batch is None:
            _position(self.original.backend, self.base_prefix, [])
            if self.original._cfg_active() and self.original.guidance_backend is not None:
                prompt = list(self.original._guidance_prompt_tokens())
                _position(
                    self.original.guidance_backend,
                    [*prompt, *self.base_visible], [],
                )
        self.original._invalidate_observation()
        self.original._invalidate_guidance()
        self._close_batches()

    def _close_batches(self) -> None:
        if self._primary_batch is not None:
            self._primary_batch.close()
            self._primary_batch = None
        if self._guidance_batch is not None:
            self._guidance_batch.close()
            self._guidance_batch = None

def beam_menu(
    io,
    beam: BeamSearch,
    *,
    at_edge: bool = False,
    promote_on_select: bool = False,
) -> tuple[str, tuple[PolicyAction, ...] | None]:
    notice = ""

    def advance_at_cursor(expand) -> bool:
        """Keep the highlighted leaderboard row stable as the frontier changes."""
        paths = beam.ordered_paths()
        cursor_row = next(
            (index for index, path in enumerate(paths)
             if path.label == beam.selected_label),
            0,
        )
        has_live_paths = expand()
        paths = beam.ordered_paths()
        if paths:
            beam.set_selection(paths[min(cursor_row, len(paths) - 1)].label)
        return has_live_paths

    def kill_path(label: str) -> bool:
        """Keep the highlight in place when pruning its selected branch."""
        paths = beam.ordered_paths()
        cursor_row = next(
            (index for index, path in enumerate(paths)
             if path.label == beam.selected_label),
            0,
        )
        was_selected = label == beam.selected_label
        if not beam.kill(label):
            return False
        paths = beam.ordered_paths()
        if was_selected and paths:
            beam.set_selection(paths[min(cursor_row, len(paths) - 1)].label)
        return True

    while True:
        response = io.read_beam(beam.view_state(notice=notice, at_edge=at_edge))
        notice = ""
        if response is None:
            return "edge", None
        command = response.command.strip().lower()
        beam.set_selection(response.selected_label)
        if at_edge:
            if command in {"return", "esc"}:
                beam.discard()
                return "discard", None
            if command in {"c", "continue", "resume"}:
                at_edge = False
                continue
            if command == "discard":
                beam.discard()
                return "discard", None
            if command == "q":
                beam.discard()
                return "quit", None
        else:
            if command in {"return", "esc"}:
                beam.discard()
                return "discard", None
            if command in {"p", "protect"}:
                try:
                    notice = beam.toggle_protection(beam.selected_label)
                except EditorError as exc:
                    notice = str(exc)
                continue
            if command in {"f", "families"}:
                notice = beam.toggle_family_metadata()
                continue
            if command in {"", "]"}:
                if not advance_at_cursor(beam.expand):
                    notice = "No live branches remain; choose an EOS branch or open Beam EDGE."
                continue
            if command == "k":
                if beam.selected_label is None:
                    notice = "Select a branch before killing it."
                    continue
                try:
                    if not kill_path(beam.selected_label):
                        return "discard", None
                except EditorError as exc:
                    notice = str(exc)
                continue
            if command.startswith(("advance ", "hold ", "a ")):
                _, _, count = command.partition(" ")
                try:
                    if not count.strip().isdigit():
                        raise EditorError("use `advance N` with a step count from 1 to 256")
                    if not advance_at_cursor(
                        lambda: beam.advance(int(count.strip()))
                    ):
                        notice = "No live branches remain; choose an EOS branch or open Beam EDGE."
                except EditorError as exc:
                    notice = str(exc)
                continue
            if command in {"advance", "hold", "a"}:
                notice = "Use `advance N` or `hold N` with a step count from 1 to 256."
                continue
            if command in {"rewind", "r", "["}:
                if not beam.rewind():
                    notice = "No beam expansion to rewind."
                continue
            if command == "q":
                at_edge = True
                continue
            if command.startswith("kill "):
                try:
                    if not kill_path(command.split(maxsplit=1)[1]):
                        return "discard", None
                except (EditorError, IndexError) as exc:
                    notice = str(exc)
                continue
            if command in {"?", "help"}:
                ranking = (
                    "Ranks by cumulative policy log-p; sampler temperature, "
                    "filters, and draw noise are ignored. "
                )
                notice = (
                    "Enter/ ] expands one token; " + ranking
                    + "`advance N` steps and `rewind` undo them. `kill ID` removes a branch; "
                    "`k` kills the selected branch. Select a branch to commit; EOS commits text "
                    "before the EOG token; q opens options."
                )
                notice += " `p` toggles a reserved slot for the selected lineage."
                notice += " `f` toggles root and visible-text family details."
                continue
            target = command[7:].strip() if command.startswith("select ") else command
            if command == "select":
                target = beam.selected_label or ""
            try:
                if target:
                    return "select", beam.select(target, promote=promote_on_select)
            except EditorError as exc:
                notice = str(exc)
                continue
        notice = "Resolve the beam before changing the episode."
