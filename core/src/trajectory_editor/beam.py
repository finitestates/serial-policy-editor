"""Interactive deterministic and stochastic beams over policy scores."""

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
from .core.sampling import conditional_gumbel_top_k
from .episode_engine import EpisodeEngine
from .terminal_contracts import BeamViewRow, BeamViewState


class BeamRequested(Exception):
    def __init__(
        self,
        width: int,
        *,
        stochastic: bool = False,
    ) -> None:
        super().__init__(width, stochastic)
        self.width = width
        self.stochastic = stochastic


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


class BeamSearch:
    """Maintain a temporary deterministic or Gumbel-Top-k frontier."""

    def __init__(
        self,
        engine: EpisodeEngine,
        width: int = 5,
        *,
        stochastic: bool = False,
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
        self.stochastic = bool(stochastic)
        self.base_visible = list(engine.visible_token_ids)
        self.base_prefix = list(engine.token_ids)
        self.shared_context = engine.backend.render(self.base_prefix, special=True)
        self.selected_outcomes: tuple[ActionOutcome, ...] = ()
        self.active: list[BeamPath] = []
        self.finished: list[BeamPath] = []
        self._history: list[_Checkpoint] = []
        self._killed_paths: set[tuple[int, ...]] = set()
        self._next_label = 0
        self._primary_batch: BatchedInferenceSession | None = None
        self._guidance_batch: BatchedInferenceSession | None = None
        self._context_tail = _recent_context(self.shared_context)
        self.selected_label: str | None = None
        self.closed = False

        shared_observation = engine.observe()
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
                "root", root_engine, None, 0.0, 0.0, None, None,
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
            log_probabilities = adjusted - log_normalizer
            ids = np.arange(len(adjusted), dtype=np.int64)
            eog_ids = set(path.engine.backend.eog_token_ids())
            path_tokens = self._path_token_ids(path)
            is_eog = np.isin(ids, tuple(eog_ids)) if eog_ids else np.zeros(ids.shape, dtype=bool)

            if self.stochastic:
                for token_id, score in conditional_gumbel_top_k(
                    log_probabilities,
                    count=self.width,
                    parent_score=path.score,
                    parent_log_probability=path.model_log_probability,
                    seed=path.engine.sampling.seed,
                    stream_fingerprint=path.engine.stream_fingerprint,
                    aligned_step=observation.sampling_boundary,
                    prefix_token_ids=path_tokens,
                ):
                    if (*path_tokens, token_id) in self._killed_paths:
                        continue
                    log_probability = float(log_probabilities[token_id])
                    candidates.append(self._candidate(
                        path,
                        observation,
                        token_id,
                        log_probability,
                        bool(is_eog[token_id]),
                        parent_order,
                        score=score,
                    ))
                continue

            order = np.lexsort((ids, -log_probabilities))

            live_count = 0
            for index in order:
                token_id = int(ids[index])
                log_probability = float(log_probabilities[index])
                if (
                    is_eog[index]
                    or (*path_tokens, token_id) in self._killed_paths
                ):
                    continue
                candidates.append(self._candidate(
                    path, observation, token_id, log_probability, False, parent_order
                ))
                live_count += 1
                if live_count == self.width:
                    break

            # Keep terminal candidates available even when their log-probability
            # is below this parent's top-width live continuations.
            for index in np.flatnonzero(is_eog):
                token_id = int(ids[index])
                log_probability = float(log_probabilities[index])
                if (*path_tokens, token_id) in self._killed_paths:
                    continue
                candidates.append(self._candidate(
                    path, observation, token_id, log_probability, True, parent_order
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
        score: float | None = None,
    ) -> _Candidate:
        model_rank = observation.policy_calculations.raw_rank(token_id)
        model_log_probability = parent.model_log_probability + log_probability
        return _Candidate(
            parent=parent,
            observation=observation,
            token_id=token_id,
            model_rank=model_rank,
            log_probability=log_probability,
            model_log_probability=model_log_probability,
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
        self._history.append(_Checkpoint(tuple(self.active), tuple(self.finished)))
        return self._expand_one()

    def advance(self, steps: int) -> bool:
        """Expand up to ``steps`` times, recording one rewind checkpoint."""
        if self.closed:
            raise EditorError("beam search is already closed")
        if type(steps) is not int or not 1 <= steps <= 256:
            raise EditorError("beam advance must be between 1 and 256 steps")
        if not self.active:
            return False
        self._history.append(_Checkpoint(tuple(self.active), tuple(self.finished)))
        for _ in range(steps):
            if not self._expand_one():
                break
        return bool(self.active)

    def _expand_one(self) -> bool:
        if self.stochastic:
            return self._expand_stochastic_one()
        if not self.active:
            return False
        previous_active = tuple(self.active)
        candidates = self._candidates()
        live_candidates = sorted(
            (candidate for candidate in candidates if not candidate.is_eog),
            key=lambda candidate: candidate.ordering,
        )[:self.width]

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

    def _expand_stochastic_one(self) -> bool:
        """Keep one Gumbel-ranked frontier of live prefixes and finished leaves."""
        if not self.active:
            return False
        previous_active = tuple(self.active)
        candidates = self._candidates()
        frontier: list[tuple[tuple, BeamPath | _Candidate]] = [
            ((-path.score, 0, path.label), path)
            for path in self.finished
        ]
        frontier.extend(
            (
                (-candidate.score, 1, *candidate.ordering[1:]),
                candidate,
            )
            for candidate in candidates
        )
        frontier.sort(key=lambda entry: entry[0])
        kept = [entry[1] for entry in frontier[:self.width]]

        selected_candidates = [
            item for item in kept if isinstance(item, _Candidate)
        ]
        self._expand_finished([
            candidate for candidate in selected_candidates if candidate.is_eog
        ])
        self._expand_live([
            candidate for candidate in selected_candidates if not candidate.is_eog
        ])
        kept_finished = [item for item in kept if isinstance(item, BeamPath)]
        self.finished = [*kept_finished, *self._new_finished]
        self.finished.sort(key=lambda path: (-path.score, path.label))

        for path in previous_active:
            path.engine._invalidate_observation()
        self._retain_selection()
        return bool(self.active)

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
        rows: list[BeamViewRow] = []
        for path in paths:
            generated = path.engine.visible_token_ids[len(self.base_visible):]
            continuation = self._safe_text(path.engine.backend.render(generated))
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
            ))

        if self.stochastic:
            title = (
                f"BEAM   STOCHASTIC · width {self.width} · depth {depth} · "
                "sequence Gumbel-Top-k"
            )
        else:
            title = (
                f"BEAM   width {self.width} · depth {depth} · "
                "score: cumulative model log-p"
            )
        return BeamViewState(
            title=title,
            shared_context=self._context_tail,
            rows=tuple(rows),
            selected_label=self.selected_label,
            notice=notice,
            at_edge=at_edge,
            stochastic=self.stochastic,
        )

    def kill(self, label: str) -> bool:
        """Remove a branch like SIGKILL, without recording an episode action."""
        path = self._find_path(label)
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
                if beam.stochastic:
                    ranking = (
                        "Gumbel-Top-k samples without replacement; live and EOS share width. "
                        "Uses policy softmax; sampler temperature, filters, and draw settings are ignored. "
                    )
                else:
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
