"""Persistence-free live sessions and lightweight branch records.

Live forks form a branch tree.  The session owns one active engine/backend and
keeps a compact, canonical record for every retained branch. A cache snapshot
is optional acceleration only; a branch's action prefix is its identity and
always suffices for reconstruction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .core.actions import Accept, PolicyAction, SetSampler, sampler_after_action
from .core.errors import EditorError
from .core.results import ActionOutcome, ReplayExpectation
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine
from .episode_live_history import truncate_live_history
from .run_loop import TapeStep
from .fresh_episode import fresh_root_from


@dataclass(frozen=True)
class BranchIdentity:
    """Stable identity and lineage of one in-memory branch."""

    branch_id: str
    parent_id: str | None = None
    fork_boundary: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, str) or not self.branch_id:
            raise EditorError("branch id must be a nonempty string")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id
        ):
            raise EditorError("branch parent id must be a nonempty string or null")
        if self.fork_boundary is not None and (
            type(self.fork_boundary) is not int or self.fork_boundary < 0
        ):
            raise EditorError("fork boundary must be a nonnegative integer or null")


@dataclass(frozen=True)
class BranchNode:
    """Identity node retained for compatible branch-tree presentation."""

    identity: BranchIdentity
    inherited_tape: tuple[TapeStep, ...] = ()
    inherited_outcomes: tuple[ActionOutcome, ...] = ()


class BranchTree:
    """Small identity tree shared by every branch handle in a session."""

    def __init__(self, root: BranchNode) -> None:
        self._nodes: dict[str, BranchNode] = {root.identity.branch_id: root}

    @property
    def nodes(self) -> Mapping[str, BranchNode]:
        return MappingProxyType(dict(self._nodes))

    def node(self, branch_id: str) -> BranchNode:
        try:
            return self._nodes[branch_id]
        except KeyError as exc:
            raise EditorError(f"unknown branch {branch_id!r}") from exc

    def add(self, node: BranchNode) -> None:
        identity = node.identity
        if identity.branch_id in self._nodes:
            raise EditorError(f"branch {identity.branch_id!r} already exists")
        if identity.parent_id is None or identity.parent_id not in self._nodes:
            raise EditorError("a fork branch must name an existing parent")
        self._nodes[identity.branch_id] = node


@dataclass(frozen=True)
class _BackendCacheSnapshot:
    """Opaque in-memory cache data associated with one exact token prefix."""

    prefix_token_ids: tuple[int, ...]
    value: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class BranchState:
    """Canonical root-relative state needed to reactivate one live branch."""

    identity: BranchIdentity
    initial_token_ids: tuple[int, ...]
    initial_sampling: SamplerConfig
    stream_fingerprint: str | None
    visible_token_ids: tuple[int, ...]
    tape: tuple[TapeStep, ...]
    outcomes: tuple[ActionOutcome, ...]
    local_action_start: int = 0
    terminal_token_id: int | None = None
    terminal_reason: str | None = None
    status: str = "open"
    backend_cache_snapshot: _BackendCacheSnapshot | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.initial_token_ids:
            raise EditorError("branch state requires initial token ids")
        if not isinstance(self.initial_sampling, SamplerConfig):
            raise EditorError("branch state requires its initial sampler configuration")
        if self.stream_fingerprint is not None and not isinstance(self.stream_fingerprint, str):
            raise EditorError("branch stream fingerprint must be a string or null")
        if len(self.tape) != len(self.outcomes):
            raise EditorError("branch tape and outcomes must align")
        if not 0 <= self.local_action_start <= len(self.tape):
            raise EditorError("branch local action offset is invalid")
        previous_boundary = 0
        for outcome in self.outcomes:
            if outcome.boundary_before != previous_boundary:
                raise EditorError("branch outcomes must have contiguous root boundaries")
            if outcome.boundary_after < outcome.boundary_before:
                raise EditorError("branch outcome boundaries are invalid")
            previous_boundary = outcome.boundary_after
        if previous_boundary != len(self.visible_token_ids):
            raise EditorError("branch outcomes do not describe the visible prefix")

    @property
    def prefix_token_ids(self) -> tuple[int, ...]:
        return (*self.initial_token_ids, *self.visible_token_ids)

    @property
    def boundary(self) -> int:
        return len(self.visible_token_ids)

    @property
    def history_tape(self) -> tuple[TapeStep, ...]:
        return self.tape

    @property
    def history_outcomes(self) -> tuple[ActionOutcome, ...]:
        return self.outcomes

    @property
    def local_tape(self) -> tuple[TapeStep, ...]:
        return self.tape[self.local_action_start :]

    @property
    def local_outcomes(self) -> tuple[ActionOutcome, ...]:
        return self.outcomes[self.local_action_start :]


@dataclass(frozen=True)
class RewindState:
    boundary: int
    discarded_tape: tuple[TapeStep, ...]
    discarded_outcomes: tuple[ActionOutcome, ...]


@dataclass(frozen=True)
class ForkState:
    parent: BranchIdentity
    child: BranchIdentity
    boundary: int


def _sampler_after_tape(
    initial: SamplerConfig, tape: tuple[TapeStep, ...]
) -> SamplerConfig:
    sampling = initial
    for step in tape:
        sampling = sampler_after_action(sampling, step.action)
    return sampling


class LiveSession:
    """One active engine plus lightweight, independently reactivatable branches."""

    def __init__(
        self,
        engine: EpisodeEngine,
        *,
        prompt: str | None = None,
        environment_stamp: Mapping[str, Any] | None = None,
        branch_id: str | None = None,
        initial_state: BranchState | None = None,
    ) -> None:
        if not isinstance(engine, EpisodeEngine):
            raise TypeError("engine must be an EpisodeEngine")
        self._engine = engine
        self._backend = engine.backend
        self._guidance_backend = engine.guidance_backend
        self.prompt = engine.initial_text if prompt is None else prompt
        if not isinstance(self.prompt, str):
            raise EditorError("prompt must be a string")
        environment = dict(environment_stamp or {})
        environment.pop("sampler", None)
        self.environment_stamp = MappingProxyType(environment)
        self.session_id = f"live-session-{uuid4().hex}"
        if initial_state is None:
            identity = BranchIdentity(branch_id or self._new_branch_id())
            root = BranchState(
                identity=identity,
                initial_token_ids=tuple(engine.initial_token_ids),
                initial_sampling=engine.sampling,
                stream_fingerprint=engine.stream_fingerprint,
                visible_token_ids=tuple(engine.visible_token_ids),
                tape=(),
                outcomes=(),
            )
        else:
            if branch_id is not None:
                raise ValueError("branch_id cannot be combined with initial_state")
            if initial_state.initial_token_ids != tuple(engine.initial_token_ids):
                raise ValueError("initial state token root does not match the engine")
            if initial_state.visible_token_ids != tuple(engine.visible_token_ids):
                raise ValueError("initial state visible prefix does not match the engine")
            root = initial_state
            identity = root.identity
        self._branches: dict[str, BranchState] = {identity.branch_id: root}
        self._tree = BranchTree(
            BranchNode(identity, root.history_tape, root.history_outcomes)
        )
        self._active_id = identity.branch_id
        self._active_identity = identity
        self._detached = False
        self._rewinds: dict[str, RewindState | None] = {identity.branch_id: None}
        self._forks: dict[str, ForkState | None] = {identity.branch_id: None}
        self._discarded = False

    @staticmethod
    def _new_branch_id() -> str:
        return f"live-{uuid4().hex}"

    def _require_session(self) -> None:
        if self._discarded:
            raise EditorError("the live session was discarded")

    def _state(self, branch_id: str | None = None) -> BranchState:
        self._require_session()
        identifier = self._active_id if branch_id is None else branch_id
        try:
            return self._branches[identifier]
        except KeyError as exc:
            raise EditorError(f"unknown live branch {identifier!r}") from exc

    def _require_live_branch(self, branch_id: str | None = None) -> None:
        state = self._state(branch_id)
        if state.status == "quit":
            raise EditorError("the live branch has been quit")
        if state.terminal_reason is not None:
            raise EditorError("the live branch has ended")

    def _capture_cache(self, prefix: tuple[int, ...]) -> _BackendCacheSnapshot | None:
        """Use an optional backend cache API without making it semantic state."""
        for name in ("snapshot_cache", "capture_cache_snapshot", "snapshot_session_cache"):
            capture = getattr(self._backend, name, None)
            if not callable(capture):
                continue
            try:
                return _BackendCacheSnapshot(prefix, capture())
            except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
                return None
        return None

    def _restore_cache(self, snapshot: _BackendCacheSnapshot | None, prefix: tuple[int, ...]) -> bool:
        if snapshot is None or snapshot.prefix_token_ids != prefix:
            return False
        for name in ("restore_cache", "restore_cache_snapshot", "restore_session_cache"):
            restore = getattr(self._backend, name, None)
            if not callable(restore):
                continue
            try:
                return restore(snapshot.value) is not False
            except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
                return False
        return False

    def _capture_active(self, *, capture_cache: bool = False) -> BranchState:
        """Freeze the active engine's token and terminal state into its record."""
        self._require_session()
        if self._detached:
            raise EditorError("the live session is detached from its shared backend")
        state = self._branches[self._active_id]
        status = state.status
        if self._engine.ended and status == "open":
            status = "completed"
        snapshot = state.backend_cache_snapshot
        prefix = (*state.initial_token_ids, *self._engine.visible_token_ids)
        if capture_cache:
            snapshot = self._capture_cache(prefix)
        captured = replace(
            state,
            visible_token_ids=tuple(self._engine.visible_token_ids),
            terminal_token_id=self._engine.terminal_token_id,
            terminal_reason=self._engine.terminal_reason,
            status=status,
            backend_cache_snapshot=snapshot,
        )
        self._branches[self._active_id] = captured
        return captured

    def _activate(self, branch_id: str) -> EpisodeEngine:
        self._require_session()
        if branch_id == self._active_id and not self._detached:
            return self._engine
        if not self._detached:
            self._capture_active(capture_cache=True)
            self._engine._invalidate_guidance()
        state = self._state(branch_id)
        prefix = state.prefix_token_ids
        self._restore_cache(state.backend_cache_snapshot, prefix)
        engine = EpisodeEngine(
            self._backend,
            sampling=_sampler_after_tape(state.initial_sampling, state.tape),
            initial_text=self.prompt,
            initial_token_ids=state.initial_token_ids,
            stream_fingerprint=state.stream_fingerprint,
            guidance_backend=self._guidance_backend,
        )
        engine.visible_token_ids = list(state.visible_token_ids)
        engine.terminal_token_id = state.terminal_token_id
        engine.terminal_reason = state.terminal_reason
        self._engine = engine
        self._active_id = branch_id
        self._active_identity = state.identity
        self._detached = False
        return engine

    def activate(self, branch_id: str) -> EpisodeEngine:
        """Reactivate a retained branch on this session's one backend."""
        return self._activate(branch_id)

    def branch_state(self, branch_id: str | None = None) -> BranchState:
        identifier = self._active_id if branch_id is None else branch_id
        if not self._discarded and not self._detached and identifier == self._active_id:
            self._capture_active()
        return self._state(identifier)

    @property
    def branch_states(self) -> Mapping[str, BranchState]:
        if not self._discarded and not self._detached:
            self._capture_active()
        return MappingProxyType(dict(self._branches))

    def branch_handle(self, branch_id: str) -> "LiveBranch":
        return LiveBranch(self, self._state(branch_id).identity)

    @property
    def engine(self) -> EpisodeEngine:
        self._require_session()
        if self._detached:
            raise EditorError("the live session is detached; activate it before using its engine")
        return self._engine

    @property
    def is_detached(self) -> bool:
        return self._detached

    def suspend(self) -> BranchState:
        """Capture this root before its shared backends are reused elsewhere."""

        self._require_session()
        if not self._detached:
            state = self._capture_active(capture_cache=True)
            self._engine._invalidate_guidance()
            self._detached = True
        else:
            state = self._branches[self._active_id]
        return state

    @property
    def branch(self) -> BranchIdentity:
        return self._active_identity if self._discarded else self._state().identity

    @property
    def branch_tree(self) -> BranchTree:
        return self._tree

    @property
    def status(self) -> str:
        return "discarded" if self._discarded else self._state().status

    @property
    def is_discarded(self) -> bool:
        return self._discarded

    @property
    def sampler(self) -> SamplerConfig:
        return self.engine.sampling

    @sampler.setter
    def sampler(self, value: SamplerConfig) -> None:
        self.set_sampler(value)

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.environment_stamp

    def _tape_for(self, branch_id: str, *, local: bool) -> tuple[TapeStep, ...]:
        if self._discarded:
            return ()
        state = self.branch_state(branch_id)
        return state.local_tape if local else state.history_tape

    def _outcomes_for(self, branch_id: str, *, local: bool) -> tuple[ActionOutcome, ...]:
        if self._discarded:
            return ()
        state = self.branch_state(branch_id)
        return state.local_outcomes if local else state.history_outcomes

    @property
    def tape(self) -> tuple[TapeStep, ...]:
        return self._tape_for(self._active_id, local=True)

    @property
    def tape_steps(self) -> tuple[TapeStep, ...]:
        return self.tape

    @property
    def outcomes(self) -> tuple[ActionOutcome, ...]:
        return self._outcomes_for(self._active_id, local=True)

    @property
    def recorded_outcomes(self) -> tuple[ActionOutcome, ...]:
        return self.outcomes

    @property
    def history_tape(self) -> tuple[TapeStep, ...]:
        return self._tape_for(self._active_id, local=False)

    @property
    def history_outcomes(self) -> tuple[ActionOutcome, ...]:
        return self._outcomes_for(self._active_id, local=False)

    @property
    def root_initial_token_ids(self) -> tuple[int, ...]:
        return () if self._discarded else self.branch_state().initial_token_ids

    @property
    def history_visible_token_ids(self) -> tuple[int, ...]:
        return () if self._discarded else self.branch_state().visible_token_ids

    @property
    def rewind_state(self) -> RewindState | None:
        return None if self._discarded else self._rewinds.get(self._active_id)

    @property
    def fork_state(self) -> ForkState | None:
        return None if self._discarded else self._forks.get(self._active_id)

    def set_sampler(
        self,
        sampler: SamplerConfig,
        *,
        _branch_id: str | None = None,
    ) -> None:
        branch_id = self._active_id if _branch_id is None else _branch_id
        self._require_live_branch(branch_id)
        engine = self._activate(branch_id)
        if engine.sampling != sampler:
            self.generate(SetSampler(sampler), _branch_id=branch_id)

    def generate(
        self,
        action: PolicyAction | None = None,
        *,
        expectation: ReplayExpectation | None = None,
        divergence_policy: str = "handoff",
        replay: bool = False,
        _branch_id: str | None = None,
    ) -> ActionOutcome:
        branch_id = self._active_id if _branch_id is None else _branch_id
        self._require_live_branch(branch_id)
        engine = self._activate(branch_id)
        resolved_action = Accept() if action is None else action
        state = self._branches[branch_id]
        outcome = engine.apply(
            resolved_action,
            expectation=expectation,
            divergence_policy=divergence_policy,
            replay=replay,
        )
        expectation_to_record = (
            expectation if replay and expectation is not None else outcome.expectation()
        )
        status = "completed" if engine.ended else state.status
        # Build this replacement atomically: the new prefix and the new action
        # must arrive together so every BranchState stays self-consistent.
        self._branches[branch_id] = replace(
            state,
            visible_token_ids=tuple(engine.visible_token_ids),
            tape=(*state.tape, TapeStep(resolved_action, expectation_to_record)),
            outcomes=(*state.outcomes, outcome),
            terminal_token_id=engine.terminal_token_id,
            terminal_reason=engine.terminal_reason,
            status=status,
            backend_cache_snapshot=None,
        )
        return outcome

    def adopt_promoted_outcomes(
        self, outcomes: tuple[ActionOutcome, ...]
    ) -> None:
        """Record already-applied speculative actions as ordinary live history."""
        self._require_live_branch()
        if not outcomes:
            return
        if self._detached:
            raise EditorError("the live session is detached from its shared backend")
        state = self._branches[self._active_id]
        engine = self._engine
        boundary = state.boundary
        for outcome in outcomes:
            if outcome.boundary_before != boundary:
                raise EditorError("promoted outcomes do not continue the live branch")
            if outcome.status == "handed-off":
                raise EditorError("a handed-off outcome cannot be promoted as live history")
            boundary = outcome.boundary_after
        if boundary != engine.boundary or boundary != len(engine.visible_token_ids):
            raise EditorError("promoted outcomes do not match the active engine prefix")
        if tuple(engine.visible_token_ids[:state.boundary]) != state.visible_token_ids:
            raise EditorError("promoted engine does not share the live branch prefix")

        promoted_tape = tuple(
            TapeStep(item.action, item.expectation()) for item in outcomes
        )
        self._branches[self._active_id] = replace(
            state,
            visible_token_ids=tuple(engine.visible_token_ids),
            tape=(*state.tape, *promoted_tape),
            outcomes=(*state.outcomes, *outcomes),
            terminal_token_id=engine.terminal_token_id,
            terminal_reason=engine.terminal_reason,
            status="completed" if engine.ended else state.status,
            backend_cache_snapshot=None,
        )

    def rewind(self, boundary: int, *, _branch_id: str | None = None) -> RewindState:
        branch_id = self._active_id if _branch_id is None else _branch_id
        self._require_live_branch(branch_id)
        engine = self._activate(branch_id)
        if type(boundary) is not int or boundary < 0 or boundary > engine.boundary:
            raise EditorError(f"rewind boundary must be between 0 and {engine.boundary}")
        state = self._capture_active()
        prefix = truncate_live_history(
            state.tape, state.outcomes, boundary
        )
        engine.rewind_to(boundary)
        retained_tape = prefix.retained_tape
        engine.sampling = _sampler_after_tape(state.initial_sampling, retained_tape)
        engine.trajectory.set_stream_fingerprint(state.stream_fingerprint)
        rewind = RewindState(
            boundary,
            prefix.discarded_tape,
            prefix.discarded_outcomes,
        )
        self._branches[branch_id] = replace(
            state,
            visible_token_ids=tuple(engine.visible_token_ids),
            tape=retained_tape,
            outcomes=prefix.retained_outcomes,
            # Rewinding before a branch's original fork point removes some or
            # all inherited actions.  The first remaining action is then the
            # new local divergence point; keeping the old start index makes
            # the BranchState invalid (and would hide newly generated actions).
            local_action_start=min(state.local_action_start, len(prefix.retained_tape)),
            terminal_token_id=None,
            terminal_reason=None,
            status="open",
            backend_cache_snapshot=None,
        )
        self._rewinds[branch_id] = rewind
        return rewind

    def fork(
        self,
        *,
        boundary: int | None = None,
        branch_id: str | None = None,
        _branch_id: str | None = None,
    ) -> "LiveBranch":
        """Create a child state record on this session's active backend."""
        source_id = self._active_id if _branch_id is None else _branch_id
        self._require_live_branch(source_id)
        engine = self._activate(source_id)
        target = engine.boundary if boundary is None else boundary
        if type(target) is not int or target < 0 or target > engine.boundary:
            raise EditorError(f"fork boundary must be between 0 and {engine.boundary}")
        source = self._capture_active(capture_cache=True)
        prefix = truncate_live_history(
            source.tape, source.outcomes, target
        )
        identity = BranchIdentity(branch_id or self._new_branch_id(), source.identity.branch_id, target)
        child = replace(
            source,
            identity=identity,
            visible_token_ids=source.visible_token_ids[:target],
            tape=prefix.retained_tape,
            outcomes=prefix.retained_outcomes,
            local_action_start=len(prefix.retained_tape),
            terminal_token_id=None,
            terminal_reason=None,
            status="open",
            backend_cache_snapshot=None,
        )
        self._tree.add(BranchNode(identity, child.history_tape, child.history_outcomes))
        self._branches[identity.branch_id] = child
        self._rewinds[identity.branch_id] = None
        fork = ForkState(source.identity, identity, target)
        self._forks[source_id] = fork
        self._forks[identity.branch_id] = fork
        return LiveBranch(self, identity)

    def quit(self, reason: str = "menu-end", *, _branch_id: str | None = None) -> None:
        branch_id = self._active_id if _branch_id is None else _branch_id
        self._require_live_branch(branch_id)
        engine = self._activate(branch_id)
        engine.terminate(reason)
        state = self._capture_active()
        self._branches[branch_id] = replace(state, status="quit", backend_cache_snapshot=None)

    def discard(self) -> None:
        """Forget every branch and invalidate every outstanding branch handle."""
        if self._discarded:
            return
        self._branches.clear()
        self._rewinds.clear()
        self._forks.clear()
        self._discarded = True



class LiveBranch:
    """A lightweight branch-bound view over a shared :class:`LiveSession`."""

    def __init__(self, session: LiveSession, identity: BranchIdentity) -> None:
        self._session = session
        self._identity = identity

    @property
    def session(self) -> LiveSession:
        return self._session

    @property
    def branch(self) -> BranchIdentity:
        return self._identity

    @property
    def branch_tree(self) -> BranchTree:
        return self._session.branch_tree

    @property
    def branch_state(self) -> BranchState:
        return self._session.branch_state(self._identity.branch_id)

    @property
    def engine(self) -> EpisodeEngine:
        return self._session.activate(self._identity.branch_id)

    @property
    def prompt(self) -> str:
        return self._session.prompt

    @property
    def environment_stamp(self) -> Mapping[str, Any]:
        return self._session.environment_stamp

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.environment_stamp

    @property
    def status(self) -> str:
        return "discarded" if self._session.is_discarded else self.branch_state.status

    @property
    def is_discarded(self) -> bool:
        return self._session.is_discarded

    @property
    def sampler(self) -> SamplerConfig:
        return self.engine.sampling

    @sampler.setter
    def sampler(self, value: SamplerConfig) -> None:
        self.set_sampler(value)

    @property
    def tape(self) -> tuple[TapeStep, ...]:
        return self._session._tape_for(self._identity.branch_id, local=True)

    @property
    def tape_steps(self) -> tuple[TapeStep, ...]:
        return self.tape

    @property
    def outcomes(self) -> tuple[ActionOutcome, ...]:
        return self._session._outcomes_for(self._identity.branch_id, local=True)

    @property
    def recorded_outcomes(self) -> tuple[ActionOutcome, ...]:
        return self.outcomes

    @property
    def history_tape(self) -> tuple[TapeStep, ...]:
        return self._session._tape_for(self._identity.branch_id, local=False)

    @property
    def history_outcomes(self) -> tuple[ActionOutcome, ...]:
        return self._session._outcomes_for(self._identity.branch_id, local=False)

    @property
    def root_initial_token_ids(self) -> tuple[int, ...]:
        return () if self._session.is_discarded else self.branch_state.initial_token_ids

    @property
    def history_visible_token_ids(self) -> tuple[int, ...]:
        return () if self._session.is_discarded else self.branch_state.visible_token_ids

    @property
    def rewind_state(self) -> RewindState | None:
        return None if self._session.is_discarded else self._session._rewinds.get(self._identity.branch_id)

    @property
    def fork_state(self) -> ForkState | None:
        return None if self._session.is_discarded else self._session._forks.get(self._identity.branch_id)

    def activate(self) -> EpisodeEngine:
        return self._session.activate(self._identity.branch_id)

    def set_sampler(self, sampler: SamplerConfig) -> None:
        self._session.set_sampler(
            sampler,
            _branch_id=self._identity.branch_id,
        )

    def generate(self, action: PolicyAction | None = None, **kwargs: Any) -> ActionOutcome:
        return self._session.generate(action, _branch_id=self._identity.branch_id, **kwargs)

    def rewind(self, boundary: int) -> RewindState:
        return self._session.rewind(boundary, _branch_id=self._identity.branch_id)

    def fork(self, **kwargs: Any) -> "LiveBranch":
        return self._session.fork(_branch_id=self._identity.branch_id, **kwargs)

    def quit(self, reason: str = "menu-end") -> None:
        self._session.quit(reason, _branch_id=self._identity.branch_id)

    def discard(self) -> None:
        self._session.discard()


@dataclass(frozen=True)
class LiveRosterEntry:
    """One stable global address in a persistence-free live roster."""

    number: int
    session: LiveSession
    branch_id: str

    @property
    def identity(self) -> BranchIdentity:
        return self.session.branch_state(self.branch_id).identity

    @property
    def state(self) -> BranchState:
        return self.session.branch_state(self.branch_id)


class LiveSessionRoster:
    """Keep unrelated live roots and their local branch trees together."""

    def __init__(self, root: LiveSession) -> None:
        if not isinstance(root, LiveSession):
            raise TypeError("root must be a LiveSession")
        self._sessions: dict[str, LiveSession] = {root.session_id: root}
        self._addresses: dict[int, tuple[str, str]] = {}
        self._numbers: dict[tuple[str, str], int] = {}
        self._next_number = 1
        self._active_session_id = root.session_id
        self._register(root, root.branch.branch_id)

    @property
    def active_session(self) -> LiveSession:
        try:
            return self._sessions[self._active_session_id]
        except KeyError as exc:  # pragma: no cover - only possible after misuse
            raise EditorError("the live roster has no active session") from exc

    @property
    def active_branch(self) -> LiveBranch:
        session = self.active_session
        return session.branch_handle(session.branch.branch_id)

    def _register(self, session: LiveSession, branch_id: str) -> int:
        key = (session.session_id, branch_id)
        existing = self._numbers.get(key)
        if existing is not None:
            return existing
        number = self._next_number
        self._next_number += 1
        self._numbers[key] = number
        self._addresses[number] = key
        return number

    def add_session(self, session: LiveSession) -> int:
        """Retain another in-memory root without changing the active one."""
        if not isinstance(session, LiveSession):
            raise TypeError("session must be a LiveSession")
        existing = self._sessions.get(session.session_id)
        if existing is not None and existing is not session:
            raise EditorError("a different session already uses this session id")
        self._sessions[session.session_id] = session
        return self._register(session, session.branch.branch_id)

    def register_branch(self, branch: LiveBranch) -> int:
        return self._register(branch.session, branch.branch.branch_id)

    def number_for(self, session: LiveSession, branch_id: str) -> int:
        return self._register(session, branch_id)

    def entries(self) -> tuple[LiveRosterEntry, ...]:
        return tuple(
            LiveRosterEntry(number, self._sessions[session_id], branch_id)
            for number, (session_id, branch_id) in sorted(self._addresses.items())
        )

    def resolve(self, reference: str | int) -> LiveRosterEntry:
        cleaned = str(reference).strip()
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]
        if cleaned.isdigit() and int(cleaned) in self._addresses:
            number = int(cleaned)
            session_id, branch_id = self._addresses[number]
            return LiveRosterEntry(number, self._sessions[session_id], branch_id)
        for entry in self.entries():
            if entry.branch_id == cleaned:
                return entry
        raise EditorError(f"unknown live branch {reference!r}")

    def switch(self, reference: str | int) -> LiveSession:
        entry = self.resolve(reference)
        current = self.active_session
        if current is not entry.session:
            current.suspend()
        entry.session.activate(entry.branch_id)
        self._active_session_id = entry.session.session_id
        return entry.session

    def fork(self, boundary: int | None = None) -> LiveBranch:
        branch = self.active_session.fork(boundary=boundary)
        self.register_branch(branch)
        return branch

    def new_root(self, prompt: str) -> LiveSession:
        source = self.active_session
        source_engine = source.engine
        environment = dict(source.environment_stamp)
        environment.pop("source_episode_id", None)
        environment.pop("model_change_boundary", None)
        source.suspend()
        try:
            engine = fresh_root_from(source_engine, prompt)
            root = LiveSession(
                engine,
                prompt=prompt,
                environment_stamp=environment,
            )
        except Exception:
            source.activate(source.branch.branch_id)
            raise
        self._sessions[root.session_id] = root
        self._register(root, root.branch.branch_id)
        self._active_session_id = root.session_id
        return root

    def discard(self) -> None:
        for session in tuple(self._sessions.values()):
            session.discard()


__all__ = [
    "BranchIdentity",
    "BranchNode",
    "BranchState",
    "BranchTree",
    "ForkState",
    "LiveBranch",
    "LiveRosterEntry",
    "LiveSession",
    "LiveSessionRoster",
    "RewindState",
]
