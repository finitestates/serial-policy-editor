"""Temporary, storage-neutral previews of several raw-rank continuations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.actions import Accept, PolicyAction, SelectRawRank
from .core.backend import BatchedInferenceSession
from .core.errors import EditorError
from .core.results import ActionOutcome
from .episode_engine import EpisodeEngine, TokenPrefixSnapshot
from .terminal_contracts import PromptRequest
from .teacher_commands import parse_chord


class ChordRequested(Exception):
    def __init__(self, ranks: tuple[int, ...]) -> None:
        super().__init__(ranks)
        self.ranks = ranks


def _position(backend, base: list[int], suffix: list[int]) -> None:
    truncate = getattr(backend, "truncate_to", None)
    if callable(truncate):
        try:
            if truncate(len(base)) is not False:
                if suffix:
                    backend.eval(suffix)
                return
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Cache truncation is an optimization; reconcile the full prefix
            # below when the adapter cannot safely crop this cache instance.
            pass
    branch = getattr(backend, "branch_to_prefix", None)
    if callable(branch):
        branch(base)
        if suffix:
            backend.eval(suffix)
    else:
        backend.reset([*base, *suffix])


def _recent_context(text: str, *, lines: int = 4) -> str:
    """Keep a small logical-line tail without assuming a terminal width."""
    safe = "".join(
        char if char == "\n" or char.isprintable()
        else "    " if char == "\t"
        else f"\\x{ord(char):02x}"
        for char in text
    )
    return "\n".join(safe.split("\n")[-lines:])


@dataclass
class ChordPath:
    label: str
    starting_rank: int
    engine: EpisodeEngine
    actions: list[PolicyAction] = field(default_factory=list)
    outcomes: list[ActionOutcome] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    prefix_snapshots: list[TokenPrefixSnapshot] = field(default_factory=list)

    @property
    def state(self) -> str:
        if self.engine.ended:
            return "EOG"
        return "live"


class Chord:
    """Owns preview engines; the episode engine and histories stay untouched."""

    def __init__(self, engine: EpisodeEngine, ranks: tuple[int, ...]) -> None:
        if engine.ended:
            raise EditorError("chord requires a live decision boundary")
        if engine._speculative_accept_prefix is not None:
            engine.discard_speculative_accept()
            engine._ensure_backend_positioned()
        self.original = engine
        self.base_visible = list(engine.visible_token_ids)
        self.base_prefix = list(engine.token_ids)
        self.shared_context = engine.backend.render(self.base_prefix, special=True)
        shared_prefix_snapshot = engine._observation_prefix_snapshot()
        self.paths: list[ChordPath] = []
        self._active_path: ChordPath | None = None
        self._primary_batch: BatchedInferenceSession | None = None
        self._guidance_batch: BatchedInferenceSession | None = None
        self._context_tail = _recent_context(self.shared_context)
        self.rounds: list[tuple[int, ...]] = []
        # Prepare sampler-driven model controls and one current base
        # observation before adapters create their independent lane caches.
        shared_observation = engine.observe()
        self.selected_outcomes: tuple[ActionOutcome, ...] = ()
        self.closed = False
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
                    [self.base_prefix for _ in ranks]
                )
                if guidance_active:
                    guidance_prefix = [
                        *engine._guidance_prompt_tokens(), *self.base_visible
                    ]
                    try:
                        self._guidance_batch = guidance_factory(
                            [guidance_prefix for _ in ranks]
                        )
                    except BaseException:
                        self._primary_batch.close()
                        self._primary_batch = None
                        raise
            for index, rank in enumerate(ranks):
                preview = EpisodeEngine(
                    (
                        self._primary_batch.lane(index)
                        if self._primary_batch is not None
                        else engine.backend
                    ),
                    sampling=engine.sampling,
                    initial_text=engine.initial_text,
                    initial_token_ids=engine.initial_token_ids,
                    stream_fingerprint=engine.stream_fingerprint,
                    guidance_backend=(
                        self._guidance_batch.lane(index)
                        if self._guidance_batch is not None
                        else engine.guidance_backend
                    ),
                )
                preview.visible_token_ids = self.base_visible
                preview._prefix_snapshot = shared_prefix_snapshot
                preview._prefix_snapshot_boundary = engine.boundary
                preview._prefix_snapshot_dirty = False
                preview._activation_runtime_key = engine._activation_runtime_key
                preview._observation = shared_observation
                preview._observation_key = preview._decision_key()
                path = ChordPath(chr(ord("a") + index), rank, preview)
                path.prefix_snapshots.append(shared_prefix_snapshot)
                self.paths.append(path)
                self._activate(path)
                outcome = preview.apply(SelectRawRank(rank))
                path.actions.append(SelectRawRank(rank))
                path.outcomes.append(outcome)
                path.token_ids.extend(outcome.resolved_token_ids)
                path.prefix_snapshots.append(preview._observation_prefix_snapshot())
            if self._primary_batch is not None:
                self._flush_batches()
        except BaseException:
            self.discard()
            raise

    def _activate(self, path: ChordPath) -> None:
        if self._active_path is path:
            # Leave the current decision lazy until the next display or action.
            return
        self._active_path = None
        if self._primary_batch is not None:
            self._active_path = path
            return
        suffix = list(path.engine.visible_token_ids[len(self.base_visible):])
        _position(self.original.backend, self.base_prefix, suffix)
        if path.engine._cfg_active() and path.engine.guidance_backend is not None:
            prompt = list(path.engine._guidance_prompt_tokens())
            _position(
                path.engine.guidance_backend,
                [*prompt, *self.base_visible], suffix,
            )
        self._active_path = path

    @staticmethod
    def _primary_prefix(path: ChordPath) -> list[int]:
        return [*path.engine.initial_token_ids, *path.engine.visible_token_ids]

    def _flush_batches(self) -> None:
        """Advance all live chord lanes, then make their next observations ready."""
        if self._primary_batch is None:
            return
        live = [index for index, path in enumerate(self.paths) if path.state == "live"]
        for index, path in enumerate(self.paths):
            self._primary_batch.lane(index).branch_to_prefix(
                self._primary_prefix(path)
            )
            if self._guidance_batch is not None:
                guidance_prefix = [
                    *path.engine._guidance_prompt_tokens(),
                    *path.engine.visible_token_ids,
                ]
                self._guidance_batch.lane(index).branch_to_prefix(guidance_prefix)
        self._primary_batch.flush(live)
        if self._guidance_batch is not None:
            guided = [
                index for index, path in enumerate(self.paths)
                if path.state == "live" and path.engine._cfg_active()
            ]
            self._guidance_batch.flush(guided)

    def _rebuild_batches(self) -> None:
        """Re-prefill live lanes after a non-append edit such as rewind."""
        if self._primary_batch is None:
            return
        live = [index for index, path in enumerate(self.paths) if path.state == "live"]
        for index, path in enumerate(self.paths):
            self._primary_batch.lane(index).reset(self._primary_prefix(path))
            if self._guidance_batch is not None:
                self._guidance_batch.lane(index).reset([
                    *path.engine._guidance_prompt_tokens(),
                    *path.engine.visible_token_ids,
                ])
        self._primary_batch.rebuild(live)
        if self._guidance_batch is not None:
            guided = [
                index for index, path in enumerate(self.paths)
                if path.state == "live" and path.engine._cfg_active()
            ]
            self._guidance_batch.rebuild(guided)

    def _close_batches(self) -> None:
        if self._primary_batch is not None:
            self._primary_batch.close()
            self._primary_batch = None
        if self._guidance_batch is not None:
            self._guidance_batch.close()
            self._guidance_batch = None

    def advance(self) -> bool:
        advanced: list[int] = []
        for index, path in enumerate(self.paths):
            if path.state != "live":
                continue
            if self._primary_batch is None:
                self._activate(path)
            outcome = path.engine.apply(Accept())
            path.actions.append(Accept())
            path.outcomes.append(outcome)
            path.token_ids.extend(outcome.resolved_token_ids)
            path.prefix_snapshots.append(path.engine._observation_prefix_snapshot())
            advanced.append(index)
        if advanced:
            self.rounds.append(tuple(advanced))
            if self._primary_batch is not None:
                self._flush_batches()
        return bool(advanced)

    def rewind(self) -> bool:
        if not self.rounds:
            return False
        # Rewind repositions the shared backend outside _activate(). Keep the
        # last rewound live path active and refresh its displayed decision.
        self._active_path = None
        active: ChordPath | None = None
        for index in self.rounds.pop():
            path = self.paths[index]
            path.actions.pop()
            path.outcomes.pop()
            path.token_ids.pop()
            path.prefix_snapshots.pop()
            path.engine.rewind_to(
                len(self.base_visible) + sum(
                    not self.original.backend.is_eog(token_id)
                    for token_id in path.token_ids
                ),
                _defer_backend_positioning=True,
            )
            path.engine._prefix_snapshot = path.prefix_snapshots[-1]
            path.engine._prefix_snapshot_boundary = path.engine.boundary
            path.engine._prefix_snapshot_dirty = False
            active = path
        if self._primary_batch is not None:
            self._rebuild_batches()
        self._active_path = active
        return True

    def _find_path(self, label: str) -> ChordPath:
        key = label.strip().lower()
        for path in self.paths:
            if key in {path.label, str(path.starting_rank)}:
                return path
        raise EditorError("select a chord path by letter or starting raw rank")

    def promote(self, label: str) -> tuple[PolicyAction, ...]:
        """Make one already-generated preview the live engine trajectory."""
        if self.closed:
            raise EditorError("chord is already closed")
        path = self._find_path(label)
        if self._primary_batch is not None:
            if path.state == "live":
                activation_changed = (
                    path.engine._activation_backend_key_for(path.engine.sampling)
                    != path.engine._activation_runtime_key
                )
                if activation_changed:
                    path.engine._prepare_activation_runtime()
                if activation_changed or not path.engine._cached_observation_is_current():
                    self._rebuild_batches()
                path.engine.observe()
        else:
            observation_is_current = (
                path.engine._observation is not None
                and path.engine._observation_key == path.engine._decision_key()
            )
            if self._active_path is path:
                if path.state == "live":
                    path.engine.observe()
            elif path.state == "live" and not observation_is_current:
                # A control edit or rewind invalidated this shared boundary.
                self._activate(path)
        boundary = len(self.base_visible)
        for outcome in path.outcomes:
            if outcome.boundary_before != boundary:
                raise EditorError("chord outcomes do not continue the shared prefix")
            boundary = outcome.boundary_after
        if (
            boundary != path.engine.boundary
            or tuple(path.engine.visible_token_ids[:len(self.base_visible)])
            != tuple(self.base_visible)
        ):
            raise EditorError("chord outcomes do not match the selected preview boundary")
        if self._primary_batch is not None:
            suffix = list(path.engine.visible_token_ids[len(self.base_visible):])
            _position(self.original.backend, self.base_prefix, suffix)
            if self.original._cfg_active() and self.original.guidance_backend is not None:
                prompt = list(self.original._guidance_prompt_tokens())
                _position(
                    self.original.guidance_backend,
                    [*prompt, *self.base_visible],
                    suffix,
                )
            path.engine.backend = self.original.backend
            path.engine.guidance_backend = self.original.guidance_backend
            self._close_batches()
        self.original.adopt_preview_state(path.engine)
        self.selected_outcomes = tuple(path.outcomes)
        self.closed = True
        return tuple(path.actions)

    def select(
        self, label: str, *, promote: bool = False
    ) -> tuple[PolicyAction, ...]:
        if promote:
            return self.promote(label)
        path = self._find_path(label)
        actions = tuple(path.actions)
        self.discard()
        return actions

    def discard(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._primary_batch is None:
            _position(self.original.backend, self.base_prefix, [])
        self.original._invalidate_observation()
        self.original._invalidate_guidance()
        if (
            self._primary_batch is None
            and self.original._cfg_active()
            and self.original.guidance_backend is not None
        ):
            prompt = list(self.original._guidance_prompt_tokens())
            _position(
                self.original.guidance_backend,
                [*prompt, *self.base_visible], [],
            )
        self._close_batches()

    def display(self) -> str:
        rows = []
        for path in self.paths:
            visible = path.engine.visible_token_ids[len(self.base_visible):]
            latest = path.engine.backend.render(visible)
            safe = "".join(
                char if char == "\n" or char.isprintable()
                else "    " if char == "\t"
                else f"\\x{ord(char):02x}"
                for char in latest
            )
            heading = f"{path.label}  rank {path.starting_rank}  {path.state.upper()}"
            rows.append(heading)
            for line in (safe.split("\n") if safe else ["(no visible continuation)"]):
                rows.append("   " + line)
        return f"Shared context (last 4 lines):\n{self._context_tail}\n\nPaths:\n" + "\n".join(rows)


def chord_menu(
    io, chord: Chord, *, at_edge: bool = False, promote_on_select: bool = False
) -> tuple[str, tuple[PolicyAction, ...] | None]:
    notice = ""
    while True:
        body = chord.display()
        if notice:
            body += "\n\n" + notice
            notice = ""
        prompt = (
            "Chord EDGE: c: resume chord | discard: restore episode | q: quit editor | ?: help > "
            if at_edge else
            "Chord: Enter or ]: advance live paths | [: rewind one round | "
            "a–z or starting rank: choose and commit | q: options | ?: help > "
        )
        raw = io.prompt(PromptRequest(prompt, body=body, isolated=True))
        if raw is None:
            return "edge", None
        command = raw.strip().lower()
        if at_edge:
            if command in {"c", "continue", "resume"}:
                at_edge = False
                continue
            if command == "discard":
                chord.discard()
                return "discard", None
            if command == "q":
                chord.discard()
                return "quit", None
        else:
            if command in {"", "]"}:
                if not chord.advance():
                    notice = "All chord paths are locked."
                continue
            if command in {"rewind", "r", "["}:
                if not chord.rewind():
                    notice = "No chord round to rewind."
                continue
            if command == "q":
                at_edge = True
                continue
            try:
                if command in {path.label for path in chord.paths} or command.isdecimal():
                    return "select", chord.select(command, promote=promote_on_select)
            except EditorError as exc:
                notice = str(exc)
                continue
        if command in {"?", "help"}:
            notice = ("Enter or ] advances live paths; [ or rewind undoes one round. "
                      "Choose a letter or starting rank to commit that path's actions "
                      "and drop the other previews. q opens Chord EDGE options: "
                      "c resumes the chord, discard restores the episode, "
                      "and q quits the editor.")
        else:
            notice = "Resolve the chord before changing the episode."
