"""Temporary, storage-neutral previews of several raw-rank continuations."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from .core.actions import Accept, PolicyAction, SelectRawRank
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
    if callable(truncate) and truncate(len(base)) is not False:
        if suffix:
            backend.eval(suffix)
        return
    branch = getattr(backend, "branch_to_prefix", None)
    if callable(branch):
        branch(base)
        if suffix:
            backend.eval(suffix)
    else:
        backend.reset([*base, *suffix])


def _recent_context(text: str, *, width: int, lines: int = 4) -> str:
    """Keep the last visible screen rows of the chord's shared prefix."""
    safe = "".join(
        char if char == "\n" or char.isprintable()
        else "    " if char == "\t"
        else f"\\x{ord(char):02x}"
        for char in text
    )
    rows = [
        row
        for logical_line in safe.split("\n")
        for row in (
            textwrap.wrap(
                logical_line, width=max(1, width),
                break_long_words=True, break_on_hyphens=False,
                replace_whitespace=False, drop_whitespace=False,
            ) or [""]
        )
    ]
    clipped = len(rows) > lines
    tail = rows[-lines:]
    if clipped:
        tail[0] = "…" + tail[0][:max(0, width - 1)]
    return "\n".join(tail)


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
        if self.engine.checkpointed:
            return "budget reached"
        return "live"


class Chord:
    """Owns preview engines; the episode engine and histories stay untouched."""

    def __init__(self, engine: EpisodeEngine, ranks: tuple[int, ...]) -> None:
        if engine.ended or engine.checkpointed:
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
        self._context_cache: tuple[int, str] | None = None
        self.rounds: list[tuple[int, ...]] = []
        # All paths start at one decision; keep its logits stable while cache
        # activation truncates and switches the shared backend prefix.
        shared_observation = (
            engine._observation
            if engine._observation is not None
            and engine._observation_key == engine._decision_key()
            else None
        )
        self.selected_outcomes: tuple[ActionOutcome, ...] = ()
        self.closed = False
        try:
            for index, rank in enumerate(ranks):
                preview = EpisodeEngine(
                    engine.backend,
                    sampling=engine.sampling,
                    max_tokens=engine.max_tokens,
                    initial_text=engine.initial_text,
                    initial_token_ids=engine.initial_token_ids,
                    stream_fingerprint=engine.stream_fingerprint,
                    backend_positioned=True,
                    guidance_backend=engine.guidance_backend,
                )
                preview.visible_token_ids = self.base_visible
                preview._prefix_snapshot = shared_prefix_snapshot
                preview._prefix_snapshot_boundary = engine.boundary
                preview._prefix_snapshot_dirty = False
                preview.checkpoint_boundary = engine.checkpoint_boundary
                preview._activation_runtime_key = engine._activation_runtime_key
                preview._activation_validation_key = engine._activation_validation_key
                if shared_observation is not None:
                    preview._observation = shared_observation
                    preview._observation_key = preview._decision_key()
                path = ChordPath(chr(ord("a") + index), rank, preview)
                path.prefix_snapshots.append(shared_prefix_snapshot)
                self.paths.append(path)
                self._activate(path)
                if shared_observation is None:
                    shared_observation = preview.observe()
                outcome = preview.apply(SelectRawRank(rank))
                path.actions.append(SelectRawRank(rank))
                path.outcomes.append(outcome)
                path.token_ids.extend(outcome.resolved_token_ids)
                path.prefix_snapshots.append(preview._observation_prefix_snapshot())
        except BaseException:
            self.discard()
            raise

    def _activate(self, path: ChordPath) -> None:
        if self._active_path is path:
            # Leave the current decision lazy until the next display or action.
            return
        self._active_path = None
        suffix = list(path.engine.visible_token_ids[len(self.base_visible):])
        _position(self.original.backend, self.base_prefix, suffix)
        path.engine._backend_positioned = True
        if path.engine._cfg_active() and path.engine.guidance_backend is not None:
            prompt = list(path.engine._guidance_prompt_tokens())
            _position(
                path.engine.guidance_backend,
                [*prompt, *self.base_visible], suffix,
            )
            path.engine._guidance_evaluated_prefix = tuple(
                [*prompt, *path.engine.visible_token_ids]
            )
            path.engine.guidance_backend._spe_cfg_owner = path.engine._guidance_owner
        self._active_path = path

    def advance(self) -> bool:
        advanced: list[int] = []
        for index, path in enumerate(self.paths):
            if path.state != "live":
                continue
            self._activate(path)
            outcome = path.engine.apply(Accept())
            path.actions.append(Accept())
            path.outcomes.append(outcome)
            path.token_ids.extend(outcome.resolved_token_ids)
            path.prefix_snapshots.append(path.engine._observation_prefix_snapshot())
            advanced.append(index)
        if advanced:
            self.rounds.append(tuple(advanced))
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
        backend_positioned = self._active_path is path
        observation_is_current = (
            path.engine._observation is not None
            and path.engine._observation_key == path.engine._decision_key()
        )
        if backend_positioned:
            if path.state == "live":
                path.engine.observe()
        elif path.state == "live" and not observation_is_current:
            # Rebuild only if a control edit or rewind invalidated the cached
            # boundary. The ordinary path selection case stays display-only.
            self._activate(path)
            backend_positioned = True
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
        self.original.adopt_preview_state(
            path.engine, backend_positioned=backend_positioned
        )
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
        _position(self.original.backend, self.base_prefix, [])
        self.original._backend_positioned = True
        self.original._invalidate_observation()
        self.original._invalidate_guidance()
        if self.original._cfg_active() and self.original.guidance_backend is not None:
            prompt = list(self.original._guidance_prompt_tokens())
            _position(
                self.original.guidance_backend,
                [*prompt, *self.base_visible], [],
            )
            self.original._guidance_evaluated_prefix = tuple(
                [*prompt, *self.base_visible]
            )
            self.original.guidance_backend._spe_cfg_owner = self.original._guidance_owner

    def display(self, *, width: int = 100) -> str:
        width = max(1, width)
        indent = " " * min(3, width - 1)
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
            rows.extend(textwrap.wrap(heading, width=width, subsequent_indent=indent,
                                      break_long_words=True, break_on_hyphens=False))
            for line in (safe.split("\n") if safe else ["(no visible continuation)"]):
                wrapped = textwrap.wrap(
                    line, width=width - len(indent), break_long_words=True,
                    break_on_hyphens=False, replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""]
                rows.extend(indent + part for part in wrapped)
        if self._context_cache is None or self._context_cache[0] != width:
            self._context_cache = (width, _recent_context(self.shared_context, width=width))
        context = self._context_cache[1]
        return f"Shared context (last 4 lines):\n{context}\n\nPaths:\n" + "\n".join(rows)


class ActionSequencePolicy:
    def __init__(self, actions: tuple[PolicyAction, ...]) -> None:
        self.actions = iter(actions)

    def choose(self, engine, observation) -> PolicyAction:
        del engine, observation
        return next(self.actions)


def chord_menu(
    io, chord: Chord, *, at_edge: bool = False, promote_on_select: bool = False
) -> tuple[str, tuple[PolicyAction, ...] | None]:
    notice = ""
    while True:
        size = io.terminal_size()
        columns = size[0] if size is not None else 100
        width = max(1, columns - 2)
        body = chord.display(width=width)
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
