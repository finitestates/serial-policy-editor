"""Temporary, storage-neutral previews of several raw-rank continuations."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from .core.actions import Accept, PolicyAction, SelectRawRank
from .core.errors import EditorError
from .episode_engine import EpisodeEngine
from .terminal_contracts import PromptRequest


class ChordRequested(Exception):
    def __init__(self, ranks: tuple[int, ...]) -> None:
        super().__init__(ranks)
        self.ranks = ranks


def parse_chord(raw: str, vocabulary_size: int) -> tuple[int, ...] | None:
    parts = raw.strip().split()
    if not parts or parts[0].lower() != "chord":
        return None
    if not 2 <= len(parts) - 1 <= 26:
        raise EditorError("use chord RANK RANK [RANK ...] (up to 26 paths)")
    if any(not part.isdecimal() for part in parts[1:]):
        raise EditorError("chord ranks must be positive integers")
    ranks = tuple(int(part) for part in parts[1:])
    if len(set(ranks)) != len(ranks):
        raise EditorError("chord ranks must be distinct")
    if any(rank < 1 or rank > vocabulary_size for rank in ranks):
        raise EditorError(f"chord ranks must be between 1 and {vocabulary_size}")
    return ranks


def _position(backend, base: list[int], suffix: list[int]) -> None:
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
    token_ids: list[int] = field(default_factory=list)

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
        self.original = engine
        self.base_visible = list(engine.visible_token_ids)
        self.base_prefix = list(engine.token_ids)
        self.shared_context = engine.backend.render(self.base_prefix, special=True)
        self.paths: list[ChordPath] = []
        self.rounds: list[tuple[int, ...]] = []
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
                    coordinate_offset=engine.coordinate_offset,
                    backend_positioned=True,
                    guidance_backend=engine.guidance_backend,
                )
                preview.visible_token_ids = self.base_visible
                preview.checkpoint_boundary = engine.checkpoint_boundary
                preview._activation_runtime_key = engine._activation_runtime_key
                preview._activation_validation_key = engine._activation_validation_key
                path = ChordPath(chr(ord("a") + index), rank, preview)
                self.paths.append(path)
                self._activate(path)
                outcome = preview.apply(SelectRawRank(rank))
                path.actions.append(SelectRawRank(rank))
                path.token_ids.extend(outcome.resolved_token_ids)
        except BaseException:
            self.discard()
            raise

    def _activate(self, path: ChordPath) -> None:
        suffix = list(path.engine.visible_token_ids[len(self.base_visible):])
        _position(self.original.backend, self.base_prefix, suffix)
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

    def advance(self) -> bool:
        advanced: list[int] = []
        for index, path in enumerate(self.paths):
            if path.state != "live":
                continue
            self._activate(path)
            outcome = path.engine.apply(Accept())
            path.actions.append(Accept())
            path.token_ids.extend(outcome.resolved_token_ids)
            advanced.append(index)
        if advanced:
            self.rounds.append(tuple(advanced))
        return bool(advanced)

    def rewind(self) -> bool:
        if not self.rounds:
            return False
        for index in self.rounds.pop():
            path = self.paths[index]
            path.actions.pop()
            path.token_ids.pop()
            path.engine.rewind_to(
                len(self.base_visible) + sum(
                    not self.original.backend.is_eog(token_id)
                    for token_id in path.token_ids
                )
            )
        return True

    def select(self, label: str) -> tuple[PolicyAction, ...]:
        key = label.strip().lower()
        for path in self.paths:
            if key in {path.label, str(path.starting_rank)}:
                actions = tuple(path.actions)
                self.discard()
                return actions
        raise EditorError("select a chord path by letter or starting raw rank")

    def discard(self) -> None:
        if self.closed:
            return
        self.closed = True
        _position(self.original.backend, self.base_prefix, [])
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
        context = _recent_context(self.shared_context, width=width)
        return f"Shared context (last 4 lines):\n{context}\n\nPaths:\n" + "\n".join(rows)


class ActionSequencePolicy:
    def __init__(self, actions: tuple[PolicyAction, ...]) -> None:
        self.actions = iter(actions)

    def choose(self, engine, observation) -> PolicyAction:
        del engine, observation
        return next(self.actions)


def chord_menu(io, chord: Chord, *, at_edge: bool = False) -> tuple[str, tuple[PolicyAction, ...] | None]:
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
            "Chord: Enter: advance live paths | rewind: undo one round | "
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
            if command == "":
                if not chord.advance():
                    notice = "All chord paths are locked."
                continue
            if command in {"rewind", "r"}:
                if not chord.rewind():
                    notice = "No chord round to rewind."
                continue
            if command == "q":
                at_edge = True
                continue
            try:
                if command in {path.label for path in chord.paths} or command.isdecimal():
                    return "select", chord.select(command)
            except EditorError as exc:
                notice = str(exc)
                continue
        if command in {"?", "help"}:
            notice = ("Enter advances live paths; rewind undoes one round. "
                      "Choose a letter or starting rank to commit that path's actions "
                      "and drop the other previews. q opens Chord EDGE options: "
                      "c resumes the chord, discard restores the episode, "
                      "and q quits the editor.")
        else:
            notice = "Resolve the chord before changing the episode."
