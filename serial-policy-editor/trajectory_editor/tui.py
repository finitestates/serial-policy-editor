"""Terminal rendering and command parsing, kept outside the editor domain."""

from __future__ import annotations

import importlib.util
import json
import pydoc
import re
import sys
import termios
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Protocol

from .domain import Candidate, ChoiceSet, EditAction, EditorError, InsertMode
from .ui_themes import resolve_live_theme


# Internal result used by the live prompt only; it is never parsed as a user
# command.  Keeping it outside the normal command grammar distinguishes Enter
# in seamless review from Escape, which returns to the live surface.
SEAMLESS_REACTIVATE = "\x1e"


class IO(Protocol):
    def read(self, prompt: str) -> str | None: ...

    def read_key(self, prompt: str) -> str | None: ...

    def write(self, text: str = "", *, end: str = "\n") -> None: ...

    def page(self, text: str) -> None: ...


@dataclass(frozen=True)
class ChoiceFeedback:
    """Structured current-choice feedback for the live decision surface."""

    category: str
    title: str
    lines: tuple[str, ...] = ()
    completion_commands: tuple[str, ...] = ()
    initial_tab_command: str | None = None


@dataclass(frozen=True)
class BoundaryReview:
    """Read-only projection of one already durable teacher boundary."""

    active_aligned_step: int
    aligned_step: int
    context_text_tail: str
    context_token_sha256: str
    position: Mapping[str, Any]
    next_token: Mapping[str, Any] | None = None


class TerminalIO:
    def __init__(
        self,
        *,
        live_choices: bool | None = None,
        live_theme: str | None = None,
    ) -> None:
        self._live_theme = resolve_live_theme(live_theme)
        requested = True if live_choices is None else live_choices
        self._live_choices = bool(
            requested
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and importlib.util.find_spec("prompt_toolkit") is not None
        )
        self._live_session: object | None = None

    @property
    def supports_live_choices(self) -> bool:
        return self._live_choices

    @property
    def live_theme(self) -> str:
        return self._live_theme

    @contextmanager
    def live_session(self) -> Iterator[object | None]:
        """Own the alternate screen for one complete interactive episode."""
        if not self._live_choices:
            yield None
            return

        from .live_tui import PersistentFullscreenSession

        previous = self._live_session
        if previous is not None:
            raise RuntimeError("live session is already active")
        with PersistentFullscreenSession() as session:
            self._live_session = session
            try:
                yield session
            finally:
                self._live_session = previous

    def read_choice(
        self,
        choice: ChoiceSet,
        *,
        remaining_tokens: int | None,
        candidates: tuple[Candidate, ...],
        display_candidates: tuple[Candidate, ...] | None = None,
        resolve_insertion: Callable[[str, InsertMode], str],
        resolve_candidate: Callable[[int], Candidate] | None = None,
        target_token_id: int | None = None,
        feedback: ChoiceFeedback | None = None,
        initial_command: str | None = None,
        review: BoundaryReview | None = None,
        seamless: bool = False,
        reactivate_on_review_enter: bool = False,
        search_lens_active: bool = False,
        policy_active: bool = False,
        show_policy_rank: bool = False,
        sort_by_policy: bool = False,
    ) -> str | None:
        if not self._live_choices:
            raise RuntimeError("live choice input is not available")
        from .live_tui import read_live_choice

        live_session = self._live_session

        return read_live_choice(
            choice,
            remaining_tokens=remaining_tokens,
            candidates=candidates,
            display_candidates=display_candidates,
            resolve_insertion=resolve_insertion,
            resolve_candidate=resolve_candidate,
            target_token_id=target_token_id,
            feedback=feedback,
            initial_command=initial_command,
            review=review,
            seamless=seamless,
            reactivate_on_review_enter=reactivate_on_review_enter,
            search_lens_active=search_lens_active,
            policy_active=policy_active,
            show_policy_rank=show_policy_rank,
            sort_by_policy=sort_by_policy,
            theme=self._live_theme,
            input_device=(
                getattr(live_session, "input_device", None)
                if live_session is not None
                else None
            ),
            output_device=(
                getattr(live_session, "output_device", None)
                if live_session is not None
                else None
            ),
        )

    def read_live_edge_command(
        self,
        *,
        episode_id: str,
        boundary: int,
        current_budget: int | None,
        remaining_tokens: int | None,
        sampler_summary: str,
    ) -> str | None:
        """Read one command from the structured live-edge surface."""
        if not self._live_choices:
            raise RuntimeError("live edge input is not available")
        from .edge_tui import read_live_edge_command

        live_session = self._live_session

        return read_live_edge_command(
            episode_id=episode_id,
            boundary=boundary,
            current_budget=current_budget,
            remaining_tokens=remaining_tokens,
            sampler_summary=sampler_summary,
            theme=self._live_theme,
            input_device=(
                getattr(live_session, "input_device", None)
                if live_session is not None
                else None
            ),
            output_device=(
                getattr(live_session, "output_device", None)
                if live_session is not None
                else None
            ),
        )

    def read(self, prompt: str) -> str | None:
        try:
            return input(prompt)
        except EOFError:
            return None

    def read_key(self, prompt: str) -> str | None:
        """Read one unbuffered key without echoing it on an interactive TTY."""
        if not sys.stdin.isatty():
            value = self.read(prompt)
            if value is None:
                return None
            return "\n" if value == "" else value[:1]
        print(prompt, end="", flush=True)
        descriptor = sys.stdin.fileno()
        prior = termios.tcgetattr(descriptor)
        current = termios.tcgetattr(descriptor)
        current[3] &= ~(termios.ICANON | termios.ECHO)
        current[6][termios.VMIN] = 1
        current[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, current)
            value = sys.stdin.read(1)
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, prior)
            print(flush=True)
        if value == "\x03":
            raise KeyboardInterrupt
        if value in {"", "\x04"}:
            return None
        return value

    def write(self, text: str = "", *, end: str = "\n") -> None:
        print(text, end=end, flush=True)

    def page(self, text: str) -> None:
        pydoc.pager(text)


class CommandKind(str, Enum):
    EDIT = "edit"
    HOLD = "hold"
    NOTE_BEFORE = "note-before"
    NOTE_AFTER = "note-after"
    FINISH = "finish"
    TEACHER_EOG = "teacher-eog"
    MAIN_MENU = "main-menu"
    MENU_EXPAND = "menu-expand"
    TOKEN_SEARCH = "token-search"
    TOKEN_SEARCH_VIEW = "token-search-view"
    CONTEXT = "context"
    POLICY_VIEW = "policy-view"
    POLICY_COLUMN = "policy-column"
    REVIEW_BACK = "review-back"
    REVIEW_FORWARD = "review-forward"
    REVIEW_EXIT = "review-exit"
    FORK = "fork"
    HELP = "help"


class ForkAddressKind(str, Enum):
    CURRENT = "current"
    ABSOLUTE = "absolute"
    RELATIVE_BACKWARD = "relative-backward"


@dataclass(frozen=True)
class ForkAddress:
    kind: ForkAddressKind
    value: int | None = None


@dataclass(frozen=True)
class TeacherCommand:
    kind: CommandKind
    action: EditAction | None = None
    hold_tokens: int | None = None
    hold_boundary: str | None = None
    note: str | None = None
    invoked_as: str | None = None
    additional_rows: int | None = None
    search_query: str | None = None
    search_direction: str | None = None
    search_rows: int | None = None
    search_rank: int | None = None
    context_characters: int | str | None = None
    force: bool = False
    fork_address: ForkAddress | None = None
    warning: str | None = None


HELP_TEXT = """Commands:
  Tab / Shift-Tab   move down/up through the current table's visual order;
                    a search lens cycles only within its neighborhood
                    the first Tab selects the sampled proposal's raw rank
                    Enter remains the only commit action
                    --manual-acceptance leaves the command blank instead
  accept             commit the sampled proposal
  1..N              commit a candidate; the proposal rank records acceptance
  t TEXT            insert continuation text (adds a joining space if needed)
  x TEXT            insert exact text
                    after `t ` or `x `, Tab inserts a literal tab character
  h [N]              release control for N tokens (default: configured limit)
  h . [N]            hold through first token containing . ! ?, capped at N
  h | [N]            hold through first token containing a newline, capped at N
                    matching tokens stay whole; no lookahead or trailing tokens
  m                  return to the main table without disclosing rows
  m N                return to the main table and reveal N more ranked rows
  /TERM              find one exact token and show its raw-rank neighborhood
  /"\\n"              JSON escapes preserve exact whitespace/control characters
  ms N               explore the neighborhood of raw rank N
  Ctrl+G             explore the numeric rank currently in the input
  ms                 return to the active token-search neighborhood
  ms + [N]           expand toward larger ranks / lower raw probability
  ms - [N]           expand toward smaller ranks / higher raw probability
  c [N|all]          page more of the current context (default: 2000 chars)
  v                  toggle raw/policy ordering of disclosed rows
  V                  toggle the policy-rank column independently
                     numeric selections accept any raw rank in the vocabulary
  [ / ]              review the previous/next durable token boundary
                      bare f forks the reviewed boundary; Esc returns live
  f                  fork a child from the current boundary
  f N                fork a child from absolute token boundary N
  f - N              fork N token boundaries backward
  n [TEXT]          add a note before the current decision
  p [TEXT]          add a note after the most recent committed update
  e | eog            preview and confirm a teacher-selected EOG
  e! | eog!          commit a teacher-selected EOG immediately
  q | finish         open the live edge menu
  h end              deprecated alias for finish
  ?                 show this help

Structured short commands may omit separating spaces: h5, h.5, h|5,
f-5, m10, ms+10, ms-10, and c900 are equivalent to their spaced forms. Text-bearing
commands /, t, x, n, and p keep their whitespace exactly as entered.
The older newline-hold spellings h / N and h/N remain accepted as deprecated
aliases for h | N and h|N.
"""


_COMPACT_HOLD_BOUNDARY = re.compile(
    r"^h\s*([.|/])(?:\s*(\d+))?$",
    re.IGNORECASE,
)
_COMPACT_HOLD_COUNT = re.compile(r"^h\s*(\d+)$", re.IGNORECASE)
_COMPACT_MENU_COUNT = re.compile(r"^m\s*(\d+)$", re.IGNORECASE)
_COMPACT_SEARCH_VIEW = re.compile(
    r"^ms\s*([+-])(?:\s*(\d+))?$",
    re.IGNORECASE,
)
_COMPACT_CONTEXT_COUNT = re.compile(r"^c\s*(\d+)$", re.IGNORECASE)
_COMPACT_FORK_ADDRESS = re.compile(
    r"^f\s*(-?)\s*(\d+)$",
    re.IGNORECASE,
)


def normalize_command_syntax(raw: str) -> str:
    """Normalize enumerated compact spellings without touching text payloads.

    This is deliberately a closed grammar rather than general whitespace
    removal.  In particular, /, t, x, n, and p retain the ordinary parser's
    whitespace-sensitive payload behavior.
    """
    if raw.startswith("/") or (
        len(raw) >= 2
        and raw[:1].lower() in {"t", "x", "n", "p"}
        and raw[1:2] == " "
    ):
        return raw

    command = raw.strip()

    match = _COMPACT_HOLD_BOUNDARY.fullmatch(command)
    if match is not None:
        boundary, count = match.groups()
        boundary = "|" if boundary == "/" else boundary
        return f"h {boundary}" + (f" {count}" if count is not None else "")

    match = _COMPACT_HOLD_COUNT.fullmatch(command)
    if match is not None:
        return f"h {match.group(1)}"

    match = _COMPACT_SEARCH_VIEW.fullmatch(command)
    if match is not None:
        direction, count = match.groups()
        return f"ms {direction}" + (f" {count}" if count is not None else "")

    match = _COMPACT_MENU_COUNT.fullmatch(command)
    if match is not None:
        return f"m {match.group(1)}"

    match = _COMPACT_CONTEXT_COUNT.fullmatch(command)
    if match is not None:
        return f"c {match.group(1)}"

    match = _COMPACT_FORK_ADDRESS.fullmatch(command)
    if match is not None:
        direction, count = match.groups()
        return f"f {direction} {count}" if direction else f"f {count}"

    return command


def parse_fork_address(raw: str) -> ForkAddress | None:
    """Parse one fork address, or return None when raw is not a fork command."""
    command = normalize_command_syntax(raw)
    lower = command.lower()
    parts = lower.split()
    if lower in {"f", "fork"}:
        return ForkAddress(ForkAddressKind.CURRENT)
    if not lower.startswith(("f ", "fork ", "f-", "fork-")):
        return None
    if parts[0] not in {"f", "fork"}:
        return None
    operands = parts[1:]
    if len(operands) == 1:
        if operands[0].startswith("+"):
            raise EditorError(
                "use f, f N, or f - N"
            )
        try:
            step = int(operands[0])
        except ValueError as exc:
            raise EditorError("use f, f N, or f - N") from exc
        if step < 0:
            magnitude = -step
            if magnitude < 1:
                raise EditorError("relative fork distance must be at least 1")
            return ForkAddress(ForkAddressKind.RELATIVE_BACKWARD, magnitude)
        return ForkAddress(ForkAddressKind.ABSOLUTE, step)
    if len(operands) == 2 and operands[0] == "-":
        try:
            magnitude = int(operands[1])
        except ValueError as exc:
            raise EditorError("relative fork distance must be an integer") from exc
        if magnitude < 1:
            raise EditorError("relative fork distance must be at least 1")
        return ForkAddress(ForkAddressKind.RELATIVE_BACKWARD, magnitude)
    raise EditorError("use f, f N, or f - N")


def parse_command(
    raw: str,
    *,
    menu_size: int,
    default_hold_tokens: int,
    vocabulary_size: int | None = None,
    default_search_radius: int = 3,
) -> TeacherCommand:
    if raw.startswith("/"):
        payload = raw[1:]
        if not payload:
            raise EditorError("token search requires text after /")
        if payload.startswith('"'):
            try:
                decoded = json.loads(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise EditorError(
                    "quoted token searches must be one valid JSON string, "
                    'for example /"\\n"'
                ) from exc
            if not isinstance(decoded, str):
                raise EditorError("quoted token search payload must be a JSON string")
            query = decoded
        else:
            query = payload
        if not query:
            raise EditorError("token search cannot query an empty string")
        return TeacherCommand(
            CommandKind.TOKEN_SEARCH,
            search_query=query,
            invoked_as=raw,
        )

    original_command = raw.strip()
    command = normalize_command_syntax(raw)
    lower = command.lower()
    if not command:
        raise EditorError("enter a command; use ? for help")
    if lower in {"?", "help"}:
        return TeacherCommand(CommandKind.HELP)
    if command == "[":
        return TeacherCommand(CommandKind.REVIEW_BACK, invoked_as=command)
    if command == "]":
        return TeacherCommand(CommandKind.REVIEW_FORWARD, invoked_as=command)
    if raw == "\x1b":
        return TeacherCommand(CommandKind.MAIN_MENU, invoked_as="escape")
    if lower in {"q", "quit", "finish"}:
        return TeacherCommand(CommandKind.FINISH, invoked_as=lower)
    if lower in {"e", "eog"}:
        return TeacherCommand(CommandKind.TEACHER_EOG, invoked_as=lower)
    if lower in {"e!", "eog!"}:
        return TeacherCommand(
            CommandKind.TEACHER_EOG,
            invoked_as=lower,
            force=True,
        )
    if lower == "accept":
        return TeacherCommand(CommandKind.EDIT, action=EditAction.accept())
    if command.isdigit():
        rank = int(command)
        if rank < 1 or (vocabulary_size is not None and rank > vocabulary_size):
            maximum = f"..{vocabulary_size}" if vocabulary_size is not None else " or greater"
            raise EditorError(f"rank must be 1{maximum}")
        return TeacherCommand(CommandKind.EDIT, action=EditAction.select(rank))
    fork_address = parse_fork_address(raw)
    if fork_address is not None:
        return TeacherCommand(
            CommandKind.FORK,
            invoked_as=original_command,
            fork_address=fork_address,
        )
    if lower in {"h", "hold"} or lower.startswith(("h ", "hold ")):
        parts = command.split()
        if len(parts) > 3:
            raise EditorError("use h, h N, h . [N], h | [N], or finish")
        if len(parts) == 1:
            return TeacherCommand(CommandKind.HOLD, hold_tokens=default_hold_tokens)
        if parts[1].lower() in {"e", "end", "eog"}:
            if len(parts) != 2:
                raise EditorError("use h end without additional arguments")
            return TeacherCommand(CommandKind.FINISH, invoked_as=lower)
        if parts[1] in {".", "|", "/"}:
            boundary = "sentence" if parts[1] == "." else "newline"
            if len(parts) == 2:
                tokens = default_hold_tokens
            else:
                try:
                    tokens = int(parts[2])
                except ValueError as exc:
                    raise EditorError("boundary hold cap must be an integer") from exc
            if tokens < 1:
                raise EditorError("boundary hold cap must be at least 1")
            return TeacherCommand(
                CommandKind.HOLD,
                hold_tokens=tokens,
                hold_boundary=boundary,
                warning=(
                    "newline holds now use h | [N]; h / [N] is deprecated"
                    if re.fullmatch(
                        r"(?:h|hold)\s*/.*",
                        original_command,
                        re.IGNORECASE,
                    )
                    else None
                ),
            )
        if len(parts) != 2:
            raise EditorError("use h, h N, h . [N], h | [N], or finish")
        try:
            tokens = int(parts[1])
        except ValueError as exc:
            raise EditorError(
                "hold length must be an integer; use h N to delegate N tokens"
            ) from exc
        if tokens < 1:
            raise EditorError("hold length must be at least 1")
        return TeacherCommand(CommandKind.HOLD, hold_tokens=tokens)
    if lower in {"m", "more"}:
        return TeacherCommand(CommandKind.MAIN_MENU, invoked_as=lower)
    if lower.startswith(("m ", "more ")):
        parts = command.split()
        if len(parts) != 2:
            raise EditorError("use m or m N")
        try:
            rows = int(parts[1])
        except ValueError as exc:
            raise EditorError("menu expansion must be a positive integer") from exc
        if rows < 1:
            raise EditorError("menu expansion must add at least 1 row")
        return TeacherCommand(
            CommandKind.MENU_EXPAND,
            additional_rows=rows,
        )
    if lower == "ms" or lower.startswith("ms "):
        parts = command.split()
        if len(parts) == 1:
            return TeacherCommand(CommandKind.TOKEN_SEARCH_VIEW)
        if len(parts) == 2 and parts[1].isdigit():
            rank = int(parts[1])
            if rank < 1 or (vocabulary_size is not None and rank > vocabulary_size):
                raise EditorError("neighborhood rank is outside the vocabulary")
            return TeacherCommand(CommandKind.TOKEN_SEARCH_VIEW, search_rank=rank)
        if len(parts) not in {2, 3} or parts[1] not in {"+", "-"}:
            raise EditorError("use ms, ms N, ms + [N], or ms - [N]")
        if len(parts) == 2:
            rows = default_search_radius
        else:
            try:
                rows = int(parts[2])
            except ValueError as exc:
                raise EditorError("search expansion must be a positive integer") from exc
        if rows < 1:
            raise EditorError("search expansion must add at least 1 row")
        return TeacherCommand(
            CommandKind.TOKEN_SEARCH_VIEW,
            search_direction=parts[1],
            search_rows=rows,
        )
    if lower in {"c", "context"} or lower.startswith(("c ", "context ")):
        parts = command.split()
        if len(parts) > 2:
            raise EditorError("use c, c N, or c all")
        if len(parts) == 1:
            return TeacherCommand(CommandKind.CONTEXT, context_characters=2000)
        if parts[1].lower() in {"all", "full"}:
            return TeacherCommand(CommandKind.CONTEXT, context_characters="all")
        try:
            characters = int(parts[1])
        except ValueError as exc:
            raise EditorError("context extent must be a positive integer or all") from exc
        if characters < 1:
            raise EditorError("context extent must be at least 1 character")
        return TeacherCommand(CommandKind.CONTEXT, context_characters=characters)
    if command == "V" or lower in {"policy-column", "policy-rank-column"}:
        return TeacherCommand(CommandKind.POLICY_COLUMN, invoked_as=command)
    if lower in {"v", "policy-view", "policy-sort"}:
        return TeacherCommand(CommandKind.POLICY_VIEW, invoked_as=lower)
    if lower == "n" or lower.startswith("n "):
        note = raw[2:] if len(raw) >= 2 and raw[1:2].isspace() else None
        return TeacherCommand(CommandKind.NOTE_BEFORE, note=note)
    if lower == "p" or lower.startswith("p "):
        note = raw[2:] if len(raw) >= 2 and raw[1:2].isspace() else None
        return TeacherCommand(CommandKind.NOTE_AFTER, note=note)
    if len(raw) >= 2 and raw[:2].lower() == "t ":
        return TeacherCommand(
            CommandKind.EDIT,
            action=EditAction.insert(raw[2:], InsertMode.CONTINUATION),
        )
    if len(raw) >= 2 and raw[:2].lower() == "x ":
        return TeacherCommand(
            CommandKind.EDIT,
            action=EditAction.insert(raw[2:], InsertMode.EXACT),
        )
    raise EditorError("unknown command; use ? for help")


def display_choice(
    io: IO,
    choice: ChoiceSet,
    *,
    remaining_tokens: int | None = None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
) -> None:
    io.write("\n" + "=" * 72)
    remaining = (
        f" | remaining: {remaining_tokens}"
        if remaining_tokens is not None
        else ""
    )
    io.write(
        f"Step {choice.aligned_step}{remaining} | "
        f"context tail: {choice.context_text_tail!r}"
    )
    io.write(
        f"Sampled proposal: {choice.proposal_text!r} "
        f"(id={choice.proposal_token_id}, raw={choice.proposal_raw_probability:.2%}, "
        f"decoder={choice.proposal_decoder_probability:.2%}"
        + (
            f", policy-rank={choice.proposal_policy_rank}"
            if policy_active and choice.proposal_policy_rank is not None
            else ""
        )
        + ")"
    )
    display_candidates(
        io,
        choice.candidates,
        heading=True,
        show_policy_rank=show_policy_rank,
        sort_by_policy=sort_by_policy,
    )
    display_actions(io)


def display_candidates(
    io: IO,
    candidates: tuple,
    *,
    heading: bool = False,
    target_token_id: int | None = None,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
) -> None:
    ordered = tuple(candidates)
    if sort_by_policy:
        ordered = tuple(
            sorted(
                ordered,
                key=lambda candidate: (
                    candidate.policy_rank
                    if candidate.policy_rank is not None
                    else candidate.rank,
                    candidate.rank,
                ),
            )
        )
    if heading:
        policy = "  pol-rank" if show_policy_rank else ""
        io.write(f"\n  rank{policy}   raw-p  decode-p  token-id  text")
    for candidate in ordered:
        decoder = (
            f"{candidate.decoder_probability:7.2%}"
            if candidate.decoder_probability > 0.0
            else "     --"
        )
        suffix = " [END]" if candidate.is_eog else ""
        if target_token_id is not None and candidate.token_id == target_token_id:
            suffix += " [MATCH]"
        policy = (
            f"  {candidate.policy_rank:>8}"
            if show_policy_rank and candidate.policy_rank is not None
            else ""
        )
        io.write(
            f"  {candidate.rank:>4}{policy}  {candidate.raw_probability:6.2%}  "
            f"{decoder}  {candidate.token_id:>8}  {candidate.text!r}{suffix}"
        )


def display_actions(io: IO) -> None:
    io.write(
        "\nActions: accept | rank | t TEXT | x TEXT | h [N] | h . [N] | h | [N] | "
        "[ / ] review | f [N|+N|-N] | m [N] | /TERM | "
        "ms [+|- [N]] | c [N|all] | v order | V policy column | "
        "n [note-before] | p [note-after] | e | e! | q | ?"
    )
