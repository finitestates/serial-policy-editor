"""Terminal rendering and command parsing, kept outside the editor domain."""

from __future__ import annotations

import importlib.util
import math
import json
import pydoc
import re
import sys
import termios
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
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


class _SessionOutput(StringIO):
    """Hold incidental print output until fullscreen exits, showing live status."""

    def __init__(self, session, original):
        super().__init__()
        self.session = session
        self.original = original

    def write(self, text):
        result = super().write(text)
        self.session.write(text, end="")
        return result

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()


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
        """Keep one terminal application alive throughout the interactive loop."""
        if not self._live_choices:
            yield None
            return
        if self._live_session is not None:
            raise RuntimeError("live session is already active")
        from .persistent_tui import PersistentTerminalSession

        session = PersistentTerminalSession(theme=self._live_theme)
        stdout, stderr = sys.stdout, sys.stderr
        captured_out = _SessionOutput(session, stdout)
        captured_err = _SessionOutput(session, stderr)
        try:
            with session:
                self._live_session = session
                try:
                    with redirect_stdout(captured_out), redirect_stderr(captured_err):
                        yield session
                finally:
                    self._live_session = None
        finally:
            # CLI summaries and errors belong to the restored normal screen.
            # Prompt-toolkit writes through the output captured before redirection.
            stdout.write(captured_out.getvalue())
            stderr.write(captured_err.getvalue())
            stdout.flush()
            stderr.flush()

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
        from .live_tui import ChoiceViewState, read_live_choice

        options = dict(
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
        )
        if self._live_session is not None:
            return self._live_session.read_choice(ChoiceViewState(choice, **options))
        return read_live_choice(choice, theme=self._live_theme, **options)

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
        from .edge_tui import EdgeViewState, read_live_edge_command

        options = dict(
            episode_id=episode_id, boundary=boundary, current_budget=current_budget,
            remaining_tokens=remaining_tokens, sampler_summary=sampler_summary,
        )
        if self._live_session is not None:
            return self._live_session.read_edge(EdgeViewState(**options))
        return read_live_edge_command(theme=self._live_theme, **options)

    def read(self, prompt: str) -> str | None:
        if self._live_session is not None:
            return self._live_session.read(prompt)
        try:
            return input(prompt)
        except EOFError:
            return None

    def read_key(self, prompt: str) -> str | None:
        """Read one unbuffered key without echoing it on an interactive TTY."""
        if self._live_session is not None:
            return self._live_session.read(prompt, single_key=True)
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
        if self._live_session is not None:
            if len(text.splitlines()) > 6:
                self._live_session.page(text)
            else:
                self._live_session.write(text, end=end)
            return
        print(text, end=end, flush=True)

    def page(self, text: str) -> None:
        if self._live_session is not None:
            self._live_session.page(text)
            return
        pydoc.pager(text)


class CommandKind(str, Enum):
    BIAS = "bias"
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
    bias_operator: str | None = None
    bias_amount: float | None = None
    bias_targets: tuple[str, ...] | None = None
    bias_target_bare: tuple[bool, ...] | None = None
    bias_prefix: str | None = None
    bias_last: int | None = None
    bias_triggers: tuple[str, ...] | None = None
    bias_until: str | None = None
    bias_stop_text: str | None = None
    bias_stop_token: int | None = None
    bias_group_name: str | None = None
    bias_group_members: tuple[str, ...] | None = None
    bias_group_member_bare: tuple[bool, ...] | None = None

    @property
    def bias_text(self) -> str | None:
        """Backward-compatible scalar view of bias_targets."""
        return self.bias_targets[0] if self.bias_targets is not None and len(self.bias_targets) == 1 else None

    @property
    def bias_texts(self) -> tuple[str, ...] | None:
        """Backward-compatible batch view of bias_targets."""
        return self.bias_targets if self.bias_targets is not None and len(self.bias_targets) > 1 else None


HELP_TEXT = """Commands:
  Tab / Shift-Tab   move down/up through the current table's visual order;
                    a search lens cycles only within its neighborhood
                    the first Tab selects the sampled proposal's raw rank
                    Enter remains the only commit action
                    --manual-acceptance leaves the command blank instead
  accept             commit the sampled proposal
  b wings +          bare words/phrases imply the usual leading space
  b {wings, scales, claws} +0.5          bias several targets at once
  b {wings, scales} + after {dragon, wyvern} until "."
                    braces are comma-separated human text; quoted items stay exact
  b nautical -> {anchor, steamship, wharf}
                    create or append a durable runtime bias group
  N- after dragon until "\n"            ranked target with an exact one-token stop
  b " TEXT" +/-[N]  quoted text remains exact; = clears the exact bias
  bl X +/-[N]       bias the last X context tokens; bl 1 is a single-token bias
  N+/-[X] ... " P"  bias ranked token N only after the tokenized prefix P
  N+ / N-           adjust token bias by the default step without advancing
  N+0.5 / N-0.5     adjust by an explicit amount; N= clears that token bias
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
aliases for h | N and h|N. Bias lists may also use the older exact JSON form
[" target", " other"]. Scoped `until .` / `until |` remain compatibility aliases
for sentence/newline lifetimes; quoted `until "..."` names one exact stop token.
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


def _split_human_bias_group(raw: str) -> tuple[str, ...]:
    """Split a {...} group on commas outside JSON-quoted strings."""
    body = raw.strip()[1:-1]
    items = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(body):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == ',':
            items.append(body[start:index])
            start = index + 1
    if quoted or escaped:
        raise EditorError("unterminated quoted text in bias group")
    items.append(body[start:])
    if any(not item.strip() for item in items):
        raise EditorError("bias groups cannot contain empty alternatives")
    return tuple(items)


def _parse_bias_text(raw: str, *, label: str, return_bare: bool = False):
    """Decode exact JSON text or friendlier bare/braced text.

    Bare text is continuation-oriented: surrounding whitespace is stripped and a
    single leading space is supplied automatically. Quoted JSON strings are exact.
    The legacy JSON-list form remains exact for backwards compatibility.
    """
    value = raw.strip()
    if not value:
        raise EditorError(f"{label} cannot be empty")
    bare_flags: tuple[bool, ...]
    if value.startswith('['):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            raise EditorError(f"{label} must be valid JSON") from exc
        if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
            raise EditorError(f"{label} JSON form must be a list of strings")
        result = tuple(decoded)
        bare_flags = tuple(False for _ in result)
    elif value.startswith('{'):
        if not value.endswith('}'):
            raise EditorError(f"unterminated {label} group")
        result_list = []
        bare_list = []
        for item in _split_human_bias_group(value):
            item = item.strip()
            if item.startswith('"'):
                try:
                    decoded = json.loads(item)
                except ValueError as exc:
                    raise EditorError(f"quoted {label} must be a valid JSON string") from exc
                if not isinstance(decoded, str):
                    raise EditorError(f"quoted {label} must be a JSON string")
                result_list.append(decoded)
                bare_list.append(False)
            else:
                if '"' in item or any(character in item for character in '{}[]'):
                    raise EditorError(f"invalid bare {label}: {item!r}")
                result_list.append(' ' + item)
                bare_list.append(True)
        result = tuple(result_list)
        bare_flags = tuple(bare_list)
    elif value.startswith('"'):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            raise EditorError(f"{label} must be a valid JSON string") from exc
        if not isinstance(decoded, str):
            raise EditorError(f"{label} must be a JSON string")
        result = (decoded,)
        bare_flags = (False,)
    else:
        if any(character in value for character in '{}[]"'):
            raise EditorError(f"invalid bare {label}")
        result = (' ' + value.strip(),)
        bare_flags = (True,)
    if not result or any(not item for item in result):
        raise EditorError(f"{label} alternatives cannot be empty")
    return (result, bare_flags) if return_bare else result


def _parse_bias_stop(raw: str, *, vocabulary_size: int) -> tuple[str | None, str | None, int | None]:
    """Return (legacy lifetime, exact stop text, explicit stop token)."""
    value = raw.strip()
    if value == '.':
        return 'sentence', None, None
    if value == '|':
        return 'newline', None, None
    if value.startswith('#'):
        try:
            token = int(value[1:])
        except ValueError as exc:
            raise EditorError("stop token IDs use until #N") from exc
        if not 0 <= token < vocabulary_size:
            raise EditorError("stop token ID is outside the vocabulary")
        return None, None, token
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            raise EditorError("until must be a valid JSON string, ., |, or #N") from exc
        if not isinstance(decoded, str) or not decoded:
            raise EditorError("stop text must be a nonempty JSON string")
        return None, decoded, None
    raise EditorError('use until "TOKEN" (or legacy until . / until |)')


def parse_bias_command(raw: str, *, vocabulary_size: int) -> TeacherCommand | None:
    """Parse bias edits without interpreting quoted text as another command."""
    quoted = r'"(?:[^"\\]|\\.)*"'
    group_match = re.fullmatch(
        r"b\s+(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s*->\s*(?P<members>\{.*\})",
        raw.strip(),
    )
    if group_match is not None:
        members, bare_flags = _parse_bias_text(
            group_match.group("members"), label="bias group members", return_bare=True
        )
        return TeacherCommand(
            CommandKind.BIAS,
            bias_group_name=group_match.group("name"),
            bias_group_members=members,
            bias_group_member_bare=bare_flags,
        )
    adjustment = r"(?P<op>[+\-=])\s*(?P<amount>\d+(?:\.\d*)?|\.\d+)?"
    stop = rf"(?:{quoted}|\#[0-9]+|[.|])"
    scope = rf"(?:\s+after\s+(?P<triggers>.+?)\s+until\s+(?P<until>{stop}))?"
    patterns = (
        rf"b\s+(?P<text>.+?)\s*{adjustment}{scope}",
        rf"bl\s+(?P<last>\d+)\s*{adjustment}",
        rf"(?P<rank>\d+)\s*{adjustment}(?:\s*\.\.\.\s*(?P<prefix>{quoted}))?{scope}",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, raw.strip())
        if match is None:
            continue
        fields = match.groupdict()
        operator, amount = fields["op"], fields["amount"]
        rank = int(fields["rank"]) if fields.get("rank") is not None else None
        last = int(fields["last"]) if fields.get("last") is not None else None
        if rank is not None and not 1 <= rank <= vocabulary_size:
            raise EditorError("bias rank is outside the vocabulary")
        if last is not None and last < 1:
            raise EditorError("bl requires a positive token count")
        if operator == "=" and amount is not None:
            raise EditorError("use = without an amount to clear a bias")
        value = float(amount) if amount is not None else None
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise EditorError("bias adjustment must be finite and positive")

        targets = None
        target_bare = None
        if fields.get("text") is not None:
            targets, target_bare = _parse_bias_text(
                fields["text"], label="bias target", return_bare=True
            )
            if len(set(targets)) != len(targets):
                raise EditorError("bias target alternatives cannot contain duplicates")

        prefix = None
        if fields.get("prefix") is not None:
            try:
                prefix = json.loads(fields["prefix"])
            except ValueError as exc:
                raise EditorError("bias prefix must be a valid JSON string") from exc
            if not isinstance(prefix, str) or not prefix:
                raise EditorError("bias prefix cannot be empty")

        triggers = None
        until = None
        stop_text = None
        stop_token = None
        if fields.get("triggers") is not None:
            if prefix is not None:
                raise EditorError("Use a b target for scoped multi-token rules")
            triggers = _parse_bias_text(fields["triggers"], label="bias trigger")
            until, stop_text, stop_token = _parse_bias_stop(
                fields["until"], vocabulary_size=vocabulary_size)

        return TeacherCommand(CommandKind.BIAS, search_rank=rank,
            bias_operator=operator, bias_amount=value, bias_last=last,
            bias_targets=targets, bias_target_bare=target_bare,
            bias_prefix=prefix, bias_triggers=triggers,
            bias_until=until, bias_stop_text=stop_text, bias_stop_token=stop_token)
    return None

def parse_command(
    raw: str,
    *,
    menu_size: int,
    default_hold_tokens: int,
    vocabulary_size: int | None = None,
    default_search_radius: int = 3,
) -> TeacherCommand:
    bias_command = parse_bias_command(raw, vocabulary_size=vocabulary_size or menu_size)
    if bias_command is not None:
        return bias_command
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
        if candidate.bias:
            suffix += f" [bias {candidate.bias:+g}]"
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
