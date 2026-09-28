"""Teacher command grammar, normalization, and shared help text."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum

from .core.errors import EditorError
from .core.ui import EditAction, InsertMode


class CommandKind(str, Enum):
    CHORD = "chord"
    BEAM = "beam"
    BIAS = "bias"
    SAMPLER = "sampler"
    REROLL = "reroll"
    DRAW = "draw"
    EDIT = "edit"
    PHRASE = "phrase"
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
    LOGIT_VIEW = "logit-view"
    PROBABILITY_VIEW = "probability-view"
    COLUMN_FOCUS = "column-focus"
    OVERLAY_TOGGLE = "overlay-toggle"
    REVIEW_BACK = "review-back"
    REVIEW_FORWARD = "review-forward"
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
    phrase_text: str | None = None
    phrase_mode: str | None = None
    phrase_force: bool = False
    hold_tokens: int | None = None
    hold_boundary: str | None = None
    note: str | None = None
    invoked_as: str | None = None
    overlay: str | None = None
    additional_rows: int | None = None
    search_query: str | None = None
    search_direction: str | None = None
    search_rows: int | None = None
    search_rank: int | None = None
    context_characters: int | str | None = None
    force: bool = False
    fork_address: ForkAddress | None = None
    bias_operator: str | None = None
    bias_status: bool = False
    bias_amount: float | None = None
    bias_targets: tuple[str, ...] | None = None
    bias_group_name: str | None = None
    bias_group_members: tuple[str, ...] | None = None
    bias_group_member_literal: tuple[bool, ...] | None = None
    bias_group_remove_name: str | None = None
    bias_group_remove_members: tuple[str, ...] | None = None
    bias_group_remove_member_literal: tuple[bool, ...] | None = None
    bias_inspect_group: str | None = None
    bias_inspect_token: int | None = None
    bias_token_id: int | None = None
    chord_ranks: tuple[int, ...] | None = None
    beam_width: int | None = None
    beam_stochastic: bool = False
    beam_skip_rank_ranges: tuple[tuple[int, int], ...] = ()
    beam_add_model_ranks: tuple[int, ...] = ()
    sampler_text: str | None = None
    reroll_seed: int | None = None
    draw_raw_rank: int | None = None


class CommandState(str, Enum):
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
    READY = "ready"


@dataclass(frozen=True)
class CommandInterpretation:
    """Syntax only. A ready command may still fail episode validation."""

    raw: str
    state: CommandState
    command: TeacherCommand | None = None
    message: str = ""


HELP_TEXT = """Commands:
  READY means command syntax is understood; episode checks still happen on Enter.
  Tab / Shift-Tab   move down/up through the current table's visual order;
                    a search lens cycles only within its neighborhood
                    the first Tab selects the sampled proposal's backend rank
                    Enter remains the only commit action
                    --manual-acceptance leaves the command blank instead
  accept             commit the sampled proposal
  groups               list groups; groups NAME shows members and token routes
  b NAME -> {terms}    create or add members to a named group
  b NAME remove {terms} remove members from a group
  b NAME +0.5          add 0.5 logit to each matching member
  b NAME =-0.5         set the group's total adjustment; off sets it to zero
  b {NAME, OTHER} +     adjust several existing groups by the default 0.5
                      each group contributes once per token; groups and direct tokens add
  b token #ID          explain active and inactive sources for one token
  RANK+ / RANK-        adjust a selected token directly by the default 0.5
  RANK=VALUE           set or clear a direct token adjustment
  terms add one-space, lowercase, sentence-case, and uppercase variants
                      no plural or generated title-case variants
                      use literal:"text" in a group to disable surface variants
  multi-token terms bias only the final token after an exact prefix match
  s top_k=20|none change sampler settings; changes are part of the action tape
  s gumbel_top_k=5 shows five Gumbel-ranked candidates (proposal stays first)
  s gumbel_noise_address=model-rank selects Gumbel-Max and addresses noise by model rank
  s gumbel_noise_scale=0.5 selects Gumbel-Max and scales post-filter noise
  s draw_kernel=gaussian-max gaussian_noise_std=1 uses Gaussian-noise argmax
  s {JSON}        replace all sampler settings from a complete SamplerConfig record
  reroll [SEED]   change the draw seed as a replayable action
  draw RAW_RANK   find a seed that draws the token at this raw rank
  1..N              commit a candidate; the proposal rank records acceptance
  chord RANK RANK... preview temporary continuations; choose a letter or starting rank
                    to commit its actions and drop the other previews
  beam [WIDTH] [skip RANKS] [add RANKS]
                    cumulative log-p beam (default width: 5; maximum: 100)
  beam stochastic [WIDTH] [skip RANKS]
                    open Gumbel-Top-k sampling without replacement
  gbeam [WIDTH] [skip RANKS]
                    short form for stochastic beam
                    skip filters one-based model ranks at the first step only;
                    the beam fills to its requested width from remaining candidates
                    when available, and skipped tokens remain available later.
                    RANKS accepts forms such as 1-5 or 1 3 10.
                    In deterministic beam, add forces root model ranks into the
                    initial beam and reserves one slot for each lineage; width stays fixed.
                    In the viewer, p protects a deterministic lineage and f shows family
                    metadata. Enter expands; k/Backspace kills the selected path; kill ID targets one;
                    select ID commits a path
  t TEXT            insert continuation text (adds a joining space if needed)
  x TEXT            insert exact text
                    after `t ` or `x `, Tab inserts a literal tab character
  check TEXT        commit a continuation phrase only if every token is within the shift bound
  checkx TEXT       same check for an exact phrase without implicit whitespace
  force TEXT        force a continuation phrase with temporary per-token policy shifts
  forcex TEXT       force an exact phrase without implicit whitespace
  h [N]              release control for N tokens (default: configured limit)
  h . [N]            hold through first token containing . ! ?, capped at N
  h | [N]            hold through first token containing a newline, capped at N
                    matching tokens stay whole; no lookahead or trailing tokens
  m                  return to the main table without disclosing rows
  m N                return to the main table and reveal N more ranked rows
  /TERM              find one exact token and show its backend-rank neighborhood
  /"\\n"              JSON escapes preserve exact whitespace/control characters
  ms N               explore the neighborhood of backend rank N
  Ctrl+G             explore the numeric rank currently in the input
  ms                 return to the active token-search neighborhood
  ms + [N]           expand toward larger ranks / lower backend probability
  ms - [N]           expand toward smaller ranks / higher backend probability
  c                  cycle middle-column focus: logit → gap_k1 → margin → z → pct → decode_pct
  C                  clear all overlays and shortcuts
  overlay NAME       toggle any named overlay alongside the others
  context [N|all]    page more of the current context (default: 2000 chars);
                     c N / c all still work (bare c is column focus)
  v                  cycle candidate order: model → policy → Gumbel (Gumbel-Max only)
  V                  toggle policy diagnostics independently of ordering
  l                  cycle logits: none / model / gap@raw1
  L                  toggle model logits + gap@raw1 together
  %                  toggle model soft-max % overlays (model-p / decode-p [/ pol-p])
                     default table is identity-only: rank | token-id | text
                     overlays combine with l / % shortcuts and column focus
                     Δrank = backend rank - policy rank; positive means promoted.
                     model-gap = model logit minus the raw rank-1 model logit;
                     raw rank 1 is therefore always +0.000.
                     pol-p is before temperature/filtering; decode-p is final.
                     numeric selections accept any backend rank in the vocabulary
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
  ?                 show this help

Structured short commands may omit separating spaces: h5, h.5, h|5,
f-5, m10, ms+10, ms-10, and c900 are equivalent to their spaced forms. Text-bearing
commands /, t, x, n, and p keep their whitespace exactly as entered.
"""


_COMPACT_HOLD_BOUNDARY = re.compile(
    r"^h\s*([.|])(?:\s*(\d+))?$",
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


def _decode_bias_member(raw: str, *, label: str) -> tuple[str, bool]:
    value = raw.strip()
    literal = value.startswith("literal:")
    if literal:
        value = value[len("literal:"):].strip()
        if not value.startswith('"'):
            raise EditorError('literal members use literal:"text"')
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            raise EditorError(f"{label} must be a valid JSON string") from exc
        if not isinstance(decoded, str):
            raise EditorError(f"{label} must be a JSON string")
        text = decoded
    else:
        if any(character in value for character in '{}[]"'):
            raise EditorError(f"invalid bare {label}: {value!r}")
        text = value.strip()
    if not text:
        raise EditorError(f"{label} cannot be empty")
    return text, literal


def _parse_bias_members(raw: str, *, label: str) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    value = raw.strip()
    if value.startswith("{"):
        if not value.endswith("}"):
            raise EditorError(f"unterminated {label} group")
        items = _split_human_bias_group(value)
    else:
        items = (value,)
    decoded = tuple(_decode_bias_member(item, label=label) for item in items)
    members = tuple(text for text, _literal in decoded)
    literal = tuple(is_literal for _text, is_literal in decoded)
    if len({(text, mode) for text, mode in zip(members, literal)}) != len(members):
        raise EditorError(f"{label} cannot contain duplicate entries")
    return members, literal


def _bias_adjustment(raw: str) -> tuple[str, float | None] | None:
    match = re.fullmatch(
        r"(?P<op>off|[+\-=])\s*(?P<amount>[+-]?(?:\d+(?:\.\d*)?|\.\d+))?",
        raw,
    )
    if match is None:
        return None
    operator, amount_text = match.group("op"), match.group("amount")
    if operator == "off":
        if amount_text is not None:
            raise EditorError("use off without an amount")
        return operator, None
    amount = None if amount_text is None else float(amount_text)
    if amount is not None and not math.isfinite(amount):
        raise EditorError("bias amount must be finite")
    if operator == "=":
        if amount is None:
            raise EditorError("setting a bias requires an amount, for example =0.5")
    elif amount is not None and amount <= 0:
        raise EditorError("bias adjustment must be finite and positive")
    return operator, amount


def _bias_prefix_message(raw: str) -> str | None:
    """Keep partial group creation in the editor's incomplete state."""
    value = raw.strip()
    group_prefix = re.fullmatch(
        r"b\s+[A-Za-z_][A-Za-z0-9_.-]*\s*->(?:\s*(?P<members>\{.*))?",
        value,
    )
    if group_prefix is None:
        return None
    members = group_prefix.group("members")
    if members is None:
        return "Type group members after `->`, for example `{term}`."
    if not members.endswith("}"):
        return "Finish the bias group with `}`."
    return None


def _required_bias_adjustment(raw: str) -> tuple[str, float | None]:
    """Parse a captured adjustment and turn malformed operators into input errors."""
    parsed = _bias_adjustment(raw)
    if parsed is None:
        raise EditorError("use +[amount], -[amount], =[amount], or off")
    return parsed


def parse_bias_command(raw: str, *, vocabulary_size: int) -> TeacherCommand | None:
    """Parse group, member, token, and attribution commands."""
    value = raw.strip()
    if value in {"b", "groups"}:
        return TeacherCommand(CommandKind.BIAS, bias_status=True)

    inspect_group = re.fullmatch(
        r"(?:groups|b\s+group)\s+(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)", value
    )
    if inspect_group:
        return TeacherCommand(
            CommandKind.BIAS,
            bias_inspect_group=inspect_group.group("name"),
        )

    inspect_token = re.fullmatch(r"b\s+token\s+#(?P<token>[0-9]+)", value)
    if inspect_token:
        token_id = int(inspect_token.group("token"))
        if token_id >= vocabulary_size:
            raise EditorError("token ID is outside the vocabulary")
        return TeacherCommand(CommandKind.BIAS, bias_inspect_token=token_id)

    add_group = re.fullmatch(
        r"b\s+(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s*->\s*(?P<members>\{.*\})",
        value,
    )
    if add_group:
        members, literals = _parse_bias_members(
            add_group.group("members"), label="bias group members"
        )
        return TeacherCommand(
            CommandKind.BIAS,
            bias_group_name=add_group.group("name"),
            bias_group_members=members,
            bias_group_member_literal=literals,
        )

    remove_group = re.fullmatch(
        r"b\s+(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s+remove\s+(?P<members>\{.*\})",
        value,
    )
    if remove_group:
        members, literals = _parse_bias_members(
            remove_group.group("members"), label="bias group members"
        )
        return TeacherCommand(
            CommandKind.BIAS,
            bias_group_remove_name=remove_group.group("name"),
            bias_group_remove_members=members,
            bias_group_remove_member_literal=literals,
        )

    direct_token = re.fullmatch(
        r"b\s+token\s+#(?P<token>[0-9]+)(?:\s+(?P<adjustment>off|[+\-=].*))?",
        value,
    )
    if direct_token:
        token_id = int(direct_token.group("token"))
        if token_id >= vocabulary_size:
            raise EditorError("token ID is outside the vocabulary")
        adjustment = direct_token.group("adjustment")
        if adjustment is None:
            return TeacherCommand(CommandKind.BIAS, bias_inspect_token=token_id)
        operator, amount = _required_bias_adjustment(adjustment)
        if operator is None:
            raise EditorError("use +[amount], -[amount], =[amount], or off")
        return TeacherCommand(
            CommandKind.BIAS,
            bias_operator=operator,
            bias_amount=amount,
            bias_token_id=token_id,
        )

    group_adjust = re.fullmatch(
        r"b\s+(?P<targets>\{.*\}|[A-Za-z_][A-Za-z0-9_.-]*)\s+"
        r"(?P<adjustment>off|[+\-=].*|[+\-])",
        value,
    )
    if group_adjust:
        target_text = group_adjust.group("targets")
        names, literals = _parse_bias_members(target_text, label="bias group target")
        if any(literals):
            raise EditorError("bias adjustments target group names, not literal text")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) for name in names):
            raise EditorError("bias group names must be simple names")
        operator, amount = _required_bias_adjustment(
            group_adjust.group("adjustment")
        )
        if operator is None:
            raise EditorError("use +[amount], -[amount], =[amount], or off")
        return TeacherCommand(
            CommandKind.BIAS,
            bias_operator=operator,
            bias_amount=amount,
            bias_targets=names,
        )

    rank_adjust = re.fullmatch(
        r"(?P<rank>[0-9]+)\s*(?P<adjustment>off|[+\-=].*|[+\-])",
        value,
    )
    if rank_adjust:
        rank = int(rank_adjust.group("rank"))
        if not 1 <= rank <= vocabulary_size:
            raise EditorError("bias rank is outside the vocabulary")
        operator, amount = _required_bias_adjustment(
            rank_adjust.group("adjustment")
        )
        if operator is None:
            raise EditorError("use +[amount], -[amount], =[amount], or off")
        return TeacherCommand(
            CommandKind.BIAS,
            search_rank=rank,
            bias_operator=operator,
            bias_amount=amount,
        )
    return None


def parse_chord(raw: str, vocabulary_size: int) -> tuple[int, ...] | None:
    """Recognize chord syntax without importing its runtime simulation."""
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


def _parse_beam_rank_ranges(
    parts: list[str], *, option: str = "skip"
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for part in parts:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise EditorError(
                f"beam {option} ranks must be positive integers or ranges such as 1-5"
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < first:
            raise EditorError(f"beam {option} ranges must be positive and ascend")
        ranges.append((first, last))

    merged: list[tuple[int, int]] = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return tuple(merged)


def format_beam_rank_ranges(ranges: tuple[tuple[int, int], ...]) -> str:
    return " ".join(
        str(first) if first == last else f"{first}-{last}"
        for first, last in ranges
    )


def parse_beam(
    raw: str,
    *,
    vocabulary_size: int | None = None,
) -> tuple[int, bool, tuple[tuple[int, int], ...], tuple[int, ...]] | None:
    """Recognize deterministic and stochastic beam forms with root controls."""
    parts = raw.strip().split()
    if not parts or parts[0].lower() not in {"beam", "gbeam"}:
        return None
    short_stochastic = parts[0].lower() == "gbeam"
    stochastic = short_stochastic
    cursor = 1
    if (
        not short_stochastic
        and cursor < len(parts)
        and parts[cursor].lower() == "stochastic"
    ):
        stochastic = True
        cursor += 1

    width = 5
    if cursor < len(parts) and parts[cursor].isdecimal():
        width = int(parts[cursor])
        cursor += 1
    if not 1 <= width <= 100:
        raise EditorError("beam width must be between 1 and 100")
    rank_options: dict[str, tuple[tuple[int, int], ...]] = {}
    while cursor < len(parts):
        option = parts[cursor].lower()
        if option not in {"skip", "add"}:
            raise EditorError(
                "use beam [WIDTH] [skip RANKS] [add RANKS], beam stochastic "
                "[WIDTH] [skip RANKS], or gbeam [WIDTH] [skip RANKS]"
            )
        if option in rank_options:
            raise EditorError(f"beam {option} may be specified only once")
        cursor += 1
        rank_start = cursor
        while cursor < len(parts) and parts[cursor].lower() not in {"skip", "add"}:
            cursor += 1
        if rank_start == cursor:
            raise EditorError(f"beam {option} needs one or more model ranks")
        rank_options[option] = _parse_beam_rank_ranges(
            parts[rank_start:cursor], option=option
        )

    skip_ranges = rank_options.get("skip", ())
    add_ranges = rank_options.get("add", ())
    if stochastic and add_ranges:
        raise EditorError("beam add is available only in deterministic beam mode")
    for option, ranges in (("skip", skip_ranges), ("add", add_ranges)):
        if vocabulary_size is not None and any(
            last > vocabulary_size for _, last in ranges
        ):
            raise EditorError(
                f"beam {option} ranks must be between 1 and {vocabulary_size}"
            )
    if (
        vocabulary_size is not None
        and skip_ranges == ((1, vocabulary_size),)
    ):
        raise EditorError("beam root skip ranks cannot exclude the whole vocabulary")

    add_count = sum(last - first + 1 for first, last in add_ranges)
    if add_count > width:
        raise EditorError("beam add cannot reserve more roots than the beam width")
    add_ranks = tuple(
        rank
        for first, last in add_ranges
        for rank in range(first, last + 1)
    )
    for rank in add_ranks:
        if any(first <= rank <= last for first, last in skip_ranges):
            raise EditorError(f"model rank {rank} cannot be both skipped and added")
    return width, stochastic, skip_ranges, add_ranks


def _beam_prefix_message(raw: str) -> str | None:
    """Return guidance while a beam command is still being typed."""
    parts = raw.strip().lower().split()
    if not parts:
        return None
    if len(parts) == 1:
        word = parts[0]
        if word == "g":
            return "Continue typing gbeam or groups."
        if len(word) >= 2:
            for candidate in ("beam", "gbeam", "groups"):
                if candidate.startswith(word) and candidate != word:
                    return f"Continue typing {candidate}."
    if parts[0] == "beam" and len(parts) == 2:
        option = parts[1]
        if option and "stochastic".startswith(option) and option != "stochastic":
            return "Finish typing stochastic, or enter a beam width."
    if parts[0] in {"beam", "gbeam"}:
        option_prefixes = {
            "skip": {"s", "sk", "ski", "skip"},
            "add": {"a", "ad", "add"},
        }
        partial = next(
            (name for name, forms in option_prefixes.items() if parts[-1] in forms),
            None,
        )
        if partial is not None:
            prefix = parts[:-1]
            if prefix[0] == "beam" and len(prefix) > 1 and prefix[1] == "stochastic":
                prefix = [prefix[0], *prefix[2:]]
            header_ok = len(prefix) in {1, 2} and (
                len(prefix) == 1
                or (prefix[1].isdecimal() and 1 <= int(prefix[1]) <= 100)
            )
            # An `add` clause may follow a complete skip list.
            if partial == "add" and len(prefix) >= 3 and prefix[1].isdecimal():
                header_ok = prefix[2] == "skip" and len(prefix) >= 4 and all(
                    re.fullmatch(r"\d+(?:-\d+)?", item) for item in prefix[3:]
                )
            if header_ok:
                if parts[-1] == partial:
                    return f"Type one or more one-based model ranks to {partial} at the first step."
                return f"Finish typing {partial}, then enter one or more one-based model ranks."
    return None


def parse_command(
    raw: str,
    *,
    menu_size: int,
    default_hold_tokens: int,
    vocabulary_size: int | None = None,
    default_search_radius: int = 3,
) -> TeacherCommand:
    beam = parse_beam(raw, vocabulary_size=vocabulary_size)
    if beam is not None:
        (
            beam_width,
            beam_stochastic,
            beam_skip_rank_ranges,
            beam_add_model_ranks,
        ) = beam
        return TeacherCommand(
            CommandKind.BEAM,
            beam_width=beam_width,
            beam_stochastic=beam_stochastic,
            beam_skip_rank_ranges=beam_skip_rank_ranges,
            beam_add_model_ranks=beam_add_model_ranks,
        )
    chord_ranks = parse_chord(raw, vocabulary_size or menu_size)
    if chord_ranks is not None:
        return TeacherCommand(CommandKind.CHORD, chord_ranks=chord_ranks)
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
    if lower == "s" or lower.startswith("s "):
        payload = raw.strip()[1:].strip()
        return TeacherCommand(
            CommandKind.SAMPLER,
            sampler_text=payload or None,
        )
    if re.match(r"draw(?:\s|$)", lower):
        rank_text = command[4:].strip()
        if not rank_text:
            raise EditorError("draw requires a raw rank; usage: draw RAW_RANK")
        if not rank_text.isdecimal():
            raise EditorError("draw raw rank must be a positive integer")
        rank = int(rank_text)
        if rank < 1 or (vocabulary_size is not None and rank > vocabulary_size):
            maximum = vocabulary_size if vocabulary_size is not None else "the vocabulary size"
            raise EditorError(f"draw raw rank must be between 1 and {maximum}")
        return TeacherCommand(CommandKind.DRAW, draw_raw_rank=rank)
    reroll_match = re.fullmatch(r"reroll(?:\s+([+-]?\d+))?", command, re.IGNORECASE)
    if reroll_match is not None:
        seed_text = reroll_match.group(1)
        return TeacherCommand(
            CommandKind.REROLL,
            reroll_seed=int(seed_text) if seed_text is not None else None,
        )
    phrase_command = raw.lstrip()
    phrase_lower = phrase_command.lower()
    for spelling, force, mode in (
        ("checkx", False, "exact"),
        ("check", False, "continuation"),
        ("forcex", True, "exact"),
        ("force", True, "continuation"),
    ):
        prefix = spelling + " "
        if phrase_lower == spelling:
            raise EditorError(f"{spelling} requires phrase text")
        if phrase_lower.startswith(prefix):
            text = phrase_command[len(prefix):]
            if not text:
                raise EditorError(f"{spelling} requires phrase text")
            return TeacherCommand(
                CommandKind.PHRASE,
                phrase_text=text,
                phrase_mode=mode,
                phrase_force=force,
                invoked_as=spelling,
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
        if parts[1] in {".", "|"}:
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
    # Bare c / C reclaim column focus; context keeps `context`, `c N`, `c all`.
    if command == "C" or lower in {"column-clear", "column-focus-clear"}:
        return TeacherCommand(CommandKind.COLUMN_FOCUS, invoked_as="C")
    if lower == "c" or lower in {"column-focus", "column-cycle"}:
        return TeacherCommand(CommandKind.COLUMN_FOCUS, invoked_as=command)
    if lower.startswith("overlay "):
        from .candidate_columns import OVERLAYS

        name = lower.split(maxsplit=1)[1].strip()
        if name not in OVERLAYS or not OVERLAYS[name].wired:
            raise EditorError("unknown overlay; use pct, decode_pct, logit, gap_k1, margin_neighbor, or z")
        return TeacherCommand(CommandKind.OVERLAY_TOGGLE, overlay=name)
    if lower == "context" or lower.startswith(("c ", "context ")):
        parts = command.split()
        if len(parts) > 2:
            raise EditorError("use context, c N, or c all")
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
    if command == "V" or lower in {"policy-column", "policy-columns", "policy-rank-column"}:
        return TeacherCommand(CommandKind.POLICY_COLUMN, invoked_as=command)
    if command == "L":
        return TeacherCommand(CommandKind.LOGIT_VIEW, invoked_as="L")
    if lower in {"l", "logit", "logits", "logit-view"}:
        return TeacherCommand(CommandKind.LOGIT_VIEW, invoked_as=command)
    if command == "%" or lower in {"pct", "probs", "probabilities", "probability-view"}:
        return TeacherCommand(CommandKind.PROBABILITY_VIEW, invoked_as=command)
    if lower in {"v", "policy-view", "policy-sort"}:
        return TeacherCommand(CommandKind.POLICY_VIEW, invoked_as=lower)
    if lower == "n" or lower.startswith("n "):
        note = raw[2:] if len(raw) >= 2 and raw[1:2].isspace() else None
        return TeacherCommand(CommandKind.NOTE_BEFORE, note=note)
    if lower == "p" or lower.startswith("p "):
        note = raw[2:] if len(raw) >= 2 and raw[1:2].isspace() else None
        return TeacherCommand(CommandKind.NOTE_AFTER, note=note)
    if len(raw) >= 2 and raw[:2].lower() == "t ":
        if not raw[2:]:
            raise EditorError("t requires insertion text")
        return TeacherCommand(
            CommandKind.EDIT,
            action=EditAction.insert(raw[2:], InsertMode.CONTINUATION),
        )
    if len(raw) >= 2 and raw[:2].lower() == "x ":
        if not raw[2:]:
            raise EditorError("x requires insertion text")
        return TeacherCommand(
            CommandKind.EDIT,
            action=EditAction.insert(raw[2:], InsertMode.EXACT),
        )
    raise EditorError("unknown command; use ? for help")


def interpret_command(
    raw: str,
    *,
    menu_size: int,
    default_hold_tokens: int,
    vocabulary_size: int,
    default_search_radius: int = 3,
    implicit_accept: bool = True,
) -> CommandInterpretation:
    """Classify a draft and carry the exact command to the submit path.

    Blank input selects the sampled proposal only in an active choice. Other
    prefixes are incomplete only when more input can make them valid.
    """
    if not raw.strip() and implicit_accept:
        return CommandInterpretation(
            raw, CommandState.READY,
            TeacherCommand(CommandKind.EDIT, action=EditAction.accept()),
        )

    stripped = raw.strip()
    lower = stripped.lower()
    beam_prefix = _beam_prefix_message(raw)
    if beam_prefix is not None:
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message=beam_prefix)
    bias_prefix = _bias_prefix_message(raw)
    if bias_prefix is not None:
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message=bias_prefix)
    if raw == "/":
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type token text after /.")
    if lower in {"t", "x"} or (len(raw) == 2 and raw[:2].lower() in {"t ", "x "}):
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type text after the insertion command.")
    if lower in {"check", "checkx", "force", "forcex"} or any(
        raw.lstrip().lower() == spelling + " "
        for spelling in ("check", "checkx", "force", "forcex")
    ):
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type phrase text after the command.")
    if lower == "overlay":
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type a wired overlay name.")
    if lower == "draw":
        return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type a raw rank after draw.")
    chord_parts = stripped.split()
    if chord_parts and chord_parts[0].lower() == "chord" and len(chord_parts) < 3:
        if len(chord_parts) == 1:
            return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type at least two distinct raw ranks.")
        if chord_parts[1].isdecimal() and 1 <= int(chord_parts[1]) <= vocabulary_size:
            return CommandInterpretation(raw, CommandState.INCOMPLETE, message="Type another distinct raw rank.")
    try:
        command = parse_command(
            raw,
            menu_size=menu_size,
            default_hold_tokens=default_hold_tokens,
            vocabulary_size=vocabulary_size,
            default_search_radius=default_search_radius,
        )
    except EditorError as exc:
        return CommandInterpretation(raw, CommandState.INVALID, message=str(exc))
    except (TypeError, ValueError, IndexError):
        # The draft is untrusted text. If a parser branch misses an input
        # validation case, keep it in the normal feedback path instead of
        # letting malformed input terminate the interactive UI.
        return CommandInterpretation(
            raw,
            CommandState.INVALID,
            message="malformed command input; use ? for help",
        )
    return CommandInterpretation(raw, CommandState.READY, command)
