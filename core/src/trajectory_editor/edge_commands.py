"""Pure parsing for the command entered at a live EDGE.

This module deliberately knows only about command syntax.  In particular, it
does not resolve episode or branch references, check boundary ranges, or decide
whether a command is available in a durable or ephemeral session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypeAlias

from .core.errors import EditorError


class EdgeCommandParseError(EditorError):
    """Raised when an EDGE command does not match the command grammar."""


@dataclass(frozen=True, slots=True)
class ContinueCommand:
    """Continue with the currently configured allowance."""


@dataclass(frozen=True, slots=True)
class BudgetCommand:
    """Set the next continuation allowance, or remove it with ``None``."""

    tokens: int | None


@dataclass(frozen=True, slots=True)
class SamplerCommand:
    """Pass an optional raw sampler override to the sampler-aware caller."""

    text: str | None = None


@dataclass(frozen=True, slots=True)
class RewindCommand:
    boundary: int


@dataclass(frozen=True, slots=True)
class ForkCommand:
    boundary: int


@dataclass(frozen=True, slots=True)
class SwitchCommand:
    """Switch using an unresolved branch or episode reference."""

    reference: str


@dataclass(frozen=True, slots=True)
class NewCommand:
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class EndCommand:
    pass


@dataclass(frozen=True, slots=True)
class QuitCommand:
    pass


@dataclass(frozen=True, slots=True)
class ProjectCommand:
    pass


@dataclass(frozen=True, slots=True)
class ForkMapCommand:
    pass


@dataclass(frozen=True, slots=True)
class BranchesCommand:
    pass


@dataclass(frozen=True, slots=True)
class ListCommand:
    """List the current context, optionally including finished entries."""

    include_finished: bool = False


@dataclass(frozen=True, slots=True)
class RenameCommand:
    """Rename the current durable episode; the caller owns the capability."""

    title: str


@dataclass(frozen=True, slots=True)
class ExportCommand:
    path: Path


@dataclass(frozen=True, slots=True)
class SaveCommand:
    workspace: Path
    reference: str | None = None


@dataclass(frozen=True, slots=True)
class SaveFamilyCommand:
    workspace: Path
    root_reference: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayCommand:
    """Replay an unresolved source reference through an optional boundary."""

    source: str
    until: int | None = None


@dataclass(frozen=True, slots=True)
class ReplaySelectionCommand:
    """Open a source replay map and let the caller choose its boundary."""

    source: str


EdgeCommand: TypeAlias = (
    ContinueCommand
    | BudgetCommand
    | SamplerCommand
    | RewindCommand
    | ForkCommand
    | SwitchCommand
    | NewCommand
    | EndCommand
    | QuitCommand
    | ProjectCommand
    | ForkMapCommand
    | BranchesCommand
    | ListCommand
    | RenameCommand
    | ExportCommand
    | SaveCommand
    | SaveFamilyCommand
    | ReplayCommand
    | ReplaySelectionCommand
)


# Short names mirror the command vocabulary while the ``*Command`` names make
# the public type union self-documenting.
Continue = ContinueCommand
Budget = BudgetCommand
Sampler = SamplerCommand
Rewind = RewindCommand
Fork = ForkCommand
Switch = SwitchCommand
New = NewCommand
End = EndCommand
Quit = QuitCommand
Project = ProjectCommand
ForkMap = ForkMapCommand
Branches = BranchesCommand
List = ListCommand
Rename = RenameCommand
Export = ExportCommand
Save = SaveCommand
SaveFamily = SaveFamilyCommand
Replay = ReplayCommand
ReplaySelection = ReplaySelectionCommand

# Descriptive aliases for callers that refer to the ``m`` form as a replay
# map command, while keeping one concrete tagged variant in the union.
NameCommand = RenameCommand
ReplayMapCommand = ReplaySelectionCommand


def _parse_error(message: str) -> NoReturn:
    raise EdgeCommandParseError(message)


def _integer(value: str, *, label: str) -> int:
    try:
        return int(value)
    except ValueError:
        _parse_error(f"{label} must be an integer")


def _require_arity(parts: list[str], expected: int | range, *, usage: str) -> None:
    if len(parts) not in (expected if isinstance(expected, range) else {expected}):
        _parse_error(f"use {usage}")


def parse_edge_command(raw: str) -> EdgeCommand:
    """Parse one raw EDGE command without performing runtime side effects.

    Blank input is an explicit :class:`ContinueCommand`.  Command names and
    their documented aliases are case-insensitive; payloads retain their raw
    spelling.  A returned reference is intentionally not resolved, so a value
    such as ``#7`` remains exactly ``"#7"``.
    """

    if not isinstance(raw, str):
        _parse_error("EDGE command must be text")

    text = raw.strip()
    if not text:
        return ContinueCommand()

    parts = text.split()
    command = parts[0].lower()

    if command in {"c", "continue"}:
        _require_arity(parts, 1, usage="c or continue")
        return ContinueCommand()

    if command in {"n", "next"}:
        _require_arity(parts, 2, usage="n N, n off, next N, or next off")
        allowance = parts[1].lower()
        if allowance in {"off", "none", "unlimited"}:
            return BudgetCommand(None)
        tokens = _integer(parts[1], label="budget")
        if tokens < 1:
            _parse_error("budget must be a positive integer")
        return BudgetCommand(tokens)

    if command in {"s", "sampler"}:
        payload = text[len(parts[0]) :].strip()
        return SamplerCommand(payload or None)

    if command == "rewind":
        _require_arity(parts, 2, usage="rewind N")
        return RewindCommand(_integer(parts[1], label="rewind boundary"))

    if command in {"f", "fork"}:
        _require_arity(parts, 2, usage="f N or fork N")
        return ForkCommand(_integer(parts[1], label="fork boundary"))

    if command == "switch":
        _require_arity(parts, 2, usage="switch REFERENCE")
        return SwitchCommand(parts[1])

    # A bare #N/reference is the compact switch spelling.  Restrict this to a
    # single token so an accidental extra argument cannot be silently dropped.
    if len(parts) == 1 and parts[0].startswith("#"):
        return SwitchCommand(parts[0])

    if command == "new":
        prompt = text[len(parts[0]) :].lstrip()
        return NewCommand(prompt or None)

    if command in {"e", "end"}:
        _require_arity(parts, 1, usage="e or end")
        return EndCommand()

    if command in {"q", "quit"}:
        _require_arity(parts, 1, usage="q or quit")
        return QuitCommand()

    if command in {"p", "project", "r", "review"}:
        _require_arity(parts, 1, usage="p or project")
        return ProjectCommand()

    if command in {"fm", "fork-map", "forkmap"}:
        _require_arity(parts, 1, usage="fm or fork-map")
        return ForkMapCommand()

    if command == "ls":
        if len(parts) == 1:
            return ListCommand()
        if len(parts) == 2 and parts[1].lower() == "all":
            return ListCommand(include_finished=True)
        _parse_error("use ls or ls all")

    if command == "branches":
        _require_arity(parts, 1, usage="branches")
        return BranchesCommand()

    if command == "name":
        title = text[len(parts[0]) :].lstrip()
        if not title:
            _parse_error("name requires a title")
        return RenameCommand(title)

    if command == "export":
        _require_arity(parts, 2, usage="export FILE")
        return ExportCommand(Path(parts[1]))

    if command in {"save-family", "savefamily"}:
        _require_arity(parts, range(2, 4), usage="save-family WORKSPACE [ROOT_ID]")
        return SaveFamilyCommand(
            Path(parts[1]),
            parts[2] if len(parts) == 3 else None,
        )

    if command == "save" and len(parts) >= 2 and parts[1].lower() == "family":
        _require_arity(parts, range(3, 5), usage="save family WORKSPACE [ROOT_ID]")
        return SaveFamilyCommand(
            Path(parts[2]),
            parts[3] if len(parts) == 4 else None,
        )

    if command == "save":
        _require_arity(parts, range(2, 4), usage="save WORKSPACE [ID]")
        return SaveCommand(
            Path(parts[1]),
            parts[2] if len(parts) == 3 else None,
        )

    if command in {"spr", "replay"}:
        if len(parts) == 3 and parts[2] == "m":
            return ReplaySelectionCommand(parts[1])
        _require_arity(parts, 2 if len(parts) == 2 else 4, usage="spr EPISODE [--until Y]")
        if len(parts) == 2:
            return ReplayCommand(parts[1])
        if parts[2] != "--until":
            _parse_error("use spr EPISODE [--until Y]")
        return ReplayCommand(
            parts[1],
            _integer(parts[3], label="replay boundary"),
        )

    _parse_error("unknown EDGE command")


__all__ = [
    "Branches",
    "BranchesCommand",
    "Budget",
    "BudgetCommand",
    "Continue",
    "ContinueCommand",
    "EdgeCommand",
    "EdgeCommandParseError",
    "End",
    "EndCommand",
    "Export",
    "ExportCommand",
    "Fork",
    "ForkCommand",
    "ForkMap",
    "ForkMapCommand",
    "List",
    "ListCommand",
    "NameCommand",
    "New",
    "NewCommand",
    "Project",
    "ProjectCommand",
    "Quit",
    "QuitCommand",
    "Replay",
    "ReplayMapCommand",
    "ReplayCommand",
    "ReplaySelection",
    "ReplaySelectionCommand",
    "Rename",
    "RenameCommand",
    "Rewind",
    "RewindCommand",
    "Sampler",
    "SamplerCommand",
    "Save",
    "SaveCommand",
    "SaveFamily",
    "SaveFamilyCommand",
    "Switch",
    "SwitchCommand",
    "parse_edge_command",
]
