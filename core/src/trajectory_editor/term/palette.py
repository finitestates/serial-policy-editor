"""Searchable command templates for the active request."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from ..edge_help import edge_help
from ..teacher_commands import HELP_TEXT
from ..terminal_contracts import BeamViewState, ChoiceViewState, EdgeViewState


@dataclass(frozen=True)
class PaletteEntry:
    title: str
    insert: str
    help: str


def _teacher_palette_entries() -> tuple[PaletteEntry, ...]:
    """Derive searchable command examples from the shared teacher help text."""
    replacements = {
        "RANK+ / RANK-": ("1+", "Adjust the selected token's bias."),
        "RANK=VALUE": ("1=0", "Set or clear a direct token adjustment."),
        "1..N": ("1", "Commit a candidate by raw rank."),
        "s top_k=20|none": ("s top_k=20", "Change sampler settings."),
        "t TEXT": ("t ", "Insert continuation text."),
        "x TEXT": ("x ", "Insert exact text."),
        "check TEXT": ("check ", "Commit a checked continuation phrase."),
        "checkx TEXT": ("checkx ", "Commit a checked exact phrase."),
        "force TEXT": ("force ", "Force a continuation phrase."),
        "forcex TEXT": ("forcex ", "Force exact text."),
        "h [N]": ("h", "Release control for the requested token count."),
        "f N": ("f ", "Fork at an absolute boundary."),
        "f - N": ("f - ", "Fork relative to this boundary."),
        "n [TEXT]": ("n ", "Add a note before the current decision."),
        "p [TEXT]": ("p ", "Add a note after the most recent update."),
        "e | eog": ("e", "Preview and confirm a teacher selected end token."),
        "e! | eog!": ("e!", "Commit a teacher selected end token."),
        "q | finish": ("q", "Open the live edge menu."),
        "[ / ]": ("[", "Review a token boundary."),
    }
    result: list[PaletteEntry] = []
    for line in HELP_TEXT.splitlines():
        if not line.startswith("  "):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        head = parts[0]
        if head in {"READY", "Tab", "Ctrl+G", "numeric", "terms", "multi-token", "after"}:
            continue
        if head not in {
            "accept", "groups", "b", "RANK+", "RANK=VALUE", "s", "reroll",
            "draw", "1..N", "chord", "beam", "gbeam", "t", "x", "check",
            "checkx", "force", "forcex", "h", "m", "/TERM", "ms", "c", "C",
            "overlay", "context", "v", "V", "l", "L", "%", "[", "f", "n",
            "p", "e", "e!", "q",
        }:
            continue
        title, _, description = stripped.partition("  ")
        if not description:
            description = parts[1] if len(parts) > 1 else title
        insert, help_text = replacements.get(title, (title, description.strip()))
        if title.startswith(("b NAME", "b {", "b token")):
            insert = "b "
        elif title.startswith(("beam", "gbeam")):
            insert = title.split()[0]
        elif title.startswith(("chord",)):
            insert = "chord "
        elif title.startswith(("reroll", "draw", "overlay", "context", "ms")):
            insert = title.split()[0] + " "
        elif title.startswith(("m N", "m ")):
            insert = "m "
        elif title.startswith("/TERM"):
            insert = "/"
        elif title.startswith("[ / ]"):
            insert = "["
        result.append(PaletteEntry(title, insert, help_text.strip()))
    known = {entry.title for entry in result}
    for title, insert, detail in (
        ("sampler settings", "s ", "Change one or more sampler settings."),
        ("token bias", "b ", "Change a group or token bias."),
        ("fork boundary", "f", "Fork at the current boundary."),
        ("help", "", "Show the full command list."),
    ):
        if title not in known:
            result.append(PaletteEntry(title, insert, detail))
    return tuple(result)


TEACHER_PALETTE_ENTRIES = _teacher_palette_entries()
BEAM_PALETTE_ENTRIES = (
    PaletteEntry("resume", "resume", "Continue from the beam edge."),
    PaletteEntry("select <label>", "select ", "Commit the selected beam branch."),
    PaletteEntry("advance N", "advance 1", "Advance one or more beam steps."),
    PaletteEntry("rewind", "rewind", "Return one beam step."),
    PaletteEntry("kill <label>", "kill ", "Remove a beam branch."),
    PaletteEntry("protect", "protect", "Protect the selected deterministic lineage."),
    PaletteEntry("families", "families", "Toggle branch family details."),
    PaletteEntry("return", "return", "Return to the teacher decision."),
)
HELP_PALETTE_ENTRY = PaletteEntry("help", "", "Show the full command list.")


def edge_insert_command(command: str) -> str:
    for prefix, insert in (
        ("s ", "s "), ("reroll", "reroll "), ("f ", "f "), ("#", "#"),
        ("new ", "new "), ("name ", "name "), ("rewind ", "rewind "),
        ("export ", "export "), ("save", "save "), ("spr", "spr "),
        ("switch ", "switch "),
    ):
        if command.startswith(prefix):
            return insert
    if " / " in command:
        return command.split(" / ", 1)[0]
    return command.split()[0]


def palette_entries(state: Any) -> tuple[PaletteEntry, ...]:
    if isinstance(state, EdgeViewState):
        commands = tuple(
            PaletteEntry(item.command, edge_insert_command(item.command), item.description)
            for item in edge_help(state.mode)
        )
        return (*commands, HELP_PALETTE_ENTRY)
    if isinstance(state, BeamViewState):
        return (*BEAM_PALETTE_ENTRIES, HELP_PALETTE_ENTRY)
    if isinstance(state, ChoiceViewState):
        return TEACHER_PALETTE_ENTRIES
    return ()


def fuzzy_score(query: str, title: str) -> tuple[float, tuple[int, ...]]:
    """Score an in-order subsequence match; 0 means no match.

    Returns the score and the matched character positions for highlighting.
    Consecutive matches, word starts, and a prefix match score higher.
    """
    query = query.lower().strip()
    if not query:
        return 1.0, ()
    lowered = title.lower()
    positions: list[int] = []
    start = 0
    for character in query:
        if character == " ":
            continue
        found = lowered.find(character, start)
        if found < 0:
            return 0.0, ()
        positions.append(found)
        start = found + 1
    score = 1.0
    for previous, current in pairwise(positions):
        if current == previous + 1:
            score += 2.0
    for position in positions:
        if position == 0 or not lowered[position - 1].isalnum():
            score += 1.5
    if lowered.startswith(query):
        score += 5.0
    score -= len(title) * 0.01
    return score, tuple(positions)


def search(entries: tuple[PaletteEntry, ...], query: str) -> list[tuple[PaletteEntry, tuple[int, ...]]]:
    scored = []
    for order, entry in enumerate(entries):
        score, positions = fuzzy_score(query, entry.title)
        if score > 0:
            scored.append((-score, order, entry, positions))
    scored.sort(key=lambda item: (item[0], item[1]) if query.strip() else (0, item[1]))
    return [(entry, positions) for _score, _order, entry, positions in scored]
