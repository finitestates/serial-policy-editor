"""Collect initial and replacement text prompts for episode surfaces."""

from __future__ import annotations

from pathlib import Path

from .terminal_contracts import PromptRequest, TerminalProtocol


def read_new_prompt(io: TerminalProtocol) -> str | None:
    """Compose an initial or replacement prompt in the owning terminal."""

    return io.prompt(PromptRequest("New prompt > ", multiline=True, isolated=True))


def read_prompt_file(path: Path) -> str:
    """Keep line breaks and trailing newlines supplied in a prompt file."""

    with path.open("r", encoding="utf-8", newline="") as source:
        return source.read()
