"""Collect initial and replacement text prompts for episode surfaces."""

from __future__ import annotations

from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator


def read_initial_prompt() -> str:
    """Open the multiline composer used when no initial prompt was supplied."""

    return prompt(
        "Write at least one character. Press Escape then Enter to continue.\n\n",
        multiline=True,
        validator=Validator.from_callable(
            lambda text: len(text) >= 1,
            error_message="Write at least one character.",
        ),
        validate_while_typing=False,
    )


def read_new_prompt(io: Any) -> str | None:
    """Use a live multiline composer when available for a bare EDGE ``new``."""

    reader = getattr(io, "read_multiline_prompt", None)
    if callable(reader):
        return reader()
    # Lightweight test/script IOs do not own a prompt-toolkit surface.
    return io.read("New prompt > ")


__all__ = ["read_initial_prompt", "read_new_prompt"]
