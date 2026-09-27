"""Canonical in-memory state for one sequential token episode.

This object deliberately knows nothing about model backends, persistence, or
terminal UI. It owns the token ledger and the small amount of state needed to
describe the current live branch and replay boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import EditorError


@dataclass
class TrajectoryState:
    """Mutable token-trajectory state shared by the runtime and adapters."""

    initial_token_ids: tuple[int, ...]
    initial_text: str
    visible_token_ids: list[int] = field(default_factory=list)
    terminal_token_id: int | None = None
    terminal_reason: str | None = None
    stream_fingerprint: str | None = None

    def __post_init__(self) -> None:
        self.initial_token_ids = tuple(self.initial_token_ids)
        self.visible_token_ids = list(self.visible_token_ids)
        if not self.initial_token_ids:
            raise EditorError("trajectory requires an initial token ledger")
        if any(type(value) is not int or value < 0 for value in self.initial_token_ids):
            raise EditorError("trajectory initial token IDs must be nonnegative integers")
        if any(type(value) is not int or value < 0 for value in self.visible_token_ids):
            raise EditorError("trajectory visible token IDs must be nonnegative integers")
        if not isinstance(self.initial_text, str):
            raise EditorError("trajectory initial text must be a string")
        if self.stream_fingerprint is not None and not isinstance(self.stream_fingerprint, str):
            raise EditorError("trajectory stream fingerprint must be a string or null")

    @property
    def boundary(self) -> int:
        return len(self.visible_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return [*self.initial_token_ids, *self.visible_token_ids]

    @property
    def ended(self) -> bool:
        return self.terminal_reason is not None

    def set_stream_fingerprint(self, stream_fingerprint: str | None) -> None:
        """Restore the sampler stream identity associated with the live branch."""
        if stream_fingerprint is not None and not isinstance(stream_fingerprint, str):
            raise EditorError("stream fingerprint must be a string or null")
        self.stream_fingerprint = stream_fingerprint

    def rewind_to(self, boundary: int) -> list[int]:
        """Retain the live branch through ``boundary`` and clear termination."""

        if type(boundary) is not int or boundary < 0 or boundary > self.boundary:
            raise EditorError(
                f"rewind boundary must be between 0 and {self.boundary}"
            )
        self.visible_token_ids = self.visible_token_ids[:boundary]
        self.terminal_token_id = None
        self.terminal_reason = None
        return list(self.visible_token_ids)

    def terminate(self, reason: str = "menu-end") -> None:
        """Seal the trajectory without manufacturing a terminal token."""

        if self.ended:
            return
        if not isinstance(reason, str) or not reason:
            raise EditorError("termination reason must be nonempty")
        self.terminal_reason = reason
