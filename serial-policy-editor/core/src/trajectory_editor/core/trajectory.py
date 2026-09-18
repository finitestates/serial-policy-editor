"""Canonical in-memory state for one sequential token episode.

This object deliberately knows nothing about model backends, persistence, or
terminal UI. It owns the token ledger and the small amount of state needed to
describe the current live branch and replay coordinate.
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
    coordinate_offset: int = 0
    stream_fingerprint: str | None = None
    max_tokens: int | None = None
    checkpoint_boundary: int | None = None

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
        if type(self.coordinate_offset) is not int or self.coordinate_offset < 0:
            raise EditorError("trajectory coordinate offset must be nonnegative")
        if self.stream_fingerprint is not None and not isinstance(self.stream_fingerprint, str):
            raise EditorError("trajectory stream fingerprint must be a string or null")
        if self.max_tokens is not None and (
            type(self.max_tokens) is not int or self.max_tokens < 1
        ):
            raise EditorError("trajectory max_tokens must be positive or null")
        if self.checkpoint_boundary is not None and (
            type(self.checkpoint_boundary) is not int or self.checkpoint_boundary < 0
        ):
            raise EditorError("trajectory checkpoint boundary must be nonnegative or null")

    @property
    def boundary(self) -> int:
        return len(self.visible_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return [*self.initial_token_ids, *self.visible_token_ids]

    @property
    def remaining(self) -> int | None:
        if self.checkpoint_boundary is None:
            return None
        return max(0, self.checkpoint_boundary - self.boundary)

    @property
    def checkpointed(self) -> bool:
        return self.terminal_reason is None and self.remaining == 0

    @property
    def ended(self) -> bool:
        return self.terminal_reason is not None

    def set_budget(
        self,
        max_tokens: int | None,
        checkpoint_boundary: int | None,
    ) -> None:
        """Update the allowance state restored by a lifecycle adapter."""

        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise EditorError("trajectory max_tokens must be positive or null")
        if checkpoint_boundary is not None and (
            type(checkpoint_boundary) is not int or checkpoint_boundary < 0
        ):
            raise EditorError("trajectory checkpoint boundary must be nonnegative or null")
        self.max_tokens = max_tokens
        self.checkpoint_boundary = checkpoint_boundary

    def set_coordinates(self, *, stream_fingerprint: str | None, coordinate_offset: int) -> None:
        """Restore the replay coordinate associated with the live branch."""

        if stream_fingerprint is not None and not isinstance(stream_fingerprint, str):
            raise EditorError("trajectory stream fingerprint must be a string or null")
        if type(coordinate_offset) is not int or coordinate_offset < 0:
            raise EditorError("trajectory coordinate offset must be nonnegative")
        self.stream_fingerprint = stream_fingerprint
        self.coordinate_offset = coordinate_offset

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
