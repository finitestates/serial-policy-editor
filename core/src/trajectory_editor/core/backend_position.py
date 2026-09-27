"""Backend cache positions and comparisons with the committed token ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackendPosition:
    """A backend's current token prefix and the physical cache range it reports.

    ``token_ids`` records the exact sequence submitted through the adapter.
    Cache APIs report positions and depth, not token identity, so the adapter's
    input ledger supplies identity while the backend reports where its cache is.
    ``cursor`` is the next absolute token position after the evaluated prefix.
    """

    token_ids: tuple[int, ...]
    cursor: int
    cache_start: int | None
    cache_end: int | None
    cache_reusable: bool
    logits_valid: bool


AlignmentStatus = Literal[
    "aligned",
    "logits-stale",
    "backend-behind",
    "backend-ahead",
    "diverged",
    "ledger-mismatch",
    "unknown",
]


@dataclass(frozen=True)
class PositionComparison:
    """Relationship between one backend position and a desired token prefix."""

    status: AlignmentStatus
    common_prefix_length: int
    backend_cursor: int
    ledger_cursor: int


def longest_common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the shared token count without materializing either prefix."""

    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def compare_backend_position(
    position: BackendPosition, token_ids: tuple[int, ...] | list[int]
) -> PositionComparison:
    """Say whether a reported backend state matches, trails, or diverges.

    A length-only match is insufficient: two branches can occupy the same
    absolute position with different tokens. Compare the adapter's exact
    sequence identity as well as the backend-reported cache range.
    """

    target = tuple(token_ids)
    common = longest_common_prefix(position.token_ids, target)
    consistent = position.cursor >= 0
    if position.cache_reusable:
        consistent = consistent and (
            position.cache_start is not None
            and position.cache_end is not None
            and 0 <= position.cache_start <= position.cache_end
            and position.cache_end + 1 == position.cursor
        )
    if not consistent:
        status: AlignmentStatus = "unknown"
    elif position.cursor != len(position.token_ids):
        # Keep the exact cache cursor distinct from the adapter's submitted
        # token ledger. The backend adapter can often crop/continue from this
        # state without throwing the whole cache away.
        status = "ledger-mismatch"
    elif common == len(position.token_ids) == len(target):
        status = "aligned" if position.logits_valid else "logits-stale"
    elif common == len(position.token_ids):
        status = "backend-behind"
    elif common == len(target):
        status = "backend-ahead"
    else:
        status = "diverged"
    return PositionComparison(
        status=status,
        common_prefix_length=common,
        backend_cursor=position.cursor,
        ledger_cursor=len(target),
    )


def position_report(
    backend, token_ids: tuple[int, ...] | list[int]
) -> PositionComparison:
    """Compare a backend's reported position with a desired token ledger."""

    position = getattr(backend, "position", None)
    if not callable(position):
        target = tuple(token_ids)
        return PositionComparison("unknown", 0, -1, len(target))
    return compare_backend_position(position(), token_ids)


def position_backend(
    backend, token_ids: tuple[int, ...] | list[int]
) -> bool:
    """Use the backend's cache-aware prefix operation when it has one.

    Return false when the adapter exposes only the minimal reset operation so
    the owning runtime can perform its explicit full-prefix fallback.
    """

    prefix = list(token_ids)
    comparison = position_report(backend, prefix)
    if comparison.status == "aligned":
        return True
    branch = getattr(backend, "branch_to_prefix", None)
    if not callable(branch):
        return False
    branch(prefix)
    return True


__all__ = [
    "AlignmentStatus",
    "BackendPosition",
    "PositionComparison",
    "compare_backend_position",
    "longest_common_prefix",
    "position_backend",
    "position_report",
]
