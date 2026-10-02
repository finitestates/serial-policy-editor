"""Small deterministic rectangle and split-allocation primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    """A terminal-cell rectangle with a zero-based origin."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("rectangle origin must be nonnegative")
        if self.width < 0 or self.height < 0:
            raise ValueError("rectangle dimensions must be nonnegative")


@dataclass(frozen=True, slots=True)
class SplitItem:
    """One named child in a sequential split.

    Minima are reserved by ``priority`` (lower values win). Children then grow
    toward ``preferred`` by ``grow_priority``. Flexible children receive any
    remaining rows; an uncapped flexible child is needed when the declared
    children must cover the complete parent.
    """

    name: str
    minimum: int = 0
    preferred: int | None = None
    priority: int = 0
    grow_priority: int | None = None
    flex: bool = False


def allocate_vertical_split(parent: Rect, items: Sequence[SplitItem]) -> dict[str, Rect]:
    """Split a parent into sequential full-width child rectangles.

    Children retain declaration order from top to bottom. If minima cannot all
    fit, lower-priority children receive a zero-height rectangle. Any remaining
    rows go to flexible children. The function is pure and raises when the
    declared sizes leave an unexplained tail in the parent.
    """

    if len({item.name for item in items}) != len(items):
        raise ValueError("split child names must be unique")
    for item in items:
        if not item.name:
            raise ValueError("split child names must be non-empty")
        if item.minimum < 0:
            raise ValueError(f"{item.name}: minimum must be nonnegative")
        if item.preferred is not None and item.preferred < item.minimum:
            raise ValueError(f"{item.name}: preferred must be at least minimum")

    sizes = {item.name: 0 for item in items}
    remaining = parent.height
    kept: set[str] = set()

    for item in sorted(items, key=lambda child: child.priority):
        if item.minimum <= remaining:
            sizes[item.name] = item.minimum
            remaining -= item.minimum
            kept.add(item.name)

    growth = sorted(
        enumerate(items),
        key=lambda pair: (
            pair[1].grow_priority if pair[1].grow_priority is not None else pair[1].priority,
            pair[0],
        ),
    )
    for _index, item in growth:
        if item.name not in kept or item.preferred is None:
            continue
        extra = min(remaining, item.preferred - sizes[item.name])
        sizes[item.name] += extra
        remaining -= extra

    for _index, item in growth:
        if remaining <= 0:
            break
        if item.name not in kept or not item.flex:
            continue
        capacity = None if item.preferred is None else item.preferred - sizes[item.name]
        extra = remaining if capacity is None else min(remaining, max(0, capacity))
        sizes[item.name] += extra
        remaining -= extra

    if remaining:
        raise ValueError(
            f"vertical split leaves {remaining} rows outside its children; "
            "add a flexible child or increase a preferred size"
        )

    regions: dict[str, Rect] = {}
    cursor = parent.y
    for item in items:
        size = sizes[item.name]
        regions[item.name] = Rect(parent.x, cursor, parent.width, size)
        cursor += size
    return regions
