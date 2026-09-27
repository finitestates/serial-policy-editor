"""Named bias groups, token routes, and source-aware bias evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core.errors import EditorError


GROUP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _token_ids(value: Any, *, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EditorError(f"{path} must be a token-id sequence")
    tokens = tuple(value)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise EditorError(f"{path} must contain nonnegative integer token IDs")
    return tokens


def member_surfaces(text: str, *, literal: bool = False) -> tuple[str, ...]:
    """Return the finite visible forms compiled for one group member."""
    if not isinstance(text, str) or not text:
        raise EditorError("bias group member text must be nonempty")
    if literal:
        return (text,)
    forms = [text, text.lower(), text[:1].upper() + text[1:], text.upper()]
    result: list[str] = []
    for form in forms:
        for surface in (form, " " + form):
            if surface not in result:
                result.append(surface)
    return tuple(result)


@dataclass(frozen=True)
class BiasRoute:
    """One tokenized member form, retaining its printable surface."""

    token_ids: tuple[int, ...]
    surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", _token_ids(self.token_ids, path="bias route"))
        if (
            not isinstance(self.surfaces, Sequence)
            or isinstance(self.surfaces, (str, bytes, bytearray))
            or not self.surfaces
            or any(not isinstance(surface, str) or not surface for surface in self.surfaces)
        ):
            raise EditorError("bias route surfaces must be nonempty strings")
        object.__setattr__(self, "surfaces", tuple(dict.fromkeys(self.surfaces)))

    @classmethod
    def from_record(cls, value: Any) -> "BiasRoute":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias route must be an object")
        unknown = set(value) - {"token_ids", "surfaces"}
        if unknown:
            raise EditorError(f"unknown bias route fields: {', '.join(sorted(unknown))}")
        if "token_ids" not in value or "surfaces" not in value:
            raise EditorError("bias route requires token_ids and surfaces")
        return cls(token_ids=value["token_ids"], surfaces=value["surfaces"])

    def to_dict(self) -> dict[str, Any]:
        return {"token_ids": list(self.token_ids), "surfaces": list(self.surfaces)}


@dataclass(frozen=True)
class BiasMember:
    """A phrase and the bounded token routes compiled from its surface forms."""

    text: str
    routes: tuple[BiasRoute, ...]
    literal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise EditorError("bias group member text must be nonempty")
        if type(self.literal) is not bool:
            raise EditorError("bias group member literal must be a boolean")
        try:
            routes = tuple(BiasRoute.from_record(route) for route in self.routes)
        except TypeError as exc:
            raise EditorError("bias group member routes must be a list") from exc
        if not routes:
            raise EditorError("bias group member must have at least one token route")
        if len({route.token_ids for route in routes}) != len(routes):
            raise EditorError("bias group member routes must have distinct token IDs")
        object.__setattr__(self, "routes", tuple(sorted(routes, key=lambda route: route.token_ids)))

    @classmethod
    def compile(
        cls,
        text: str,
        backend: Any,
        *,
        literal: bool = False,
    ) -> "BiasMember":
        by_route: dict[tuple[int, ...], list[str]] = {}
        for surface in member_surfaces(text, literal=literal):
            tokens = tuple(backend.tokenize(surface, add_bos=False, special=False))
            if not tokens or any(
                type(token) is not int
                or token < 0
                or token >= backend.vocabulary_size()
                or backend.is_eog(token)
                for token in tokens
            ):
                raise EditorError(
                    f"bias member form {surface!r} produced no ordinary model tokens"
                )
            by_route.setdefault(tokens, []).append(surface)
        routes = tuple(
            BiasRoute(token_ids=tokens, surfaces=tuple(surfaces))
            for tokens, surfaces in by_route.items()
        )
        return cls(text=text, routes=routes, literal=literal)

    @classmethod
    def from_record(cls, value: Any) -> "BiasMember":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias group member must be an object")
        unknown = set(value) - {"text", "routes", "literal"}
        if unknown:
            raise EditorError(f"unknown bias member fields: {', '.join(sorted(unknown))}")
        if "text" not in value or "routes" not in value:
            raise EditorError("bias group member requires text and routes")
        return cls(
            text=value["text"],
            routes=value["routes"],
            literal=value.get("literal", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "literal": self.literal,
            "routes": [route.to_dict() for route in self.routes],
        }


@dataclass(frozen=True)
class BiasGroup:
    """A named collection of phrase members sharing one logit adjustment."""

    name: str
    members: tuple[BiasMember, ...]
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not GROUP_NAME_RE.fullmatch(self.name):
            raise EditorError(
                "bias group names must begin with a letter or underscore and "
                "contain only letters, numbers, underscores, periods, or hyphens"
            )
        if type(self.bias) not in (int, float) or not math.isfinite(self.bias):
            raise EditorError("bias group amount must be finite")
        try:
            members = tuple(BiasMember.from_record(member) for member in self.members)
        except TypeError as exc:
            raise EditorError("bias group members must be a list") from exc
        if not members:
            raise EditorError("bias groups must contain at least one member")
        if len({(member.text, member.literal) for member in members}) != len(members):
            raise EditorError("bias group members must be distinct")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "bias", float(self.bias))

    @classmethod
    def from_record(cls, value: Any) -> "BiasGroup":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias group must be an object")
        unknown = set(value) - {"name", "members", "bias"}
        if unknown:
            raise EditorError(f"unknown bias group fields: {', '.join(sorted(unknown))}")
        if "name" not in value or "members" not in value:
            raise EditorError("bias group requires name and members")
        return cls(name=value["name"], members=value["members"], bias=value.get("bias", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bias": self.bias,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class BiasToken:
    """A direct, context-free adjustment for one tokenizer token ID."""

    token_id: int
    bias: float

    def __post_init__(self) -> None:
        if type(self.token_id) is not int or self.token_id < 0:
            raise EditorError("bias token ID must be a nonnegative integer")
        if type(self.bias) not in (int, float) or not math.isfinite(self.bias):
            raise EditorError("token bias amount must be finite")
        object.__setattr__(self, "bias", float(self.bias))

    @classmethod
    def from_record(cls, value: Any) -> "BiasToken":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("token bias must be an object")
        unknown = set(value) - {"token_id", "bias"}
        if unknown:
            raise EditorError(f"unknown token bias fields: {', '.join(sorted(unknown))}")
        if "token_id" not in value or "bias" not in value:
            raise EditorError("token bias requires token_id and bias")
        return cls(token_id=value["token_id"], bias=value["bias"])

    def to_dict(self) -> dict[str, Any]:
        return {"token_id": self.token_id, "bias": self.bias}


@dataclass(frozen=True)
class BiasMemberRoute:
    """A group's member route and whether it matches the current prefix."""

    member_text: str
    member_literal: bool
    route: BiasRoute
    active: bool


@dataclass(frozen=True)
class BiasContribution:
    """One source's potential or active adjustment to a token."""

    token_id: int
    amount: float
    active: bool
    source: str
    group_name: str | None = None
    member_routes: tuple[BiasMemberRoute, ...] = ()


def _endswith(history: Sequence[int], prefix: Sequence[int]) -> bool:
    if len(history) < len(prefix):
        return False
    if not prefix:
        return True
    return tuple(history[-len(prefix):]) == tuple(prefix)


def bias_contributions(
    groups: Sequence[BiasGroup],
    token_biases: Sequence[BiasToken],
    history: Sequence[int] | None,
    *,
    include_inactive: bool = False,
) -> tuple[BiasContribution, ...]:
    """Return stable, attributable bias sources for the supplied context."""
    if history is None and any(
        group.bias != 0.0
        and any(
            len(route.token_ids) > 1
            for member in group.members
            for route in member.routes
        )
        for group in groups
    ):
        raise EditorError("multi-token group biases require exact context token IDs")
    normalized_history = () if history is None else tuple(history)
    result: list[BiasContribution] = []

    for token_bias in token_biases:
        token_bias = BiasToken.from_record(token_bias)
        if token_bias.bias:
            result.append(BiasContribution(
                token_id=token_bias.token_id,
                amount=token_bias.bias,
                active=True,
                source=f"token #{token_bias.token_id}",
            ))

    for group_value in groups:
        group = BiasGroup.from_record(group_value)
        if not group.bias:
            continue
        token_routes: dict[int, list[BiasMemberRoute]] = {}
        for member in group.members:
            for route in member.routes:
                target = route.token_ids[-1]
                active = _endswith(normalized_history, route.token_ids[:-1])
                token_routes.setdefault(target, []).append(BiasMemberRoute(
                    member_text=member.text,
                    member_literal=member.literal,
                    route=route,
                    active=active,
                ))
        for token_id, member_routes in token_routes.items():
            active = any(member_route.active for member_route in member_routes)
            if active or include_inactive:
                result.append(BiasContribution(
                    token_id=token_id,
                    amount=group.bias,
                    active=active,
                    source=f"group {group.name!r}",
                    group_name=group.name,
                    member_routes=tuple(member_routes),
                ))

    return tuple(sorted(
        result,
        key=lambda item: (
            item.token_id,
            item.source,
            tuple(
                (route.member_text, route.member_literal, route.route.token_ids)
                for route in item.member_routes
            ),
        ),
    ))


def active_biases(
    groups: Sequence[BiasGroup],
    token_biases: Sequence[BiasToken],
    history: Sequence[int] | None,
) -> dict[int, float]:
    """Sum active sources once per group and target token."""
    result: dict[int, float] = {}
    for contribution in bias_contributions(groups, token_biases, history):
        if contribution.active:
            result[contribution.token_id] = (
                result.get(contribution.token_id, 0.0) + contribution.amount
            )
    return result
