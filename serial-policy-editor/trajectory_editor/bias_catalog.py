"""Model-specific term catalogs for human-readable bias definitions.

Compilation remains side-effect free.  Runtime consumers can validate a
catalog against their loaded tokenizer and turn entries into logical bias
rules without recompiling human-readable terms.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from types import MappingProxyType

from .domain import EditorError


CATALOG_FORMAT = "spe-bias-catalog-v1"
LEVELS = ("minimal", "standard", "exhaustive")
MODES = ("tail", "path", "beheaded")
COMPILE_MODES = ("auto", *MODES)
ALLOCATIONS = (
    "legacy",
    "full",
    "equal",
    "information",
    "information_amplified",
    "naive_chaining",
)
DEFAULT_ALLOCATION = "legacy"
ROUTE_POLICIES = ("all", "cohesive")
ROUTE_CLASSES = ("direct", "word_aligned", "cohesive", "fragmented")
DEFAULT_LEVEL = "standard"
DEFAULT_MAX_ROUTES = 4096
DEFAULT_STANDARD_MAX_ROUTE_TOKENS = 2
DEFAULT_EXHAUSTIVE_MAX_ROUTE_TOKENS = 8
DEFAULT_ROUTE_POLICY = "all"
DEFAULT_MIN_ROUTE_PIECE_CHARS = 3
GROUP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
SurfaceIndex = Mapping[str, Sequence[tuple[str, Sequence[int]]]]


def _error(path: str, message: str) -> EditorError:
    return EditorError(f"{path}: {message}")


def normalize_scalar(value: str, *, path: str = "term") -> str:
    """Normalize a semantic term without requiring tokenizer-ready spacing."""

    if not isinstance(value, str):
        raise _error(path, "must be a string")
    result = unicodedata.normalize("NFC", value).strip()
    if not result:
        raise _error(path, "must not be empty")
    return result


def _normalize_surface(value: str, *, path: str = "surface") -> str:
    """Normalize a generated surface while preserving intentional leading space."""

    if not isinstance(value, str):
        raise _error(path, "must be a string")
    result = unicodedata.normalize("NFC", value)
    if not result.strip():
        raise _error(path, "must not be empty")
    return result


def normalize_name(value: str, *, path: str = "name") -> str:
    result = normalize_scalar(value, path=path)
    if result.startswith("@"):
        raise _error(path, "must not start with @")
    if not GROUP_NAME_RE.fullmatch(result):
        raise _error(path, "must be an identifier-like name")
    return result


def _as_bool(value: Any, *, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    raise _error(path, "must be a boolean")


def _as_int(value: Any, *, path: str, minimum: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise _error(path, "must be an integer") from exc
    if result < minimum:
        raise _error(path, f"must be at least {minimum}")
    return result


def _as_scale(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(path, "must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise _error(path, "must be a finite nonnegative number")
    return result


def _as_string_list(value: Any, *, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (normalize_scalar(value, path=path),)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise _error(path, "must be a string or list of strings")
    result = tuple(normalize_scalar(item, path=f"{path}[{index}]")
                  for index, item in enumerate(value))
    if not result:
        raise _error(path, "must not be empty")
    return result


def _parse_level(value: Any, *, path: str) -> str:
    result = str(value).strip().lower()
    if result not in LEVELS:
        raise _error(path, f"must be one of {', '.join(LEVELS)}")
    return result


def _parse_mode(value: Any, *, path: str) -> str:
    result = str(value).strip().lower()
    if result not in COMPILE_MODES:
        raise _error(path, f"must be one of {', '.join(COMPILE_MODES)}")
    return result


def _parse_allocation(value: Any, *, path: str) -> str:
    result = str(value).strip().lower()
    if result not in ALLOCATIONS:
        raise _error(path, f"must be one of {', '.join(ALLOCATIONS)}")
    return result


def _parse_route_policy(value: Any, *, path: str) -> str:
    result = str(value).strip().lower()
    if result not in ROUTE_POLICIES:
        raise _error(path, f"must be one of {', '.join(ROUTE_POLICIES)}")
    return result


_WORD_BOUNDARY_MARKERS = frozenset(("▁", "Ġ"))


def _token_content(token_text: str) -> str:
    content = str(token_text).strip()
    while content and content[0] in _WORD_BOUNDARY_MARKERS:
        content = content[1:].lstrip()
    return content


def _starts_word_boundary(token_text: str) -> bool:
    text = str(token_text)
    return bool(text and (text[0].isspace() or text[0] in _WORD_BOUNDARY_MARKERS))


def _ends_word_boundary(token_text: str) -> bool:
    text = str(token_text)
    return bool(text and text[-1].isspace())


def _is_possessive_suffix(token_text: str) -> bool:
    return _token_content(token_text).lower() in {"'s", "’s"}


def _has_short_head(token_text: str) -> bool:
    """Recognize a bare boundary or one- or two-letter route head."""

    content = _token_content(token_text)
    if not content:
        return True
    return 1 <= len(content) <= 2 and all(character.isalpha() for character in content)


def _route_is_cohesive(token_texts: Sequence[str], *, min_piece_chars: int) -> bool:
    """Keep whole-word-like pieces and sufficiently informative subword chunks.

    A token is whole-word-like when it begins and ends at a word boundary in
    the route.  This keeps decompositions such as ``port`` + `` of`` +
    `` call`` while rejecting alternatives such as ``o`` + ``f`` or a run of
    tiny subword fragments.  Punctuation-only pieces do not make a route less
    cohesive.
    """

    for index, raw_text in enumerate(token_texts):
        content = _token_content(raw_text)
        if not content or not any(character.isalnum() for character in content):
            continue
        content_chars = sum(character.isalnum() for character in content)
        whole_word = (
            (index == 0 or _starts_word_boundary(raw_text))
            and (
                index == len(token_texts) - 1
                or _starts_word_boundary(token_texts[index + 1])
                or _ends_word_boundary(raw_text)
            )
        )
        if (
            content_chars < min_piece_chars
            and not whole_word
            and not _is_possessive_suffix(raw_text)
        ):
            return False
    return True


def _route_prefix_is_cohesive(
    token_texts: Sequence[str],
    *,
    min_piece_chars: int,
) -> bool:
    """Reject a cohesive route as soon as a completed tiny piece is invalid."""

    for index, raw_text in enumerate(token_texts[:-1]):
        content = _token_content(raw_text)
        if not content or not any(character.isalnum() for character in content):
            continue
        content_chars = sum(character.isalnum() for character in content)
        if content_chars >= min_piece_chars:
            continue
        whole_word = (
            (index == 0 or _starts_word_boundary(raw_text))
            and (
                _starts_word_boundary(token_texts[index + 1])
                or _ends_word_boundary(raw_text)
            )
        )
        if not whole_word and not _is_possessive_suffix(raw_text):
            return False
    return True


def _is_word_aligned_route(token_texts: Sequence[str]) -> bool:
    """Return whether each token occupies one whitespace-delimited word."""

    for index, raw_text in enumerate(token_texts):
        content = _token_content(raw_text)
        if any(character.isspace() for character in content):
            return False
        if index == 0:
            continue
        previous = token_texts[index - 1]
        if not (_starts_word_boundary(raw_text) or _ends_word_boundary(previous)):
            return False
    return True


def _route_class(token_texts: Sequence[str], *, min_piece_chars: int) -> str:
    if len(token_texts) == 1:
        return "direct"
    if _is_word_aligned_route(token_texts):
        return "word_aligned"
    if _route_is_cohesive(token_texts, min_piece_chars=min_piece_chars):
        return "cohesive"
    return "fragmented"


def _is_preferred_route(route_class: str) -> bool:
    return route_class != "fragmented"


def _route_mode(
    default_mode: str,
    token_texts: Sequence[str],
    *,
    allow_beheaded: bool,
) -> str:
    if allow_beheaded and token_texts and _has_short_head(token_texts[0]):
        return "beheaded"
    return default_mode


def _title_case(value: str) -> str:
    """Title-case a surface while keeping terminal possessive ``'s`` lower-case."""

    titled = value.title()
    return re.sub(r"(['’])S(?=\b)", r"\1s", titled)


@dataclass(frozen=True)
class CompileOptions:
    level: str = DEFAULT_LEVEL
    mode: str = "auto"
    head_scale: float = 1.0
    continuation_scale: float = 1.0
    cases: tuple[str, ...] = ("original", "lower", "title", "sentence")
    leading_space: bool = True
    plural: bool = True
    suffixes: tuple[str, ...] = ()
    max_routes: int = DEFAULT_MAX_ROUTES
    max_route_tokens: int | None = None
    route_policy: str = DEFAULT_ROUTE_POLICY
    min_route_piece_chars: int = DEFAULT_MIN_ROUTE_PIECE_CHARS
    allocation: str = DEFAULT_ALLOCATION
    allocation_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise EditorError(f"unknown compilation level {self.level!r}")
        if self.mode not in COMPILE_MODES:
            raise EditorError(f"unknown compilation mode {self.mode!r}")
        if self.allocation not in ALLOCATIONS:
            raise EditorError(f"unknown bias allocation {self.allocation!r}")
        if (
            type(self.allocation_floor) not in (int, float)
            or not math.isfinite(self.allocation_floor)
            or self.allocation_floor < 0
            or self.allocation_floor > 1
        ):
            raise EditorError("allocation_floor must be between 0 and 1")
        for name in ("head_scale", "continuation_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise EditorError(f"{name} must be a finite nonnegative number")
        allowed_cases = {"original", "lower", "title", "sentence", "upper"}
        if not self.cases or any(case not in allowed_cases for case in self.cases):
            raise EditorError("cases must contain supported case names")
        if self.max_routes < 1:
            raise EditorError("max_routes must be positive")
        if self.max_route_tokens is not None and self.max_route_tokens < 1:
            raise EditorError("max_route_tokens must be positive")
        if self.route_policy not in ROUTE_POLICIES:
            raise EditorError(f"unknown route policy {self.route_policy!r}")
        if self.min_route_piece_chars < 1:
            raise EditorError("min_route_piece_chars must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, base: "CompileOptions" | None = None,
                     path: str = "options") -> "CompileOptions":
        current = base or cls()
        if value is None:
            return current
        if not isinstance(value, Mapping):
            raise _error(path, "must be a mapping")
        cases_value = value.get("cases", current.cases)
        cases = _as_string_list(cases_value, path=f"{path}.cases")
        suffixes_value = value.get("suffixes", current.suffixes)
        suffixes = () if suffixes_value in (None, (), []) else _as_string_list(
            suffixes_value, path=f"{path}.suffixes"
        )
        leading = value.get("leading_space", current.leading_space)
        if isinstance(leading, str) and leading.strip().lower() in {"both", "true", "yes", "on"}:
            leading = True
        elif isinstance(leading, str) and leading.strip().lower() in {"none", "false", "no", "off"}:
            leading = False
        else:
            leading = _as_bool(leading, path=f"{path}.leading_space")
        max_route_tokens = value.get("max_route_tokens", current.max_route_tokens)
        if max_route_tokens is not None:
            max_route_tokens = _as_int(max_route_tokens, path=f"{path}.max_route_tokens", minimum=1)
        return cls(
            level=_parse_level(value.get("level", current.level), path=f"{path}.level"),
            mode=_parse_mode(value.get("mode", current.mode), path=f"{path}.mode"),
            allocation=_parse_allocation(
                value.get("allocation", current.allocation),
                path=f"{path}.allocation",
            ),
            allocation_floor=_as_scale(
                value.get("allocation_floor", current.allocation_floor),
                path=f"{path}.allocation_floor",
            ),
            head_scale=_as_scale(
                value.get("head_scale", current.head_scale),
                path=f"{path}.head_scale",
            ),
            continuation_scale=_as_scale(
                value.get("continuation_scale", current.continuation_scale),
                path=f"{path}.continuation_scale",
            ),
            cases=cases,
            leading_space=leading,
            plural=_as_bool(value.get("plural", current.plural), path=f"{path}.plural"),
            suffixes=suffixes,
            max_routes=_as_int(value.get("max_routes", current.max_routes),
                               path=f"{path}.max_routes", minimum=1),
            max_route_tokens=max_route_tokens,
            route_policy=_parse_route_policy(
                value.get("route_policy", current.route_policy),
                path=f"{path}.route_policy",
            ),
            min_route_piece_chars=_as_int(
                value.get("min_route_piece_chars", current.min_route_piece_chars),
                path=f"{path}.min_route_piece_chars",
                minimum=1,
            ),
        )

    def route_limit(self) -> int:
        if self.max_route_tokens is not None:
            return self.max_route_tokens
        if self.level == "standard":
            return DEFAULT_STANDARD_MAX_ROUTE_TOKENS
        if self.level == "exhaustive":
            return DEFAULT_EXHAUSTIVE_MAX_ROUTE_TOKENS
        return 1


@dataclass(frozen=True)
class PrefixReferenceStats:
    """Reference lexical mass used by experimental route allocation.

    References are surface strings rather than model-generated continuations.
    A prefix's mass is the sum of reference weights for surfaces beginning
    with that prefix.  The compiler can therefore build these statistics once
    and the runtime never needs to consult the reference universe.
    """

    references: tuple[tuple[str, float], ...]
    _prefix_masses: Mapping[str, float] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        normalized: list[tuple[str, float]] = []
        for surface, weight in self.references:
            if not isinstance(surface, str) or not surface:
                raise EditorError("reference surfaces must be nonempty strings")
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise EditorError("reference weights must be finite positive numbers")
            normalized.append((surface, float(weight)))
        if not normalized:
            raise EditorError("reference collection must not be empty")
        object.__setattr__(self, "references", tuple(normalized))
        prefix_masses: dict[str, float] = {}
        for surface, weight in normalized:
            for index in range(len(surface) + 1):
                prefix = surface[:index]
                prefix_masses[prefix] = prefix_masses.get(prefix, 0.0) + weight
        object.__setattr__(self, "_prefix_masses", MappingProxyType(prefix_masses))

    @property
    def root_mass(self) -> float:
        return self._prefix_masses[""]

    def mass(self, prefix: str) -> float:
        return self._prefix_masses.get(prefix, 0.0)


@dataclass(frozen=True)
class ReferencePriorRoute:
    """One model-token route retained for the experimental online prior."""

    text: str
    token_ids: tuple[int, ...]
    weight: float

    def __post_init__(self) -> None:
        if not self.text or not self.token_ids:
            raise EditorError("reference prior routes require text and token IDs")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise EditorError("reference prior route token IDs must be nonnegative integers")
        if type(self.weight) not in (int, float) or not math.isfinite(self.weight) or self.weight <= 0:
            raise EditorError("reference prior route weights must be finite positive numbers")
        object.__setattr__(self, "weight", float(self.weight))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "weight": self.weight,
        }


def build_prefix_reference_stats(
    references: Sequence[str] | Mapping[str, float],
) -> PrefixReferenceStats:
    """Build raw- or frequency-weighted surface-prefix reference statistics."""

    if isinstance(references, Mapping):
        try:
            values = tuple((str(surface), float(weight))
                           for surface, weight in references.items())
        except (TypeError, ValueError) as exc:
            raise EditorError("reference weights must be numeric") from exc
    elif isinstance(references, Sequence) and not isinstance(references, (str, bytes, bytearray)):
        values = tuple((str(surface), 1.0) for surface in references)
    else:
        raise EditorError("reference collection must be a list or mapping")
    return PrefixReferenceStats(values)


def _reference_values(
    reference: Sequence[str] | Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if isinstance(reference, Mapping):
        try:
            return tuple((str(surface), float(weight))
                         for surface, weight in reference.items())
        except (TypeError, ValueError) as exc:
            raise EditorError("reference weights must be numeric") from exc
    if isinstance(reference, Sequence) and not isinstance(
        reference, (str, bytes, bytearray)
    ):
        return tuple((str(surface), 1.0) for surface in reference)
    raise EditorError("reference collection must be a list or mapping")


def _compile_reference_prior_routes(
    backend: Any,
    reference: Sequence[str] | Mapping[str, float] | None,
) -> tuple[ReferencePriorRoute, ...]:
    """Compile explicit reference surfaces for the optional online prior.

    The compiler keeps exact and leading-space forms as separate routes, but
    splits a surface's weight between them so adding the automatic boundary
    variant does not double its importance.
    """

    if reference is None:
        return ()
    routes: list[ReferencePriorRoute] = []
    for surface, weight in _reference_values(reference):
        if not surface.strip():
            raise EditorError("reference surfaces must be nonempty strings")
        candidates = (surface,) if surface.startswith(" ") else (surface, f" {surface}")
        compiled: list[tuple[str, tuple[int, ...]]] = []
        for candidate in candidates:
            try:
                token_ids = tuple(int(token) for token in backend.tokenize(
                    candidate, add_bos=False, special=False
                ))
            except (RuntimeError, TypeError, ValueError):
                continue
            if token_ids:
                compiled.append((candidate, token_ids))
        if not compiled:
            continue
        route_weight = float(weight) / len(compiled)
        for candidate, token_ids in compiled:
            routes.append(ReferencePriorRoute(candidate, token_ids, route_weight))
    return tuple(routes)


@dataclass(frozen=True)
class CompiledRoute:
    token_ids: tuple[int, ...]
    texts: tuple[str, ...]
    token_texts: tuple[str, ...]
    mode: str
    strategies: tuple[str, ...]
    sources: tuple[str, ...] = ()
    head_scale: float = 1.0
    continuation_scale: float = 1.0
    route_class: str = "fragmented"
    allocation: str = DEFAULT_ALLOCATION
    edge_weights: tuple[float, ...] = ()
    allocation_diagnostics: tuple[tuple[float, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.route_class not in ROUTE_CLASSES:
            raise EditorError(f"unknown route class {self.route_class!r}")
        if self.allocation not in ALLOCATIONS:
            raise EditorError(f"unknown bias allocation {self.allocation!r}")
        edge_weights = tuple(float(weight) for weight in self.edge_weights)
        if edge_weights and len(edge_weights) != len(self.token_ids):
            raise EditorError("route edge_weights must align with route token_ids")
        if any(not math.isfinite(weight) or weight < 0 for weight in edge_weights):
            raise EditorError("route edge_weights must be finite nonnegative numbers")
        diagnostics = tuple(tuple(float(value) for value in row)
                            for row in self.allocation_diagnostics)
        if diagnostics and len(diagnostics) != len(self.token_ids):
            raise EditorError("route allocation diagnostics must align with route token_ids")
        if any(
            len(row) != 4
            or any(not math.isfinite(value) for value in row)
            or row[0] < 0
            or row[1] < 0
            or not 0 <= row[2] <= 1
            or row[3] < 0
            for row in diagnostics
        ):
            raise EditorError("invalid route allocation diagnostics")
        object.__setattr__(self, "edge_weights", edge_weights)
        object.__setattr__(self, "allocation_diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "texts": list(self.texts),
            "token_texts": list(self.token_texts),
            "mode": self.mode,
            "strategies": list(self.strategies),
            "route_class": self.route_class,
            **({"sources": list(self.sources)} if self.sources else {}),
            **({"head_scale": self.head_scale} if self.head_scale != 1.0 else {}),
            **({"continuation_scale": self.continuation_scale}
               if self.continuation_scale != 1.0 else {}),
            **({"allocation": self.allocation} if self.allocation != DEFAULT_ALLOCATION else {}),
            **({"edge_weights": list(self.edge_weights)} if self.edge_weights else {}),
            **({
                "allocation_diagnostics": [
                    {
                        "token_id": self.token_ids[index],
                        "token": (
                            self.token_texts[index]
                            if index < len(self.token_texts) else ""
                        ),
                        "remaining_mass": row[0],
                        "information": row[1],
                        "phi": row[2],
                        "edge_weight": row[3],
                    }
                    for index, row in enumerate(self.allocation_diagnostics)
                ]
            } if self.allocation_diagnostics else {}),
        }


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    kind: Literal["term", "group"]
    routes: tuple[CompiledRoute, ...]
    mode: str | None = None
    source: str | None = None
    members: tuple[str, ...] = ()
    level: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "routes": [route.to_dict() for route in self.routes],
        }
        if self.source is not None:
            result["source"] = self.source
        if self.mode is not None:
            result["default_mode"] = self.mode
        if self.level is not None:
            result["level"] = self.level
        if self.members:
            result["members"] = list(self.members)
        if self.warnings:
            result["warnings"] = list(self.warnings)
        return result


@dataclass(frozen=True)
class BiasCatalog:
    model: Mapping[str, Any]
    compiler: Mapping[str, Any]
    entries: Mapping[str, CatalogEntry]
    reference_prior_routes: tuple[ReferencePriorRoute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, Mapping):
            raise EditorError("catalog model metadata must be a mapping")
        if not isinstance(self.entries, Mapping):
            raise EditorError("catalog entries must be a mapping")
        for name, entry in self.entries.items():
            if name != entry.name:
                raise EditorError("catalog entry names must match their keys")

    def resolve(self, name: str) -> CatalogEntry | None:
        return self.entries.get(name)

    def require(self, name: str) -> CatalogEntry:
        entry = self.resolve(name)
        if entry is None:
            raise EditorError(f"catalog has no entry named {name!r}")
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CATALOG_FORMAT,
            "model": dict(self.model),
            "compiler": dict(self.compiler),
            "entries": {name: entry.to_dict() for name, entry in sorted(self.entries.items())},
            **({
                "reference_prior_routes": [
                    route.to_dict() for route in self.reference_prior_routes
                ]
            } if self.reference_prior_routes else {}),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BiasCatalog":
        if not isinstance(value, Mapping) or value.get("format") != CATALOG_FORMAT:
            raise EditorError(f"catalog must use format {CATALOG_FORMAT}")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, Mapping):
            raise EditorError("catalog entries must be a mapping")
        raw_prior_routes = value.get("reference_prior_routes", ())
        if not isinstance(raw_prior_routes, Sequence) or isinstance(
            raw_prior_routes, (str, bytes, bytearray)
        ):
            raise _error("reference_prior_routes", "must be a list")
        prior_routes: list[ReferencePriorRoute] = []
        for index, raw_route in enumerate(raw_prior_routes):
            route_path = f"reference_prior_routes[{index}]"
            if not isinstance(raw_route, Mapping):
                raise _error(route_path, "must be a mapping")
            raw_ids = raw_route.get("token_ids")
            if not isinstance(raw_ids, Sequence) or isinstance(
                raw_ids, (str, bytes, bytearray)
            ):
                raise _error(f"{route_path}.token_ids", "must be a list")
            try:
                token_ids = tuple(int(token) for token in raw_ids)
            except (TypeError, ValueError) as exc:
                raise _error(f"{route_path}.token_ids", "must contain integers") from exc
            prior_routes.append(ReferencePriorRoute(
                text=str(raw_route.get("text", "")),
                token_ids=token_ids,
                weight=float(raw_route.get("weight", 0.0)),
            ))
        entries: dict[str, CatalogEntry] = {}
        for raw_name, raw_entry in raw_entries.items():
            if not isinstance(raw_entry, Mapping):
                raise _error(f"entries.{raw_name}", "must be a mapping")
            kind = raw_entry.get("kind")
            if kind not in {"term", "group"}:
                raise _error(f"entries.{raw_name}.kind", "must be term or group")
            name = (normalize_name(str(raw_name), path="entries.name")
                    if kind == "group" else normalize_scalar(str(raw_name), path="entries.name"))
            routes = _routes_from_dict(raw_entry.get("routes", ()), path=f"entries.{name}.routes")
            members = tuple(str(item) for item in raw_entry.get("members", ()))
            warnings = tuple(str(item) for item in raw_entry.get("warnings", ()))
            mode = raw_entry.get("default_mode")
            if mode is not None and mode not in MODES:
                raise _error(
                    f"entries.{name}.default_mode",
                    f"must be one of {', '.join(MODES)}",
                )
            entries[name] = CatalogEntry(
                name=name,
                kind=kind,
                routes=routes,
                mode=mode,
                source=raw_entry.get("source"),
                members=members,
                level=raw_entry.get("level"),
                warnings=warnings,
            )
        return cls(
            model=dict(value.get("model", {})),
            compiler=dict(value.get("compiler", {})),
            entries=entries,
            reference_prior_routes=tuple(prior_routes),
        )

    @classmethod
    def from_json(cls, text: str) -> "BiasCatalog":
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise EditorError(f"invalid catalog JSON: {exc}") from exc
        return cls.from_dict(value)

    @classmethod
    def merge(cls, catalogs: Sequence["BiasCatalog"]) -> "BiasCatalog":
        if not catalogs:
            raise EditorError("at least one catalog is required for merge")
        first = catalogs[0]
        identity = _model_identity(first.model)
        entries: dict[str, CatalogEntry] = {}
        prior_routes: list[ReferencePriorRoute] = []
        for catalog in catalogs:
            if _model_identity(catalog.model) != identity:
                raise EditorError("cannot merge catalogs compiled for different tokenizers")
            for name, entry in catalog.entries.items():
                existing = entries.get(name)
                if existing is None:
                    entries[name] = entry
                    continue
                if existing.kind != entry.kind:
                    raise EditorError(f"catalog entry collision for {name!r}")
                if existing.kind == "term" and (
                    existing.source != entry.source or existing.mode != entry.mode
                ):
                    raise EditorError(f"catalog term collision for {name!r}")
                entries[name] = _merge_entries(existing, entry)
            prior_routes.extend(catalog.reference_prior_routes)
        unique_prior_routes: dict[tuple[str, tuple[int, ...]], ReferencePriorRoute] = {}
        for route in prior_routes:
            key = (route.text, route.token_ids)
            existing = unique_prior_routes.get(key)
            if existing is None:
                unique_prior_routes[key] = route
            else:
                unique_prior_routes[key] = ReferencePriorRoute(
                    text=route.text,
                    token_ids=route.token_ids,
                    weight=existing.weight + route.weight,
                )
        return cls(
            model=first.model,
            compiler={
                **dict(first.compiler),
                "merged_catalogs": len(catalogs),
            },
            entries=entries,
            reference_prior_routes=tuple(unique_prior_routes.values()),
        )


def _model_identity(model: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        model.get("backend"),
        model.get("vocabulary_size"),
        model.get("tokenizer_fingerprint"),
    )


def _merge_routes(left: Sequence[CompiledRoute], right: Sequence[CompiledRoute]) -> tuple[CompiledRoute, ...]:
    merged: dict[
        tuple[tuple[int, ...], str, float, float, str, str, tuple[float, ...]],
        CompiledRoute,
    ] = {}
    for route in (*left, *right):
        key = (
            route.token_ids,
            route.mode,
            route.head_scale,
            route.continuation_scale,
            route.route_class,
            route.allocation,
            route.edge_weights,
        )
        current = merged.get(key)
        if current is None:
            merged[key] = route
            continue
        merged[key] = CompiledRoute(
            token_ids=current.token_ids,
            texts=tuple(dict.fromkeys((*current.texts, *route.texts))),
            token_texts=current.token_texts,
            mode=current.mode,
            strategies=tuple(dict.fromkeys((*current.strategies, *route.strategies))),
            sources=tuple(dict.fromkeys((*current.sources, *route.sources))),
            head_scale=current.head_scale,
            continuation_scale=current.continuation_scale,
            route_class=current.route_class,
            allocation=current.allocation,
            edge_weights=current.edge_weights,
            allocation_diagnostics=current.allocation_diagnostics,
        )
    return tuple(merged.values())


def _merge_entries(left: CatalogEntry, right: CatalogEntry) -> CatalogEntry:
    return CatalogEntry(
        name=left.name,
        kind=left.kind,
        routes=_merge_routes(left.routes, right.routes),
        mode=left.mode,
        source=left.source,
        members=tuple(dict.fromkeys((*left.members, *right.members))),
        level=left.level,
        warnings=tuple(dict.fromkeys((*left.warnings, *right.warnings))),
    )


def _routes_from_dict(value: Any, *, path: str) -> tuple[CompiledRoute, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(path, "must be a list")
    routes: list[CompiledRoute] = []
    for index, raw in enumerate(value):
        route_path = f"{path}[{index}]"
        if not isinstance(raw, Mapping):
            raise _error(route_path, "must be a mapping")
        token_ids = raw.get("token_ids")
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes, bytearray)) or not token_ids:
            raise _error(f"{route_path}.token_ids", "must be a nonempty list")
        try:
            ids = tuple(int(token) for token in token_ids)
        except (TypeError, ValueError) as exc:
            raise _error(f"{route_path}.token_ids", "must contain integers") from exc
        mode = raw.get("mode", "tail")
        if mode not in MODES:
            raise _error(f"{route_path}.mode", f"must be one of {', '.join(MODES)}")
        texts = tuple(str(text) for text in raw.get("texts", ()))
        token_texts = tuple(str(text) for text in raw.get("token_texts", ()))
        strategies = tuple(str(strategy) for strategy in raw.get("strategies", ()))
        sources = tuple(str(source) for source in raw.get("sources", ()))
        route_class = str(raw.get("route_class", "fragmented"))
        if route_class not in ROUTE_CLASSES:
            raise _error(
                f"{route_path}.route_class",
                f"must be one of {', '.join(ROUTE_CLASSES)}",
            )
        raw_diagnostics = raw.get("allocation_diagnostics", ())
        diagnostics: list[tuple[float, float, float, float]] = []
        if raw_diagnostics:
            if not isinstance(raw_diagnostics, Sequence) or isinstance(
                raw_diagnostics, (str, bytes, bytearray)
            ):
                raise _error(f"{route_path}.allocation_diagnostics", "must be a list")
            for diagnostic_index, diagnostic in enumerate(raw_diagnostics):
                if not isinstance(diagnostic, Mapping):
                    raise _error(
                        f"{route_path}.allocation_diagnostics[{diagnostic_index}]",
                        "must be a mapping",
                    )
                diagnostics.append((
                    float(diagnostic.get("remaining_mass", 0.0)),
                    float(diagnostic.get("information", 0.0)),
                    float(diagnostic.get("phi", 0.0)),
                    float(diagnostic.get("edge_weight", 0.0)),
                ))
        edge_weights = raw.get("edge_weights", ())
        if edge_weights is None:
            edge_weights = ()
        if not isinstance(edge_weights, Sequence) or isinstance(
            edge_weights, (str, bytes, bytearray)
        ):
            raise _error(f"{route_path}.edge_weights", "must be a list")
        routes.append(CompiledRoute(
            ids,
            texts,
            token_texts,
            mode,
            strategies,
            sources,
            _as_scale(raw.get("head_scale", 1.0), path=f"{route_path}.head_scale"),
            _as_scale(
                raw.get("continuation_scale", 1.0),
                path=f"{route_path}.continuation_scale",
            ),
            route_class,
            _parse_allocation(raw.get("allocation", DEFAULT_ALLOCATION),
                              path=f"{route_path}.allocation"),
            tuple(float(weight) for weight in edge_weights),
            tuple(diagnostics),
        ))
    return tuple(routes)


def tokenizer_fingerprint(backend: Any) -> str:
    """Fingerprint the addressable token text exposed by a backend."""

    digest = hashlib.sha256()
    size = int(backend.vocabulary_size())
    for token_id in range(size):
        text = str(backend.token_text(token_id))
        is_eog = bool(getattr(backend, "is_eog", lambda _token: False)(token_id))
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\1" if is_eog else b"\0")
        digest.update(b"\2")
    return digest.hexdigest()


def catalog_model_metadata(backend: Any, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    info = dict(provenance or {})
    if not info:
        method = getattr(backend, "provenance", None)
        if callable(method):
            info = dict(method(include_model_sha256=False))
    model: dict[str, Any] = {
        key: info[key]
        for key in ("backend", "filename", "file_size_bytes", "model_path", "vocabulary_size")
        if key in info
    }
    model["vocabulary_size"] = int(backend.vocabulary_size())
    model["tokenizer_fingerprint"] = info.get("tokenizer_fingerprint") or tokenizer_fingerprint(backend)
    return model


def validate_catalog(catalog: BiasCatalog, backend: Any, provenance: Mapping[str, Any] | None = None) -> BiasCatalog:
    """Ensure a catalog addresses the tokenizer currently used at runtime."""

    model = catalog.model
    vocabulary_size = model.get("vocabulary_size")
    if type(vocabulary_size) is not int or vocabulary_size != int(backend.vocabulary_size()):
        raise EditorError("bias catalog vocabulary does not match the loaded model")
    expected_fingerprint = model.get("tokenizer_fingerprint")
    if expected_fingerprint is not None:
        actual_fingerprint = tokenizer_fingerprint(backend)
        if expected_fingerprint != actual_fingerprint:
            raise EditorError("bias catalog tokenizer fingerprint does not match the loaded model")
    info = dict(provenance or {})
    if not info:
        method = getattr(backend, "provenance", None)
        if callable(method):
            info = dict(method(include_model_sha256=False))
    expected_backend = model.get("backend")
    if expected_backend is not None and info.get("backend") not in (None, expected_backend):
        raise EditorError("bias catalog backend does not match the loaded model")
    return catalog


def _pluralize_word(word: str) -> str:
    if not word:
        return word
    lowered = word.lower()
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if lowered.endswith("y") and len(word) > 1 and lowered[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _last_word_span(text: str) -> tuple[str, str]:
    match = re.search(r"([^\s]+)$", text)
    if match is None:
        return text, ""
    return text[:match.start()], match.group(1)


def _apply_suffix(text: str, suffix: str) -> str:
    prefix, last = _last_word_span(text)
    if not last:
        return text + suffix
    return prefix + last + suffix


def generate_forms(source: str, options: CompileOptions, explicit_forms: Sequence[str] = ()) -> tuple[str, ...]:
    """Generate semantic surface forms before tokenizer route search."""

    bases: list[str] = [source, *explicit_forms]
    expanded: list[str] = []
    for base in bases:
        base = normalize_scalar(base)
        for case in options.cases:
            if case == "original":
                value = base
            elif case == "lower":
                value = base.lower()
            elif case == "title":
                value = _title_case(base)
            elif case == "sentence":
                value = base.capitalize()
            else:
                value = base.upper()
            expanded.append(value)
            if options.plural:
                expanded.append(_pluralize_word(value))
            for suffix in options.suffixes:
                expanded.append(_apply_suffix(value, suffix))
    result: list[str] = []
    seen: set[str] = set()
    for value in expanded:
        for candidate in (value, f" {value}") if options.leading_space else (value,):
            candidate = _normalize_surface(candidate)
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return tuple(result)


def _default_reference_surfaces(backend: Any, extra: Sequence[str]) -> tuple[str, ...]:
    """Return a tokenizer-local reference universe for the raw baseline."""

    surfaces: dict[str, None] = {}
    for token_id in range(int(backend.vocabulary_size())):
        if bool(getattr(backend, "is_eog", lambda _token: False)(token_id)):
            continue
        text = str(backend.token_text(token_id))
        if not text or (text.startswith("<") and text.endswith(">")):
            continue
        surfaces.setdefault(text, None)
    for surface in extra:
        if surface:
            surfaces.setdefault(surface, None)
    return tuple(surfaces)


def _reference_stats_for_backend(
    backend: Any,
    extra: Sequence[str],
    reference: Sequence[str] | Mapping[str, float] | None = None,
) -> PrefixReferenceStats:
    if reference is None:
        return build_prefix_reference_stats({
            surface: 1.0
            for surface in _default_reference_surfaces(backend, extra)
        })
    # An explicitly supplied lexicon defines the reference universe.  Do not
    # mix unweighted tokenizer vocabulary entries into a frequency-weighted
    # lexicon.  Add generated target forms only when absent so an omitted
    # target still has a finite terminal mass.
    values: dict[str, float] = dict(_reference_values(reference))
    for surface in extra:
        values.setdefault(surface, 1.0)
    return build_prefix_reference_stats(values)


def _render_route_prefix(route: Sequence[int], backend: Any) -> str:
    try:
        return str(backend.render(list(route), special=False))
    except (RuntimeError, TypeError, ValueError):
        return "".join(str(backend.token_text(token)) for token in route)


def allocate_route(
    route: Sequence[int],
    backend: Any,
    *,
    strategy: str,
    reference_stats: PrefixReferenceStats | None = None,
    allocation_floor: float = 0.05,
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float, float], ...]]:
    """Allocate one unit of bias across a route and return debug diagnostics.

    ``legacy`` deliberately returns no weights so the existing mode and
    head/continuation behavior remains authoritative at runtime.  The other
    strategies produce explicit edge weights that are consumed by the same
    runtime matcher.
    """

    if strategy not in ALLOCATIONS:
        raise EditorError(f"unknown bias allocation {strategy!r}")
    if not route:
        raise EditorError("cannot allocate bias across an empty route")
    if type(allocation_floor) not in (int, float) or not math.isfinite(allocation_floor):
        raise EditorError("allocation_floor must be a finite number")
    if allocation_floor < 0 or allocation_floor > 1:
        raise EditorError("allocation_floor must be between 0 and 1")
    if strategy == DEFAULT_ALLOCATION:
        return (), ()
    if strategy == "naive_chaining":
        weights = (1.0,) if len(route) == 1 else tuple(
            0.5 * index for index in range(len(route))
        )
        return weights, tuple(
            (0.0, 0.0, 0.0, weight) for weight in weights
        )
    information_strategy = strategy in {"information", "information_amplified"}
    if information_strategy and reference_stats is None:
        raise EditorError("information allocation requires reference statistics")

    if reference_stats is None:
        masses = [0.0] * (len(route) + 1)
    else:
        prefix_texts = [""] + [
            _render_route_prefix(route[:index], backend)
            for index in range(1, len(route) + 1)
        ]
        root_mass = reference_stats.root_mass
        masses = [reference_stats.mass(prefix) for prefix in prefix_texts]
    information = [0.0]
    for mass in masses[1:]:
        if mass <= 0:
            value = information[-1]
        else:
            value = -math.log(mass / root_mass)
        information.append(max(information[-1], value))

    terminal_information = information[-1]
    if terminal_information <= 1e-12:
        phi = [index / len(route) for index in range(len(route) + 1)]
    else:
        phi = [min(1.0, max(0.0, value / terminal_information))
               for value in information]
        for index in range(1, len(phi)):
            phi[index] = max(phi[index], phi[index - 1])
        phi[-1] = 1.0

    if strategy == "full":
        weights = [1.0] * len(route)
    elif strategy == "equal":
        weights = [1.0 / len(route)] * len(route)
    else:
        weights = [max(0.0, phi[index + 1] - phi[index])
                   for index in range(len(route))]
        total = sum(weights)
        if total <= 1e-12:
            weights = [1.0 / len(route)] * len(route)
        else:
            weights[-1] += 1.0 - total
    if information_strategy and allocation_floor > 0:
        if len(route) * allocation_floor >= 1.0:
            weights = [1.0 / len(route)] * len(route)
        else:
            total = sum(weights)
            residual = 1.0 - len(route) * allocation_floor
            weights = [
                allocation_floor + residual * (weight / total)
                for weight in weights
            ]
    if strategy == "information_amplified" and len(route) >= 3:
        # Preserve the information allocation's shape, but let later edges
        # benefit from specificity accumulated by the prefix.  This is
        # intentionally not renormalized: the experiment should be able to
        # exert more total pressure on a long, increasingly specific route.
        weights = [
            weight * (1.0 + phi[index + 1])
            for index, weight in enumerate(weights)
        ]

    diagnostics = tuple(
        (masses[index + 1], information[index + 1], phi[index + 1], weights[index])
        for index in range(len(route))
    )
    return tuple(weights), diagnostics


def _surface_index(backend: Any) -> SurfaceIndex:
    result: dict[str, dict[str, list[int]]] = {}
    size = int(backend.vocabulary_size())
    for token_id in range(size):
        if bool(getattr(backend, "is_eog", lambda _token: False)(token_id)):
            continue
        text = str(backend.token_text(token_id))
        if not text or text.startswith("<") and text.endswith(">"):
            continue
        result.setdefault(text[0], {}).setdefault(text, []).append(token_id)
    return {
        first: tuple(sorted(
            ((piece, tuple(token_ids)) for piece, token_ids in pieces.items()),
            key=lambda item: (-len(item[0]), item[0]),
        ))
        for first, pieces in result.items()
    }


def _enumerate_routes(
    surface: str,
    backend: Any,
    index: SurfaceIndex,
    *,
    max_tokens: int,
    route_policy: str = DEFAULT_ROUTE_POLICY,
    min_route_piece_chars: int = DEFAULT_MIN_ROUTE_PIECE_CHARS,
) -> tuple[tuple[int, ...], ...]:
    routes: list[tuple[int, ...]] = []
    frontier: list[tuple[int, tuple[int, ...]]] = [(0, ())]
    while frontier:
        position, tokens = frontier.pop()
        if position == len(surface):
            try:
                rendered = backend.render(list(tokens), special=False)
            except (RuntimeError, TypeError, ValueError):
                rendered = None
            if rendered == surface:
                if route_policy == "cohesive":
                    token_texts = tuple(
                        str(backend.token_text(token)) for token in tokens
                    )
                    if not _route_is_cohesive(
                        token_texts,
                        min_piece_chars=min_route_piece_chars,
                    ):
                        continue
                routes.append(tokens)
            continue
        if len(tokens) >= max_tokens:
            continue
        for piece, token_ids in index.get(surface[position], ()):
            if not piece or not surface.startswith(piece, position):
                continue
            next_position = position + len(piece)
            for token_id in reversed(token_ids):
                candidate = (*tokens, int(token_id))
                if route_policy == "cohesive":
                    token_texts = tuple(
                        str(backend.token_text(token)) for token in candidate
                    )
                    if not _route_prefix_is_cohesive(
                        token_texts,
                        min_piece_chars=min_route_piece_chars,
                    ):
                        continue
                frontier.append((next_position, candidate))
    return tuple(dict.fromkeys(routes))


def _route_quality(
    route: Sequence[int],
    token_texts: Sequence[str],
    all_routes: Sequence[Sequence[int]],
    *,
    route_class: str,
    min_piece_chars: int,
) -> tuple[Any, ...]:
    """Return a deterministic, tokenizer-local quality key for a route.

    Lower keys are preferred. Direct and word-aligned routes outrank cohesive
    subword routes, which outrank fragmented routes. Token count, piece size,
    local fan-out, and token IDs provide deterministic tie breakers without
    requiring corpus or model-probability information.
    """

    content_lengths = tuple(
        sum(character.isalnum() for character in _token_content(text))
        for text in token_texts
    )
    tiny_piece_count = sum(length < min_piece_chars for length in content_lengths)
    boundary_breaks = sum(
        index > 0
        and not _starts_word_boundary(token_texts[index])
        and not _ends_word_boundary(token_texts[index - 1])
        for index in range(len(token_texts))
    )
    head = route[0]
    head_fanout = sum(
        1 for candidate in all_routes if candidate and candidate[0] == head
    )
    return (
        ROUTE_CLASSES.index(route_class),
        len(route),
        boundary_breaks,
        tiny_piece_count,
        -max(content_lengths, default=0),
        -content_lengths[0] if content_lengths else 0,
        head_fanout,
        tuple(route),
    )


def _canonical_route(surface: str, backend: Any) -> tuple[int, ...]:
    try:
        tokens = tuple(int(token) for token in backend.tokenize(surface, add_bos=False, special=False))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EditorError(f"could not tokenize {surface!r}: {exc}") from exc
    if not tokens:
        raise EditorError(f"term {surface!r} produced no tokens")
    if any(token < 0 or token >= int(backend.vocabulary_size()) for token in tokens):
        raise EditorError(f"term {surface!r} produced an invalid token ID")
    if any(bool(getattr(backend, "is_eog", lambda _token: False)(token)) for token in tokens):
        raise EditorError(f"term {surface!r} produced a special end-of-generation token")
    return tokens


def compile_term(
    name: str,
    source: str,
    backend: Any,
    *,
    options: CompileOptions,
    explicit_forms: Sequence[str] = (),
    surface_index: SurfaceIndex | None = None,
    reference_stats: PrefixReferenceStats | None = None,
) -> CatalogEntry:
    name = normalize_scalar(name, path="term name")
    source = normalize_scalar(source, path=f"term {name!r}")
    mode = options.mode
    if mode == "auto":
        mode = "tail" if any(character.isspace() for character in source) else "path"
    forms = generate_forms(source, options, explicit_forms)
    index = surface_index if surface_index is not None else _surface_index(backend)
    if reference_stats is None:
        reference_stats = _reference_stats_for_backend(backend, forms)
    form_specs = tuple((form, _canonical_route(form, backend)) for form in forms)

    def candidates_for(
        route_policy: str,
    ) -> list[tuple[str, tuple[tuple[int, ...], ...]]]:
        result: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        for form, tokenizer_route in form_specs:
            max_tokens = max(options.route_limit(), len(tokenizer_route))
            enumerated = _enumerate_routes(
                form,
                backend,
                index,
                max_tokens=max_tokens,
                route_policy=route_policy,
                min_route_piece_chars=options.min_route_piece_chars,
            )
            candidates = tuple(dict.fromkeys((tokenizer_route, *enumerated)))
            classified = {
                route: _route_class(
                    tuple(str(backend.token_text(token)) for token in route),
                    min_piece_chars=options.min_route_piece_chars,
                )
                for route in candidates
            }
            if route_policy == "cohesive":
                candidates = tuple(
                    route for route in candidates
                    if _is_preferred_route(classified[route])
                )
            if not candidates:
                result.append((form, ()))
                continue
            all_candidates = tuple(candidates)
            ordered = sorted(
                candidates,
                key=lambda route: _route_quality(
                    route,
                    tuple(str(backend.token_text(token)) for token in route),
                    all_candidates,
                    route_class=classified[route],
                    min_piece_chars=options.min_route_piece_chars,
                ),
            )
            if options.level == "minimal":
                ordered = ordered[:1]
            result.append((form, tuple(ordered)))
        return result

    form_routes = candidates_for(options.route_policy)
    fallback_tail = (
        options.route_policy == "cohesive"
        and not any(routes for _, routes in form_routes)
    )
    warning: str | None = None
    if fallback_tail:
        warning = (
            f"term {name!r} has no cohesive routes; "
            "using a tail-only fallback"
        )
        form_routes = [
            (form, routes[:1])
            for form, routes in candidates_for("all")
        ]

    route_map: dict[tuple[int, ...], CompiledRoute] = {}
    route_classes: dict[tuple[int, ...], str] = {}
    for form, routes in form_routes:
        for route in routes:
            token_texts = tuple(str(backend.token_text(token)) for token in route)
            route_class = _route_class(
                token_texts,
                min_piece_chars=options.min_route_piece_chars,
            )
            route_classes[route] = route_class
            route_mode = "tail" if fallback_tail else _route_mode(
                "path" if options.mode == "beheaded" else mode,
                token_texts,
                allow_beheaded=options.mode in {"auto", "beheaded"},
            )
            head_scale = 0.0 if route_mode == "beheaded" else options.head_scale
            allocation = DEFAULT_ALLOCATION if fallback_tail else options.allocation
            edge_weights, allocation_diagnostics = allocate_route(
                route,
                backend,
                strategy=allocation,
                reference_stats=reference_stats,
                allocation_floor=options.allocation_floor,
            )
            strategy = "fallback" if fallback_tail else (
                "preferred" if _is_preferred_route(route_class) else "derived"
            )
            existing = route_map.get(route)
            if existing is None:
                route_map[route] = CompiledRoute(
                    token_ids=route,
                    texts=(form,),
                    token_texts=token_texts,
                    mode=route_mode,
                    strategies=(strategy,),
                    sources=(name,),
                    head_scale=head_scale,
                    continuation_scale=options.continuation_scale,
                    route_class=route_class,
                    allocation=allocation,
                    edge_weights=edge_weights,
                    allocation_diagnostics=allocation_diagnostics,
                )
            else:
                route_map[route] = CompiledRoute(
                    token_ids=route,
                    texts=tuple(dict.fromkeys((*existing.texts, form))),
                    token_texts=existing.token_texts,
                    mode=existing.mode,
                    strategies=tuple(dict.fromkeys((*existing.strategies, strategy))),
                    sources=existing.sources,
                    head_scale=existing.head_scale,
                    continuation_scale=existing.continuation_scale,
                    route_class=existing.route_class,
                    allocation=existing.allocation,
                    edge_weights=existing.edge_weights,
                    allocation_diagnostics=existing.allocation_diagnostics,
                )

    def select_buckets(
        buckets: Sequence[Sequence[tuple[int, ...]]],
        selected_order: list[tuple[int, ...]],
        selected: set[tuple[int, ...]],
    ) -> None:
        remaining = [list(bucket) for bucket in buckets if bucket]
        while len(selected_order) < options.max_routes and remaining:
            progressed = False
            next_remaining: list[list[tuple[int, ...]]] = []
            for bucket in remaining:
                while bucket and bucket[0] in selected:
                    bucket.pop(0)
                if not bucket:
                    continue
                route = bucket.pop(0)
                progressed = True
                if route not in selected:
                    selected.add(route)
                    selected_order.append(route)
                if bucket:
                    next_remaining.append(bucket)
                if len(selected_order) >= options.max_routes:
                    break
            remaining = next_remaining
            if not progressed:
                break

    preferred_buckets: list[list[tuple[int, ...]]] = []
    derived_buckets: list[list[tuple[int, ...]]] = []
    all_buckets: list[list[tuple[int, ...]]] = []
    for form, routes in form_routes:
        del form
        all_buckets.append(list(routes))
        preferred_buckets.append([
            route for route in routes
            if _is_preferred_route(route_classes[route])
        ])
        derived_buckets.append([
            route for route in routes
            if not _is_preferred_route(route_classes[route])
        ])

    selected_order: list[tuple[int, ...]] = []
    selected: set[tuple[int, ...]] = set()
    if options.allocation == DEFAULT_ALLOCATION:
        select_buckets(preferred_buckets, selected_order, selected)
        select_buckets(derived_buckets, selected_order, selected)
    else:
        # Experimental allocation evaluates every accepted route by the same
        # edge-scoring rule; do not spend the route budget on a preferred
        # bucket before considering derived alternatives.
        select_buckets(all_buckets, selected_order, selected)

    return CatalogEntry(
        name=name,
        kind="term",
        routes=tuple(route_map[route] for route in selected_order),
        mode="tail" if fallback_tail else mode,
        source=source,
        level=options.level,
        warnings=(warning,) if warning else (),
    )


@dataclass(frozen=True)
class _TermSpec:
    name: str
    source: str
    options: CompileOptions
    explicit_forms: tuple[str, ...] = ()


@dataclass(frozen=True)
class _GroupSpec:
    name: str
    members: tuple[str, ...]
    level: str | None = None


def _term_spec_from_item(item: Any, *, base: CompileOptions, path: str) -> _TermSpec:
    if isinstance(item, str):
        source = normalize_scalar(item, path=path)
        return _TermSpec(source, source, base)
    if not isinstance(item, Mapping) or len(item) != 1:
        raise _error(path, "must be a string or one-entry mapping")
    raw_name, raw_spec = next(iter(item.items()))
    name = normalize_scalar(str(raw_name), path=f"{path}.name")
    if raw_spec is None:
        raw_spec = {}
    if isinstance(raw_spec, str):
        raw_spec = {"text": raw_spec}
    if not isinstance(raw_spec, Mapping):
        raise _error(f"{path}.{name}", "must be a mapping or source string")
    source = normalize_scalar(str(raw_spec.get("text", name)), path=f"{path}.{name}.text")
    options = CompileOptions.from_mapping(raw_spec, base=base, path=f"{path}.{name}")
    explicit = ()
    if "forms" in raw_spec:
        explicit = _as_string_list(raw_spec["forms"], path=f"{path}.{name}.forms")
    return _TermSpec(name, source, options, explicit)


def _parse_source(source: Any) -> tuple[CompileOptions, dict[str, _TermSpec], dict[str, _GroupSpec], tuple[str, ...]]:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        source = {"terms": list(source)}
    if not isinstance(source, Mapping):
        raise EditorError("catalog source must be a YAML mapping or a list of terms")

    reserved = {"defaults", "terms", "groups", "term_options"}
    if not (set(source) & reserved):
        if all(isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
               for value in source.values()):
            source = {"groups": source}
        else:
            raise EditorError("catalog source requires terms/groups or a mapping of groups")

    defaults_raw = source.get("defaults", {})
    defaults = CompileOptions.from_mapping(defaults_raw, path="defaults")
    term_options = source.get("term_options", {})
    if term_options is None:
        term_options = {}
    if not isinstance(term_options, Mapping):
        raise EditorError("term_options must be a mapping")

    terms: dict[str, _TermSpec] = {}
    raw_terms = source.get("terms", ())
    if isinstance(raw_terms, Mapping):
        term_items = [{name: spec} for name, spec in raw_terms.items()]
    elif isinstance(raw_terms, Sequence) and not isinstance(raw_terms, (str, bytes, bytearray)):
        term_items = list(raw_terms)
    else:
        raise EditorError("terms must be a list or mapping")
    for index, item in enumerate(term_items):
        spec = _term_spec_from_item(item, base=defaults, path=f"terms[{index}]")
        override = term_options.get(spec.name)
        if override is not None:
            options = CompileOptions.from_mapping(override, base=spec.options,
                                                   path=f"term_options.{spec.name}")
            spec = _TermSpec(spec.name, spec.source, options, spec.explicit_forms)
        if spec.name in terms and terms[spec.name] != spec:
            raise EditorError(f"duplicate term {spec.name!r}")
        terms[spec.name] = spec

    groups: dict[str, _GroupSpec] = {}
    raw_groups = source.get("groups", {})
    if raw_groups is None:
        raw_groups = {}
    if not isinstance(raw_groups, Mapping):
        raise EditorError("groups must be a mapping")
    for raw_name, raw_spec in raw_groups.items():
        name = normalize_name(str(raw_name), path="group name")
        level: str | None = None
        if isinstance(raw_spec, Mapping):
            raw_members = raw_spec.get("members", ())
            if "level" in raw_spec:
                level = _parse_level(raw_spec["level"], path=f"groups.{name}.level")
        else:
            raw_members = raw_spec
        if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes, bytearray)):
            raise _error(f"groups.{name}.members", "must be a list")
        members = tuple(normalize_scalar(str(member), path=f"groups.{name}.members[{index}]")
                        for index, member in enumerate(raw_members))
        if name in groups:
            raise EditorError(f"duplicate group {name!r}")
        groups[name] = _GroupSpec(name, members, level)

    ungrouped = tuple(terms)
    return defaults, terms, groups, ungrouped


def compile_catalog(
    source: Any,
    backend: Any,
    *,
    provenance: Mapping[str, Any] | None = None,
    default_level: str | None = None,
    options_override: Mapping[str, Any] | None = None,
    reference: Sequence[str] | Mapping[str, float] | None = None,
) -> BiasCatalog:
    """Compile a YAML-shaped source object against one tokenizer/backend."""

    defaults, terms, groups, ungrouped = _parse_source(source)
    override: dict[str, Any] = dict(options_override or {})
    if default_level is not None:
        override["level"] = default_level
    if override:
        defaults = CompileOptions.from_mapping(override, base=defaults,
                                                path="compiler override")
        terms = {
            name: _TermSpec(spec.name, spec.source,
                            CompileOptions.from_mapping(override, base=spec.options,
                                                        path=f"terms.{name}"),
                            spec.explicit_forms)
            for name, spec in terms.items()
        }
    index = _surface_index(backend)
    compiled_terms: dict[str, CatalogEntry] = {}

    def ensure_implicit(member: str, *, level: str | None = None) -> str:
        name = normalize_scalar(member, path="group member")
        if name in terms or name in groups:
            return name
        if name.startswith("@"):
            target = name[1:]
            if target not in terms and target not in groups:
                raise EditorError(f"unknown catalog reference @{target}")
            return target
        options = defaults
        if level is not None:
            options = CompileOptions.from_mapping({"level": level}, base=options,
                                                   path=f"implicit term {name!r}")
        terms[name] = _TermSpec(name, name, options)
        return name

    def compile_entry(name: str, stack: tuple[str, ...] = ()) -> CatalogEntry:
        if name in compiled_terms:
            return compiled_terms[name]
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise EditorError(f"catalog group cycle: {cycle}")
        if name in terms:
            spec = terms[name]
            entry = compile_term(
                spec.name,
                spec.source,
                backend,
                options=spec.options,
                explicit_forms=spec.explicit_forms,
                surface_index=index,
                reference_stats=reference_stats,
            )
            compiled_terms[name] = entry
            return entry
        group = groups.get(name)
        if group is None:
            raise EditorError(f"unknown catalog entry {name!r}")
        routes: list[CompiledRoute] = []
        members: list[str] = []
        for member in group.members:
            target = ensure_implicit(member, level=group.level)
            if target not in members:
                members.append(target)
            child = compile_entry(target, (*stack, name))
            routes.extend(
                CompiledRoute(
                    token_ids=route.token_ids,
                    texts=route.texts,
                    token_texts=route.token_texts,
                    mode=route.mode,
                    strategies=route.strategies,
                    sources=tuple(dict.fromkeys((*route.sources, target))),
                    head_scale=route.head_scale,
                    continuation_scale=route.continuation_scale,
                    route_class=route.route_class,
                    allocation=route.allocation,
                    edge_weights=route.edge_weights,
                    allocation_diagnostics=route.allocation_diagnostics,
                )
                for route in child.routes
            )
        entry = CatalogEntry(
            name=name,
            kind="group",
            routes=_merge_routes((), routes),
            members=tuple(members),
            level=group.level,
            warnings=tuple(dict.fromkeys(
                warning
                for member in members
                for warning in compile_entry(member).warnings
            )),
        )
        compiled_terms[name] = entry
        return entry

    if "global" in terms:
        raise EditorError("the term name 'global' is reserved for the automatic global group")
    if "global" in groups:
        groups["global"] = _GroupSpec(
            "global",
            tuple(dict.fromkeys((*ungrouped, *groups["global"].members))),
            groups["global"].level,
        )
    elif ungrouped:
        groups["global"] = _GroupSpec("global", ungrouped)

    # Resolve implicit group terms before building the shared reference
    # universe, so every generated form can contribute to prefix ambiguity.
    for group in tuple(groups.values()):
        for member in group.members:
            ensure_implicit(member, level=group.level)
    reference_surfaces = tuple(
        form
        for spec in terms.values()
        for form in generate_forms(spec.source, spec.options, spec.explicit_forms)
    )
    reference_stats = _reference_stats_for_backend(
        backend,
        reference_surfaces,
        reference,
    )
    reference_prior_routes = _compile_reference_prior_routes(backend, reference)

    for name in tuple(terms):
        compile_entry(name)
    for name in tuple(groups):
        compile_entry(name)

    model = catalog_model_metadata(backend, provenance)
    warnings = tuple(dict.fromkeys(
        warning
        for entry in compiled_terms.values()
        for warning in entry.warnings
    ))
    return BiasCatalog(
        model=model,
        compiler={
            "format_version": 1,
            "default_level": defaults.level,
            "default_mode": defaults.mode,
            "default_allocation": defaults.allocation,
            "default_allocation_floor": defaults.allocation_floor,
            "default_head_scale": defaults.head_scale,
            "default_continuation_scale": defaults.continuation_scale,
            "default_route_policy": defaults.route_policy,
            "default_min_route_piece_chars": defaults.min_route_piece_chars,
            "levels": list(LEVELS),
            "modes": list(COMPILE_MODES),
            "allocations": list(ALLOCATIONS),
            "route_policies": list(ROUTE_POLICIES),
            "route_classes": list(ROUTE_CLASSES),
            "reference_prior": (
                "compiled-token-routes-v1" if reference_prior_routes else "none"
            ),
            "route_selection": (
                "preferred-routes-first-v2"
                if defaults.allocation == DEFAULT_ALLOCATION
                else "all-routes-round-robin-v1"
            ),
            **({"warnings": list(warnings)} if warnings else {}),
            "default_route_limits": {
                "standard_max_route_tokens": DEFAULT_STANDARD_MAX_ROUTE_TOKENS,
                "exhaustive_max_route_tokens": DEFAULT_EXHAUSTIVE_MAX_ROUTE_TOKENS,
                "max_routes": DEFAULT_MAX_ROUTES,
            },
        },
        entries=compiled_terms,
        reference_prior_routes=reference_prior_routes,
    )


def load_yaml_source(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise EditorError("PyYAML is required to compile bias catalogs") from exc
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise EditorError(f"could not read YAML source {path}: {exc}") from exc
    if value is None:
        raise EditorError(f"YAML source {path} is empty")
    return value


def load_catalog(path: Path) -> BiasCatalog:
    try:
        return BiasCatalog.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditorError(f"could not read catalog {path}: {exc}") from exc
