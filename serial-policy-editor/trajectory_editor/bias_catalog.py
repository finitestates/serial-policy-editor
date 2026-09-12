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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .domain import EditorError


CATALOG_FORMAT = "spe-bias-catalog-v1"
LEVELS = ("minimal", "standard", "exhaustive")
MODES = ("tail", "path")
COMPILE_MODES = ("auto", *MODES)
DEFAULT_LEVEL = "standard"
DEFAULT_MAX_ROUTES = 4096
DEFAULT_STANDARD_MAX_ROUTE_TOKENS = 2
DEFAULT_EXHAUSTIVE_MAX_ROUTE_TOKENS = 8
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

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise EditorError(f"unknown compilation level {self.level!r}")
        if self.mode not in COMPILE_MODES:
            raise EditorError(f"unknown compilation mode {self.mode!r}")
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
class CompiledRoute:
    token_ids: tuple[int, ...]
    texts: tuple[str, ...]
    token_texts: tuple[str, ...]
    mode: str
    strategies: tuple[str, ...]
    sources: tuple[str, ...] = ()
    head_scale: float = 1.0
    continuation_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "texts": list(self.texts),
            "token_texts": list(self.token_texts),
            "mode": self.mode,
            "strategies": list(self.strategies),
            **({"sources": list(self.sources)} if self.sources else {}),
            **({"head_scale": self.head_scale} if self.head_scale != 1.0 else {}),
            **({"continuation_scale": self.continuation_scale}
               if self.continuation_scale != 1.0 else {}),
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
        return result


@dataclass(frozen=True)
class BiasCatalog:
    model: Mapping[str, Any]
    compiler: Mapping[str, Any]
    entries: Mapping[str, CatalogEntry]

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
            mode = raw_entry.get("default_mode")
            if mode is not None and mode not in MODES:
                raise _error(f"entries.{name}.default_mode", "must be tail or path")
            entries[name] = CatalogEntry(
                name=name,
                kind=kind,
                routes=routes,
                mode=mode,
                source=raw_entry.get("source"),
                members=members,
                level=raw_entry.get("level"),
            )
        return cls(
            model=dict(value.get("model", {})),
            compiler=dict(value.get("compiler", {})),
            entries=entries,
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
        return cls(
            model=first.model,
            compiler={
                **dict(first.compiler),
                "merged_catalogs": len(catalogs),
            },
            entries=entries,
        )


def _model_identity(model: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        model.get("backend"),
        model.get("vocabulary_size"),
        model.get("tokenizer_fingerprint"),
    )


def _merge_routes(left: Sequence[CompiledRoute], right: Sequence[CompiledRoute]) -> tuple[CompiledRoute, ...]:
    merged: dict[tuple[tuple[int, ...], str, float, float], CompiledRoute] = {}
    for route in (*left, *right):
        key = (
            route.token_ids,
            route.mode,
            route.head_scale,
            route.continuation_scale,
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
            raise _error(f"{route_path}.mode", "must be tail or path")
        texts = tuple(str(text) for text in raw.get("texts", ()))
        token_texts = tuple(str(text) for text in raw.get("token_texts", ()))
        strategies = tuple(str(strategy) for strategy in raw.get("strategies", ()))
        sources = tuple(str(source) for source in raw.get("sources", ()))
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
                value = base.title()
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
                routes.append(tokens)
            continue
        if len(tokens) >= max_tokens:
            continue
        for piece, token_ids in index.get(surface[position], ()):
            if not piece or not surface.startswith(piece, position):
                continue
            next_position = position + len(piece)
            for token_id in reversed(token_ids):
                frontier.append((next_position, (*tokens, int(token_id))))
    return tuple(dict.fromkeys(routes))


def _route_quality(
    route: Sequence[int],
    token_texts: Sequence[str],
    all_routes: Sequence[Sequence[int]],
) -> tuple[Any, ...]:
    """Return a deterministic, tokenizer-local quality key for an alternate route.

    Lower keys are preferred.  The route count and tiny-piece penalties favor
    cohesive decompositions; the leading-piece and local fan-out terms prefer
    routes whose first edge carries more of the term and is less ambiguous.
    Token IDs only break otherwise identical ties.
    """

    content_lengths = tuple(len(text.replace(" ", "")) for text in token_texts)
    tiny_piece_count = sum(length <= 1 for length in content_lengths)
    head = route[0]
    head_fanout = sum(
        1 for candidate in all_routes if candidate and candidate[0] == head
    )
    return (
        len(route),
        tiny_piece_count,
        -content_lengths[0],
        -min(content_lengths),
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
) -> CatalogEntry:
    name = normalize_scalar(name, path="term name")
    source = normalize_scalar(source, path=f"term {name!r}")
    mode = options.mode
    if mode == "auto":
        mode = "tail" if any(character.isspace() for character in source) else "path"
    forms = generate_forms(source, options, explicit_forms)
    index = surface_index if surface_index is not None else _surface_index(backend)
    route_map: dict[tuple[int, ...], CompiledRoute] = {}
    canonical_order: list[tuple[int, ...]] = []
    canonical_routes: set[tuple[int, ...]] = set()
    alternate_buckets: list[list[tuple[int, ...]]] = []

    for form in forms:
        canonical = _canonical_route(form, backend)
        if canonical not in canonical_routes:
            canonical_routes.add(canonical)
            canonical_order.append(canonical)

        candidates: list[tuple[tuple[int, ...], str]] = [(canonical, "canonical")]
        form_alternates: list[tuple[int, ...]] = []
        if options.level != "minimal":
            for route in _enumerate_routes(
                form,
                backend,
                index,
                max_tokens=options.route_limit(),
            ):
                if route != canonical:
                    candidates.append((route, "alternate"))
                    form_alternates.append(route)

        for route, strategy in candidates:
            token_texts = tuple(str(backend.token_text(token)) for token in route)
            existing = route_map.get(route)
            if existing is None:
                route_map[route] = CompiledRoute(
                    token_ids=route,
                    texts=(form,),
                    token_texts=token_texts,
                    mode=mode,
                    strategies=(strategy,),
                    sources=(name,),
                    head_scale=options.head_scale,
                    continuation_scale=options.continuation_scale,
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
                )

        if form_alternates:
            unique = tuple(dict.fromkeys(form_alternates))
            alternate_buckets.append(sorted(
                unique,
                key=lambda route: _route_quality(
                    route,
                    route_map[route].token_texts,
                    tuple(candidate for candidate, _ in candidates),
                ),
            ))

    if len(canonical_order) > options.max_routes:
        raise EditorError(
            f"canonical routes for term {name!r} exceed "
            f"max_routes={options.max_routes}"
        )

    selected_order = list(canonical_order)
    selected = set(selected_order)
    while len(selected_order) < options.max_routes and alternate_buckets:
        progressed = False
        remaining_buckets: list[list[tuple[int, ...]]] = []
        for bucket in alternate_buckets:
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
                remaining_buckets.append(bucket)
            if len(selected_order) >= options.max_routes:
                break
        alternate_buckets = remaining_buckets
        if not progressed:
            break

    return CatalogEntry(
        name=name,
        kind="term",
        routes=tuple(route_map[route] for route in selected_order),
        mode=mode,
        source=source,
        level=options.level,
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
                )
                for route in child.routes
            )
        entry = CatalogEntry(
            name=name,
            kind="group",
            routes=_merge_routes((), routes),
            members=tuple(members),
            level=group.level,
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

    for name in tuple(terms):
        compile_entry(name)
    for name in tuple(groups):
        compile_entry(name)

    model = catalog_model_metadata(backend, provenance)
    return BiasCatalog(
        model=model,
        compiler={
            "format_version": 1,
            "default_level": defaults.level,
            "default_mode": defaults.mode,
            "default_head_scale": defaults.head_scale,
            "default_continuation_scale": defaults.continuation_scale,
            "levels": list(LEVELS),
            "modes": list(COMPILE_MODES),
            "default_route_limits": {
                "standard_max_route_tokens": DEFAULT_STANDARD_MAX_ROUTE_TOKENS,
                "exhaustive_max_route_tokens": DEFAULT_EXHAUSTIVE_MAX_ROUTE_TOKENS,
                "max_routes": DEFAULT_MAX_ROUTES,
            },
        },
        entries=compiled_terms,
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
