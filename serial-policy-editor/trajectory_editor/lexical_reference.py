"""Standalone lexical priors: human surfaces and relative positive weights.

No group or learner is needed. Token routes are compiled once and stored with
the steering preset, so importing a preset requires no original YAML file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .domain import EditorError


def compile_reference(reference, backend) -> tuple[tuple[tuple[int, ...], float], ...]:
    if isinstance(reference, Mapping):
        items = tuple(reference.items())
    elif isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        items = tuple((text, 1.0) for text in reference)
    else:
        raise EditorError("reference weights must be a term-to-weight mapping or a list of terms")
    if not items:
        raise EditorError("reference weights must not be empty")
    normalized = []
    for text, raw_weight in items:
        if not isinstance(text, str) or not text.strip():
            raise EditorError("reference terms must be nonempty strings")
        try:
            weight = float(raw_weight)
        except (ValueError, TypeError) as exc:
            raise EditorError(f"reference weight for {text!r} must be numeric") from exc
        if isinstance(raw_weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise EditorError(f"reference weight for {text!r} must be finite and positive")
        normalized.append((text, weight))
    # Normalize before accumulating variants to avoid overflow. Multiplying
    # every supplied weight by the same constant changes no policy behavior.
    scale = max(weight for _, weight in normalized)
    result = {}
    for text, weight in normalized:
        forms = (text,) if text.startswith(" ") else (text, f" {text}")
        routes = set()
        for form in forms:
            route = tuple(backend.tokenize(form, add_bos=False, special=False))
            if not route or any(type(t) is not int or not 0 <= t < backend.vocabulary_size()
                                or backend.is_eog(t) for t in route):
                raise EditorError(f"reference term {form!r} does not produce ordinary model tokens")
            routes.add(route)
        mass = (weight / scale) / len(routes)
        if mass == 0:
            raise EditorError("reference weight range is too large to represent")
        for route in routes:
            result[route] = result.get(route, 0.0) + mass
    return tuple(sorted(result.items()))


def load_reference(path: Path, backend):
    import yaml

    try:
        # Keep lexical keys such as yes/on/123 as text.
        source = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise EditorError(f"could not read reference weights: {exc}") from exc
    return compile_reference(source, backend)
