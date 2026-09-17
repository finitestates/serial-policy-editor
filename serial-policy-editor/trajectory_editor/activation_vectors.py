"""Portable activation-vector artifacts.

The first activation interface intentionally operates at the model's final
hidden/output-head boundary.  It is a real hidden-state difference, while its
runtime application is the equivalent linear output-head logit adjustment.
The same artifact envelope can also carry the layerwise F32 directions emitted
by llama.cpp's cvector-generator.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .domain import EditorError, SamplingConfig


FORMAT = "spe-activation-vector-v1"
KIND = "activation"
OUTPUT_LAYER = "output"
RUNTIME_POSITION = "current"
CONTROL_VECTOR_LAYER = "control-vector"
CONTROL_VECTOR_POSITION = "layers"
CAPTURE_POSITIONS = {"first", "last"}
MODEL_IDENTITY_FIELDS = (
    "backend",
    "adapter",
    "filename",
    "file_size_bytes",
    "vocabulary_size",
    "tokenizer_fingerprint",
    "model_type",
    "activation_width",
    "activation_layer_count",
)


def _vector(value: Any, name: str = "activation_vector") -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise EditorError(f"{name} must be an array")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EditorError(f"{name} must contain numbers") from exc
    if not result:
        raise EditorError(f"{name} must not be empty")
    if not all(math.isfinite(item) for item in result):
        raise EditorError(f"{name} must contain finite numbers")
    return result


def model_identity(
    provenance: Mapping[str, Any], *, activation_width: int | None = None
) -> dict[str, Any]:
    """Keep stable model facts needed to interpret an activation vector."""
    result = {
        name: provenance[name]
        for name in MODEL_IDENTITY_FIELDS
        if name != "activation_width"
        and name in provenance
        and provenance[name] is not None
    }
    if activation_width is None:
        activation_width = provenance.get("activation_width")
    if activation_width is not None:
        result["activation_width"] = int(activation_width)
    return result


def model_identity_json(model: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(model), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def model_identity_from_json(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise EditorError("activation vector model identity is malformed") from exc
    if not isinstance(parsed, dict):
        raise EditorError("activation vector model identity must be an object")
    return parsed


def activation_vector_digest_for(
    vector: Sequence[float],
    *,
    model: Mapping[str, Any] | str | None = None,
    layer: str = OUTPUT_LAYER,
    position: str = RUNTIME_POSITION,
    strength: float = 1.0,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> str:
    """Return the canonical content digest used by activation artifacts.

    Runtime state must not rely on a caller-provided label alone.  Keeping the
    digest construction here also makes artifact and live sampler identities
    agree without making the domain module import the artifact type.
    """
    if model is None:
        model_value: Mapping[str, Any] = {}
    elif isinstance(model, str):
        model_value = model_identity_from_json(model)
    elif isinstance(model, Mapping):
        model_value = model
    else:
        raise EditorError("activation vector model identity must be an object")
    payload = {
        "model": dict(model_value),
        "layer": layer,
        "position": position,
        "strength": float(strength),
        "vector": [float(value) for value in vector],
        "layer_start": layer_start,
        "layer_end": layer_end,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_model_compatibility(
    left: Mapping[str, Any], right: Mapping[str, Any], *, label: str
) -> None:
    for name in MODEL_IDENTITY_FIELDS:
        left_value = left.get(name)
        right_value = right.get(name)
        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            raise EditorError(
                f"incompatible activation models: {label} differs in {name}"
            )


def assert_model_compatible(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, label: str
) -> None:
    """Validate a saved activation model identity against a loaded model."""
    _check_model_compatibility(expected, actual, label=label)


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in kwargs.items() if name in names}


class _GGUFReader:
    """Small defensive reader for the F32 direction tensors in cvector GGUFs."""

    _VALUE_TYPES = {
        0: "B",   # UINT8
        1: "b",   # INT8
        2: "H",   # UINT16
        3: "h",   # INT16
        4: "I",   # UINT32
        5: "i",   # INT32
        6: "f",   # FLOAT32
        7: "?",   # BOOL
        10: "Q",  # UINT64
        11: "q",  # INT64
        12: "d",  # FLOAT64
    }

    def __init__(self, path: Path) -> None:
        try:
            self.data = path.read_bytes()
        except OSError as exc:
            raise EditorError(f"could not read cvector GGUF: {exc}") from exc
        self.path = path
        self.offset = 0

    def read(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        if self.offset + size > len(self.data):
            raise EditorError("cvector GGUF is truncated")
        value = struct.unpack_from("<" + fmt, self.data, self.offset)[0]
        self.offset += size
        return value

    def string(self) -> str:
        size = int(self.read("Q"))
        if size > len(self.data) - self.offset:
            raise EditorError("cvector GGUF contains an invalid string")
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EditorError("cvector GGUF contains a non-UTF-8 string") from exc

    def value(self, value_type: int) -> Any:
        if value_type == 8:  # STRING
            return self.string()
        if value_type == 9:  # ARRAY
            element_type = int(self.read("I"))
            count = int(self.read("Q"))
            if count > 10_000_000:
                raise EditorError("cvector GGUF metadata array is too large")
            return [self.value(element_type) for _ in range(count)]
        fmt = self._VALUE_TYPES.get(value_type)
        if fmt is None:
            raise EditorError(f"unsupported cvector GGUF metadata type {value_type}")
        return self.read(fmt)

    def tensor(self) -> tuple[str, tuple[int, ...], int, int]:
        name = self.string()
        dimensions = int(self.read("I"))
        if dimensions < 1 or dimensions > 4:
            raise EditorError("cvector GGUF tensor has an invalid rank")
        shape = tuple(int(self.read("Q")) for _ in range(dimensions))
        if any(value < 1 for value in shape):
            raise EditorError("cvector GGUF tensor has an invalid shape")
        tensor_type = int(self.read("I"))
        offset = int(self.read("Q"))
        return name, shape, tensor_type, offset


def _load_cvector_gguf(path: Path) -> tuple[dict[str, Any], tuple[float, ...], dict[str, Any]]:
    reader = _GGUFReader(path)
    if reader.data[:4] != b"GGUF":
        raise EditorError("cvector input is not a GGUF file")
    reader.offset = 4
    version = int(reader.read("I"))
    if version not in {2, 3}:
        raise EditorError(f"unsupported cvector GGUF version {version}")
    tensor_count = int(reader.read("Q"))
    metadata_count = int(reader.read("Q"))
    if tensor_count > 100_000 or metadata_count > 100_000:
        raise EditorError("cvector GGUF header is implausibly large")
    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key = reader.string()
        metadata[key] = reader.value(int(reader.read("I")))
    tensors = [reader.tensor() for _ in range(tensor_count)]
    alignment = int(metadata.get("general.alignment", 32))
    if alignment < 1 or alignment > 4096:
        raise EditorError("cvector GGUF alignment is invalid")
    data_start = (reader.offset + alignment - 1) // alignment * alignment
    if data_start > len(reader.data):
        raise EditorError("cvector GGUF has no tensor data")

    directions: dict[int, tuple[float, ...]] = {}
    width: int | None = None
    for name, shape, tensor_type, offset in tensors:
        if not name.startswith("direction."):
            continue
        try:
            layer = int(name.split(".", 1)[1])
        except (IndexError, ValueError) as exc:
            raise EditorError(f"invalid cvector direction tensor name {name!r}") from exc
        if layer < 1 or len(shape) != 1 or tensor_type != 0:
            raise EditorError("cvector directions must be unique one-dimensional F32 tensors")
        if layer in directions:
            raise EditorError(f"duplicate cvector direction layer {layer}")
        if width is None:
            width = shape[0]
        elif width != shape[0]:
            raise EditorError("cvector direction widths do not match")
        start = data_start + offset
        end = start + shape[0] * 4
        if start < data_start or end > len(reader.data):
            raise EditorError("cvector direction tensor is outside the GGUF data")
        values = tuple(float(value) for value in struct.unpack_from(
            "<" + "f" * shape[0], reader.data, start
        ))
        if not all(math.isfinite(value) for value in values):
            raise EditorError("cvector direction tensor contains non-finite values")
        directions[layer] = values
    if not directions or width is None:
        raise EditorError("cvector GGUF contains no direction tensors")
    expected_layers = list(range(1, max(directions) + 1))
    if sorted(directions) != expected_layers:
        raise EditorError("cvector direction layers must be contiguous from layer 1")
    layer_count = int(metadata.get("controlvector.layer_count", len(directions)))
    if layer_count != len(directions):
        raise EditorError("cvector layer count metadata does not match direction tensors")
    model_hint = metadata.get("controlvector.model_hint")
    model = {
        "backend": "llama.cpp",
        "model_type": model_hint if isinstance(model_hint, str) else None,
        "activation_width": width,
        "activation_layer_count": layer_count,
    }
    model = {key: value for key, value in model.items() if value is not None}
    vector = tuple(value for layer in expected_layers for value in directions[layer])
    source = {
        "type": "llama-cvector-gguf",
        "adapter": "llama-cvector",
        "path": str(path),
        "gguf_version": version,
        "model_hint": model_hint,
        "layer_count": layer_count,
        "direction_layers": expected_layers,
    }
    return model, vector, source


@dataclass(frozen=True)
class ActivationVectorArtifact:
    """A model-matched output direction or llama.cpp layerwise cvector."""

    model: Mapping[str, Any]
    vector: tuple[float, ...]
    layer: str = OUTPUT_LAYER
    position: str = RUNTIME_POSITION
    strength: float = 1.0
    method: str = "prompt-difference-v1"
    source: Mapping[str, Any] | None = None
    layer_start: int | None = None
    layer_end: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, Mapping):
            raise EditorError("activation artifact model must be an object")
        object.__setattr__(self, "model", dict(self.model))
        vector = _vector(self.vector)
        object.__setattr__(self, "vector", vector)
        if self.layer == OUTPUT_LAYER:
            if self.position != RUNTIME_POSITION:
                raise EditorError(f"activation position must be {RUNTIME_POSITION!r}")
            if self.layer_start is not None or self.layer_end is not None:
                raise EditorError("output activation vectors cannot specify a layer range")
        elif self.layer == CONTROL_VECTOR_LAYER:
            if self.position != CONTROL_VECTOR_POSITION:
                raise EditorError(
                    f"control-vector activation position must be {CONTROL_VECTOR_POSITION!r}"
                )
            if (
                type(self.layer_start) is not int
                or self.layer_start < 1
                or type(self.layer_end) is not int
                or self.layer_end < self.layer_start
            ):
                raise EditorError("control-vector activation layer range is invalid")
        else:
            raise EditorError(
                f"activation layer must be {OUTPUT_LAYER!r} or {CONTROL_VECTOR_LAYER!r}"
            )
        if not isinstance(self.method, str) or not self.method:
            raise EditorError("activation artifact method must be nonempty")
        if (
            type(self.strength) not in (int, float)
            or not math.isfinite(float(self.strength))
            or float(self.strength) < 0.0
        ):
            raise EditorError("activation strength must be finite and nonnegative")
        object.__setattr__(self, "strength", float(self.strength))
        if self.source is not None:
            if not isinstance(self.source, Mapping):
                raise EditorError("activation artifact source must be an object")
            object.__setattr__(self, "source", dict(self.source))

        width = self.model.get("activation_width")
        if width is not None:
            if type(width) is not int or width < 1:
                raise EditorError("activation model width must be a positive integer")
            if self.layer == OUTPUT_LAYER and width != len(vector):
                raise EditorError("activation vector dimension does not match model width")
            if self.layer == CONTROL_VECTOR_LAYER and len(vector) % width:
                raise EditorError("control-vector activation dimension is not layer-aligned")
        layer_count = self.model.get("activation_layer_count")
        if layer_count is not None and (
            type(layer_count) is not int or layer_count < 1
        ):
            raise EditorError("activation model layer count must be a positive integer")
        if self.layer == CONTROL_VECTOR_LAYER:
            if width is None:
                raise EditorError("control-vector activation requires model width")
            if layer_count is not None and len(vector) // int(width) != layer_count:
                raise EditorError("control-vector activation layer count does not match model")
            if layer_count is not None and self.layer_end > layer_count:
                raise EditorError("control-vector activation layer range exceeds model layers")

    @property
    def dimension(self) -> int:
        return len(self.vector)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.vector, dtype=np.float64)))

    @property
    def digest(self) -> str:
        return activation_vector_digest_for(
            self.vector,
            model=self.model,
            layer=self.layer,
            position=self.position,
            strength=self.strength,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActivationVectorArtifact":
        if not isinstance(value, Mapping):
            raise EditorError("activation artifact must be an object")
        if value.get("format") != FORMAT:
            raise EditorError(f"activation artifact must use format {FORMAT}")
        if value.get("kind") != KIND:
            raise EditorError(f"activation artifact kind must be {KIND!r}")
        model = value.get("model")
        if not isinstance(model, Mapping):
            raise EditorError("activation artifact requires model metadata")
        return cls(
            model=model,
            vector=value.get("vector"),
            layer=value.get("layer", OUTPUT_LAYER),
            position=value.get("position", RUNTIME_POSITION),
            strength=value.get("strength", 1.0),
            method=value.get("method", "prompt-difference-v1"),
            source=value.get("source"),
            layer_start=value.get("layer_start"),
            layer_end=value.get("layer_end"),
        )

    @classmethod
    def from_path(cls, path: Path) -> "ActivationVectorArtifact":
        if path.suffix.lower() == ".gguf":
            return cls.from_cvector_path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EditorError(f"could not read activation artifact: {exc}") from exc
        artifact = cls.from_mapping(value)
        declared_digest = value.get("digest")
        if declared_digest is not None and declared_digest != artifact.digest:
            raise EditorError("activation artifact digest does not match its contents")
        return artifact

    @classmethod
    def from_cvector_path(
        cls, path: Path, *, strength: float = 1.0
    ) -> "ActivationVectorArtifact":
        model, vector, source = _load_cvector_gguf(path)
        layer_count = int(model["activation_layer_count"])
        return cls(
            model=model,
            vector=vector,
            layer=CONTROL_VECTOR_LAYER,
            position=CONTROL_VECTOR_POSITION,
            strength=strength,
            method="llama-cvector-generator-v1",
            source=source,
            layer_start=1,
            layer_end=layer_count,
        )

    @classmethod
    def from_prompt_pair(
        cls,
        backend: Any,
        provenance: Mapping[str, Any],
        prompt_a: str,
        prompt_b: str,
        *,
        capture_position: str = "last",
        normalize: bool = True,
    ) -> "ActivationVectorArtifact":
        if capture_position not in CAPTURE_POSITIONS:
            raise EditorError(
                "activation capture position must be first or last"
            )
        if not isinstance(prompt_a, str) or not prompt_a:
            raise EditorError("prompt A must be nonempty")
        if not isinstance(prompt_b, str) or not prompt_b:
            raise EditorError("prompt B must be nonempty")
        capture = getattr(backend, "activation_snapshot", None)
        width_method = getattr(backend, "activation_width", None)
        if not callable(capture) or not callable(width_method):
            raise EditorError(
                "the loaded backend does not expose output activation snapshots"
            )
        try:
            kwargs = {"layer": OUTPUT_LAYER, "position": capture_position}
            first = np.asarray(
                capture(prompt_a, **_supported_kwargs(capture, kwargs)),
                dtype=np.float64,
            )
            second = np.asarray(
                capture(prompt_b, **_supported_kwargs(capture, kwargs)),
                dtype=np.float64,
            )
            width = int(width_method())
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EditorError(f"could not capture activation pair: {exc}") from exc
        if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
            raise EditorError("activation snapshots must be equal one-dimensional vectors")
        if first.shape[0] != width:
            raise EditorError(
                f"activation snapshot width {first.shape[0]} does not match backend width {width}"
            )
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise EditorError("activation snapshots must be finite")
        delta = first - second
        raw_norm = float(np.linalg.norm(delta))
        if normalize and raw_norm > 0.0:
            delta = delta / raw_norm
        return cls(
            model=model_identity(provenance, activation_width=width),
            vector=tuple(float(value) for value in delta),
            source={
                "type": "prompt-pair",
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "capture_position": capture_position,
                "normalized": bool(normalize),
                "raw_delta_norm": raw_norm,
            },
        )

    @classmethod
    def from_prompt_pairs(
        cls,
        backend: Any,
        provenance: Mapping[str, Any],
        pairs: Sequence[tuple[str, str]],
        *,
        capture_position: str = "last",
        normalize: bool = True,
        strength: float = 1.0,
        source: Mapping[str, Any] | None = None,
    ) -> "ActivationVectorArtifact":
        """Average positive-minus-negative activation differences.

        Each pair is captured independently, then the raw differences are
        averaged before optional unit normalization.  Averaging the raw
        differences keeps a long or unusually energetic example from being
        silently given a larger direction merely because it was normalized
        first.
        """
        if not pairs:
            raise EditorError("at least one activation prompt pair is required")
        if capture_position not in CAPTURE_POSITIONS:
            raise EditorError("activation capture position must be first or last")
        capture = getattr(backend, "activation_snapshot", None)
        width_method = getattr(backend, "activation_width", None)
        if not callable(capture) or not callable(width_method):
            raise EditorError(
                "the loaded backend does not expose output activation snapshots"
            )

        deltas: list[np.ndarray] = []
        pair_norms: list[float] = []
        try:
            width = int(width_method())
            if width < 1:
                raise ValueError("activation width must be positive")
            kwargs = {"layer": OUTPUT_LAYER, "position": capture_position}
            for positive_prompt, negative_prompt in pairs:
                if not isinstance(positive_prompt, str) or not positive_prompt:
                    raise ValueError("positive prompts must be nonempty strings")
                if not isinstance(negative_prompt, str) or not negative_prompt:
                    raise ValueError("negative prompts must be nonempty strings")
                positive = np.asarray(
                    capture(positive_prompt, **_supported_kwargs(capture, kwargs)),
                    dtype=np.float64,
                )
                negative = np.asarray(
                    capture(negative_prompt, **_supported_kwargs(capture, kwargs)),
                    dtype=np.float64,
                )
                if positive.ndim != 1 or negative.ndim != 1 or positive.shape != negative.shape:
                    raise ValueError(
                        "activation snapshots must be equal one-dimensional vectors"
                    )
                if positive.shape[0] != width:
                    raise ValueError(
                        f"activation snapshot width {positive.shape[0]} does not match backend width {width}"
                    )
                if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
                    raise ValueError("activation snapshots must be finite")
                delta = positive - negative
                deltas.append(delta)
                pair_norms.append(float(np.linalg.norm(delta)))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EditorError(f"could not capture activation pairs: {exc}") from exc

        aggregate = np.mean(np.stack(deltas, axis=0), axis=0)
        aggregate_norm = float(np.linalg.norm(aggregate))
        if normalize and aggregate_norm > 0.0:
            aggregate = aggregate / aggregate_norm
        source_payload = dict(source or {})
        source_payload.setdefault("type", "prompt-pairs")
        source_payload.update(
            {
                "pair_count": len(deltas),
                "capture_position": capture_position,
                "normalized": bool(normalize),
                "pair_delta_norms": pair_norms,
                "aggregate_delta_norm": aggregate_norm,
            }
        )
        return cls(
            model=model_identity(provenance, activation_width=width),
            vector=tuple(float(value) for value in aggregate),
            strength=strength,
            method="prompt-pairs-mean-v1",
            source=source_payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": KIND,
            "model": dict(self.model),
            "layer": self.layer,
            "position": self.position,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "strength": self.strength,
            "method": self.method,
            "vector": list(self.vector),
            "digest": self.digest,
            **({"source": dict(self.source)} if self.source is not None else {}),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def write(self, path: Path) -> None:
        try:
            path.write_text(self.to_json() + "\n", encoding="utf-8")
        except OSError as exc:
            raise EditorError(f"could not write activation artifact: {exc}") from exc

    def validate_against_backend(
        self, backend: Any, provenance: Mapping[str, Any]
    ) -> np.ndarray:
        width_method = getattr(backend, "activation_width", None)
        if not callable(width_method):
            raise EditorError("the loaded backend does not expose activation width metadata")
        loaded_model = model_identity(provenance, activation_width=int(width_method()))
        assert_model_compatible(self.model, loaded_model, label="loaded model")
        if self.layer == CONTROL_VECTOR_LAYER:
            control_width = getattr(backend, "activation_control_vector_width", None)
            control_layers = getattr(backend, "activation_control_vector_layer_count", None)
            if not callable(control_width) or not callable(control_layers):
                raise EditorError(
                    "the loaded backend does not expose llama.cpp control-vector runtime support"
                )
            if int(control_width()) != int(self.model.get("activation_width", -1)):
                raise EditorError("control-vector width does not match the loaded model")
            if int(control_width()) * int(control_layers()) != self.dimension:
                raise EditorError("control-vector dimension does not match the loaded model layers")
            if self.layer_end > int(control_layers()):
                raise EditorError("control-vector layer range exceeds the loaded model")
            return np.zeros(int(backend.vocabulary_size()), dtype=np.float64)
        adjustment = getattr(backend, "activation_logit_adjustments", None)
        if not callable(adjustment):
            raise EditorError("the loaded backend does not expose output activation runtime support")
        if int(width_method()) != self.dimension:
            raise EditorError(
                f"activation vector dimension {self.dimension} does not match loaded model width {width_method()}"
            )
        try:
            values = adjustment(
                np.asarray(self.vector, dtype=np.float32),
                **_supported_kwargs(
                    adjustment,
                    {"layer": self.layer, "position": self.position},
                ),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EditorError(f"could not validate activation vector: {exc}") from exc
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != int(backend.vocabulary_size()):
            raise EditorError("activation logit adjustment does not match vocabulary")
        if not np.all(np.isfinite(values)):
            raise EditorError("activation logit adjustment is not finite")
        return values

    def apply_to_sampling(
        self, sampling: SamplingConfig, *, strength: float | None = None
    ) -> SamplingConfig:
        return replace_activation_sampling(
            sampling,
            vector=self.vector,
            strength=self.strength if strength is None else strength,
            layer=self.layer,
            position=self.position,
            model=self.model,
            digest=self.digest,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )


def replace_activation_sampling(
    sampling: SamplingConfig,
    *,
    vector: tuple[float, ...] | list[float],
    strength: float,
    layer: str,
    position: str,
    model: Mapping[str, Any],
    digest: str,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> SamplingConfig:
    """Attach an activation artifact's effective state to a sampler config."""
    from dataclasses import replace

    return replace(
        sampling,
        activation_vector=tuple(float(value) for value in vector),
        activation_vector_strength=float(strength),
        activation_vector_layer=layer,
        activation_vector_position=position,
        activation_vector_model=model_identity_json(model),
        activation_vector_digest=digest,
        activation_vector_layer_start=layer_start,
        activation_vector_layer_end=layer_end,
    )


def assert_compatible(
    left: ActivationVectorArtifact, right: ActivationVectorArtifact
) -> None:
    _check_model_compatibility(left.model, right.model, label="artifacts")
    if left.layer != right.layer:
        raise EditorError("incompatible activation layers")
    if left.position != right.position:
        raise EditorError("incompatible activation positions")
    if (left.layer_start, left.layer_end) != (right.layer_start, right.layer_end):
        raise EditorError("incompatible activation layer ranges")


def blend_artifacts(
    artifacts: list[ActivationVectorArtifact],
    weights: list[float],
    *,
    source: Mapping[str, Any] | None = None,
) -> ActivationVectorArtifact:
    if not artifacts:
        raise EditorError("blend requires at least one activation artifact")
    if len(weights) != len(artifacts):
        raise EditorError("blend weights must match the number of artifacts")
    if any(not math.isfinite(float(weight)) for weight in weights):
        raise EditorError("blend weights must be finite numbers")
    first = artifacts[0]
    for artifact in artifacts[1:]:
        assert_compatible(first, artifact)
    vector = np.zeros(first.dimension, dtype=np.float64)
    for weight, artifact in zip(weights, artifacts):
        vector += float(weight) * artifact.strength * np.asarray(
            artifact.vector, dtype=np.float64
        )
    return ActivationVectorArtifact(
        model=first.model,
        vector=tuple(float(value) for value in vector),
        layer=first.layer,
        position=first.position,
        strength=1.0,
        method="weighted-blend-v1",
        source=source,
        layer_start=first.layer_start,
        layer_end=first.layer_end,
    )
