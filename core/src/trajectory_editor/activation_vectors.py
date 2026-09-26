"""Portable steering-vector artifacts.

This module contains two deliberately distinct artifact families: output-head
steering vectors, which are post-normalization output-head-input directions
projected into vocabulary logits, and hidden-state vectors, which are layerwise
directions installed inside a compatible model runtime.

The implementation file retains its historical name for now so the runtime
refactor can proceed independently of the public terminology. The artifact
format and command surface do not retain that ambiguity.
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

from .core.errors import EditorError
from .core.sampler_config import SamplerConfig


FORMAT = "spe-steering-vector-v1"
OUTPUT_HEAD_KIND = "output-head-steering-vector"
HIDDEN_STATE_KIND = "hidden-state-vector"
HIDDEN_STATE_COORDINATE = "canonical-decoder-block-output-v1"
OUTPUT_LAYER = "output"
RUNTIME_POSITION = "current"
CONTROL_VECTOR_LAYER = "control-vector"
CONTROL_VECTOR_POSITION = "layers"
CAPTURE_POSITIONS = {"first", "last"}


def _vector(value: Any, name: str = "steering_vector") -> tuple[float, ...]:
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


def steering_vector_digest_for(
    vector: Sequence[float],
    *,
    layer: str = OUTPUT_LAYER,
    position: str = RUNTIME_POSITION,
    strength: float = 1.0,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> str:
    """Return the canonical content digest used by steering artifacts."""
    payload = {
        "layer": layer,
        "position": position,
        "coordinate": (
            HIDDEN_STATE_COORDINATE
            if layer == CONTROL_VECTOR_LAYER
            else "output-head-current-v1"
        ),
        "strength": float(strength),
        "vector": [float(value) for value in vector],
        "layer_start": layer_start,
        "layer_end": layer_end,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _load_cvector_gguf(path: Path) -> tuple[tuple[float, ...], dict[str, Any]]:
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
    native_layer_count = int(metadata.get("controlvector.layer_count", len(directions)))
    if native_layer_count != len(directions):
        raise EditorError("cvector layer count metadata does not match direction tensors")
    # llama.cpp's native cvector buffer starts at native slot 1.  Native slot
    # 1 is applied after canonical decoder block 2, while native slot N-1 is
    # applied after canonical decoder block N.  Make that translation explicit
    # instead of presenting native slot numbers as human-facing layer numbers.
    layer_count = native_layer_count + 1
    canonical_vector = (0.0,) * width + tuple(
        value for layer in expected_layers for value in directions[layer]
    )
    model_hint = metadata.get("controlvector.model_hint")
    source = {
        "type": "llama-cvector-gguf",
        "adapter": "llama-cvector",
        "path": str(path),
        "gguf_version": version,
        "model_hint": model_hint,
        "native_layer_count": native_layer_count,
        "canonical_layer_count": layer_count,
        "direction_layers": expected_layers,
        "canonical_layer_range": [2, layer_count],
    }
    return canonical_vector, source


@dataclass(frozen=True)
class SteeringVectorArtifact:
    """An output-head or hidden-state steering vector."""

    vector: tuple[float, ...]
    layer: str = OUTPUT_LAYER
    position: str = RUNTIME_POSITION
    strength: float = 1.0
    method: str = "prompt-difference-v1"
    source: Mapping[str, Any] | None = None
    layer_start: int | None = None
    layer_end: int | None = None

    def __post_init__(self) -> None:
        vector = _vector(self.vector)
        object.__setattr__(self, "vector", vector)
        if self.layer == OUTPUT_LAYER:
            if self.position != RUNTIME_POSITION:
                raise EditorError(
                    f"output-head steering position must be {RUNTIME_POSITION!r}"
                )
            if self.layer_start is not None or self.layer_end is not None:
                raise EditorError(
                    "output-head steering vectors cannot specify a layer range"
                )
        elif self.layer == CONTROL_VECTOR_LAYER:
            if self.position != CONTROL_VECTOR_POSITION:
                raise EditorError(
                    f"hidden-state vector position must be {CONTROL_VECTOR_POSITION!r}"
                )
            if (
                type(self.layer_start) is not int
                or self.layer_start < 1
                or type(self.layer_end) is not int
                or self.layer_end < self.layer_start
            ):
                raise EditorError("hidden-state vector layer range is invalid")
        else:
            raise EditorError(
                f"steering vector target must be {OUTPUT_LAYER!r} or {CONTROL_VECTOR_LAYER!r}"
            )
        if not isinstance(self.method, str) or not self.method:
            raise EditorError("steering vector artifact method must be nonempty")
        if (
            type(self.strength) not in (int, float)
            or not math.isfinite(float(self.strength))
            or float(self.strength) < 0.0
        ):
            raise EditorError("steering vector strength must be finite and nonnegative")
        object.__setattr__(self, "strength", float(self.strength))
        if self.source is not None:
            if not isinstance(self.source, Mapping):
                raise EditorError("steering vector artifact source must be an object")
            object.__setattr__(self, "source", dict(self.source))

    @property
    def dimension(self) -> int:
        return len(self.vector)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.vector, dtype=np.float64)))

    @property
    def digest(self) -> str:
        return steering_vector_digest_for(
            self.vector,
            layer=self.layer,
            position=self.position,
            strength=self.strength,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )

    @property
    def kind(self) -> str:
        return (
            OUTPUT_HEAD_KIND
            if self.layer == OUTPUT_LAYER
            else HIDDEN_STATE_KIND
        )

    @property
    def target_description(self) -> str:
        return (
            "output head"
            if self.kind == OUTPUT_HEAD_KIND
            else "hidden-state layers"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SteeringVectorArtifact":
        if not isinstance(value, Mapping):
            raise EditorError("steering vector artifact must be an object")
        if value.get("format") != FORMAT:
            raise EditorError(f"steering vector artifact must use format {FORMAT}")
        schema_version = value.get("schema_version")
        if type(schema_version) is not int or schema_version != 3:
            raise EditorError("steering vector artifact schema version must be 3")
        kind = value.get("kind")
        if kind not in {OUTPUT_HEAD_KIND, HIDDEN_STATE_KIND}:
            raise EditorError(
                "steering vector artifact kind must be "
                f"{OUTPUT_HEAD_KIND!r} or {HIDDEN_STATE_KIND!r}"
            )
        if kind == OUTPUT_HEAD_KIND:
            layer = OUTPUT_LAYER
            position = RUNTIME_POSITION
            layer_start = layer_end = None
        else:
            layer = CONTROL_VECTOR_LAYER
            position = CONTROL_VECTOR_POSITION
            layer_start = value.get("layer_start")
            layer_end = value.get("layer_end")
        return cls(
            vector=value.get("vector"),
            layer=layer,
            position=position,
            strength=value.get("strength", 1.0),
            method=value.get("method", "prompt-difference-v1"),
            source=value.get("source"),
            layer_start=layer_start,
            layer_end=layer_end,
        )

    @classmethod
    def from_path(cls, path: Path) -> "SteeringVectorArtifact":
        if path.suffix.lower() == ".gguf":
            return cls.from_cvector_path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EditorError(f"could not read steering vector artifact: {exc}") from exc
        artifact = cls.from_mapping(value)
        declared_digest = value.get("digest")
        if declared_digest is not None and declared_digest != artifact.digest:
            raise EditorError("steering vector artifact digest does not match its contents")
        return artifact

    @classmethod
    def from_cvector_path(
        cls, path: Path, *, strength: float = 1.0
    ) -> "SteeringVectorArtifact":
        vector, source = _load_cvector_gguf(path)
        layer_count = int(source["canonical_layer_count"])
        if layer_count < 2:
            raise EditorError(
                "cvector GGUF has no canonical decoder-block output layer that llama.cpp can steer"
            )
        return cls(
            vector=vector,
            layer=CONTROL_VECTOR_LAYER,
            position=CONTROL_VECTOR_POSITION,
            strength=strength,
            method="llama-cvector-generator-v1",
            source=source,
            # Native direction.N is the native slot N, which is the output of
            # canonical block N+1.  The importer has already padded and
            # translated those chunks into the canonical layer layout.
            layer_start=2,
            layer_end=layer_count,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "format": FORMAT,
            "schema_version": 3,
            "kind": self.kind,
            "strength": self.strength,
            "method": self.method,
            "vector": list(self.vector),
            "digest": self.digest,
        }
        if self.kind == HIDDEN_STATE_KIND:
            result.update(
                {
                    "layer_start": self.layer_start,
                    "layer_end": self.layer_end,
                }
            )
        if self.source is not None:
            result["source"] = dict(self.source)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def write(self, path: Path) -> None:
        try:
            path.write_text(self.to_json() + "\n", encoding="utf-8")
        except OSError as exc:
            raise EditorError(f"could not write steering vector artifact: {exc}") from exc

    def validate_against_backend(self, backend: Any) -> np.ndarray:
        if self.layer == CONTROL_VECTOR_LAYER:
            setter = getattr(backend, "set_activation_control_vector", None)
            if not callable(setter):
                raise EditorError(
                    "the loaded backend does not expose llama.cpp control-vector runtime support"
                )
            # The backend is the authority for whether a vector can be
            # installed. Externally produced vectors may be intentional
            # experiments; the runtime setter reports application failures.
            return np.zeros(int(backend.vocabulary_size()), dtype=np.float64)
        width_method = getattr(backend, "activation_width", None)
        if not callable(width_method):
            raise EditorError(
                "the loaded backend does not expose hidden-state width metadata"
            )
        adjustment = getattr(backend, "activation_logit_adjustments", None)
        if not callable(adjustment):
            raise EditorError(
                "the loaded backend does not expose output-head steering runtime support"
            )
        if int(width_method()) != self.dimension:
            raise EditorError(
                f"output-head steering vector dimension {self.dimension} does not match loaded model width {width_method()}"
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
            raise EditorError(f"could not validate output-head steering vector: {exc}") from exc
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != int(backend.vocabulary_size()):
            raise EditorError("output-head steering adjustment does not match vocabulary")
        if not np.all(np.isfinite(values)):
            raise EditorError("output-head steering adjustment is not finite")
        return values

    def apply_to_sampling(
        self, sampling: SamplerConfig, *, strength: float | None = None
    ) -> SamplerConfig:
        return replace_steering_sampling(
            sampling,
            vector=self.vector,
            strength=self.strength if strength is None else strength,
            layer=self.layer,
            position=self.position,
            digest=self.digest,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )


def replace_steering_sampling(
    sampling: SamplerConfig,
    *,
    vector: tuple[float, ...] | list[float],
    strength: float,
    layer: str,
    position: str,
    digest: str,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> SamplerConfig:
    """Attach a steering artifact's effective state to a sampler config."""
    from dataclasses import replace

    return replace(
        sampling,
        activation_vector=tuple(float(value) for value in vector),
        activation_vector_strength=float(strength),
        activation_vector_layer=layer,
        activation_vector_position=position,
        activation_vector_digest=digest,
        activation_vector_layer_start=layer_start,
        activation_vector_layer_end=layer_end,
    )
