"""Portable standalone artifacts for token-preference vectors."""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .domain import EditorError, SamplingConfig
from .token_preference_features import (
    DEFAULT_PROJECTION_CHUNK_SIZE,
    TokenPreferenceCoordinateIdentity,
)


FORMAT = "spe-token-preference-vector-v1"
KIND = "token-preference"
MODEL_IDENTITY_FIELDS = (
    "backend",
    "filename",
    "file_size_bytes",
    "vocabulary_size",
    "tokenizer_fingerprint",
)


def _vector(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise EditorError(f"{name} must be an array")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EditorError(f"{name} must contain numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise EditorError(f"{name} must contain finite numbers")
    return result


def model_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only stable model identity fields in a portable artifact."""
    return {
        name: provenance[name]
        for name in MODEL_IDENTITY_FIELDS
        if name in provenance and provenance[name] is not None
    }


def _check_model_compatibility(
    left: Mapping[str, Any], right: Mapping[str, Any], *, label: str
) -> None:
    for name in MODEL_IDENTITY_FIELDS:
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is not None and right_value is not None and left_value != right_value:
            raise EditorError(f"incompatible token preference models: {label} differs in {name}")


def _check_coordinate_compatibility(
    left: TokenPreferenceCoordinateIdentity | None,
    right: TokenPreferenceCoordinateIdentity | None,
    *,
    label: str,
) -> None:
    if left is None or right is None:
        return
    for name, left_value, right_value in zip(
        (
            "model_fingerprint",
            "embedding_width",
            "dimension",
            "projection_seed",
            "feature_scheme",
            "whitening_scheme",
            "whitening_ridge",
        ),
        left.basis_key,
        right.basis_key,
    ):
        if left_value is not None and right_value is not None and left_value != right_value:
            raise EditorError(
                f"incompatible token preference coordinates: {label} differs in {name}"
            )


def assert_compatible(
    left: "TokenPreferenceVectorArtifact",
    right: "TokenPreferenceVectorArtifact",
) -> None:
    """Require artifacts to describe the same model and feature basis."""
    _check_model_compatibility(left.model, right.model, label="artifacts")
    _check_coordinate_compatibility(
        left.coordinate_identity,
        right.coordinate_identity,
        label="artifacts",
    )


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Allow small test/lightweight providers to expose a reduced signature."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in kwargs.items() if name in names}


def materialize_features(
    backend: Any,
    identity: TokenPreferenceCoordinateIdentity,
    *,
    projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
) -> np.ndarray:
    provider = getattr(backend, "token_preference_features", None)
    if not callable(provider):
        raise EditorError(
            "the loaded backend does not expose token preference features"
        )
    try:
        features = provider(
            **_supported_kwargs(
                provider,
                {
                    "feature_dimension": identity.dimension,
                    "projection_seed": identity.projection_seed,
                    "projection_chunk_size": projection_chunk_size,
                    "feature_scheme": identity.feature_scheme,
                    "whitening_ridge": identity.whitening_ridge,
                },
            )
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EditorError(f"could not materialize token preference features: {exc}") from exc
    values = np.asarray(features, dtype=np.float32)
    expected = (backend.vocabulary_size(), identity.dimension)
    if values.shape != expected:
        raise EditorError(
            "token preference features have shape "
            f"{values.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(values)):
        raise EditorError("token preference features contain non-finite values")
    return values


@dataclass(frozen=True)
class TokenPreferenceVectorArtifact:
    """A model-matched, standalone token-preference vector bundle."""

    model: Mapping[str, Any]
    coordinate_identity: TokenPreferenceCoordinateIdentity | Mapping[str, Any] | None
    token_preference_vector: tuple[float, ...] = ()
    token_preference_fast_vector: tuple[float, ...] = ()
    token_preference_strength: float = 1.0
    token_preference_fast_strength: float = 0.0
    source: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, Mapping):
            raise EditorError("token preference artifact model must be an object")
        object.__setattr__(self, "model", dict(self.model))
        identity = self.coordinate_identity
        if identity is not None and not isinstance(identity, TokenPreferenceCoordinateIdentity):
            try:
                identity = TokenPreferenceCoordinateIdentity.from_mapping(identity)
            except (KeyError, TypeError, ValueError) as exc:
                raise EditorError("token preference artifact coordinate identity is malformed") from exc
            object.__setattr__(self, "coordinate_identity", identity)
        slow = _vector(self.token_preference_vector, "token_preference_vector")
        fast = _vector(self.token_preference_fast_vector, "token_preference_fast_vector")
        object.__setattr__(self, "token_preference_vector", slow)
        object.__setattr__(self, "token_preference_fast_vector", fast)
        if slow and fast and len(slow) != len(fast):
            raise EditorError("token preference slow and fast vectors must have the same dimension")
        if (slow or fast) and identity is None:
            raise EditorError("nonempty token preference vectors require coordinate identity")
        if identity is not None and (slow or fast) and identity.dimension != len(slow or fast):
            raise EditorError("token preference vector dimension does not match coordinate identity")
        for name in (
            "token_preference_strength",
            "token_preference_fast_strength",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise EditorError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, float(value))
        if self.source is not None:
            if not isinstance(self.source, Mapping):
                raise EditorError("token preference artifact source must be an object")
            object.__setattr__(self, "source", dict(self.source))

    @property
    def dimension(self) -> int | None:
        if self.token_preference_vector or self.token_preference_fast_vector:
            return len(self.token_preference_vector or self.token_preference_fast_vector)
        return self.coordinate_identity.dimension if self.coordinate_identity else None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TokenPreferenceVectorArtifact":
        if not isinstance(value, Mapping):
            raise EditorError("token preference artifact must be an object")
        if value.get("format") != FORMAT:
            raise EditorError(f"token preference artifact must use format {FORMAT}")
        if value.get("kind") != KIND:
            raise EditorError(f"token preference artifact kind must be {KIND!r}")
        model = value.get("model")
        if not isinstance(model, Mapping):
            raise EditorError("token preference artifact requires model metadata")
        return cls(
            model=model,
            coordinate_identity=value.get("coordinate_identity"),
            token_preference_vector=value.get("token_preference_vector", ()),
            token_preference_fast_vector=value.get("token_preference_fast_vector", ()),
            token_preference_strength=value.get("token_preference_strength", 1.0),
            token_preference_fast_strength=value.get("token_preference_fast_strength", 0.0),
            source=value.get("source"),
        )

    @classmethod
    def from_path(cls, path: Path) -> "TokenPreferenceVectorArtifact":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EditorError(f"could not read token preference artifact: {exc}") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_sampling(
        cls,
        sampling: SamplingConfig,
        provenance: Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = None,
    ) -> "TokenPreferenceVectorArtifact":
        return cls(
            model=model_identity(provenance),
            coordinate_identity=sampling.token_preference_coordinate_identity,
            token_preference_vector=sampling.token_preference_vector,
            token_preference_fast_vector=sampling.token_preference_fast_vector,
            token_preference_strength=sampling.token_preference_strength,
            token_preference_fast_strength=sampling.token_preference_fast_strength,
            source=source,
        )

    @classmethod
    def from_preset_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = None,
    ) -> "TokenPreferenceVectorArtifact":
        from .bias_presets import FORMAT as BIAS_FORMAT

        if not isinstance(value, Mapping) or value.get("format") != BIAS_FORMAT:
            raise EditorError(f"bias preset must use format {BIAS_FORMAT}")
        model = value.get("model")
        if not isinstance(model, Mapping):
            raise EditorError("bias preset requires model metadata")
        sampling = SamplingConfig.from_mapping(value)
        return cls.from_sampling(sampling, model, source=source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": KIND,
            "model": dict(self.model),
            "coordinate_identity": (
                self.coordinate_identity.to_dict()
                if self.coordinate_identity is not None else None
            ),
            "token_preference_vector": list(self.token_preference_vector),
            "token_preference_fast_vector": list(self.token_preference_fast_vector),
            "token_preference_strength": self.token_preference_strength,
            "token_preference_fast_strength": self.token_preference_fast_strength,
            **({"source": dict(self.source)} if self.source is not None else {}),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    def write(self, path: Path) -> None:
        try:
            path.write_text(self.to_json() + "\n", encoding="utf-8")
        except OSError as exc:
            raise EditorError(f"could not write token preference artifact: {exc}") from exc

    def validate_against_backend(
        self,
        backend: Any,
        provenance: Mapping[str, Any],
        *,
        projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
    ) -> np.ndarray | None:
        _check_model_compatibility(self.model, model_identity(provenance), label="loaded model")
        if self.coordinate_identity is None:
            return None
        identity_method = getattr(backend, "token_preference_coordinate_identity", None)
        if callable(identity_method):
            try:
                actual = identity_method(
                    **_supported_kwargs(
                        identity_method,
                        {
                            "feature_dimension": self.coordinate_identity.dimension,
                            "projection_seed": self.coordinate_identity.projection_seed,
                            "feature_scheme": self.coordinate_identity.feature_scheme,
                            "whitening_ridge": self.coordinate_identity.whitening_ridge,
                        },
                    )
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise EditorError(f"could not validate token preference coordinates: {exc}") from exc
            if not isinstance(actual, TokenPreferenceCoordinateIdentity):
                try:
                    actual = TokenPreferenceCoordinateIdentity.from_mapping(actual)
                except (KeyError, TypeError, ValueError) as exc:
                    raise EditorError("backend returned malformed token preference coordinates") from exc
            _check_coordinate_compatibility(
                self.coordinate_identity,
                actual,
                label="loaded model",
            )
        return materialize_features(
            backend,
            self.coordinate_identity,
            projection_chunk_size=projection_chunk_size,
        )


def blend_artifacts(
    artifacts: list[TokenPreferenceVectorArtifact],
    weights: list[float],
    *,
    source: Mapping[str, Any] | None = None,
) -> TokenPreferenceVectorArtifact:
    if not artifacts:
        raise EditorError("blend requires at least one token preference artifact")
    if len(weights) != len(artifacts):
        raise EditorError("blend weights must match the number of artifacts")
    if any(not math.isfinite(float(weight)) for weight in weights):
        raise EditorError("blend weights must be finite numbers")
    first = artifacts[0]
    for artifact in artifacts[1:]:
        assert_compatible(first, artifact)
    basis = next((artifact for artifact in artifacts if artifact.dimension is not None), None)
    dimension = basis.dimension if basis is not None else None
    if dimension is None:
        raise EditorError("cannot blend empty token preference artifacts")
    slow_present = any(artifact.token_preference_vector for artifact in artifacts)
    fast_present = any(artifact.token_preference_fast_vector for artifact in artifacts)
    slow = np.zeros(dimension, dtype=np.float64)
    fast = np.zeros(dimension, dtype=np.float64)
    for weight, artifact in zip(weights, artifacts):
        if artifact.token_preference_vector:
            slow += float(weight) * artifact.token_preference_strength * np.asarray(
                artifact.token_preference_vector, dtype=np.float64
            )
        if artifact.token_preference_fast_vector:
            fast += float(weight) * artifact.token_preference_fast_strength * np.asarray(
                artifact.token_preference_fast_vector, dtype=np.float64
            )
    return TokenPreferenceVectorArtifact(
        model=basis.model,
        coordinate_identity=basis.coordinate_identity,
        token_preference_vector=tuple(float(value) for value in slow) if slow_present else (),
        token_preference_fast_vector=tuple(float(value) for value in fast) if fast_present else (),
        token_preference_strength=1.0,
        token_preference_fast_strength=1.0 if fast_present else 0.0,
        source=source,
    )
