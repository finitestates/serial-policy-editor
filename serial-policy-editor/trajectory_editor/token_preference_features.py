"""Fixed low-dimensional features derived from model token embeddings."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


DEFAULT_TOKEN_PREFERENCE_DIMENSION = 64
DEFAULT_PROJECTION_SEED = 9137
DEFAULT_PROJECTION_CHUNK_SIZE = 8192
DEFAULT_WHITENING_RIDGE = 1.0e-6
TOKEN_PREFERENCE_FEATURE_SCHEMES = (
    "random-projection-unit-v1",
    "whitened-projection-v2",
)


@dataclass(frozen=True)
class TokenPreferenceCoordinateIdentity:
    """The immutable coordinate system in which a token preference memory is expressed.

    A preference vector is not portable across feature bases.  The model and
    embedding fields are optional because lightweight providers often do not
    expose a model fingerprint; the basis-defining projection fields remain
    fully checked in that case.
    """

    model_fingerprint: str | None
    embedding_width: int | None
    dimension: int
    projection_seed: int
    feature_scheme: str
    whitening_scheme: str
    whitening_ridge: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TokenPreferenceCoordinateIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("token preference coordinate identity must be an object")
        return cls(
            model_fingerprint=value.get("model_fingerprint"),
            embedding_width=value.get("embedding_width"),
            dimension=int(value["dimension"]),
            projection_seed=int(value["projection_seed"]),
            feature_scheme=str(value["feature_scheme"]),
            whitening_scheme=str(value.get("whitening_scheme", "none-v1")),
            whitening_ridge=float(value.get("whitening_ridge", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "embedding_width": self.embedding_width,
            "dimension": self.dimension,
            "projection_seed": self.projection_seed,
            "feature_scheme": self.feature_scheme,
            "whitening_scheme": self.whitening_scheme,
            "whitening_ridge": self.whitening_ridge,
        }

    @property
    def basis_key(self) -> tuple[Any, ...]:
        """Fields that change token-feature coordinates."""
        return (
            self.model_fingerprint,
            self.embedding_width,
            self.dimension,
            self.projection_seed,
            self.feature_scheme,
            self.whitening_scheme,
            self.whitening_ridge,
        )


def coordinate_identity(
    *,
    dimension: int,
    projection_seed: int,
    feature_scheme: str,
    whitening_ridge: float,
    model_fingerprint: str | None = None,
    embedding_width: int | None = None,
) -> TokenPreferenceCoordinateIdentity:
    """Build the canonical identity for a feature request."""
    return TokenPreferenceCoordinateIdentity(
        model_fingerprint=model_fingerprint,
        embedding_width=embedding_width,
        dimension=int(dimension),
        projection_seed=int(projection_seed),
        feature_scheme=str(feature_scheme),
        whitening_scheme=(
            "eigh-ridge-global-scale-v2"
            if feature_scheme == "whitened-projection-v2" else "none-v1"
        ),
        whitening_ridge=(float(whitening_ridge)
                         if feature_scheme == "whitened-projection-v2" else 0.0),
    )


def sampling_coordinate_identity(
    sampling,
    *,
    dimension: int | None = None,
    model_fingerprint: str | None = None,
    embedding_width: int | None = None,
) -> TokenPreferenceCoordinateIdentity:
    """Return the identity requested by a sampling configuration."""
    if dimension is None:
        vectors = (
            getattr(sampling, "token_preference_vector", ())
            or getattr(sampling, "token_preference_fast_vector", ())
        )
        stored = getattr(sampling, "token_preference_coordinate_identity", None)
        dimension = len(vectors) or (stored.dimension if stored is not None else DEFAULT_TOKEN_PREFERENCE_DIMENSION)
    return coordinate_identity(
        dimension=dimension,
        projection_seed=sampling.token_preference_projection_seed,
        feature_scheme=sampling.token_preference_feature_scheme,
        whitening_ridge=sampling.token_preference_whitening_ridge,
        model_fingerprint=model_fingerprint,
        embedding_width=embedding_width,
    )


def coordinate_identity_matches(
    sampling,
    *,
    dimension: int | None = None,
    model_fingerprint: str | None = None,
    embedding_width: int | None = None,
) -> bool:
    """Check whether persisted coordinates match the currently requested basis."""
    stored = getattr(sampling, "token_preference_coordinate_identity", None)
    if stored is None:
        return True
    requested = sampling_coordinate_identity(
        sampling,
        dimension=dimension,
        model_fingerprint=model_fingerprint,
        embedding_width=embedding_width,
    )
    # A lightweight provider may not know the model fingerprint or embedding
    # width when it first creates a vector.  Unknown metadata is compatible
    # with later-discovered metadata; a conflict between two known values is
    # always incompatible.
    return all(
        left == right or left is None or right is None
        for left, right in zip(stored.basis_key, requested.basis_key)
    )


def embedding_fingerprint(embeddings: np.ndarray) -> str:
    """Return a stable fingerprint for the exact embedding matrix supplied."""
    values = np.asarray(embeddings, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def project_token_embeddings(
    embeddings: np.ndarray,
    *,
    feature_dimension: int = DEFAULT_TOKEN_PREFERENCE_DIMENSION,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
    projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
    feature_scheme: str = "random-projection-unit-v1",
    whitening_ridge: float = DEFAULT_WHITENING_RIDGE,
) -> np.ndarray:
    """Project a vocabulary-by-embedding matrix to fixed token preference features.

    The v1 branch intentionally retains the original per-token unit
    normalization.  v2 centers and whitens the projected vocabulary with a
    small eigensystem, then applies one global scale so the mean squared row
    norm is one.  Projection is performed in row chunks to bound temporary
    workspace.  The returned rows are fixed model features; no learned state
    is stored here.
    """
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("token embeddings must be a nonempty 2-D matrix")
    if type(feature_dimension) is not int or feature_dimension < 1:
        raise ValueError("feature dimension must be a positive integer")
    if type(projection_seed) is not int:
        raise ValueError("projection seed must be an integer")
    if type(projection_chunk_size) is not int or projection_chunk_size < 1:
        raise ValueError("projection chunk size must be a positive integer")
    if feature_scheme not in TOKEN_PREFERENCE_FEATURE_SCHEMES:
        raise ValueError(f"unknown preference feature scheme: {feature_scheme}")
    if (
        type(whitening_ridge) not in (int, float)
        or not math.isfinite(float(whitening_ridge))
        or float(whitening_ridge) < 0.0
    ):
        raise ValueError("whitening ridge must be finite and nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("token embeddings must be finite")

    rng = np.random.default_rng(projection_seed % (1 << 64))
    projection = rng.standard_normal(
        (values.shape[1], feature_dimension), dtype=np.float32
    )
    projection /= np.float32(math.sqrt(values.shape[1]))
    if feature_scheme == "random-projection-unit-v1":
        # Keep the original chunk-local arithmetic for persisted v1
        # trajectories. In particular, do not change the reduction shape or
        # normalization order while adding the v2 coordinate system.
        features = np.zeros(
            (values.shape[0], feature_dimension), dtype=np.float32
        )
        for start in range(0, values.shape[0], projection_chunk_size):
            stop = min(start + projection_chunk_size, values.shape[0])
            projected = values[start:stop] @ projection
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            np.divide(projected, norms, out=features[start:stop], where=norms > 1.0e-12)
    else:
        projected_rows = np.zeros(
            (values.shape[0], feature_dimension),
            dtype=np.float32,
        )
        for start in range(0, values.shape[0], projection_chunk_size):
            stop = min(start + projection_chunk_size, values.shape[0])
            projected_rows[start:stop] = values[start:stop] @ projection
        # Keep the eigendecomposition in float64.  This makes the whitening
        # transform stable for small or rank-deficient synthetic vocabularies
        # while the published feature matrix remains compact float32.
        mean = np.mean(projected_rows, axis=0, dtype=np.float64)
        centered = projected_rows.astype(np.float64) - mean
        covariance = (centered.T @ centered) / float(values.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, float(whitening_ridge))
        whitening = (
            eigenvectors * (1.0 / np.sqrt(eigenvalues))
        ) @ eigenvectors.T
        whitened = centered @ whitening
        mean_squared_norm = float(np.mean(np.sum(whitened * whitened, axis=1)))
        if mean_squared_norm > 1.0e-24:
            whitened /= math.sqrt(mean_squared_norm)
        features = np.asarray(whitened, dtype=np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("projected token features are non-finite")
    features.setflags(write=False)
    return features
