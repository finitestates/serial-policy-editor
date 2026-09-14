"""Fixed low-dimensional features derived from model token embeddings."""

from __future__ import annotations

import math

import numpy as np


DEFAULT_LATENT_DIMENSION = 64
DEFAULT_PROJECTION_SEED = 9137
DEFAULT_PROJECTION_CHUNK_SIZE = 8192


def project_token_embeddings(
    embeddings: np.ndarray,
    *,
    feature_dimension: int = DEFAULT_LATENT_DIMENSION,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
    projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
) -> np.ndarray:
    """Project a vocabulary-by-embedding matrix to fixed unit features.

    The projection is deterministic for a given embedding width, output
    dimension, and seed.  Projection is performed in row chunks to bound
    temporary workspace.  The returned rows are fixed model features; no
    learned state is stored here.
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
    if not np.all(np.isfinite(values)):
        raise ValueError("token embeddings must be finite")

    rng = np.random.default_rng(projection_seed % (1 << 64))
    projection = rng.standard_normal(
        (values.shape[1], feature_dimension), dtype=np.float32
    )
    projection /= np.float32(math.sqrt(values.shape[1]))
    features = np.zeros(
        (values.shape[0], feature_dimension),
        dtype=np.float32,
    )
    for start in range(0, values.shape[0], projection_chunk_size):
        stop = min(start + projection_chunk_size, values.shape[0])
        projected = values[start:stop] @ projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        np.divide(
            projected,
            norms,
            out=features[start:stop],
            where=norms > 1.0e-12,
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("projected token features are non-finite")
    features.setflags(write=False)
    return features
