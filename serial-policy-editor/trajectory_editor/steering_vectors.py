"""Public steering-vector artifact API.

Use this module for portable vector management.  Output-head steering and
hidden-state control are represented by the explicit ``kind`` on
``SteeringVectorArtifact``; callers should not infer the runtime surface from
an artifact filename or a generic activation label.
"""

from .activation_vectors import (
    FORMAT,
    HIDDEN_STATE_KIND,
    OUTPUT_HEAD_KIND,
    SteeringVectorArtifact,
    assert_compatible,
    blend_artifacts,
    model_identity,
    model_identity_from_json,
    model_identity_json,
    replace_steering_sampling,
    steering_vector_digest_for,
)

__all__ = [
    "FORMAT",
    "HIDDEN_STATE_KIND",
    "OUTPUT_HEAD_KIND",
    "SteeringVectorArtifact",
    "assert_compatible",
    "blend_artifacts",
    "model_identity",
    "model_identity_from_json",
    "model_identity_json",
    "replace_steering_sampling",
    "steering_vector_digest_for",
]
