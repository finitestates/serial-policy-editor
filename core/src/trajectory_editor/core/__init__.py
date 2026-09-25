"""Small dependency-light contracts for the episode runtime.

The core package contains concepts that describe and execute an episode. It
must not depend on terminal UI, persistence, backend implementations, or
research-only features.
"""

from .actions import (
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    PolicyAction,
    SelectRawRank,
    Write,
    action_from_dict,
)
from .backend import (
    BackendStateSnapshot,
    CacheMode,
    IncrementalTextBackend,
    IncrementalTextStream,
    InferenceBackend,
    SnapshotableInferenceBackend,
    require_inference_backend,
    validate_cache_mode,
)
from .candidates import Candidate
from .errors import EditorError
from .results import ActionOutcome, Divergence, ReplayExpectation, TokenEvidence
from .sampling import (
    CandidateFilterResult,
    SparseDistribution,
    StandardCandidateFilter,
    apply_candidate_filter,
    draw_token,
    position_uniform,
    position_uniform_token,
    raw_rank,
    top_raw_ids,
)
from .sampler_config import SAMPLING_POLICY_SCHEME, SamplerConfig
from .observation import ObservationStatistics
from .trajectory import TrajectoryState
from .ui import ActionKind, ChoiceSet, ContextText, EditAction, InsertMode

__all__ = [
    "Accept",
    "ActionOutcome",
    "CandidateFilterResult",
    "Candidate",
    "BackendStateSnapshot",
    "CacheMode",
    "ActionKind",
    "ChoiceSet",
    "ContextText",
    "EditAction",
    "InsertMode",
    "EditorError",
    "Divergence",
    "EndGeneration",
    "Hold",
    "IncrementalTextBackend",
    "IncrementalTextStream",
    "InferenceBackend",
    "SnapshotableInferenceBackend",
    "ObservationStatistics",
    "Phrase",
    "PolicyAction",
    "ReplayExpectation",
    "SparseDistribution",
    "SAMPLING_POLICY_SCHEME",
    "SamplerConfig",
    "StandardCandidateFilter",
    "SelectRawRank",
    "TrajectoryState",
    "TokenEvidence",
    "Write",
    "action_from_dict",
    "apply_candidate_filter",
    "draw_token",
    "position_uniform",
    "position_uniform_token",
    "raw_rank",
    "require_inference_backend",
    "top_raw_ids",
    "validate_cache_mode",
]
