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
    BatchedInferenceBackend,
    BatchedInferenceSession,
    CacheMode,
    IncrementalTextBackend,
    IncrementalTextStream,
    InferenceBackend,
    PositionAwareInferenceBackend,
    SnapshotableInferenceBackend,
    require_inference_backend,
    validate_cache_mode,
)
from .backend_batch import InferenceBatch
from .backend_position import (
    BackendPosition,
    PositionComparison,
    compare_backend_position,
    position_backend,
    position_report,
)
from .candidates import Candidate
from .errors import EditorError
from .results import ActionOutcome, Divergence, ReplayExpectation, TokenEvidence
from .sampling import (
    CandidateFilterResult,
    GUMBEL_NOISE_ADDRESSES,
    SparseDistribution,
    StandardCandidateFilter,
    apply_candidate_filter,
    draw_token,
    find_seed_for_token,
    gaussian_ranking_scores,
    gaussian_winner,
    gumbel_ranking_scores,
    gumbel_ranked_ids,
    gumbel_winner,
    position_uniform,
    position_uniform_model_rank,
    position_uniform_token,
    raw_rank,
    top_raw_ids,
)
from .sampler_config import SAMPLING_POLICY_SCHEME, SamplerConfig
from .policy_calculations import PolicyCalculations
from .episode_observation import EpisodeObservation
from .trajectory import TrajectoryState
from .ui import ActionKind, ChoiceSet, ContextText, EditAction, InsertMode

__all__ = [
    "Accept",
    "ActionOutcome",
    "CandidateFilterResult",
    "Candidate",
    "GUMBEL_NOISE_ADDRESSES",
    "EpisodeObservation",
    "BackendStateSnapshot",
    "BatchedInferenceBackend",
    "BatchedInferenceSession",
    "BackendPosition",
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
    "InferenceBatch",
    "PositionAwareInferenceBackend",
    "PositionComparison",
    "SnapshotableInferenceBackend",
    "PolicyCalculations",
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
    "find_seed_for_token",
    "gaussian_ranking_scores",
    "gaussian_winner",
    "gumbel_ranking_scores",
    "gumbel_ranked_ids",
    "gumbel_winner",
    "position_backend",
    "compare_backend_position",
    "position_report",
    "position_uniform",
    "position_uniform_model_rank",
    "position_uniform_token",
    "raw_rank",
    "require_inference_backend",
    "top_raw_ids",
    "validate_cache_mode",
]
