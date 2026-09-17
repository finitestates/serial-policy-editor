"""Public API for the policy-episode runtime."""

from .domain import Candidate, EditorError, SamplingConfig
from .episode_actions import (
    Accept,
    EndGeneration,
    Finish,
    Hold,
    SelectRawRank,
    Write,
)
from .episode_backend import EpisodeBackend
from .episode_engine import (
    ActionOutcome,
    Divergence,
    EpisodeEngine,
    Observation,
    ReplayExpectation,
    TokenEvidence,
)
from .episode_policy import EpisodeRunner, RunResult, TapeStep, WriteLearningResult
from .episode_projector import EpisodeProjection, project_episode
from .episode_store import EpisodeStore
from .token_preference import (
    TokenPreferenceConfig,
    TokenPreferenceLearner,
    TokenPreferenceResult,
)
from .token_preference_features import TokenPreferenceCoordinateIdentity
from .vector_artifacts import TokenPreferenceVectorArtifact
from .activation_vectors import ActivationVectorArtifact
from .controller_pipeline import ControllerPipeline
from .sampling import ControllerTrace, ControllerTraceStage
from .trajectory_compare import compare_episodes, render_compare_report
from .vector_impact import impact_vector, render_impact_report
from .online_learning import LearningResult, OnlineLearner, OnlineLearningConfig
from .version import VERSION

__all__ = [
    "Accept",
    "ActionOutcome",
    "Candidate",
    "Divergence",
    "EditorError",
    "EndGeneration",
    "EpisodeBackend",
    "EpisodeEngine",
    "EpisodeProjection",
    "EpisodeRunner",
    "EpisodeStore",
    "Finish",
    "Hold",
    "Observation",
    "ReplayExpectation",
    "RunResult",
    "SamplingConfig",
    "LearningResult",
    "TokenPreferenceConfig",
    "TokenPreferenceLearner",
    "TokenPreferenceResult",
    "TokenPreferenceCoordinateIdentity",
    "TokenPreferenceVectorArtifact",
    "ActivationVectorArtifact",
    "ControllerPipeline",
    "ControllerTrace",
    "ControllerTraceStage",
    "compare_episodes",
    "impact_vector",
    "OnlineLearner",
    "OnlineLearningConfig",
    "SelectRawRank",
    "TapeStep",
    "TokenEvidence",
    "Write",
    "WriteLearningResult",
    "project_episode",
    "render_compare_report",
    "render_impact_report",
    "VERSION",
]

__version__ = VERSION
