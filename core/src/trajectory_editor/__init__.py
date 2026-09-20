"""Core-only public API for Serial Policy Editor."""

from .core.actions import (
    Accept,
    EndGeneration,
    Finish,
    Hold,
    Phrase,
    SelectRawRank,
    Write,
)
from .core.backend import InferenceBackend
from .core.candidates import Candidate
from .core.errors import EditorError
from .core.observation import ControllerTrace, ControllerTraceStage
from .core.results import ActionOutcome, Divergence, ReplayExpectation, TokenEvidence
from .core.sampler_config import SamplerConfig
from .episode_backend import EpisodeBackend
from .episode_engine import EpisodeEngine, Observation
from .episode_runner import TapeStep
from .episode_session import (
    BranchIdentity,
    BranchState,
    ControlPoint,
    LiveBranch,
    LiveEpisode,
    LiveSession,
    Session,
)
from .version import VERSION

__all__ = [
    "Accept",
    "ActionOutcome",
    "Candidate",
    "ControllerTrace",
    "ControllerTraceStage",
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
    "InferenceBackend",
    "BranchIdentity",
    "BranchState",
    "ControlPoint",
    "LiveBranch",
    "Observation",
    "LiveEpisode",
    "LiveSession",
    "LiveSessionRunner",
    "Phrase",
    "ReplayExpectation",
    "RunResult",
    "SamplerConfig",
    "SelectRawRank",
    "Session",
    "TapeStep",
    "TokenEvidence",
    "Write",
    "project_episode",
    "VERSION",
]

__version__ = VERSION


def __getattr__(name: str):
    """Keep persistence adapters out of the lightweight live-session import."""
    if name in {"EpisodeRunner", "LiveSessionRunner", "RunResult"}:
        from .episode_runner import EpisodeRunner, LiveSessionRunner, RunResult
        return {
            "EpisodeRunner": EpisodeRunner,
            "LiveSessionRunner": LiveSessionRunner,
            "RunResult": RunResult,
        }[name]
    if name == "EpisodeStore":
        from .episode_store import EpisodeStore
        return EpisodeStore
    if name in {"EpisodeProjection", "project_episode"}:
        from .projector import EpisodeProjection, project_episode
        return {"EpisodeProjection": EpisodeProjection, "project_episode": project_episode}[name]
    raise AttributeError(name)
