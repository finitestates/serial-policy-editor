"""Core-only public API for Serial Policy Editor."""

from .core.actions import (
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    Reroll,
    SelectRawRank,
    SetSampler,
    Write,
)
from .core.backend import InferenceBackend
from .core.candidates import Candidate
from .core.errors import EditorError
from .core.results import ActionOutcome, Divergence, ReplayExpectation, TokenEvidence
from .core.sampler_config import SamplerConfig
from .core.episode_observation import EpisodeObservation
from .episode_engine import EpisodeEngine
from .run_loop import RunResult, TapeStep
from .episode_session import (
    BranchIdentity,
    BranchState,
    LiveBranch,
    LiveRosterEntry,
    LiveSession,
    LiveSessionRoster,
)
from .fresh_episode import fresh_root_from
from .version import VERSION

__all__ = [
    "Accept",
    "ActionOutcome",
    "Candidate",
    "Divergence",
    "EditorError",
    "EndGeneration",
    "EpisodeEngine",
    "EpisodeProjection",
    "EpisodeStore",
    "Hold",
    "InferenceBackend",
    "BranchIdentity",
    "BranchState",
    "LiveBranch",
    "EpisodeObservation",
    "LiveRosterEntry",
    "LiveSession",
    "LiveSessionRoster",
    "Phrase",
    "ReplayExpectation",
    "Reroll",
    "RunResult",
    "SamplerConfig",
    "SelectRawRank",
    "SetSampler",
    "TapeStep",
    "TokenEvidence",
    "Write",
    "project_episode",
    "fresh_root_from",
    "VERSION",
]

__version__ = VERSION


def __getattr__(name: str):
    """Keep SQLite modules out of the lightweight live-session import."""
    if name == "EpisodeStore":
        from .episode_store import EpisodeStore
        return EpisodeStore
    if name in {"EpisodeProjection", "project_episode"}:
        from .projector import EpisodeProjection, project_episode
        return {"EpisodeProjection": EpisodeProjection, "project_episode": project_episode}[name]
    raise AttributeError(name)
