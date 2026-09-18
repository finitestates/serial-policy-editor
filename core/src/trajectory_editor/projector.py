"""Canonical episode export name.

The implementation remains in ``episode_projector`` during migration so old
imports continue to work.
"""

from typing import Any

from . import episode_projector as _legacy


EpisodeProjection = _legacy.EpisodeProjection


def project_episode(*args: Any, **kwargs: Any) -> Any:
    return _legacy.project_episode(*args, **kwargs)


def project_fork_map(*args: Any, **kwargs: Any) -> Any:
    return _legacy.project_fork_map(*args, **kwargs)


def project_lineage(*args: Any, **kwargs: Any) -> Any:
    return _legacy.project_lineage(*args, **kwargs)


def project_procedure(*args: Any, **kwargs: Any) -> Any:
    return _legacy.project_procedure(*args, **kwargs)

__all__ = [
    "EpisodeProjection",
    "project_episode",
    "project_fork_map",
    "project_lineage",
    "project_procedure",
]
