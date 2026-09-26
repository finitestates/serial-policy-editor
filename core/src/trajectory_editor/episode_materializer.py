"""Explicitly materialize an in-memory live session into a workspace.

Live sessions do not need SQLite to run. This module is the opt-in bridge for
the moment a user asks to save one branch or an entire live branch family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.errors import EditorError
from .episode_session import BranchState, LiveSession
from .episode_store import EpisodeStore


def materialize_live_branch(
    store: EpisodeStore,
    session: LiveSession,
    state: BranchState,
    provenance: dict[str, Any],
    *,
    episode_id: str | None = None,
    parent_episode_id: str | None = None,
    mode: str = "ephemeral-save",
) -> str:
    """Write one canonical live branch as an independent durable episode."""

    initial = state.control_points[0]
    if initial.stream_fingerprint is None:
        raise EditorError("the selected branch has no stream fingerprint to save")
    identifier = store.create_episode(
        episode_id=episode_id,
        parent_episode_id=parent_episode_id,
        fork_boundary=state.identity.fork_boundary if parent_episode_id is not None else None,
        initial_text=session.prompt,
        initial_token_ids=state.initial_token_ids,
        sampling=initial.sampling,
        stream_fingerprint=initial.stream_fingerprint,
        max_tokens=initial.max_tokens,
        backend=provenance,
        metadata={
            "mode": mode,
            "live_session_id": session.session_id,
            "live_branch_id": state.identity.branch_id,
            "live_parent_branch_id": state.identity.parent_id,
            "live_fork_boundary": state.identity.fork_boundary,
            "live_materialization": "full-root-branch",
            "coordinate_system": "root-relative",
        },
        checkpoint_boundary=initial.checkpoint_boundary,
    )
    prior = initial
    for point in state.control_points[1:]:
        if point.stream_fingerprint is None:
            raise EditorError("the selected branch has no stream fingerprint to save")
        if (
            point.sampling,
            point.stream_fingerprint,
        ) != (
            prior.sampling,
            prior.stream_fingerprint,
        ):
            store.record_sampling_segment(
                identifier,
                start_boundary=point.boundary,
                sampling=point.sampling,
                stream_fingerprint=point.stream_fingerprint,
            )
        if (point.max_tokens, point.checkpoint_boundary) != (
            prior.max_tokens,
            prior.checkpoint_boundary,
        ):
            store.record_budget(
                identifier,
                point.boundary,
                point.max_tokens,
                point.checkpoint_boundary,
            )
        prior = point
    for ordinal, outcome in enumerate(state.history_outcomes):
        store.record_action(identifier, ordinal, outcome)
    visible_text = session.engine.backend.render(list(state.visible_token_ids))
    final = state.control_points[-1]
    if state.terminal_reason is not None:
        store.finish_episode(
            identifier,
            visible_text=visible_text,
            terminal_token_id=state.terminal_token_id,
            terminal_reason=state.terminal_reason,
        )
    else:
        store.update_episode(
            identifier,
            visible_text=visible_text,
            max_tokens=final.max_tokens,
        )
    return identifier


def save_live_branch(
    session: LiveSession,
    workspace: Path,
    provenance: dict[str, Any],
    *,
    episode_id: str | None = None,
) -> str:
    """Materialize only the currently selected branch after explicit save."""

    state = session.branch_state()
    with EpisodeStore(workspace) as store:
        return materialize_live_branch(
            store, session, state, provenance, episode_id=episode_id
        )


def save_live_family(
    session: LiveSession,
    workspace: Path,
    provenance: dict[str, Any],
    *,
    root_episode_id: str | None = None,
) -> dict[str, str]:
    """Materialize every retained live branch and map its durable lineage."""

    states = dict(session.branch_states)
    pending = dict(states)
    identifiers: dict[str, str] = {}
    with EpisodeStore(workspace) as store:
        while pending:
            progressed = False
            for branch_id, state in tuple(pending.items()):
                parent = state.identity.parent_id
                if parent is not None and parent not in identifiers:
                    continue
                identifier = materialize_live_branch(
                    store,
                    session,
                    state,
                    provenance,
                    episode_id=root_episode_id if parent is None else None,
                    parent_episode_id=identifiers.get(parent),
                    mode="ephemeral-family-save",
                )
                identifiers[branch_id] = identifier
                del pending[branch_id]
                progressed = True
            if not progressed:
                raise EditorError("live branch family has an unresolved parent")
    return identifiers


__all__ = ["materialize_live_branch", "save_live_branch", "save_live_family"]
