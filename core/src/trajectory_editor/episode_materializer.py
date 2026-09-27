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

    if state.stream_fingerprint is None:
        raise EditorError("the selected branch has no stream fingerprint to save")
    identifier = store.create_episode(
        episode_id=episode_id,
        parent_episode_id=parent_episode_id,
        fork_boundary=state.identity.fork_boundary if parent_episode_id is not None else None,
        initial_text=session.prompt,
        initial_token_ids=state.initial_token_ids,
        sampling=state.initial_sampling,
        stream_fingerprint=state.stream_fingerprint,
        backend=provenance,
        metadata={
            "mode": mode,
            "live_session_id": session.session_id,
            "live_branch_id": state.identity.branch_id,
            "live_parent_branch_id": state.identity.parent_id,
            "live_fork_boundary": state.identity.fork_boundary,
            "live_materialization": "full-root-branch",
            "boundary_system": "root-relative",
        },
    )
    for ordinal, outcome in enumerate(state.history_outcomes):
        store.record_action(identifier, ordinal, outcome)
    visible_text = session.engine.backend.render(list(state.visible_token_ids))
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
        parent_id = state.identity.parent_id
        if parent_id is not None:
            try:
                store.get_episode(parent_id)
            except EditorError:
                parent_id = None
        return materialize_live_branch(
            store,
            session,
            state,
            provenance,
            episode_id=episode_id,
            parent_episode_id=parent_id,
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
                if parent is not None and parent in states and parent not in identifiers:
                    continue
                durable_parent = identifiers.get(parent) if parent in states else parent
                if durable_parent is not None and parent not in states:
                    try:
                        store.get_episode(durable_parent)
                    except EditorError:
                        durable_parent = None
                identifier = materialize_live_branch(
                    store,
                    session,
                    state,
                    provenance,
                    episode_id=root_episode_id if parent is None else None,
                    parent_episode_id=durable_parent,
                    mode="ephemeral-family-save",
                )
                identifiers[branch_id] = identifier
                del pending[branch_id]
                progressed = True
            if not progressed:
                raise EditorError("live branch family has an unresolved parent")
    return identifiers


__all__ = ["materialize_live_branch", "save_live_branch", "save_live_family"]
