"""Restore saved episodes into the persistence-free live-session model."""

from __future__ import annotations

from typing import Any

from .core.actions import sampler_after_action
from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine
from .episode_history import materialize_stored_prefix, outcome_from_stored_action
from .episode_session import BranchIdentity, BranchState, LiveSession
from .episode_store import EpisodeStore
from .run_loop import TapeStep


def restore_live_session(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    *,
    boundary: int | None = None,
    branch_identity: BranchIdentity | None = None,
) -> LiveSession:
    """Load one saved prefix; the live session owns subsequent execution.

    Omitting ``boundary`` restores the complete saved command stream, including
    commands at its current token boundary. Supplying a boundary selects the
    state from before every command at that boundary.
    """
    episode = store.get_episode(episode_id)
    token_rows = store.tokens(episode_id)
    all_visible = tuple(
        int(row["token_id"]) for row in token_rows if bool(row["realized_visible"])
    )
    visible_boundary = len(all_visible)
    selected_boundary = visible_boundary if boundary is None else boundary
    if type(selected_boundary) is not int or not 0 <= selected_boundary <= visible_boundary:
        raise EditorError(f"restore boundary must be between 0 and {visible_boundary}")
    if tuple(engine.visible_token_ids) != all_visible[:selected_boundary]:
        raise EditorError("restored engine prefix does not match saved episode history")

    history = materialize_stored_prefix(
        store.actions(episode_id),
        token_rows,
        selected_boundary,
        include_boundary_events=boundary is None,
    )
    outcomes = tuple(outcome_from_stored_action(action) for action in history.actions)
    tape = tuple(TapeStep(outcome.action, outcome.expectation()) for outcome in outcomes)
    initial_sampling = SamplerConfig.from_record(episode["initial_sampling"])
    active_sampling = initial_sampling
    for outcome in outcomes:
        active_sampling = sampler_after_action(active_sampling, outcome.action)
    requested_sampling = engine.sampling
    engine.sampling = active_sampling
    engine.trajectory.set_stream_fingerprint(episode["initial_stream_fingerprint"])

    identity = branch_identity or BranchIdentity(
        episode_id,
        parent_id=episode.get("parent_episode_id"),
        fork_boundary=episode.get("fork_boundary"),
    )
    local_action_start = 0
    if identity.parent_id is not None and identity.fork_boundary is not None:
        local_action_start = sum(
            outcome.boundary_after < int(identity.fork_boundary)
            or (
                outcome.boundary_after == int(identity.fork_boundary)
                and outcome.boundary_before < int(identity.fork_boundary)
            )
            for outcome in outcomes
        )
    terminal_reason = (
        episode.get("terminal_reason")
        if boundary is None and branch_identity is None
        else None
    )
    state = BranchState(
        identity=identity,
        initial_token_ids=tuple(engine.initial_token_ids),
        initial_sampling=initial_sampling,
        stream_fingerprint=episode["initial_stream_fingerprint"],
        visible_token_ids=tuple(engine.visible_token_ids),
        tape=tape,
        outcomes=outcomes,
        local_action_start=local_action_start,
        terminal_token_id=(
            episode.get("terminal_token_id") if terminal_reason is not None else None
        ),
        terminal_reason=terminal_reason,
        status=("completed" if terminal_reason is not None else "open"),
    )
    session = LiveSession(
        engine,
        prompt=str(episode["initial_text"]),
        environment_stamp={
            "backend": dict(episode["backend"]),
            "source_episode_id": episode_id,
        },
        initial_state=state,
    )
    if terminal_reason is None and requested_sampling != active_sampling:
        session.set_sampler(requested_sampling)
    return session


def model_change_session(
    store: EpisodeStore,
    source_id: str,
    backend: Any,
    provenance: dict[str, Any],
    *,
    boundary: int,
    sampling: SamplerConfig,
    guidance_backend: Any | None = None,
) -> LiveSession:
    """Prepare a model-change continuation without creating an episode."""
    from uuid import uuid4

    from .core.actions import Write
    from .episode_history import visible_text_prefix
    from .episode_identity import backend_provenance_with_identity
    from .episode_lifecycle import _model_change_sampling

    episode = store.get_episode(source_id)
    token_rows = tuple(store.tokens(source_id))
    visible = tuple(
        int(row["token_id"]) for row in token_rows if bool(row["realized_visible"])
    )
    if type(boundary) is not int or not 0 <= boundary <= len(visible):
        raise EditorError(f"model-change boundary must be between 0 and {len(visible)}")
    sampler_state = store.sampling_state_at_boundary(source_id, boundary)
    destination = backend_provenance_with_identity(backend, provenance)
    root_text = str(episode["initial_text"])
    root_token_ids = backend.tokenize(root_text, add_bos=True, special=True)
    source_tokenizer = episode["backend"].get("tokenizer_id")
    same_tokenizer = (
        isinstance(source_tokenizer, str)
        and source_tokenizer == destination.get("tokenizer_id")
        and list(episode["initial_token_ids"]) == root_token_ids
    )
    safe_sampling = _model_change_sampling(sampling, same_tokenizer=same_tokenizer)

    if same_tokenizer:
        engine = EpisodeEngine(
            backend,
            sampling=safe_sampling,
            initial_text=root_text,
            initial_token_ids=episode["initial_token_ids"],
            stream_fingerprint=sampler_state["stream_fingerprint"],
            guidance_backend=guidance_backend,
        )
        engine.visible_token_ids = list(visible[:boundary])
        branch_identity = BranchIdentity(
            f"live-model-change-{uuid4().hex}",
            parent_id=source_id,
            fork_boundary=boundary,
        )
        return restore_live_session(
            store,
            source_id,
            engine,
            boundary=boundary,
            branch_identity=branch_identity,
        )

    engine = EpisodeEngine(
        backend,
        sampling=safe_sampling,
        initial_text=root_text,
        initial_token_ids=root_token_ids,
        guidance_backend=guidance_backend,
    )
    session = LiveSession(
        engine,
        prompt=root_text,
        environment_stamp={
            "backend": destination,
            "source_episode_id": source_id,
            "model_change_boundary": boundary,
        },
    )
    prior_text = visible_text_prefix(token_rows, boundary)
    if prior_text:
        session.generate(Write(prior_text, mode="exact"))
    return session


def new_live_session(
    engine: EpisodeEngine,
    *,
    prompt: str,
    provenance: dict[str, Any],
    branch_identity: BranchIdentity | None = None,
    source_episode_id: str | None = None,
) -> LiveSession:
    """Create an empty live root or an in-memory replay-derived branch."""
    environment = {"backend": dict(provenance)}
    if source_episode_id is not None:
        environment["source_episode_id"] = source_episode_id
    if branch_identity is None:
        return LiveSession(engine, prompt=prompt, environment_stamp=environment)
    state = BranchState(
        identity=branch_identity,
        initial_token_ids=tuple(engine.initial_token_ids),
        initial_sampling=engine.sampling,
        stream_fingerprint=engine.stream_fingerprint,
        visible_token_ids=tuple(engine.visible_token_ids),
        tape=(),
        outcomes=(),
    )
    return LiveSession(
        engine,
        prompt=prompt,
        environment_stamp=environment,
        initial_state=state,
    )


__all__ = ["model_change_session", "new_live_session", "restore_live_session"]
