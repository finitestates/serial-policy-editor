"""Restore saved episode rows into the persistence-free live-session model."""

from __future__ import annotations

from bisect import bisect_right
from typing import Any

from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .episode_history import materialize_stored_prefix, outcome_from_stored_action
from .episode_session import BranchIdentity, BranchState, ControlPoint, LiveSession
from .episode_store import EpisodeStore
from .run_loop import TapeStep


def restore_live_session(
    store: EpisodeStore,
    episode_id: str,
    engine,
    *,
    boundary: int | None = None,
    branch_identity: BranchIdentity | None = None,
) -> LiveSession:
    """Load durable state once, then let the live session own execution."""

    episode = store.get_episode(episode_id)
    token_rows = store.tokens(episode_id)
    all_visible = tuple(
        int(row["token_id"]) for row in token_rows if bool(row["realized_visible"])
    )
    visible_boundary = len(all_visible)
    selected_boundary = visible_boundary if boundary is None else boundary
    if type(selected_boundary) is not int or not 0 <= selected_boundary <= visible_boundary:
        raise EditorError(
            f"restore boundary must be between 0 and {visible_boundary}"
        )
    if tuple(engine.visible_token_ids) != all_visible[:selected_boundary]:
        raise EditorError("restored engine prefix does not match saved episode history")

    history = materialize_stored_prefix(
        store.actions(episode_id),
        token_rows,
        store.sampler_segments(episode_id),
        store.budget_segments(episode_id),
        selected_boundary,
    )
    outcomes = tuple(outcome_from_stored_action(action) for action in history.actions)
    tape = tuple(
        TapeStep(outcome.action, outcome.expectation()) for outcome in outcomes
    )

    samplers = list(history.sampler_segments)
    budgets = list(history.budget_segments)
    if not samplers or not budgets:
        raise EditorError("saved episode is missing initial control state")
    sampler_starts = [int(row["start_boundary"]) for row in samplers]
    budget_starts = [int(row["start_boundary"]) for row in budgets]
    boundaries = sorted(
        {
            0,
            *(start for start in sampler_starts if start <= selected_boundary),
            *(start for start in budget_starts if start <= selected_boundary),
        }
    )
    control_points: list[ControlPoint] = []
    for point_boundary in boundaries:
        sampler = samplers[bisect_right(sampler_starts, point_boundary) - 1]
        budget = budgets[bisect_right(budget_starts, point_boundary) - 1]
        control_points.append(
            ControlPoint(
                boundary=point_boundary,
                sampling=SamplerConfig.from_record(sampler["sampling"]),
                stream_fingerprint=sampler["stream_fingerprint"],
                max_tokens=budget["max_tokens"],
                checkpoint_boundary=budget["checkpoint_boundary"],
            )
        )

    current = ControlPoint(
        boundary=selected_boundary,
        sampling=engine.sampling,
        stream_fingerprint=engine.stream_fingerprint,
        max_tokens=engine.max_tokens,
        checkpoint_boundary=engine.checkpoint_boundary,
    )
    if control_points[-1].boundary == selected_boundary:
        control_points[-1] = current
    elif control_points[-1] != current:
        control_points.append(current)

    identity = branch_identity or BranchIdentity(
        episode_id,
        parent_id=episode.get("parent_episode_id"),
        fork_boundary=episode.get("fork_boundary"),
    )
    local_action_start = 0
    if identity.parent_id is not None and identity.fork_boundary is not None:
        local_action_start = sum(
            outcome.boundary_after <= int(identity.fork_boundary) for outcome in outcomes
        )
    terminal_reason = (
        episode.get("terminal_reason")
        if branch_identity is None and selected_boundary == visible_boundary
        else None
    )
    state = BranchState(
        identity=identity,
        initial_token_ids=tuple(engine.initial_token_ids),
        visible_token_ids=tuple(engine.visible_token_ids),
        tape=tape,
        outcomes=outcomes,
        control_points=tuple(control_points),
        local_action_start=local_action_start,
        terminal_token_id=(
            episode.get("terminal_token_id") if terminal_reason is not None else None
        ),
        terminal_reason=terminal_reason,
        status=("completed" if terminal_reason is not None else "open"),
    )
    return LiveSession(
        engine,
        prompt=str(episode["initial_text"]),
        environment_stamp={
            "backend": dict(episode["backend"]),
            "sampler": engine.sampling.to_dict(),
            "source_episode_id": episode_id,
        },
        initial_state=state,
    )


def model_change_session(
    store: EpisodeStore,
    source_id: str,
    backend: Any,
    provenance: dict[str, Any],
    *,
    boundary: int,
    sampling: SamplerConfig,
    max_tokens: int | None,
    guidance_backend: Any | None = None,
) -> LiveSession:
    """Prepare a model-change continuation in memory, without creating an episode."""

    from uuid import uuid4

    from .core.actions import Write
    from .episode_history import visible_text_prefix
    from .episode_identity import backend_provenance_with_identity
    from .episode_lifecycle import _model_change_sampling
    from .episode_engine import EpisodeEngine

    episode = store.get_episode(source_id)
    token_rows = tuple(store.tokens(source_id))
    visible = tuple(
        int(row["token_id"]) for row in token_rows if bool(row["realized_visible"])
    )
    if type(boundary) is not int or not 0 <= boundary <= len(visible):
        raise EditorError(f"model-change boundary must be between 0 and {len(visible)}")
    segment = store.sampling_segment(source_id, boundary)
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
    budget = store.budget_at(source_id, boundary)
    allowance = max_tokens
    if allowance is None and budget is not None:
        allowance = budget["max_tokens"]

    if same_tokenizer:
        prefix = [*episode["initial_token_ids"], *visible[:boundary]]
        branch = getattr(backend, "branch_to_prefix", None)
        if callable(branch):
            branch(prefix)
        else:
            backend.reset(prefix)
        engine = EpisodeEngine(
            backend,
            sampling=safe_sampling,
            max_tokens=allowance,
            initial_text=root_text,
            initial_token_ids=episode["initial_token_ids"],
            stream_fingerprint=segment["stream_fingerprint"],
            backend_positioned=True,
            guidance_backend=guidance_backend,
        )
        engine.visible_token_ids = list(visible[:boundary])
        if allowance is None:
            engine.trajectory.set_budget(None, None)
        else:
            engine.trajectory.set_budget(allowance, boundary + allowance)
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
        max_tokens=allowance,
        initial_text=root_text,
        initial_token_ids=root_token_ids,
        guidance_backend=guidance_backend,
    )
    session = LiveSession(
        engine,
        prompt=root_text,
        environment_stamp={
            "backend": destination,
            "sampler": safe_sampling.to_dict(),
            "source_episode_id": source_id,
            "model_change_boundary": boundary,
        },
    )
    prior_text = visible_text_prefix(token_rows, boundary)
    if prior_text:
        session.generate(Write(prior_text, mode="exact"))
    return session


def new_live_session(
    engine,
    *,
    prompt: str,
    provenance: dict[str, Any],
    branch_identity: BranchIdentity | None = None,
    source_episode_id: str | None = None,
) -> LiveSession:
    """Create an empty live root or an in-memory replay-derived branch."""

    environment = {"backend": dict(provenance), "sampler": engine.sampling.to_dict()}
    if source_episode_id is not None:
        environment["source_episode_id"] = source_episode_id
    if branch_identity is None:
        return LiveSession(engine, prompt=prompt, environment_stamp=environment)
    state = BranchState(
        identity=branch_identity,
        initial_token_ids=tuple(engine.initial_token_ids),
        visible_token_ids=tuple(engine.visible_token_ids),
        tape=(),
        outcomes=(),
        control_points=(
            ControlPoint(
                0,
                engine.sampling,
                engine.stream_fingerprint,
                engine.max_tokens,
                engine.checkpoint_boundary,
            ),
        ),
    )
    return LiveSession(
        engine,
        prompt=prompt,
        environment_stamp=environment,
        initial_state=state,
    )


__all__ = ["model_change_session", "new_live_session", "restore_live_session"]
