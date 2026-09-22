"""Core episode lifecycle operations for the terminal runtime."""
from __future__ import annotations
from dataclasses import fields, replace
from typing import Any, Callable
from .core.errors import EditorError
from .core.actions import Write
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine
from .spr_recipe import (
    ReplayControlPolicy,
    ReplayPlacement,
    SourceReplayRecipe,
    compose_replay_plan,
)
from .episode_store import EpisodeStore
from .episode_runner import ReplayPlan
from .episode_history import visible_text_prefix

SAMPLER_FIELDS = tuple(field.name for field in fields(SamplerConfig))
POLICY_FIELDS = tuple(
    field.name for field in fields(SamplerConfig)
    if field.name.startswith("activation_")
)

def _inherit_budget(store, episode_id, engine, boundary, *, notice=print):
    state = store.budget_at(episode_id, boundary)
    if state is None:
        if store.get_episode(episode_id)["max_tokens"] is not None:
            notice("Budget history is missing at this boundary; continuing with unlimited tokens.")
        engine.trajectory.set_budget(None, None)
        return
    engine.trajectory.set_budget(state["max_tokens"], state["checkpoint_boundary"])


def _model_change_sampling(sampling: SamplerConfig) -> SamplerConfig:
    """Remove controls whose token IDs belong to the old backend."""
    return replace(
        sampling,
        bias_rules=(),
        bias_groups=(),
        activation_vector=(),
        activation_vector_strength=0.0,
        activation_vector_layer_start=None,
        activation_vector_layer_end=None,
        activation_vector_model="",
        activation_vector_digest="",
    )


def _materialize_model_change_fork(
    store: EpisodeStore,
    source_id: str,
    boundary: int,
    backend: Any,
    provenance: dict[str, Any],
    *,
    sampling: SamplerConfig,
    max_tokens: int | None,
    requested_id: str | None = None,
    guidance_backend: Any | None = None,
) -> tuple[EpisodeEngine, str]:
    """Materialize a fork whose destination backend has a new tokenizer.

    The source boundary selects text from the source coordinate space.  The
    destination then starts at its own root prompt and records that retained
    text as an exact write, so the child still has boundary zero immediately
    after its root prompt.
    """
    source = store.get_episode(source_id)
    source_tokens = store.tokens(source_id)
    visible_count = sum(bool(row["realized_visible"]) for row in source_tokens)
    if type(boundary) is not int or not 0 <= boundary <= visible_count:
        raise EditorError(f"fork boundary must be between 0 and {visible_count}")
    root_text = str(source["initial_text"])
    root_token_ids = backend.tokenize(root_text, add_bos=True, special=True)
    runtime = EpisodeEngine(
        backend,
        sampling=_model_change_sampling(sampling),
        max_tokens=None,
        initial_text=root_text,
        initial_token_ids=root_token_ids,
        coordinate_offset=0,
        guidance_backend=guidance_backend,
    )
    identifier = _create_episode(
        store,
        runtime,
        backend_provenance=provenance,
        requested_id=requested_id,
        parent_episode_id=source_id,
        fork_boundary=boundary,
        mode="model-change",
        metadata={
            "model_change_from": source_id,
            "coordinate_system": "root-relative",
        },
    )
    retained_text = visible_text_prefix(source_tokens, boundary)
    if retained_text:
        outcome = runtime.apply(Write(retained_text, mode="exact"))
        store.record_action(identifier, 0, outcome)

    budget = store.budget_at(source_id, boundary)
    allowance = max_tokens
    if allowance is None and budget is not None:
        allowance = budget["max_tokens"]
    checkpoint = None if allowance is None else runtime.boundary + allowance
    runtime.trajectory.set_budget(allowance, checkpoint)
    store.record_sampling_segment(
        identifier,
        start_boundary=runtime.boundary,
        sampling=runtime.sampling,
        stream_fingerprint=runtime.stream_fingerprint,
        coordinate_offset=runtime.coordinate_offset,
    )
    store.record_budget(identifier, runtime.boundary, allowance, checkpoint)
    store.update_episode(
        identifier,
        visible_text=backend.render(runtime.visible_token_ids),
        max_tokens=allowance,
    )
    store.rename(identifier, store.label(source_id).split("  ", 1)[-1] + " · model change")
    return runtime, identifier


def _model_continuation(
    store, source_id, backend, provenance, *, guidance_backend=None,
    sampling_factory: Callable = SamplerConfig.from_record,
):
    source = store.get_episode(source_id)
    boundary = len(_visible_tokens(store, source_id))
    segment = store.sampling_segment(source_id, boundary)
    return _materialize_model_change_fork(
        store,
        source_id,
        boundary,
        backend,
        provenance,
        sampling=sampling_factory(segment["sampling"]),
        max_tokens=source["max_tokens"],
        guidance_backend=guidance_backend,
    )


def _visible_tokens(store: EpisodeStore, episode_id: str) -> list[int]:
    return [
        int(token["token_id"])
        for token in store.tokens(episode_id)
        if bool(token["realized_visible"])
    ]


def _restore_engine(
    store: EpisodeStore,
    episode_id: str,
    backend: Any,
    *,
    max_tokens: int | None,
    sampling_override: SamplerConfig | None,
    guidance_backend: Any | None = None,
    guidance_generated_prefix: list[int] | None = None,
    guidance_tokens_consumed: int = 0,
    sampling_factory: Callable = SamplerConfig.from_record,
    notice=print,
) -> EpisodeEngine:
    episode = store.get_episode(episode_id)
    if episode["status"] in {"completed", "failed"}:
        raise EditorError(
            f"episode {episode_id!r} is sealed; use --replay or --fork-from instead"
        )
    visible = _visible_tokens(store, episode_id)
    segment = store.sampling_segment(episode_id, len(visible))
    source_sampling = sampling_factory(segment["sampling"])
    sampling = sampling_override or source_sampling
    saved_budget = episode["max_tokens"] or None
    runtime = EpisodeEngine(
        backend,
        sampling=sampling,
        max_tokens=saved_budget if max_tokens is None else max_tokens,
        initial_text=str(episode["initial_text"]),
        initial_token_ids=episode["initial_token_ids"],
        stream_fingerprint=segment["stream_fingerprint"],
        coordinate_offset=segment["coordinate_offset"],
        guidance_backend=guidance_backend,
        guidance_generated_prefix=guidance_generated_prefix,
        guidance_tokens_consumed=guidance_tokens_consumed,
    )
    # These tokens are already known; only the final-position logits are needed.
    # Let the backend batch reconstruction without replaying individual moves.
    if visible:
        backend.eval(visible)
        runtime.trajectory.visible_token_ids.extend(visible)
    if max_tokens is None:
        _inherit_budget(store, episode_id, runtime, len(visible), notice=notice)
    else:
        runtime.resume(max_tokens=max_tokens, sampling=sampling)
    if sampling != source_sampling:
        store.record_sampling_segment(
            episode_id,
            start_boundary=runtime.boundary,
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=runtime.coordinate_offset,
        )
    return runtime


def _create_episode(
    store: EpisodeStore,
    engine: EpisodeEngine,
    *,
    backend_provenance: dict[str, Any],
    requested_id: str | None = None,
    parent_episode_id: str | None = None,
    fork_boundary: int | None = None,
    mode: str = "interactive",
    metadata: dict[str, Any] | None = None,
) -> str:
    payload = {"mode": mode, **(metadata or {})}
    if mode == "fork":
        # Ordinary forks retain the source prompt/context and copy their
        # inherited actions into the child.  Keep this explicit in metadata so
        # tooling can distinguish the canonical representation from old local
        # coordinate episodes without changing the compact schema.
        payload.setdefault("coordinate_system", "root-relative")
    trajectory = engine.trajectory
    return store.create_episode(
        episode_id=requested_id,
        parent_episode_id=parent_episode_id,
        fork_boundary=fork_boundary,
        initial_text=trajectory.initial_text,
        initial_token_ids=trajectory.initial_token_ids,
        sampling=engine.sampling,
        stream_fingerprint=trajectory.stream_fingerprint,
        coordinate_offset=trajectory.coordinate_offset,
        max_tokens=trajectory.max_tokens,
        backend=backend_provenance,
        metadata=payload,
        checkpoint_boundary=trajectory.checkpoint_boundary,
    )


def _rewind_episode(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    boundary: int,
    *, notice=print, sampling_factory: Callable = SamplerConfig.from_record,
) -> dict[str, Any]:
    """Restore the destination sampler as well as its retained token prefix."""
    # Read before truncation removes the future sampler segments. Unlike a
    # fork, this engine keeps its original prefix and absolute boundaries, so
    # the stored coordinate offset must not have the boundary added to it.
    segment = store.sampling_segment(episode_id, boundary)
    sampling = sampling_factory(segment["sampling"])
    engine.rewind_to(boundary)
    _inherit_budget(store, episode_id, engine, boundary, notice=notice)
    details = store.rewind_to(
        episode_id,
        boundary,
        visible_text=engine.backend.render(engine.trajectory.visible_token_ids),
        max_tokens=engine.trajectory.max_tokens,
    )
    store.record_budget(
        episode_id,
        boundary,
        engine.trajectory.max_tokens,
        engine.trajectory.checkpoint_boundary,
    )
    engine.sampling = sampling
    engine.trajectory.set_coordinates(
        stream_fingerprint=segment["stream_fingerprint"],
        coordinate_offset=segment["coordinate_offset"],
    )
    return details


def _fork_engine(
    store: EpisodeStore,
    parent_id: str,
    parent_engine: EpisodeEngine,
    target: int,
    *,
    backend: Any,
    max_tokens: int | None,
    guidance_backend: Any | None = None,
    sampling_factory: Callable = SamplerConfig.from_record,
    notice=print,
) -> EpisodeEngine:
    if not 0 <= target <= parent_engine.boundary:
        raise EditorError(f"fork boundary must be between 0 and {parent_engine.boundary}")
    prefix = [
        *parent_engine.trajectory.initial_token_ids,
        *parent_engine.trajectory.visible_token_ids[:target],
    ]
    branch = getattr(backend, "branch_to_prefix", None)
    if callable(branch):
        branch(prefix)
    else:
        backend.reset(prefix)
    segment = store.sampling_segment(parent_id, target)
    sampling = sampling_factory(segment["sampling"])
    engine = EpisodeEngine(
        backend,
        sampling=sampling,
        max_tokens=(
            parent_engine.trajectory.max_tokens
            if max_tokens is None
            else max_tokens
        ),
        # Keep the original entrance and represent the retained parent
        # actions as visible history.  Replacing initial_text with the fork
        # prefix would silently rebase the child's public boundary zero.
        initial_text=parent_engine.initial_text,
        initial_token_ids=parent_engine.initial_token_ids,
        stream_fingerprint=segment["stream_fingerprint"],
        coordinate_offset=segment["coordinate_offset"],
        backend_positioned=True,
        guidance_backend=guidance_backend,
    )
    engine.visible_token_ids = list(parent_engine.trajectory.visible_token_ids[:target])
    if max_tokens is None:
        _inherit_budget(store, parent_id, engine, target, notice=notice)
    else:
        engine.trajectory.set_budget(max_tokens, target + max_tokens)
    return engine


def _spr_engine_from_source(
    recipe: SourceReplayRecipe,
    backend: Any,
    *,
    sampling: SamplerConfig,
    max_tokens: int | None,
    control_policy: ReplayControlPolicy,
    sampling_overrides: dict[str, Any] | None = None,
    initial_token_ids: list[int],
    stream_fingerprint: str | None = None,
    coordinate_offset: int | None = None,
    guidance_backend: Any | None = None,
) -> tuple[EpisodeEngine, ReplayPlan]:
    """Build a root-entered runtime and plan from a semantic replay recipe."""

    overrides = dict(sampling_overrides or {})
    if overrides.keys() - set(SAMPLER_FIELDS):
        raise EditorError("unknown replay sampler override")
    sampling = replace(sampling, **overrides)
    root_controls = recipe.controls.effective_at(0)
    if stream_fingerprint is None:
        stream_fingerprint = root_controls.stream_fingerprint
    if coordinate_offset is None:
        coordinate_offset = root_controls.coordinate_offset
    if stream_fingerprint is None:
        raise EditorError("source replay requires a sampler stream fingerprint")
    plan = compose_replay_plan(
        recipe,
        ReplayPlacement.SOURCE_ROOT,
        control_policy,
        sampler_overrides=overrides,
    )
    runtime = EpisodeEngine(
        backend,
        sampling=sampling,
        max_tokens=max_tokens,
        initial_text=recipe.source_prompt,
        initial_token_ids=initial_token_ids,
        stream_fingerprint=stream_fingerprint,
        coordinate_offset=coordinate_offset,
        guidance_backend=guidance_backend,
    )
    return runtime, plan
