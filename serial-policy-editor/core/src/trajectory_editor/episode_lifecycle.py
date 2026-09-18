"""Core episode lifecycle operations for the terminal runtime."""
from __future__ import annotations
from dataclasses import fields, replace
from typing import Any, Callable
from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine
from .episode_store import EpisodeStore
from .episode_runner import ReplayContext, ReplayPlan, TapeStep

SAMPLER_FIELDS = tuple(field.name for field in fields(SamplerConfig))
POLICY_FIELDS = tuple(
    field.name for field in fields(SamplerConfig)
    if field.name.startswith("activation_")
)

def _inherit_budget(store, episode_id, engine, boundary, *, rebase=False, notice=print):
    state = store.budget_at(episode_id, boundary)
    if state is None:
        if store.get_episode(episode_id)["max_tokens"] is not None:
            notice("Budget history is missing at this boundary; continuing with unlimited tokens.")
        engine.trajectory.set_budget(None, None)
        return
    checkpoint = state["checkpoint_boundary"]
    checkpoint = (max(0, checkpoint - boundary)
                  if rebase and checkpoint is not None else checkpoint)
    engine.trajectory.set_budget(state["max_tokens"], checkpoint)


def _model_continuation(
    store, source_id, backend, provenance, *, guidance_backend=None,
    sampling_factory: Callable = SamplerConfig.from_record,
):
    source = store.get_episode(source_id)
    boundary = len(_visible_tokens(store, source_id))
    segment = store.sampling_segment(source_id, boundary)
    # Token IDs and evidence belong to their original tokenizer. New model,
    # new execution record, with the old text as its entrance.
    engine = EpisodeEngine(
        backend, sampling=replace(
            sampling_factory(segment["sampling"]),
            bias_rules=(), bias_groups=(),
            activation_vector=(), activation_vector_strength=0.0,
            activation_vector_layer_start=None, activation_vector_layer_end=None,
            activation_vector_model="", activation_vector_digest="",
        ),
        initial_text=source["initial_text"] + source["visible_text"],
        max_tokens=source["max_tokens"],
        guidance_backend=guidance_backend,
        guidance_generated_prefix=_visible_tokens(store, source_id),
        guidance_tokens_consumed=boundary,
    )
    _inherit_budget(store, source_id, engine, boundary, rebase=True)
    identifier = _create_episode(
        store, engine, backend_provenance=provenance, parent_episode_id=source_id,
        fork_boundary=boundary, mode="model-change",
        metadata={"model_change_from": source_id},
    )
    store.rename(identifier, store.label(source_id).split("  ", 1)[-1] + " · model change")
    return engine, identifier


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
        initial_text=backend.render(prefix, special=True),
        initial_token_ids=prefix,
        stream_fingerprint=segment["stream_fingerprint"],
        coordinate_offset=segment["coordinate_offset"] + target,
        backend_positioned=True,
        guidance_backend=guidance_backend,
        guidance_generated_prefix=parent_engine.trajectory.visible_token_ids[:target],
        guidance_tokens_consumed=target,
    )
    if max_tokens is None:
        _inherit_budget(store, parent_id, engine, target, rebase=True, notice=notice)
    return engine


def _spr_engine_from_source(
    store: EpisodeStore,
    source_id: str,
    backend: Any,
    *,
    sampling: SamplerConfig,
    max_tokens: int | None,
    until: int | None = None,
    follow_source_sampling: bool | None = None,
    sampling_overrides: dict[str, Any] | None = None,
    initial_token_ids: list[int] | None = None,
    stream_fingerprint: str | None = None,
    coordinate_offset: int | None = None,
    guidance_backend: Any | None = None,
    sampling_factory: Callable = SamplerConfig.from_record,
) -> tuple[EpisodeEngine, ReplayPlan]:
    source = store.get_episode(source_id)
    overrides = dict(sampling_overrides or {})
    if overrides.keys() - set(SAMPLER_FIELDS):
        raise EditorError("unknown replay sampler override")
    sampling = replace(sampling, **overrides)
    if follow_source_sampling is None:
        # Replays from the original entrance follow the source.  A caller that
        # supplies a current prefix is performing counterfactual SPR and keeps
        # its current sampler unless it opts into source-following explicitly.
        follow_source_sampling = initial_token_ids is None
    source_steps = store.replay_until(
        source_id, until, sampling_factory=sampling_factory
    )
    tape = [TapeStep(step["action"], step["expectation"]) for step in source_steps]
    context = ReplayContext(
        sampling=tuple(
            replace(step["sampling"], **overrides)
            if follow_source_sampling else None
            for step in source_steps
        )
    )
    # Only an explicit target allowance limits replay.
    if initial_token_ids is None:
        segment = store.sampling_segment(source_id, 0)
        initial_token_ids = list(source["initial_token_ids"])
        stream_fingerprint = segment["stream_fingerprint"] if stream_fingerprint is None else stream_fingerprint
        coordinate_offset = (
            segment["coordinate_offset"]
            if coordinate_offset is None
            else coordinate_offset
        )
        initial_text = str(source["initial_text"])
    else:
        initial_text = backend.render(initial_token_ids, special=True)
        if stream_fingerprint is None or coordinate_offset is None:
            raise EditorError("counterfactual SPR requires sampler stream coordinates")
    runtime = EpisodeEngine(
        backend,
        sampling=sampling,
        max_tokens=max_tokens,
        initial_text=initial_text,
        initial_token_ids=initial_token_ids,
        stream_fingerprint=stream_fingerprint,
        coordinate_offset=coordinate_offset,
        guidance_backend=guidance_backend,
    )
    final_sampling = (
        replace(
            sampling_factory(store.final_sampling_record(source_id))
            if until is None
            else sampling_factory(store.sampling_segment(source_id, until)["sampling"]),
            **overrides,
        )
        if follow_source_sampling else None
    )
    return runtime, ReplayPlan(
        tuple(tape), follow_source_sampling, final_sampling, context
    )
