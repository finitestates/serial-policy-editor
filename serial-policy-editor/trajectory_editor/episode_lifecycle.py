"""Shared episode lifecycle operations for terminal and headless clients."""
from __future__ import annotations
from dataclasses import replace
from typing import Any
from .domain import EditorError, SamplingConfig
from .episode_engine import EpisodeEngine
from .episode_store import EpisodeStore
from .episode_policy import TapeStep, ReplayPlan

SAMPLER_FIELDS = ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty", "seed", "logit_bias", "bias_step", "sequence_bias")

def _inherit_budget(store, episode_id, engine, boundary, *, rebase=False, notice=print):
    state = store.budget_at(episode_id, boundary)
    if state is None:
        if store.get_episode(episode_id)["max_tokens"] is not None:
            notice("Budget history is missing at this boundary; continuing with unlimited tokens.")
        engine.max_tokens = engine.checkpoint_boundary = None
        return
    engine.max_tokens = state["max_tokens"]
    checkpoint = state["checkpoint_boundary"]
    engine.checkpoint_boundary = (max(0, checkpoint - boundary)
                                  if rebase and checkpoint is not None else checkpoint)


def _model_continuation(store, source_id, backend, provenance):
    source = store.get_episode(source_id)
    boundary = len(_visible_tokens(store, source_id))
    segment = store.sampling_segment(source_id, boundary)
    # Token IDs and evidence belong to their original tokenizer. New model,
    # new execution record, with the old text as its entrance.
    engine = EpisodeEngine(
        backend, sampling=replace(SamplingConfig.from_record(segment["sampling"]), logit_bias=(), sequence_bias=()),
        initial_text=source["initial_text"] + source["visible_text"],
        max_tokens=source["max_tokens"],
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
    sampling_override: SamplingConfig | None,
    notice=print,
) -> EpisodeEngine:
    episode = store.get_episode(episode_id)
    if episode["status"] in {"completed", "failed"}:
        raise EditorError(
            f"episode {episode_id!r} is sealed; use --replay or --fork-from instead"
        )
    visible = _visible_tokens(store, episode_id)
    segment = store.sampling_segment(episode_id, len(visible))
    source_sampling = SamplingConfig.from_record(segment["sampling"])
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
    )
    # These tokens are already known; only the final-position logits are needed.
    # Let the backend batch reconstruction without replaying individual moves.
    if visible:
        backend.eval(visible)
        runtime.visible_token_ids.extend(visible)
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
    return store.create_episode(
        episode_id=requested_id,
        parent_episode_id=parent_episode_id,
        fork_boundary=fork_boundary,
        initial_text=engine.initial_text,
        initial_token_ids=engine.initial_token_ids,
        sampling=engine.sampling,
        stream_fingerprint=engine.stream_fingerprint,
        coordinate_offset=engine.coordinate_offset,
        max_tokens=engine.max_tokens,
        backend=backend_provenance,
        metadata=payload,
        checkpoint_boundary=engine.checkpoint_boundary,
    )


def _rewind_episode(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    boundary: int,
    *, notice=print,
) -> dict[str, Any]:
    """Restore the destination sampler as well as its retained token prefix."""
    # Read before truncation removes the future sampler segments. Unlike a
    # fork, this engine keeps its original prefix and absolute boundaries, so
    # the stored coordinate offset must not have the boundary added to it.
    segment = store.sampling_segment(episode_id, boundary)
    sampling = SamplingConfig.from_record(segment["sampling"])
    engine.rewind_to(boundary)
    _inherit_budget(store, episode_id, engine, boundary, notice=notice)
    details = store.rewind_to(
        episode_id,
        boundary,
        visible_text=engine.backend.render(engine.visible_token_ids),
        max_tokens=engine.max_tokens,
    )
    store.record_budget(episode_id, boundary, engine.max_tokens, engine.checkpoint_boundary)
    engine.sampling = sampling
    engine.stream_fingerprint = segment["stream_fingerprint"]
    engine.coordinate_offset = segment["coordinate_offset"]
    return details


def _fork_engine(
    store: EpisodeStore,
    parent_id: str,
    parent_engine: EpisodeEngine,
    target: int,
    *,
    backend: Any,
    max_tokens: int | None,
    notice=print,
) -> EpisodeEngine:
    if not 0 <= target <= parent_engine.boundary:
        raise EditorError(f"fork boundary must be between 0 and {parent_engine.boundary}")
    prefix = [
        *parent_engine.initial_token_ids,
        *parent_engine.visible_token_ids[:target],
    ]
    branch = getattr(backend, "branch_to_prefix", None)
    if callable(branch):
        branch(prefix)
    else:
        backend.reset(prefix)
    segment = store.sampling_segment(parent_id, target)
    sampling = SamplingConfig.from_record(segment["sampling"])
    engine = EpisodeEngine(
        backend,
        sampling=sampling,
        max_tokens=parent_engine.max_tokens if max_tokens is None else max_tokens,
        initial_text=backend.render(prefix, special=True),
        initial_token_ids=prefix,
        stream_fingerprint=segment["stream_fingerprint"],
        coordinate_offset=segment["coordinate_offset"] + target,
        backend_positioned=True,
    )
    if max_tokens is None:
        _inherit_budget(store, parent_id, engine, target, rebase=True, notice=notice)
    return engine


def _spr_engine_from_source(
    store: EpisodeStore,
    source_id: str,
    backend: Any,
    *,
    sampling: SamplingConfig,
    max_tokens: int | None,
    until: int | None = None,
    follow_source_sampling: bool | None = None,
    sampling_overrides: dict[str, Any] | None = None,
    initial_token_ids: list[int] | None = None,
    stream_fingerprint: str | None = None,
    coordinate_offset: int | None = None,
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
    tape = [TapeStep(step["action"], step["expectation"],
                     replace(step["sampling"], **overrides) if follow_source_sampling else None)
            for step in store.replay_until(source_id, until)]
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
    )
    final_sampling = (
        replace(store.final_sampling(source_id) if until is None else
                SamplingConfig.from_record(store.sampling_segment(source_id, until)["sampling"]), **overrides)
        if follow_source_sampling else None
    )
    return runtime, ReplayPlan(tuple(tape), follow_source_sampling, final_sampling)
