"""Restore stored episodes and prepare replay engines for the terminal runtime."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Callable

from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine
from .core.backend_position import position_backend
from .spr_recipe import (
    ReplaySamplerPolicy,
    ReplayPlacement,
    SourceReplayRecipe,
    compose_replay_plan,
)
from .episode_identity import tokenizer_id_for
from .episode_store import EpisodeStore
from .run_loop import ReplayPlan

SAMPLER_FIELDS = tuple(field.name for field in fields(SamplerConfig))
POLICY_FIELDS = tuple(
    field.name for field in fields(SamplerConfig)
    if field.name.startswith("activation_")
)


def _model_change_sampling(
    sampling: SamplerConfig, *, same_tokenizer: bool
) -> SamplerConfig:
    """Retain token-ID controls only when their tokenizer identity is unchanged."""
    return replace(
        sampling,
        token_biases=sampling.token_biases if same_tokenizer else (),
        bias_groups=sampling.bias_groups if same_tokenizer else (),
        activation_vector=(),
        activation_vector_strength=0.0,
        activation_vector_layer_start=None,
        activation_vector_layer_end=None,
        activation_vector_digest="",
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
    sampling_override: SamplerConfig | None,
    guidance_backend: Any | None = None,
    sampling_factory: Callable = SamplerConfig.from_record,
    current_sampling_state: dict[str, Any] | None = None,
) -> EpisodeEngine:
    episode = store.get_episode(episode_id)
    saved_tokenizer_id = episode["backend"].get("tokenizer_id")
    if (
        isinstance(saved_tokenizer_id, str)
        and saved_tokenizer_id != tokenizer_id_for(backend)
    ):
        raise EditorError(
            "episode tokenizer identity differs from the loaded backend; "
            "use a model-change continuation"
        )
    if episode["status"] in {"completed", "failed"}:
        raise EditorError(
            f"episode {episode_id!r} is sealed; use --replay or --fork-from instead"
        )
    visible = _visible_tokens(store, episode_id)
    if current_sampling_state is not None:
        if current_sampling_state.get("boundary") != len(visible):
            raise EditorError("episode changed while preparing its current sampler state")
        segment = current_sampling_state
    else:
        segment = store.current_sampling_state(episode_id)
    source_sampling = sampling_factory(segment["sampling"])
    sampling = sampling_override or source_sampling
    runtime = EpisodeEngine(
        backend,
        sampling=sampling,
        initial_text=str(episode["initial_text"]),
        initial_token_ids=episode["initial_token_ids"],
        stream_fingerprint=segment["stream_fingerprint"],
        guidance_backend=guidance_backend,
    )
    # Position against the complete ledger so a related live cache can be
    # cropped to its shared prefix instead of rebuilding the full episode.
    runtime.trajectory.visible_token_ids.extend(visible)
    if not position_backend(runtime.backend, runtime.token_ids):
        runtime.backend.reset(runtime.token_ids)
    return runtime


def _spr_engine_from_source(
    recipe: SourceReplayRecipe,
    backend: Any,
    *,
    sampling: SamplerConfig,
    sampler_policy: ReplaySamplerPolicy,
    sampling_overrides: dict[str, Any] | None = None,
    initial_token_ids: list[int],
    stream_fingerprint: str | None = None,
    guidance_backend: Any | None = None,
) -> tuple[EpisodeEngine, ReplayPlan]:
    """Build a root-entered runtime and plan from a semantic replay recipe."""

    overrides = dict(sampling_overrides or {})
    if overrides.keys() - set(SAMPLER_FIELDS):
        raise EditorError("unknown replay sampler override")
    sampling = replace(sampling, **overrides)
    if stream_fingerprint is None:
        stream_fingerprint = recipe.stream_fingerprint
    if stream_fingerprint is None:
        raise EditorError("source replay requires a sampler stream fingerprint")
    plan = compose_replay_plan(
        recipe,
        ReplayPlacement.SOURCE_ROOT,
        sampler_policy,
        sampler_overrides=overrides,
    )
    runtime = EpisodeEngine(
        backend,
        sampling=sampling,
        initial_text=recipe.source_prompt,
        initial_token_ids=initial_token_ids,
        stream_fingerprint=stream_fingerprint,
        guidance_backend=guidance_backend,
    )
    return runtime, plan
