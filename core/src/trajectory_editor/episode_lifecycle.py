"""Restore stored episodes and prepare replay engines for the terminal runtime."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Callable

from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .episode_engine import EpisodeEngine
from .spr_recipe import (
    ReplayControlPolicy,
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


def _inherit_budget(store, episode_id, engine, boundary, *, notice=print):
    state = store.budget_at(episode_id, boundary)
    if state is None:
        if store.get_episode(episode_id)["max_tokens"] is not None:
            notice("Budget history is missing at this boundary; continuing with unlimited tokens.")
        engine.trajectory.set_budget(None, None)
        return
    engine.trajectory.set_budget(state["max_tokens"], state["checkpoint_boundary"])


def _model_change_sampling(
    sampling: SamplerConfig, *, same_tokenizer: bool
) -> SamplerConfig:
    """Retain token-ID controls only when their tokenizer identity is unchanged."""
    return replace(
        sampling,
        bias_rules=sampling.bias_rules if same_tokenizer else (),
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
    max_tokens: int | None,
    sampling_override: SamplerConfig | None,
    guidance_backend: Any | None = None,
    sampling_factory: Callable = SamplerConfig.from_record,
    current_sampling_state: dict[str, Any] | None = None,
    notice=print,
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
        guidance_backend=guidance_backend,
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
    return runtime


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
        guidance_backend=guidance_backend,
    )
    return runtime, plan
