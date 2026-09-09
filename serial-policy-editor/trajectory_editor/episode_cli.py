"""Command line for the reduced Serial Policy Editor.

A live episode alternates between the ordinary teacher loop and live edges.
Token budgets, replay exhaustion, and replay divergence yield at a live edge;
only EOG or explicit ``end`` seals the episode.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import secrets
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator

from .backend_factory import BACKEND_NAMES, create_backend
from .decoder import LlamaCppSettings
from .domain import MAX_SEED, MIN_SEED, EditorError, SamplingConfig
from .episode_actions import Write
from .episode_engine import EpisodeEngine
from .episode_policy import (
    EdgeRequested,
    EpisodeRunner,
    ForkRequested,
    SeamlessEdgeRequested,
    SeamlessRewindRequested,
    TapeStep,
    ReplayPlan,
)
from .episode_projector import project_episode, project_fork_map, project_lineage, project_procedure
from .episode_store import EpisodeStore
from .episode_recovery import recover_sampler_record
from .episode_ui import InteractivePolicy
from .transformers_backend import TransformersSettings
from .tui import TerminalIO
from .ui_themes import LIVE_THEME_NAMES
from .version import VERSION


SAMPLER_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)
SAMPLER_ALIASES = {
    "temp": "temperature",
    "rep": "repeat_penalty",
    "rep_pen": "repeat_penalty",
    "repeat": "repeat_penalty",
    "presence": "presence_penalty",
    "frequency": "frequency_penalty",
}


def _random_seed() -> int:
    """Return a uniformly chosen seed from the supported signed 64-bit range."""

    return secrets.randbelow(MAX_SEED - MIN_SEED + 1) + MIN_SEED


def _read_initial_prompt() -> str:
    return prompt(
        "Write at least one character. Press Escape then Enter to continue.\n\n",
        multiline=True,
        validator=Validator.from_callable(
            lambda text: len(text) >= 1,
            error_message="Write at least one character.",
        ),
        validate_while_typing=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-editor",
        description=(
            "Interactive token-policy editor with Serial Policy Replay. "
            "Token budgets are checkpoints, not run termination."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("episodes.sqlite3"),
        help="compact episode workspace used for SPR and token evidence",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--new-prompt", metavar="TEXT")
    source.add_argument("--new-prompt-file", type=Path, metavar="FILE")
    source.add_argument(
        "--replay",
        metavar="EPISODE_ID",
        help="Serial Policy Replay of an existing episode, then yield live",
    )
    parser.add_argument("--until", type=int, metavar="Y",
                        help="replay through source token boundary Y, then return to EDGE")
    source.add_argument(
        "--resume",
        "--continue-from",
        dest="resume",
        metavar="EPISODE_ID",
        help="resume an unsealed/checkpointed episode in the same episode id",
    )
    parser.add_argument(
        "--fixed-config", action="store_true",
        help="freeze source-initial sampler settings plus explicit overrides during --replay",
    )
    source.add_argument("--fork-from", metavar="EPISODE_ID")
    source.add_argument("--project", metavar="EPISODE_ID")
    parser.add_argument("--procedure", action="store_true", help="show a manual replay procedure with --project")
    source.add_argument("--list", action="store_true", dest="list_episodes")
    parser.add_argument("--at", type=int, metavar="BOUNDARY")
    parser.add_argument(
        "--lineage",
        metavar="EPISODE_ID",
        help="show the ordinary fork family and related replays for an episode (with --list)",
    )
    parser.add_argument("--episode-id")
    parser.add_argument(
        "--output",
        type=Path,
        help="write full final episode text here when the episode is sealed",
    )

    parser.add_argument("--backend", choices=BACKEND_NAMES, default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="optional visible-token allowance before a live edge (default: unlimited)",
    )
    parser.add_argument("--table-depth", type=int, default=12)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--hold-default", type=int, default=24)
    parser.add_argument("--context-chars", type=int, default=0, help="Context character limit (0 keeps all context)")
    parser.add_argument("--plain-ui", action="store_true")
    parser.add_argument(
        "--cache",
        choices=("auto", "off"),
        default="auto",
        help="use backend cache when available (default: auto; use off for full-prefix evaluation)",
    )
    parser.add_argument(
        "--no-cache",
        dest="cache",
        action="store_const",
        const="off",
        help="disable backend caching and use full-prefix evaluation",
    )
    parser.add_argument(
        "--seamless",
        action="store_true",
        help="accepted for compatibility; live history rewind is always enabled",
    )
    parser.add_argument(
        "--manual-acceptance",
        action="store_true",
        help="leave each teacher command blank instead of prefilling the sampled proposal",
    )
    parser.add_argument("--show-policy-rank", action="store_true")
    parser.add_argument("--theme", choices=LIVE_THEME_NAMES)
    parser.add_argument(
        "--divergence-policy", choices=("handoff", "ballistic"), default="handoff"
    )

    sampling = parser.add_argument_group("sampling")
    for name, kind in (
        ("temperature", float),
        ("top_k", int),
        ("top_p", float),
        ("min_p", float),
        ("repeat_penalty", float),
        ("repeat_last_n", int),
        ("presence_penalty", float),
        ("frequency_penalty", float),
    ):
        sampling.add_argument("--" + name.replace("_", "-"), type=kind)
    seed_options = sampling.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int)
    seed_options.add_argument(
        "--random-seed",
        action="store_true",
        help="choose a random sampler seed from the supported signed 64-bit range",
    )

    llama = parser.add_argument_group("llama.cpp")
    llama.add_argument("--n-ctx", type=int, default=2048)
    llama.add_argument("--n-batch", type=int, default=256)
    llama.add_argument("--n-ubatch", type=int)
    llama.add_argument("--n-threads", type=int)
    llama.add_argument("--n-threads-batch", type=int)
    llama.add_argument("--n-gpu-layers", type=int)
    llama.add_argument("--main-gpu", type=int)
    llama.add_argument("--no-flash-attn", action="store_true")
    llama.add_argument("--no-mmap", action="store_true")
    llama.add_argument("--use-mlock", action="store_true")

    transformers = parser.add_argument_group("Transformers")
    transformers.add_argument("--transformers-device", default="auto")
    transformers.add_argument(
        "--transformers-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    transformers.add_argument(
        "--transformers-device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
    )
    transformers.add_argument(
        "--transformers-attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "flex_attention"),
    )
    transformers.add_argument(
        "--transformers-quantization",
        choices=("none", "bitsandbytes-8bit", "bitsandbytes-4bit"),
        default="none",
    )
    transformers.add_argument("--transformers-trust-remote-code", action="store_true")
    transformers.add_argument("--transformers-slow-tokenizer", action="store_true")
    transformers.add_argument("--transformers-torch-threads", type=int)
    transformers.add_argument("--transformers-torch-interop-threads", type=int)

    projection = parser.add_argument_group("projection")
    projection.add_argument(
        "--annotations", choices=("none", "inline", "footnotes"), default="none"
    )
    projection.add_argument("--with-loss", action="store_true")
    projection.add_argument("--with-rank", action="store_true")
    projection.add_argument("--with-policy-rank", action="store_true")
    projection.add_argument(
        "--full-evidence",
        action="store_true",
        help="footnote teacher-selected tokens with proposal agreement, NLL, and ranks",
    )
    projection.add_argument(
        "--with-model-probs",
        "--with-model",
        dest="with_model_probs",
        action="store_true",
        help="add raw-model and decoder probabilities to teacher-token footnotes",
    )
    projection.add_argument(
        "--with-lineage",
        action="store_true",
        help="append fork-family and replay metadata to a projection",
    )
    return parser


def _sampling_from_args(
    args: argparse.Namespace, source: SamplingConfig | None = None
) -> SamplingConfig:
    base = source if source is not None else SamplingConfig()
    return SamplingConfig(
        **{
            name: getattr(args, name)
            if getattr(args, name) is not None
            else getattr(base, name)
            for name in SAMPLER_FIELDS
        }
    )


def _sampler_override(current: SamplingConfig, raw: str) -> SamplingConfig:
    values = {name: getattr(current, name) for name in SAMPLER_FIELDS}
    pieces = raw.replace(",", " ").split()
    if not pieces:
        return current
    if len(pieces) == 1 and pieces[0].lower() in {"random", "random-seed"}:
        values["seed"] = _random_seed()
        return SamplingConfig(**values)
    for piece in pieces:
        if "=" not in piece:
            raise EditorError("sampler changes use key=value (for example top_k=20)")
        key, value = piece.split("=", 1)
        key = SAMPLER_ALIASES.get(key.strip().lower(), key.strip().lower())
        if key not in values:
            raise EditorError(f"unknown sampler field {key!r}")
        try:
            values[key] = int(value) if key in {"top_k", "repeat_last_n", "seed"} else float(value)
        except ValueError as exc:
            raise EditorError(f"invalid value for {key}: {value!r}") from exc
    return SamplingConfig(**values)


def _backend(args: argparse.Namespace):
    args.backend = args.backend or "llama.cpp"
    if args.model is None:
        raise EditorError("--model is required to start, resume, fork, or replay")
    llama = LlamaCppSettings(
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        n_gpu_layers=args.n_gpu_layers,
        main_gpu=args.main_gpu,
        flash_attn=not args.no_flash_attn,
        use_mmap=not args.no_mmap,
        use_mlock=args.use_mlock,
    )
    transformers = TransformersSettings(
        device=args.transformers_device,
        dtype=args.transformers_dtype,
        device_map=args.transformers_device_map,
        attention_implementation=args.transformers_attention_implementation,
        quantization_method=args.transformers_quantization,
        trust_remote_code=args.transformers_trust_remote_code,
        use_fast_tokenizer=not args.transformers_slow_tokenizer,
        torch_num_threads=args.transformers_torch_threads,
        torch_num_interop_threads=args.transformers_torch_interop_threads,
    )
    print(f"Loading model with {args.backend}: {args.model} ...", flush=True)
    result = create_backend(
        args.backend,
        args.model,
        llama_settings=llama,
        transformers_settings=transformers,
        cache_mode=args.cache,
    )
    print("Model loaded.", flush=True)
    return result


def _load_episode_backend(args, source, io, *, use_saved=False, current_backend=None, current_provenance=None):
    """Load a saved execution context, confirming deliberate model changes."""
    selected = copy.copy(args)
    saved = source["backend"] if source else {}
    old_path = saved.get("model_path")
    if use_saved:
        selected.model = None
        selected.backend = None
    if selected.model is None and old_path:
        selected.model = Path(old_path)
    selected.backend = selected.backend or saved.get("backend") or "llama.cpp"
    if selected.backend not in BACKEND_NAMES:
        selected.backend = args.backend or "llama.cpp"
    # Persisted launch options avoid reconstructing device/quantization settings
    # from diagnostic effective values. Explicit launch flags take precedence.
    explicit = getattr(args, "_explicit_options", set()) if not use_saved else set()
    saved_options = dict(saved.get("load_options", {}))
    if not saved_options:
        for key, value in saved.get("runtime_configuration", {}).items():
            if saved.get("backend") == "transformers":
                key = "transformers_" + key
            elif key in {"flash_attn", "use_mmap"}:
                key, value = {"flash_attn": "no_flash_attn", "use_mmap": "no_mmap"}[key], not value
            if hasattr(selected, key) and key != "seed":
                saved_options[key] = value
    for key, value in saved_options.items():
        if key not in explicit:
            setattr(selected, key, value)
    while True:
        if selected.model is None:
            path = io.read("Saved model location unavailable. Model path (Enter cancels)> ")
            if not path:
                raise EditorError("model loading cancelled")
            selected.model = Path(path).expanduser()
        changed = bool(source and old_path and (
            Path(old_path).resolve() != selected.model.resolve()
            or (saved.get("backend") in BACKEND_NAMES and saved.get("backend") != selected.backend)
        ))
        if changed:
            answer = io.read(f"Previously used {old_path} ({saved.get('backend')}). Continue with {selected.model} ({selected.backend}) in a new linked episode? [y/N]> ")
            if not answer or answer.strip().lower() not in {"y", "yes"}:
                raise EditorError("model change cancelled")
        try:
            if (current_backend is not None and current_provenance
                and current_provenance.get("model_path") == str(selected.model.resolve())
                and current_provenance.get("backend") == selected.backend
                and all(getattr(selected, key, None) == value
                        for key, value in current_provenance.get("load_options", {}).items())):
                return current_backend, current_provenance, changed
            backend = _backend(selected)
            provenance = dict(backend.provenance(include_model_sha256=False))
            provenance["model_path"] = str(selected.model.resolve())
            provenance["load_options"] = {
                key: value for key, value in vars(selected).items()
                if key.startswith("transformers_") or key in {
                    "n_ctx", "n_batch", "n_ubatch", "n_threads", "n_threads_batch",
                    "n_gpu_layers", "main_gpu", "no_flash_attn", "no_mmap", "use_mlock", "cache"
                }
            }
            return backend, provenance, changed
        except (EditorError, OSError, RuntimeError) as exc:
            if not source:
                raise
            io.write(f"Could not load {selected.model}: {exc}")
            path = io.read("Replacement model path (Enter cancels)> ")
            if not path:
                raise EditorError("model loading cancelled") from exc
            kind = io.read("Backend: llama.cpp or transformers (Enter keeps current)> ")
            if kind:
                if kind.strip() not in BACKEND_NAMES:
                    io.write("Unknown backend.")
                    continue
                selected.backend = kind.strip()
            selected.model = Path(path).expanduser()


def _inherit_budget(store, episode_id, engine, boundary, *, rebase=False):
    state = store.budget_at(episode_id, boundary)
    if state is None:
        if store.get_episode(episode_id)["max_tokens"] is not None:
            print("Budget history is missing at this boundary; continuing with unlimited tokens.")
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
        backend, sampling=SamplingConfig.from_record(segment["sampling"]),
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


def _interactive_policy(
    args: argparse.Namespace, store: EpisodeStore, episode_id: str, io: TerminalIO
) -> InteractivePolicy:
    return InteractivePolicy(
        io=io,
        menu_size=args.table_depth,
        search_radius=args.search_radius,
        default_hold_tokens=args.hold_default,
        context_characters=args.context_chars,
        manual_acceptance=args.manual_acceptance,
        show_policy_rank=args.show_policy_rank,
        store=store,
        episode_id=episode_id,
        seamless=io.supports_live_choices,
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
    sampling_override: SamplingConfig | None,
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
        _inherit_budget(store, episode_id, runtime, len(visible))
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
) -> dict[str, Any]:
    """Restore the destination sampler as well as its retained token prefix."""
    # Read before truncation removes the future sampler segments. Unlike a
    # fork, this engine keeps its original prefix and absolute boundaries, so
    # the stored coordinate offset must not have the boundary added to it.
    segment = store.sampling_segment(episode_id, boundary)
    sampling = SamplingConfig.from_record(segment["sampling"])
    engine.rewind_to(boundary)
    _inherit_budget(store, episode_id, engine, boundary)
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
        _inherit_budget(store, parent_id, engine, target, rebase=True)
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


def _sampler_summary(config: SamplingConfig) -> str:
    return (
        f"temp={config.temperature:g} top_k={config.top_k} top_p={config.top_p:g} "
        f"min_p={config.min_p:g} rep={config.repeat_penalty:g}/{config.repeat_last_n} "
        f"presence={config.presence_penalty:g} frequency={config.frequency_penalty:g} "
        f"seed={config.seed}"
    )


def _live_edge_menu(
    io: TerminalIO,
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
) -> tuple[str, Any]:
    live_surface = bool(
        getattr(io, "supports_live_choices", False)
        and callable(getattr(io, "read_live_edge_command", None))
    )
    while True:
        if live_surface:
            raw = io.read_live_edge_command(  # type: ignore[attr-defined]
                episode_id=store.label(episode_id),
                boundary=engine.boundary,
                current_budget=engine.max_tokens,
                remaining_tokens=engine.remaining,
                sampler_summary=_sampler_summary(engine.sampling),
            )
        else:
            io.write(store.label(episode_id))
            io.write("[ls / ls all] episodes  [#N] switch  [name TITLE] rename  [rewind N] delete back to N")
            io.write(
                f"\nLive edge @ boundary {engine.boundary} · {_sampler_summary(engine.sampling)}"
            )
            raw = io.read(
                "[c]ontinue  [n N/off] budget  [s key=value] sampler  "
                "([s random-seed] randomize)  [f N] fork  [fm] fork map  "
                "[spr ID [--until Y | m]] replay  [p]roject  [e]nd  [q]uit > "
            )
        if raw is None:
            return "quit", None
        text = raw.strip()
        lower = text.lower()
        if lower in {"ls", "ls all"}:
            io.page(store.workspace_list(include_finished=lower == "ls all", current=episode_id))
            selected = io.read("Episode #number (Enter returns)> ")
            if selected and selected.strip():
                try:
                    return "switch", store.resolve_id(selected.strip())
                except EditorError as exc:
                    io.write(str(exc))
            continue
        if lower.startswith("name "):
            store.rename(episode_id, text[5:])
            continue
        if lower.startswith("#"):
            try:
                return "switch", store.resolve_id(text)
            except EditorError as exc:
                io.write(str(exc))
            continue
        if lower.startswith("rewind "):
            try:
                target = int(text.split()[1])
                _rewind_episode(store, episode_id, engine, target)
                store.record_interaction(episode_id, target, "seamless-rewind", {"to_boundary": target})
            except (ValueError, EditorError) as exc:
                io.write(str(exc))
            continue
        if lower in {"q", "quit"}:
            return "quit", None
        if lower in {"e", "end"}:
            return "end", None
        if lower in {"c", "continue", ""}:
            return "continue", "keep"
        if lower in {"p", "project", "r", "review"}:
            io.page(
                project_episode(
                    store,
                    episode_id,
                    annotations="footnotes",
                    full_evidence=True,
                ).text
            )
            continue
        if lower in {"fm", "fork-map", "forkmap"}:
            io.page(project_fork_map(store, episode_id))
            while True:
                entered = io.read(
                    f"Fork boundary (0..{engine.boundary}; blank cancels) > "
                )
                if entered is None:
                    return "quit", None
                value = entered.strip()
                if not value:
                    break
                try:
                    target = int(value)
                except ValueError:
                    io.write("Fork boundary must be an integer.")
                    continue
                if not 0 <= target <= engine.boundary:
                    io.write(f"Fork boundary must be 0..{engine.boundary}.")
                    continue
                return "fork", target
            continue
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in {"n", "next"}:
            if parts[1].lower() in {"off", "none", "unlimited"}:
                return "continue", None
            try:
                budget = int(parts[1])
            except ValueError:
                io.write("Budget must be a positive integer.")
                continue
            if budget < 1:
                io.write("Budget must be at least 1.")
                continue
            return "continue", budget
        if parts and parts[0].lower() in {"s", "sampler"}:
            payload = parts[1] if len(parts) == 2 else ""
            if not payload:
                entered = io.read(
                    "sampler key=value changes (blank cancels; e.g. top_k=20 temperature=.8)> "
                )
                payload = entered or ""
            if not payload.strip():
                continue
            try:
                updated = _sampler_override(engine.sampling, payload)
            except EditorError as exc:
                io.write(f"[invalid sampler change] {exc}")
                continue
            engine.sampling = updated
            store.record_sampling_segment(
                episode_id,
                start_boundary=engine.boundary,
                sampling=updated,
                stream_fingerprint=engine.stream_fingerprint,
                coordinate_offset=engine.coordinate_offset,
            )
            store.record_interaction(
                episode_id,
                engine.boundary,
                "sampling-transition",
                {"sampling": updated.to_dict()},
            )
            continue
        if len(parts) == 2 and parts[0].lower() in {"f", "fork"}:
            try:
                target = int(parts[1])
            except ValueError:
                io.write("Fork boundary must be an integer.")
                continue
            if not 0 <= target <= engine.boundary:
                io.write(f"Fork boundary must be 0..{engine.boundary}.")
                continue
            return "fork", target
        if len(parts) == 2 and parts[0].lower() in {"spr", "replay"}:
            try:
                replay_args = parts[1].split()
                if not replay_args:
                    raise EditorError("Use spr EPISODE [--until Y | m].")
                source_id = store.resolve_id(replay_args[0])
                until = None
                if replay_args[1:] == ["m"]:
                    io.page("Source replay map (recorded output; replay may differ).\n"
                            "Source 0 inserts only the prompt; destination tokens remain individually indexed.\n"
                            + project_fork_map(store, source_id))
                    while True:
                        entered = io.read("Replay through source boundary (blank cancels) > ")
                        if entered is None or not entered.strip():
                            break
                        try:
                            until = int(entered)
                            store.replay_until(source_id, until)
                        except (ValueError, EditorError):
                            until = None
                            io.write("Choose a valid source token boundary.")
                            continue
                        break
                    if until is None:
                        continue
                elif len(replay_args) == 3 and replay_args[1] == "--until":
                    try:
                        until = int(replay_args[2])
                    except ValueError:
                        raise EditorError("Replay boundary must be an integer.") from None
                elif len(replay_args) != 1:
                    raise EditorError("Use spr EPISODE [--until Y | m].")
                source_id = recover_sampler_record(store, source_id, io)
                store.replay_until(source_id, until)
                return "spr", (source_id, until)
            except EditorError as exc:
                io.write(str(exc))
            continue
        io.write("Unknown live-edge command.")


def _seal(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    *,
    reason: str | None = None,
    output: Path | None = None,
) -> None:
    if reason is not None and not engine.ended:
        engine.terminate(reason)
    store.finish_episode(
        episode_id,
        visible_text=engine.backend.render(engine.visible_token_ids),
        terminal_token_id=engine.terminal_token_id,
        terminal_reason=engine.terminal_reason,
        status="completed",
    )
    text = engine.text
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Text: {output}", flush=True)
    else:
        print("\n--- final text ---")
        print(text)


def _print_list(store: EpisodeStore) -> None:
    print(store.workspace_list(include_finished=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    args._explicit_options = {
        action.dest for action in parser._actions
        if any(token.split("=", 1)[0] in action.option_strings for token in arguments)
    }
    try:
        if args.until is not None and args.replay is None:
            raise EditorError("--until requires --replay")
        if args.fixed_config and args.replay is None:
            raise EditorError("--fixed-config requires --replay")
        with EpisodeStore(args.workspace) as store:
            for field in ("resume", "fork_from", "replay", "project", "lineage"):
                value = getattr(args, field)
                if value:
                    setattr(args, field, store.resolve_id(value))
            if args.at is not None and args.fork_from is None:
                raise EditorError("--at is only valid with --fork-from")
            if args.procedure and not args.project:
                raise EditorError("--procedure requires --project EPISODE_ID")
            if args.lineage is not None:
                if not args.list_episodes:
                    raise EditorError("--lineage requires --list")
                print(project_lineage(store, args.lineage))
                return 0
            if args.list_episodes:
                _print_list(store)
                return 0
            if args.project:
                if args.procedure:
                    print(project_procedure(store, args.project))
                    return 0
                print(
                    project_episode(
                        store,
                        args.project,
                        annotations=args.annotations,
                        with_loss=args.with_loss,
                        with_rank=args.with_rank,
                        with_policy_rank=args.with_policy_rank,
                        full_evidence=args.full_evidence,
                        with_model_probs=args.with_model_probs,
                        with_lineage=args.with_lineage,
                    ).text
                )
                return 0
            if not any(
                (
                    args.new_prompt is not None,
                    args.new_prompt_file is not None,
                    args.replay is not None,
                    args.resume is not None,
                    args.fork_from is not None,
                )
            ):
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    raise EditorError(
                        "no episode source supplied; use --new-prompt, "
                        "--new-prompt-file, --replay, --resume, or --fork-from"
                    )

                args.new_prompt = _read_initial_prompt()

            if args.random_seed:
                args.seed = _random_seed()
                print(f"Random seed: {args.seed}", flush=True)

            io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
            for field in ("resume", "fork_from", "replay"):
                if getattr(args, field):
                    setattr(args, field, recover_sampler_record(store, getattr(args, field), io))
            source_id = args.resume or args.fork_from or args.replay
            source = store.get_episode(source_id) if source_id else None
            backend, provenance, model_changed = _load_episode_backend(args, source, io)
            requested_id = args.episode_id
            parent_id: str | None = None
            fork_boundary: int | None = None
            pending_tape: ReplayPlan | None = None

            if args.resume is not None:
                # Only explicit CLI sampler flags override the stored segment.
                explicit = any(getattr(args, name) is not None for name in SAMPLER_FIELDS)
                segment = store.sampling_segment(
                    args.resume, len(_visible_tokens(store, args.resume))
                )
                source_sampling = SamplingConfig.from_record(segment["sampling"])
                sampling = (
                    _sampling_from_args(args, source_sampling)
                    if explicit
                    else source_sampling
                )
                io.write("Restoring saved context...")
                if model_changed:
                    engine, episode_id = _model_continuation(store, args.resume, backend, provenance)
                    if args.max_tokens is not None:
                        engine.resume(max_tokens=args.max_tokens)
                    if explicit:
                        engine.sampling = sampling
                        store.record_sampling_segment(episode_id, start_boundary=0, sampling=sampling,
                            stream_fingerprint=engine.stream_fingerprint, coordinate_offset=0)
                else:
                    engine = _restore_engine(
                        store, args.resume, backend, max_tokens=args.max_tokens,
                        sampling_override=sampling if explicit else None,
                    )
                    episode_id = args.resume
            elif args.new_prompt is not None or args.new_prompt_file is not None:
                initial_text = (
                    args.new_prompt
                    if args.new_prompt is not None
                    else args.new_prompt_file.read_text(encoding="utf-8")
                )
                engine = EpisodeEngine(
                    backend,
                    sampling=_sampling_from_args(args),
                    max_tokens=args.max_tokens,
                    initial_text=initial_text,
                )
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                )
            elif args.replay is not None:
                source_segment = store.sampling_segment(args.replay, 0)
                source_sampling = SamplingConfig.from_record(source_segment["sampling"])
                overrides = {
                    name: getattr(args, name) for name in SAMPLER_FIELDS
                    if getattr(args, name) is not None
                }
                sampling = _sampling_from_args(args, source_sampling)
                replay_prefix = None
                if model_changed:
                    replay_prefix = backend.tokenize(store.get_episode(args.replay)["initial_text"], add_bos=True, special=True)
                engine, pending_tape = _spr_engine_from_source(
                    store,
                    args.replay,
                    backend,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    until=args.until,
                    follow_source_sampling=not args.fixed_config,
                    sampling_overrides=overrides,
                    initial_token_ids=replay_prefix,
                    stream_fingerprint=source_segment["stream_fingerprint"] if model_changed else None,
                    coordinate_offset=source_segment["coordinate_offset"] if model_changed else None,
                )
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                    parent_episode_id=args.replay,
                    fork_boundary=0,
                    mode="serial-policy-replay",
                    metadata={"spr_source": args.replay},
                )
            else:
                assert args.fork_from is not None
                source = store.get_episode(args.fork_from)
                visible = _visible_tokens(store, args.fork_from)
                target = len(visible) if args.at is None else args.at
                if not 0 <= target <= len(visible):
                    raise EditorError(f"fork boundary must be 0..{len(visible)}")
                prefix = [*source["initial_token_ids"], *visible[:target]]
                if model_changed:
                    retained = "".join(row["text"] for row in store.tokens(args.fork_from)
                                       if row["realized_visible"] and row["boundary"] < target)
                    prefix = backend.tokenize(source["initial_text"] + retained, add_bos=True, special=True)
                segment = store.sampling_segment(args.fork_from, target)
                source_sampling = SamplingConfig.from_record(segment["sampling"])
                explicit = any(getattr(args, name) is not None for name in SAMPLER_FIELDS)
                sampling = _sampling_from_args(args, source_sampling)
                engine = EpisodeEngine(
                    backend,
                    sampling=sampling,
                    max_tokens=source["max_tokens"] if args.max_tokens is None else args.max_tokens,
                    initial_text=backend.render(prefix, special=True),
                    initial_token_ids=prefix,
                    stream_fingerprint=segment["stream_fingerprint"],
                    coordinate_offset=segment["coordinate_offset"] + target,
                )
                if args.max_tokens is None:
                    _inherit_budget(store, args.fork_from, engine, target, rebase=True)
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                    parent_episode_id=args.fork_from,
                    fork_boundary=target,
                    mode="fork",
                )

            store.visit(episode_id)
            enter_edge = False
            while True:
                runner = EpisodeRunner(
                    engine,
                    store,
                    episode_id,
                    divergence_policy=args.divergence_policy,
                )
                try:
                    if enter_edge:
                        enter_edge = False
                        raise EdgeRequested()
                    result = runner.run(
                        tape=pending_tape,
                        live_policy=_interactive_policy(args, store, episode_id, io),
                        stop_after_tape=True,
                    )
                except EdgeRequested:
                    pending_tape = None
                    action, value = _live_edge_menu(io, store, episode_id, engine)
                except SeamlessRewindRequested as request:
                    from_boundary = engine.boundary
                    io.write(f"Restoring context at boundary {request.boundary}...")
                    details = _rewind_episode(
                        store, episode_id, engine, request.boundary
                    )
                    store.record_interaction(
                        episode_id,
                        request.boundary,
                        "seamless-rewind",
                        {
                            "from_boundary": from_boundary,
                            "to_boundary": request.boundary,
                            "trimmed_action": details["trimmed_action"],
                        },
                    )
                    pending_tape = None
                    continue
                except SeamlessEdgeRequested as request:
                    from_boundary = engine.boundary
                    io.write(f"Restoring context at boundary {request.boundary}...")
                    details = _rewind_episode(
                        store, episode_id, engine, request.boundary
                    )
                    store.record_interaction(
                        episode_id,
                        request.boundary,
                        "seamless-edge-open",
                        {
                            "from_boundary": from_boundary,
                            "to_boundary": request.boundary,
                            "trimmed_action": details["trimmed_action"],
                        },
                    )
                    pending_tape = None
                    action, value = _live_edge_menu(io, store, episode_id, engine)
                except ForkRequested as request:
                    action, value = "fork", request.boundary
                else:
                    if engine.ended:
                        _seal(store, episode_id, engine, output=args.output)
                        return 0
                    if result.outcomes and result.outcomes[-1].stop_reason == "replay-eog":
                        io.write("Replay encountered EOG; tape stopped, live edge reached.")
                    elif result.handed_off:
                        if result.handoff_reason:
                            io.write(result.handoff_reason)
                        io.write(
                            f"Execution handed off at boundary {engine.boundary}; live edge reached."
                        )
                    elif result.replay_exhausted:
                        io.write(
                            f"SPR route exhausted at boundary {engine.boundary}; live edge reached."
                        )
                    elif engine.checkpointed:
                        io.write(f"Checkpoint reached at boundary {engine.boundary}.")
                    pending_tape = None
                    action, value = _live_edge_menu(io, store, episode_id, engine)

                if action == "switch":
                    destination = str(value)
                    target_episode = store.get_episode(destination)
                    sealed = target_episode["status"] in {"completed", "failed"}
                    if sealed:
                        io.page(project_episode(store, destination).text)
                        reply = io.read("Finished episode. Fork from end? [y/N]> ")
                        if not reply or reply.strip().lower() not in {"y", "yes"}:
                            enter_edge = True
                            continue
                    store.update_episode(episode_id, visible_text=engine.backend.render(engine.visible_token_ids),
                                         max_tokens=engine.max_tokens, status="open")
                    store.record_budget(episode_id, engine.boundary, engine.max_tokens, engine.checkpoint_boundary)
                    try:
                        destination = recover_sampler_record(store, destination, io)
                        target_episode = store.get_episode(destination)
                        new_backend, new_provenance, changed = _load_episode_backend(args, target_episode, io, use_saved=True,
                            current_backend=backend, current_provenance=provenance)
                        if changed:
                            new_engine, destination = _model_continuation(store, destination, new_backend, new_provenance)
                        elif sealed:
                            visible = _visible_tokens(store, destination)
                            segment = store.sampling_segment(destination, len(visible))
                            new_engine = EpisodeEngine(new_backend,
                                initial_token_ids=[*target_episode["initial_token_ids"], *visible],
                                sampling=SamplingConfig.from_record(segment["sampling"]),
                                stream_fingerprint=segment["stream_fingerprint"],
                                coordinate_offset=segment["coordinate_offset"] + len(visible),
                                max_tokens=target_episode["max_tokens"])
                            _inherit_budget(store, destination, new_engine, len(visible), rebase=True)
                            destination = _create_episode(store, new_engine, backend_provenance=new_provenance,
                                parent_episode_id=destination, fork_boundary=len(visible), mode="fork")
                        else:
                            new_engine = _restore_engine(store, destination, new_backend,
                                max_tokens=None, sampling_override=None)
                    except (EditorError, OSError, RuntimeError) as exc:
                        # A reused backend may already have been repositioned.
                        engine.backend.reset(engine.token_ids)
                        io.write(str(exc))
                        enter_edge = True
                        continue
                    engine, backend, provenance, episode_id = new_engine, new_backend, new_provenance, destination
                    store.visit(episode_id)
                    pending_tape = None
                    enter_edge = True
                    continue
                if action == "quit":
                    store.update_episode(
                        episode_id,
                        visible_text=engine.backend.render(engine.visible_token_ids),
                        max_tokens=engine.max_tokens,
                        status="open",
                    )
                    print(
                        f"Episode {store.label(episode_id)} remains unsealed. Resume with --resume '{store.label(episode_id).split()[0]}'",
                        flush=True,
                    )
                    return 0
                if action == "end":
                    _seal(store, episode_id, engine, reason="menu-end", output=args.output)
                    return 0
                if action == "continue":
                    checkpoint_transition = engine.checkpointed or value != "keep"
                    engine.resume(max_tokens=value)
                    store.record_budget(episode_id, engine.boundary, engine.max_tokens, engine.checkpoint_boundary)
                    store.record_interaction(
                        episode_id,
                        engine.boundary,
                        "checkpoint-resume" if checkpoint_transition else "edge-continue",
                        {"next_max_tokens": engine.max_tokens},
                    )
                    pending_tape = None
                    continue
                if action == "fork":
                    target = int(value)
                    parent_id = episode_id
                    parent_engine = engine
                    engine = _fork_engine(
                        store,
                        parent_id,
                        parent_engine,
                        target,
                        backend=backend,
                        max_tokens=None,
                    )
                    episode_id = _create_episode(
                        store,
                        engine,
                        backend_provenance=provenance,
                        parent_episode_id=parent_id,
                        fork_boundary=target,
                        mode="fork",
                    )
                    pending_tape = None
                    continue
                if action == "spr":
                    source_id, until = value
                    # Snapshot before appending, including self-replay. EDGE
                    # composition inserts the source prompt as literal text;
                    # CLI replay continues to use it as initial context.
                    source_prompt = store.get_episode(source_id)["initial_text"]
                    source_steps = store.replay_until(source_id, until)
                    steps = []
                    if source_prompt:
                        steps.append(TapeStep(
                            Write(source_prompt, "exact"), None,
                            source_episode_id=source_id, source_boundary=0,
                            source_part="prompt",
                        ))
                    steps.extend(
                        TapeStep(
                            step["action"], step["expectation"],
                            source_episode_id=source_id,
                            source_boundary=step["boundary"],
                        )
                        for step in source_steps
                    )
                    pending_tape = ReplayPlan(tuple(steps), follow_source_sampling=False)
                    store.record_interaction(
                        episode_id, engine.boundary, "replay-start",
                        {"source_episode_id": source_id, "action_count": len(steps), "until": until},
                    )
                    # Keep the existing engine and ledger: boundary zero,
                    # prefix, sampler stream, and remaining budget do not move.
                    continue
                raise AssertionError(f"unhandled live-edge action {action!r}")
    except (EditorError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
