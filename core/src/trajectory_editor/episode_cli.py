"""Command line for the reduced Serial Policy Editor.

A live episode alternates between the ordinary teacher loop and live edges.
Token budgets, replay exhaustion, and replay divergence yield at a live edge;
only EOG or explicit ``end`` seals the episode.
"""

from __future__ import annotations

import argparse
import math
from contextlib import ExitStack
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Literal

from . import (
    episode_backend_loader,
    episode_prompts,
    session_runtime,
)
from .backend_factory import BACKEND_NAMES
from .decoder import KV_CACHE_TYPES
from .core.errors import EditorError
from .core.cli_config import (
    CORE_SAMPLER_FIELDS,
    add_core_sampler_arguments,
    apply_activation_artifact,
    random_seed,
    sampler_from_args,
    sampler_overrides_present,
)
from .core.sampler_config import SamplerConfig
from .episode_lifecycle import (
    POLICY_FIELDS, _inherit_budget, _visible_tokens, _restore_engine,
    _spr_engine_from_source,
)
from .episode_engine import EpisodeEngine
from .episode_replay_source import build_source_replay_recipe
from .spr_recipe import ReplayControlPolicy, ReplayPlacement, compose_replay_plan
from .projector import (
    project_episode,
    project_lineage,
    project_procedure,
)
from .episode_store import EpisodeStore
from .controller_profiles import (
    explicit_option_dests,
    load_controller_profile,
    profile_arguments,
)
from .tui import TerminalIO
from .ui_themes import LIVE_THEME_NAMES
from .version import VERSION
from .teacher_plan import TeacherTape, export_teacher_tape, load_teacher_tape_jsonl


@dataclass(frozen=True)
class LaunchSource:
    """Validated source for starting a live episode."""

    kind: Literal["new", "resume", "replay", "fork"]
    episode_id: str | None = None
    teacher_tape: TeacherTape | None = None


def _select_launch_source(args: argparse.Namespace) -> LaunchSource:
    """Validate launch flags and load any external teacher plan."""

    if args.until is not None and args.replay is None:
        raise EditorError("--until requires --replay")
    if args.fixed_config and args.replay is None:
        raise EditorError("--fixed-config requires --replay")
    if (
        args.teacher_plan_envelope is not None
        and args.teacher_plan is None
        and args.export_teacher_plan is None
    ):
        raise EditorError("--teacher-plan-envelope requires --teacher-plan or --export-teacher-plan")
    if args.export_teacher_plan is not None and any(
        value is not None
        for value in (
            args.new_prompt, args.new_prompt_file, args.replay,
            args.resume, args.fork_from, args.projector,
        )
    ):
        raise EditorError("--export-teacher-plan cannot be combined with an episode source")

    teacher_tape = None
    if args.teacher_plan is not None:
        if any(value is not None for value in (args.replay, args.resume, args.fork_from)):
            raise EditorError("--teacher-plan currently starts a new episode only")
        teacher_tape = load_teacher_tape_jsonl(
            args.teacher_plan,
            envelope_path=args.teacher_plan_envelope,
            require_observations=args.divergence_policy == "handoff",
        )
        if args.new_prompt is None and args.new_prompt_file is None:
            prompt_text = teacher_tape.envelope.get("prompt")
            if not isinstance(prompt_text, str):
                raise EditorError("--teacher-plan requires --new-prompt, --new-prompt-file, or an envelope prompt")
            args.new_prompt = prompt_text

    if args.ephemeral:
        incompatible = {
            "--resume": args.resume, "--replay": args.replay,
            "--fork-from": args.fork_from, "--projector": args.projector,
            "--export-teacher-plan": args.export_teacher_plan,
            "--list": args.list_episodes, "--lineage": args.lineage,
        }
        requested = next((flag for flag, value in incompatible.items() if value), None)
        if requested is not None:
            raise EditorError(f"--ephemeral cannot be combined with {requested}")
        if args.at is not None or args.until is not None or args.fixed_config or args.procedure:
            raise EditorError("--ephemeral accepts a new prompt and optional --teacher-plan only")

    if args.resume is not None:
        return LaunchSource("resume", args.resume, teacher_tape)
    if args.new_prompt is not None or args.new_prompt_file is not None:
        return LaunchSource("new", teacher_tape=teacher_tape)
    if args.replay is not None:
        return LaunchSource("replay", args.replay, teacher_tape)
    if args.fork_from is not None:
        return LaunchSource("fork", args.fork_from, teacher_tape)
    return LaunchSource("new", teacher_tape=teacher_tape)


def _resolve_launch_source(
    args: argparse.Namespace, store: EpisodeStore, selection: LaunchSource,
) -> LaunchSource:
    """Resolve stored IDs without opening any terminal input."""

    for field in ("resume", "fork_from", "replay", "projector", "lineage"):
        value = getattr(args, field)
        if value:
            setattr(args, field, store.resolve_id(value))
    if args.at is not None and args.fork_from is None:
        raise EditorError("--at is only valid with --fork-from")
    if args.procedure and not args.projector:
        raise EditorError("--procedure requires --projector EPISODE_ID")
    # Backend provenance historically prefers resume, then fork, then replay
    # even when a prompt is also supplied. Keep that lookup independent of the
    # execution branch chosen below.
    source_id = args.resume or args.fork_from or args.replay
    return LaunchSource(selection.kind, source_id, selection.teacher_tape)


def _needs_launch_prompt(args: argparse.Namespace, selection: LaunchSource) -> bool:
    return selection.kind == "new" and not any((
        args.new_prompt is not None, args.new_prompt_file is not None,
        args.teacher_plan is not None, args.list_episodes,
        args.lineage is not None, args.export_teacher_plan is not None,
        args.projector is not None,
    ))


def _validate_prompt_source(args: argparse.Namespace, selection: LaunchSource) -> None:
    if _needs_launch_prompt(args, selection) and (
        not sys.stdin.isatty() or not sys.stdout.isatty()
    ):
        raise EditorError(
            "no episode source supplied; use --new-prompt, "
            "--new-prompt-file, --replay, --resume, or --fork-from"
        )


def _collect_launch_prompt(args: argparse.Namespace, selection: LaunchSource,
                           io: TerminalIO) -> None:
    if _needs_launch_prompt(args, selection):
        prompt = episode_prompts.read_new_prompt(io)
        if prompt is None:
            raise EditorError("prompt entry cancelled")
        args.new_prompt = prompt


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def build_parser(
    *,
    include_vector: bool = True,
    prog: str = "policy-editor",
) -> argparse.ArgumentParser:
    """Build the core episode parser."""

    parser = argparse.ArgumentParser(
        prog=prog,
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
        help="saved episodes used for SPR, resume, and explicit EDGE saves",
    )
    parser.add_argument(
        "--ephemeral", action="store_true",
        help="run a non-durable live session; export or save a branch explicitly",
    )
    parser.add_argument(
        "--teacher-plan", type=Path, metavar="FILE",
        help="execute a portable JSONL teacher tape from a new prompt",
    )
    parser.add_argument(
        "--teacher-plan-envelope", type=Path, metavar="FILE",
        help="optional JSON sidecar describing a portable teacher tape",
    )
    parser.add_argument(
        "--export-teacher-plan", nargs=2, type=Path, metavar=("EPISODE_ID", "FILE"),
        help="export an episode as a portable JSONL teacher tape",
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
        help="continue an unsealed/checkpointed saved episode in a live session",
    )
    parser.add_argument(
        "--fixed-config", action="store_true",
        help="freeze source-initial sampler settings plus explicit overrides during --replay",
    )
    source.add_argument("--fork-from", metavar="EPISODE_ID")
    source.add_argument("--projector", metavar="EPISODE_ID")
    parser.add_argument("--procedure", action="store_true", help="show a manual replay procedure with --projector")
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
        help="write final or projected episode text here",
    )

    parser.add_argument("--backend", choices=BACKEND_NAMES, default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        metavar="FILE",
        help="load reusable CLI values from a YAML controller profile",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="optional visible-token allowance before a live edge (default: unlimited)",
    )
    parser.add_argument("--table-depth", type=int, default=12)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--hold-default", type=int, default=100)
    parser.add_argument(
        "--phrase-max-tokens",
        type=_positive_int,
        default=16,
        help="maximum tokenized length for check/force phrase commands",
    )
    parser.add_argument(
        "--phrase-max-shift",
        type=_nonnegative_float,
        default=6.0,
        help="policy-logit shift bound used by check phrase commands",
    )
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
        "--manual-acceptance",
        action="store_true",
        help="leave each teacher command blank instead of prefilling the sampled proposal",
    )
    policy_view = parser.add_mutually_exclusive_group()
    policy_view.add_argument(
        "--policy-view", "--show-policy-rank", dest="show_policy_rank", action="store_true",
        default=None, help="show policy diagnostics without changing backend-rank ordering (default: off)",
    )
    policy_view.add_argument(
        "--no-policy-view", dest="show_policy_rank", action="store_false",
        help="hide policy diagnostics; V can toggle them during the session",
    )
    parser.add_argument(
        "--logit-view",
        choices=("none", "raw", "gap", "both"),
        default="none",
        help=(
            "show model logits, model gap from raw rank 1, or both "
            "(default: none; l cycles, L toggles both)"
        ),
    )
    parser.add_argument(
        "--show-model-probabilities",
        action="store_true",
        default=False,
        help=(
            "include soft-max %% overlays (raw-p / decode-p [/ pol-p]); "
            "default identity-only table; %% toggles during the session"
        ),
    )
    parser.add_argument("--theme", choices=LIVE_THEME_NAMES)
    parser.add_argument(
        "--divergence-policy", choices=("handoff", "ballistic"), default="handoff"
    )

    add_core_sampler_arguments(parser, include_vector=include_vector)

    llama = parser.add_argument_group("llama.cpp")
    llama.add_argument("--n-ctx", type=int, default=2048)
    llama.add_argument("--n-batch", type=int, default=256)
    llama.add_argument("--n-ubatch", type=int)
    llama.add_argument("--n-threads", type=int)
    llama.add_argument("--n-threads-batch", type=int)
    llama.add_argument("--n-gpu-layers", type=int)
    llama.add_argument("--main-gpu", type=int)
    llama.add_argument("--no-flash-attn", action="store_true")
    for component in ("k", "v"):
        llama.add_argument(f"--cache-type-{component}", dest=f"type_{component}",
            choices=KV_CACHE_TYPES, help="KV cache precision (default: library default; quantized V requires Flash Attention)")
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

    return parser


def _write_text(text: str, *, output: Path | None = None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Text: {output}", flush=True)
    else:
        print("\n--- final text ---")
        print(text)


def _print_list(store: EpisodeStore) -> None:
    print(store.workspace_list(include_finished=True))


def main(
    argv: list[str] | None = None,
    *,
    include_vector: bool | None = None,
    prog: str = "policy-editor",
) -> int:
    """Run the core episode CLI."""

    if include_vector is None:
        # Proper steering-vector loading is the one optional extension exposed
        # by the normal runtime surface; research controls remain separate.
        from importlib.util import find_spec

        include_vector = find_spec(
            f"{__package__}.activation_vectors"
        ) is not None
    parser = build_parser(
        include_vector=include_vector,
        prog=prog,
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        cli_explicit = explicit_option_dests(parser, arguments)
        profile_values: dict[str, Any] = {}
        profile_applied: set[str] = set()
        if args.profile is not None:
            profile_values, _ = load_controller_profile(args.profile, parser)
            profile_tokens, profile_applied = profile_arguments(
                parser,
                profile_values,
                overridden=cli_explicit,
            )
            args = parser.parse_args([*profile_tokens, *arguments])
        args._explicit_options = cli_explicit | profile_applied
        sampling_factory = SamplerConfig.from_record
        selection = _select_launch_source(args)
        _validate_prompt_source(args, selection)
        teacher_tape = selection.teacher_tape
        if args.ephemeral:
            if args.random_seed:
                args.seed = random_seed()
                print(f"Random seed: {args.seed}", flush=True)
            io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
            with io.session():
                _collect_launch_prompt(args, selection, io)
                return session_runtime.run_new_session(
                    args,
                    io=io,
                    teacher_tape=teacher_tape,
                )
        with EpisodeStore(args.workspace) as store, ExitStack() as ui_stack:
            selection = _resolve_launch_source(args, store, selection)
            if args.lineage is not None:
                if not args.list_episodes:
                    raise EditorError("--lineage requires --list")
                print(project_lineage(store, args.lineage))
                return 0
            if args.list_episodes:
                _print_list(store)
                return 0
            if args.export_teacher_plan is not None:
                source, destination = args.export_teacher_plan
                export_teacher_tape(
                    store,
                    store.resolve_id(str(source)),
                    destination,
                    envelope_path=args.teacher_plan_envelope,
                )
                print(f"Exported teacher tape to {destination}", flush=True)
                return 0
            if args.projector:
                if args.procedure:
                    print(project_procedure(store, args.projector))
                    return 0
                projection = project_episode(
                    store,
                    args.projector,
                    annotations=getattr(args, "annotations", "none"),
                    with_loss=getattr(args, "with_loss", False),
                    with_rank=getattr(args, "with_rank", False),
                    with_policy_rank=getattr(args, "with_policy_rank", False),
                    full_evidence=getattr(args, "full_evidence", False),
                    with_model_probs=getattr(args, "with_model_probs", False),
                    with_lineage=getattr(args, "with_lineage", False),
                )
                _write_text(projection.text, output=args.output)
                return 0
            if args.random_seed:
                args.seed = random_seed()
                print(f"Random seed: {args.seed}", flush=True)

            io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
            ui_stack.enter_context(io.session())
            _collect_launch_prompt(args, selection, io)
            source_id = selection.episode_id
            source = store.get_episode(source_id) if source_id else None
            backend, provenance, model_changed = (
                episode_backend_loader.load_episode_backend(args, source, io)
            )
            cfg_guidance_backend = None
            cfg_primary_backend = None

            def cfg_backend_for(sampling, *, plan=None, historical_sampling=(),
                                primary=None, model_provenance=None):
                nonlocal cfg_guidance_backend, cfg_primary_backend
                primary = backend if primary is None else primary
                model_provenance = provenance if model_provenance is None else model_provenance
                if not episode_backend_loader.cfg_required(
                    sampling, plan=plan, historical_sampling=historical_sampling
                ):
                    return cfg_guidance_backend if cfg_primary_backend is primary else None
                if cfg_guidance_backend is None or cfg_primary_backend is not primary:
                    io.write("Loading second model copy for CFG prefix guidance...")
                    cfg_guidance_backend = (
                        episode_backend_loader.load_cfg_guidance_backend(
                            args,
                            model_provenance,
                        )
                    )
                    cfg_primary_backend = primary
                return cfg_guidance_backend

            from uuid import uuid4

            from .episode_identity import backend_provenance_with_identity
            from .episode_live_restore import model_change_session, new_live_session, restore_live_session
            from .episode_session import BranchIdentity, LiveSession, LiveSessionRoster

            args._model_changed = model_changed
            saved_tokenizer_id = source["backend"].get("tokenizer_id") if source is not None else None
            destination_tokenizer_id = provenance.get("tokenizer_id")
            tokenizer_changed = model_changed and (
                not isinstance(saved_tokenizer_id, str)
                or not isinstance(destination_tokenizer_id, str)
                or saved_tokenizer_id != destination_tokenizer_id
            )
            if tokenizer_changed:
                args.bias_rules = ()
                args.bias_groups = ()
                io.write("Tokenizer changed: token-ID biases reset; load a matching preset to apply biases.")

            activation_artifact = None
            if args.activation_strength is not None and args.activation_vector is None:
                raise EditorError("--steering-strength requires --steering-vector")
            if args.activation_vector is not None:
                from .activation_vectors import SteeringVectorArtifact
                activation_artifact = SteeringVectorArtifact.from_path(args.activation_vector)
                activation_artifact.validate_against_backend(backend)
            provenance = backend_provenance_with_identity(backend, provenance)
            backend_state = {"backend": backend, "provenance": provenance}
            session = None
            pending_tape = teacher_tape.plan if teacher_tape else None
            default_save_id = args.episode_id

            if selection.kind == "resume":
                episode_id = str(args.resume)
                segment = store.current_sampling_state(episode_id)
                source_sampling = sampling_factory(segment["sampling"])
                explicit = sampler_overrides_present(args)
                sampling = sampler_from_args(args, source_sampling) if explicit else source_sampling
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                explicit = sampling != source_sampling
                io.write("Restoring saved context...")
                visible = _visible_tokens(store, episode_id)
                historical_sampling = tuple(
                    sampling_factory(row["sampling"]) for row in store.sampler_segments(episode_id)
                )
                guidance = cfg_backend_for(sampling, historical_sampling=historical_sampling)
                if model_changed:
                    session = model_change_session(
                        store, episode_id, backend, provenance,
                        boundary=len(visible), sampling=sampling,
                        max_tokens=args.max_tokens, guidance_backend=guidance,
                    )
                else:
                    engine = _restore_engine(
                        store, episode_id, backend, max_tokens=args.max_tokens,
                        sampling_override=sampling if explicit else None,
                        guidance_backend=guidance,
                        sampling_factory=sampling_factory,
                        current_sampling_state=segment,
                    )
                    session = restore_live_session(
                        store, episode_id, engine,
                        branch_identity=BranchIdentity(
                            f"live-resume-{uuid4().hex}", episode_id, len(visible)
                        ),
                    )
            elif selection.kind == "new":
                initial_text = args.new_prompt if args.new_prompt is not None else episode_prompts.read_prompt_file(args.new_prompt_file)
                sampling = apply_activation_artifact(sampler_from_args(args), activation_artifact, args)
                engine = EpisodeEngine(
                    backend, sampling=sampling, max_tokens=args.max_tokens,
                    initial_text=initial_text,
                    guidance_backend=cfg_backend_for(sampling, plan=pending_tape),
                )
                session = LiveSession(
                    engine, prompt=initial_text,
                    environment_stamp={"backend": provenance, "sampler": sampling.to_dict()},
                )
            elif selection.kind == "replay":
                recipe = build_source_replay_recipe(
                    store, str(args.replay), args.until, sampling_factory=sampling_factory
                )
                source_sampling = recipe.controls.effective_at(0).sampling
                overrides = {
                    name: getattr(args, name) for name in CORE_SAMPLER_FIELDS
                    if getattr(args, name) is not None
                }
                sampling = apply_activation_artifact(
                    sampler_from_args(args, source_sampling), activation_artifact, args
                )
                if activation_artifact is not None:
                    for name in POLICY_FIELDS:
                        overrides[name] = getattr(sampling, name)
                replay_prefix = (
                    backend.tokenize(recipe.source_prompt, add_bos=True, special=True)
                    if model_changed else list(source["initial_token_ids"])
                )
                replay_control_policy = (
                    ReplayControlPolicy.PRESERVE_DESTINATION if args.fixed_config
                    else ReplayControlPolicy.FOLLOW_SOURCE
                )
                pending_tape = compose_replay_plan(
                    recipe,
                    ReplayPlacement.SOURCE_ROOT,
                    replay_control_policy,
                    sampler_overrides=overrides,
                )
                engine, pending_tape = _spr_engine_from_source(
                    recipe, backend, sampling=sampling, max_tokens=args.max_tokens,
                    control_policy=replay_control_policy,
                    sampling_overrides=overrides, initial_token_ids=replay_prefix,
                    guidance_backend=cfg_backend_for(sampling, plan=pending_tape),
                )
                session = new_live_session(
                    engine, prompt=recipe.source_prompt, provenance=provenance,
                    branch_identity=BranchIdentity(
                        f"live-replay-{uuid4().hex}", str(args.replay), 0
                    ),
                    source_episode_id=str(args.replay),
                )
            else:
                assert args.fork_from is not None
                episode_id = str(args.fork_from)
                source = store.get_episode(episode_id)
                visible = _visible_tokens(store, episode_id)
                target = len(visible) if args.at is None else args.at
                if not 0 <= target <= len(visible):
                    raise EditorError(f"fork boundary must be 0..{len(visible)}")
                segment = store.sampling_segment(episode_id, target)
                source_sampling = sampling_factory(segment["sampling"])
                sampling = apply_activation_artifact(
                    sampler_from_args(args, source_sampling), activation_artifact, args
                )
                history = tuple(
                    sampling_factory(row["sampling"])
                    for row in store.sampler_segments(episode_id)
                    if int(row["start_boundary"]) <= target
                )
                guidance = cfg_backend_for(sampling, historical_sampling=history)
                if model_changed:
                    session = model_change_session(
                        store, episode_id, backend, provenance,
                        boundary=target, sampling=sampling,
                        max_tokens=args.max_tokens, guidance_backend=guidance,
                    )
                else:
                    prefix = [*source["initial_token_ids"], *visible[:target]]
                    branch = getattr(backend, "branch_to_prefix", None)
                    branch(prefix) if callable(branch) else backend.reset(prefix)
                    engine = EpisodeEngine(
                        backend, sampling=sampling,
                        max_tokens=source["max_tokens"] if args.max_tokens is None else args.max_tokens,
                        initial_text=str(source["initial_text"]),
                        initial_token_ids=source["initial_token_ids"],
                        stream_fingerprint=segment["stream_fingerprint"],
                        backend_positioned=True, guidance_backend=guidance,
                    )
                    engine.visible_token_ids = list(visible[:target])
                    if args.max_tokens is None:
                        _inherit_budget(store, episode_id, engine, target)
                    else:
                        engine.trajectory.set_budget(args.max_tokens, target + args.max_tokens)
                    session = restore_live_session(
                        store, episode_id, engine, boundary=target,
                        branch_identity=BranchIdentity(
                            f"live-fork-{uuid4().hex}", episode_id, target
                        ),
                    )

            assert session is not None

            def load_saved_session(episode_id: str) -> LiveSession:
                target_episode = store.get_episode(episode_id)
                new_backend, new_provenance, changed = episode_backend_loader.load_episode_backend(
                    args, target_episode, io, use_saved=True,
                    current_backend=backend_state["backend"],
                    current_provenance=backend_state["provenance"],
                )
                new_provenance = backend_provenance_with_identity(new_backend, new_provenance)
                visible = _visible_tokens(store, episode_id)
                boundary = len(visible)
                state = store.current_sampling_state(episode_id)
                sampling = sampling_factory(state["sampling"])
                historical = tuple(
                    sampling_factory(row["sampling"]) for row in store.sampler_segments(episode_id)
                )
                guidance = cfg_backend_for(
                    sampling, primary=new_backend, model_provenance=new_provenance,
                    historical_sampling=historical,
                )
                if changed:
                    loaded = model_change_session(
                        store, episode_id, new_backend, new_provenance,
                        boundary=boundary, sampling=sampling, max_tokens=None,
                        guidance_backend=guidance,
                    )
                elif target_episode["status"] in {"completed", "failed"}:
                    saved_tokenizer = target_episode["backend"].get("tokenizer_id")
                    if isinstance(saved_tokenizer, str) and saved_tokenizer != new_provenance.get("tokenizer_id"):
                        raise EditorError(
                            "episode tokenizer identity differs from the loaded backend; use a model-change continuation"
                        )
                    prefix = [*target_episode["initial_token_ids"], *visible]
                    branch = getattr(new_backend, "branch_to_prefix", None)
                    branch(prefix) if callable(branch) else new_backend.reset(prefix)
                    engine = EpisodeEngine(
                        new_backend, sampling=sampling,
                        max_tokens=target_episode["max_tokens"] or None,
                        initial_text=str(target_episode["initial_text"]),
                        initial_token_ids=target_episode["initial_token_ids"],
                        stream_fingerprint=state["stream_fingerprint"],
                        backend_positioned=True, guidance_backend=guidance,
                    )
                    engine.visible_token_ids = list(visible)
                    _inherit_budget(store, episode_id, engine, boundary)
                    loaded = restore_live_session(
                        store, episode_id, engine,
                        branch_identity=BranchIdentity(
                            f"live-resume-{uuid4().hex}", episode_id, boundary
                        ),
                    )
                else:
                    engine = _restore_engine(
                        store, episode_id, new_backend, max_tokens=None,
                        sampling_override=None, guidance_backend=guidance,
                        sampling_factory=sampling_factory, current_sampling_state=state,
                    )
                    loaded = restore_live_session(
                        store, episode_id, engine,
                        branch_identity=BranchIdentity(
                            f"live-resume-{uuid4().hex}", episode_id, boundary
                        ),
                    )
                backend_state.update(backend=new_backend, provenance=new_provenance)
                return loaded

            return session_runtime.run_session_roster(
                args, io=io, roster=LiveSessionRoster(session),
                backend_provenance=provenance, teacher_tape=teacher_tape,
                initial_tape=pending_tape,
                store=store, load_saved_session=load_saved_session,
                default_save_id=default_save_id,
            )
    except (EditorError, OSError, RuntimeError, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
