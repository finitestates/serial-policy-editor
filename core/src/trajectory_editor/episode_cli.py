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
    edge_commands,
    ephemeral_runtime,
    episode_backend_loader,
    episode_policy_setup,
    episode_prompts,
)
from .backend_factory import BACKEND_NAMES
from .chord import ActionSequencePolicy, Chord, ChordRequested, chord_menu
from .decoder import KV_CACHE_TYPES
from .core.errors import EditorError
from .core.cli_config import (
    CORE_SAMPLER_FIELDS,
    add_core_sampler_arguments,
    apply_activation_artifact,
    random_seed,
    sampler_from_args,
    sampler_override,
    sampler_overrides_present,
)
from .core.sampler_config import SamplerConfig
from .episode_lifecycle import (
    POLICY_FIELDS, _inherit_budget, _materialize_model_change_fork,
    _model_continuation, _visible_tokens, _restore_engine,
    _create_episode, _rewind_episode, _fork_engine, _spr_engine_from_source,
)
from .episode_engine import EpisodeEngine
from .episode_replay_source import build_source_replay_recipe
from .episode_runner import (
    EdgeRequested,
    EpisodeRunner as CoreEpisodeRunner,
    ForkRequested,
    SeamlessRewindRequested,
    ReplayPlan,
)
from .spr_recipe import (
    ReplayControlPolicy,
    ReplayPlacement,
    compose_replay_plan,
)
from .projector import (
    project_episode,
    project_fork_map,
    project_lineage,
    project_procedure,
)
from .episode_store import EpisodeStore
from .edge_status import sampler_summary
from .fresh_episode import fresh_root_from
from .controller_profiles import (
    explicit_option_dests,
    load_controller_profile,
    profile_arguments,
)
from .tui import TerminalIO
from .terminal_contracts import EdgeViewState, PromptRequest
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
        help="compact episode workspace used for SPR and token evidence",
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
        help="resume an unsealed/checkpointed episode in the same episode id",
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


def _record_fork_edge_state(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
) -> None:
    """Record the child's active controls at its root-relative fork edge."""
    store.record_sampling_segment(
        episode_id,
        start_boundary=engine.boundary,
        sampling=engine.sampling,
        stream_fingerprint=engine.stream_fingerprint,
        coordinate_offset=engine.coordinate_offset,
    )
    store.record_budget(
        episode_id,
        engine.boundary,
        engine.max_tokens,
        engine.checkpoint_boundary,
    )


def _live_edge_menu(
    io: TerminalIO,
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    *,
    sampling_factory=SamplerConfig.from_record,
) -> tuple[str, Any]:
    while True:
        raw = io.read_edge(EdgeViewState(
            episode_id=store.label(episode_id),
            boundary=engine.boundary,
            current_budget=engine.max_tokens,
            remaining_tokens=engine.remaining,
            sampler_summary=sampler_summary(engine.sampling),
        ))
        if raw is None:
            return "quit", None
        try:
            command = edge_commands.parse_edge_command(raw)
        except edge_commands.EdgeCommandParseError as exc:
            io.write(str(exc))
            continue
        if isinstance(command, edge_commands.ListCommand):
            selected = io.prompt(PromptRequest(
                "Episode #number (Enter returns)> ",
                body=store.workspace_list(
                    include_finished=command.include_finished,
                    current=episode_id,
                ),
            ))
            if selected and selected.strip():
                try:
                    return "switch", store.resolve_id(selected.strip())
                except EditorError as exc:
                    io.write(str(exc))
            continue
        if isinstance(command, edge_commands.RenameCommand):
            store.rename(episode_id, command.title)
            continue
        if isinstance(command, edge_commands.SwitchCommand):
            try:
                return "switch", store.resolve_id(command.reference)
            except EditorError as exc:
                io.write(str(exc))
            continue
        if isinstance(command, edge_commands.RewindCommand):
            try:
                target = command.boundary
                _rewind_episode(
                    store, episode_id, engine, target,
                    sampling_factory=sampling_factory,
                )
                store.record_interaction(episode_id, target, "seamless-rewind", {"to_boundary": target})
            except (ValueError, EditorError) as exc:
                io.write(str(exc))
            continue
        if isinstance(command, edge_commands.QuitCommand):
            return "quit", None
        if isinstance(command, edge_commands.EndCommand):
            return "end", None
        if isinstance(command, edge_commands.ContinueCommand):
            return "continue", "keep"
        if isinstance(command, edge_commands.NewCommand):
            prompt_text = command.prompt if command.prompt else episode_prompts.read_new_prompt(io)
            if prompt_text is None:
                continue
            if not prompt_text:
                io.write("New prompt must not be empty.")
                continue
            return "new", prompt_text
        if isinstance(command, edge_commands.ProjectCommand):
            io.page(
                project_episode(
                    store,
                    episode_id,
                    annotations="footnotes",
                    full_evidence=True,
                ).text
            )
            continue
        if isinstance(command, edge_commands.ForkMapCommand):
            fork_map = project_fork_map(store, episode_id)
            while True:
                entered = io.prompt(PromptRequest(
                    f"Fork boundary (0..{engine.boundary}; blank cancels) > ",
                    body=fork_map,
                ))
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
        if isinstance(command, edge_commands.BudgetCommand):
            return "continue", command.tokens
        if isinstance(command, edge_commands.SamplerCommand):
            payload = command.text
            if payload is None:
                entered = io.read(
                    "sampler key=value changes (blank cancels; e.g. top_k=20 temperature=.8)> "
                )
                payload = entered or ""
            if not payload.strip():
                continue
            try:
                updated = sampler_override(engine.sampling, payload)
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
        if isinstance(command, edge_commands.ForkCommand):
            target = command.boundary
            if not 0 <= target <= engine.boundary:
                io.write(f"Fork boundary must be 0..{engine.boundary}.")
                continue
            return "fork", target
        if isinstance(command, edge_commands.ReplaySelectionCommand):
            try:
                source_id = store.resolve_id(command.source)
                until = None
                replay_map = (
                    "Source replay map (recorded output; replay may differ).\n"
                    "Source 0 inserts only the prompt; destination tokens remain individually indexed.\n"
                    + project_fork_map(store, source_id)
                )
                while True:
                    entered = io.prompt(PromptRequest(
                        "Replay through source boundary (blank cancels) > ",
                        body=replay_map,
                    ))
                    if entered is None or not entered.strip():
                        break
                    try:
                        until = int(entered)
                        build_source_replay_recipe(
                            store,
                            source_id,
                            until,
                            sampling_factory=sampling_factory,
                        )
                    except (ValueError, EditorError):
                        until = None
                        io.write("Choose a valid source token boundary.")
                        continue
                    break
                if until is None:
                    continue
                build_source_replay_recipe(
                    store,
                    source_id,
                    until,
                    sampling_factory=sampling_factory,
                )
                return "spr", (source_id, until)
            except EditorError as exc:
                io.write(str(exc))
            continue
        if isinstance(command, edge_commands.ReplayCommand):
            try:
                source_id = store.resolve_id(command.source)
                build_source_replay_recipe(
                    store,
                    source_id,
                    command.until,
                    sampling_factory=sampling_factory,
                )
                return "spr", (source_id, command.until)
            except EditorError as exc:
                io.write(str(exc))
            continue
        io.write("This command is not available at a durable EDGE.")


def _seal(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    *,
    reason: str,
    output: Path | None = None,
) -> None:
    engine.terminate(reason)
    store.finish_episode(
        episode_id,
        visible_text=engine.backend.render(engine.visible_token_ids),
        terminal_token_id=engine.terminal_token_id,
        terminal_reason=engine.terminal_reason,
    )
    _print_final_text(engine, output=output)


def _print_final_text(engine: EpisodeEngine, *, output: Path | None = None) -> None:
    _write_text(engine.text, output=output)


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
                return ephemeral_runtime.run_ephemeral(
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

            args._model_changed = model_changed
            if model_changed:
                args.bias_rules = ()
                args.bias_groups = ()
                io.write("Model changed: token-ID biases reset; load a matching preset to apply biases.")
            activation_artifact = None
            if args.activation_strength is not None and args.activation_vector is None:
                raise EditorError("--steering-strength requires --steering-vector")
            if args.activation_vector is not None:
                from .activation_vectors import SteeringVectorArtifact

                activation_artifact = SteeringVectorArtifact.from_path(
                    args.activation_vector
                )
                activation_artifact.validate_against_backend(backend, provenance)
            requested_id = args.episode_id
            parent_id: str | None = None
            fork_boundary: int | None = None
            pending_tape: ReplayPlan | None = teacher_tape.plan if teacher_tape else None
            if selection.kind == "resume":
                # Only explicit CLI sampler flags override the stored segment.
                explicit = sampler_overrides_present(args)
                segment = store.sampling_segment(
                    args.resume, len(_visible_tokens(store, args.resume))
                )
                source_sampling = sampling_factory(segment["sampling"])
                sampling = (
                    sampler_from_args(args, source_sampling)
                    if explicit
                    else source_sampling
                )
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                explicit = sampling != source_sampling
                io.write("Restoring saved context...")
                if model_changed:
                    engine, episode_id = _model_continuation(
                        store, args.resume, backend, provenance,
                        guidance_backend=cfg_backend_for(sampling),
                        sampling_factory=sampling_factory,
                    )
                    if args.max_tokens is not None:
                        engine.resume(max_tokens=args.max_tokens)
                    if explicit or activation_artifact is not None:
                        engine.sampling = sampling
                        store.record_sampling_segment(episode_id, start_boundary=0, sampling=sampling,
                            stream_fingerprint=engine.stream_fingerprint, coordinate_offset=0)
                else:
                    engine = _restore_engine(
                        store, args.resume, backend, max_tokens=args.max_tokens,
                        sampling_override=sampling if explicit else None,
                        guidance_backend=cfg_backend_for(sampling),
                        sampling_factory=sampling_factory,
                    )
                    episode_id = args.resume
            elif selection.kind == "new":
                initial_text = (
                    args.new_prompt
                    if args.new_prompt is not None
                    else episode_prompts.read_prompt_file(args.new_prompt_file)
                )
                sampling = sampler_from_args(args)
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                engine = EpisodeEngine(
                    backend,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    initial_text=initial_text,
                    guidance_backend=cfg_backend_for(sampling),
                )
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                )
            elif selection.kind == "replay":
                recipe = build_source_replay_recipe(
                    store,
                    args.replay,
                    args.until,
                    sampling_factory=sampling_factory,
                )
                source_sampling = recipe.controls.effective_at(0).sampling
                overrides = {
                    name: getattr(args, name) for name in CORE_SAMPLER_FIELDS
                    if getattr(args, name) is not None
                }
                sampling = sampler_from_args(args, source_sampling)
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                # Explicit steering imports apply to every replay segment, just
                # like explicit sampler flags. Unspecified fields follow source.
                if activation_artifact is not None:
                    for name in POLICY_FIELDS:
                        overrides[name] = getattr(sampling, name)
                replay_prefix = None
                if model_changed:
                    replay_prefix = backend.tokenize(
                        recipe.source_prompt,
                        add_bos=True,
                        special=True,
                    )
                engine, pending_tape = _spr_engine_from_source(
                    recipe,
                    backend,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    control_policy=(
                        ReplayControlPolicy.PRESERVE_DESTINATION
                        if args.fixed_config
                        else ReplayControlPolicy.FOLLOW_SOURCE
                    ),
                    sampling_overrides=overrides,
                    initial_token_ids=(
                        replay_prefix
                        if replay_prefix is not None
                        else list(source["initial_token_ids"])
                    ),
                    guidance_backend=cfg_backend_for(sampling),
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
                segment = store.sampling_segment(args.fork_from, target)
                source_sampling = sampling_factory(segment["sampling"])
                explicit = sampler_overrides_present(args)
                sampling = sampler_from_args(args, source_sampling)
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                if model_changed:
                    engine, episode_id = _materialize_model_change_fork(
                        store,
                        args.fork_from,
                        target,
                        backend,
                        provenance,
                        sampling=sampling,
                        max_tokens=(
                            source["max_tokens"]
                            if args.max_tokens is None
                            else args.max_tokens
                        ),
                        requested_id=requested_id,
                        guidance_backend=cfg_backend_for(sampling),
                    )
                else:
                    prefix = [*source["initial_token_ids"], *visible[:target]]
                    branch = getattr(backend, "branch_to_prefix", None)
                    if callable(branch):
                        branch(prefix)
                    else:
                        backend.reset(prefix)
                    engine = EpisodeEngine(
                        backend,
                        sampling=sampling,
                        max_tokens=source["max_tokens"] if args.max_tokens is None else args.max_tokens,
                        initial_text=str(source["initial_text"]),
                        initial_token_ids=source["initial_token_ids"],
                        stream_fingerprint=segment["stream_fingerprint"],
                        coordinate_offset=segment["coordinate_offset"],
                        backend_positioned=True,
                        guidance_backend=cfg_backend_for(sampling),
                    )
                    engine.visible_token_ids = list(visible[:target])
                    if args.max_tokens is None:
                        _inherit_budget(store, args.fork_from, engine, target)
                    else:
                        engine.trajectory.set_budget(args.max_tokens, target + args.max_tokens)
                    episode_id = _create_episode(
                        store,
                        engine,
                        backend_provenance=provenance,
                        requested_id=requested_id,
                        parent_episode_id=args.fork_from,
                        fork_boundary=target,
                        mode="fork",
                    )
                    store.copy_prefix(
                        args.fork_from,
                        episode_id,
                        target,
                        visible_text=backend.render(engine.visible_token_ids),
                        max_tokens=engine.max_tokens,
                    )
                    _record_fork_edge_state(store, episode_id, engine)

            store.visit(episode_id)
            enter_edge = False
            while True:
                if engine.guidance_backend is None:
                    engine.guidance_backend = cfg_backend_for(
                        engine.sampling, plan=pending_tape,
                        historical_sampling=(
                            sampling_factory(segment["sampling"])
                            for segment in store.sampler_segments(episode_id)
                        ),
                    )
                runner_options = {
                    "divergence_policy": args.divergence_policy,
                }
                runner = CoreEpisodeRunner(
                    engine,
                    store,
                    episode_id,
                    **runner_options,
                )
                try:
                    if enter_edge:
                        enter_edge = False
                        raise EdgeRequested()
                    result = runner.run(
                        tape=pending_tape,
                        live_policy=episode_policy_setup.durable_policy(
                            args, store, episode_id, io
                        ),
                    )
                except EdgeRequested:
                    pending_tape = None
                    action, value = _live_edge_menu(
                        io, store, episode_id, engine,
                        sampling_factory=sampling_factory,
                    )
                except ChordRequested as request:
                    pending_tape = None
                    chord = Chord(engine, request.ranks)
                    try:
                        chord_action, actions = chord_menu(io, chord)
                    finally:
                        chord.discard()
                    if chord_action == "quit":
                        store.update_episode(
                            episode_id,
                            visible_text=engine.backend.render(engine.visible_token_ids),
                            max_tokens=engine.max_tokens,
                            status="open",
                        )
                        return 0
                    if chord_action == "select":
                        assert actions is not None
                        selected = runner.run(
                            live_policy=ActionSequencePolicy(actions),
                            max_live_actions=len(actions),
                        )
                        if selected.handed_off:
                            io.write(selected.handoff_reason or "Chord selection handed off.")
                        if engine.ended:
                            _print_final_text(engine, output=args.output)
                            return 0
                    continue
                except SeamlessRewindRequested as request:
                    from_boundary = engine.boundary
                    io.write(f"Restoring context at boundary {request.boundary}...")
                    details = _rewind_episode(
                        store, episode_id, engine, request.boundary,
                        sampling_factory=sampling_factory,
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
                except ForkRequested as request:
                    action, value = "fork", request.boundary
                else:
                    if engine.ended:
                        _print_final_text(engine, output=args.output)
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
                    action, value = _live_edge_menu(
                        io, store, episode_id, engine,
                        sampling_factory=sampling_factory,
                    )

                if action == "new":
                    # Persist the current live edge before moving the shared
                    # backend to an unrelated prompt root.  The factory keeps
                    # only loaded backends, sampler settings, and the full
                    # configured tranche allowance; the new episode is not a
                    # fork or replay child.
                    store.update_episode(
                        episode_id,
                        visible_text=engine.backend.render(engine.visible_token_ids),
                        max_tokens=engine.max_tokens,
                        status="open",
                    )
                    store.record_budget(
                        episode_id,
                        engine.boundary,
                        engine.max_tokens,
                        engine.checkpoint_boundary,
                    )
                    new_engine = fresh_root_from(engine, str(value))
                    new_episode_id = _create_episode(
                        store,
                        new_engine,
                        backend_provenance=provenance,
                    )
                    engine, episode_id = new_engine, new_episode_id
                    store.visit(episode_id)
                    pending_tape = None
                    enter_edge = True
                    continue
                if action == "switch":
                    destination = str(value)
                    target_episode = store.get_episode(destination)
                    sealed = target_episode["status"] in {"completed", "failed"}
                    if sealed:
                        io.page(project_episode(store, destination).text)
                        reply = io.prompt(PromptRequest(
                            "Finished episode. Fork from end? [y/N]> "
                        ))
                        if not reply or reply.strip().lower() not in {"y", "yes"}:
                            enter_edge = True
                            continue
                    store.update_episode(episode_id, visible_text=engine.backend.render(engine.visible_token_ids),
                                         max_tokens=engine.max_tokens, status="open")
                    store.record_budget(episode_id, engine.boundary, engine.max_tokens, engine.checkpoint_boundary)
                    try:
                        target_episode = store.get_episode(destination)
                        new_backend, new_provenance, changed = (
                            episode_backend_loader.load_episode_backend(
                                args,
                                target_episode,
                                io,
                                use_saved=True,
                                current_backend=backend,
                                current_provenance=provenance,
                            )
                        )
                        if changed:
                            new_engine, destination = _model_continuation(
                                store, destination, new_backend, new_provenance,
                                guidance_backend=cfg_backend_for(
                                    sampling_factory(
                                        store.sampling_segment(
                                            destination, len(_visible_tokens(store, destination))
                                        )["sampling"]
                                    ),
                                    primary=new_backend, model_provenance=new_provenance,
                                ),
                                sampling_factory=sampling_factory,
                            )
                        elif sealed:
                            visible = _visible_tokens(store, destination)
                            segment = store.sampling_segment(destination, len(visible))
                            prefix = [*target_episode["initial_token_ids"], *visible]
                            branch = getattr(new_backend, "branch_to_prefix", None)
                            if callable(branch):
                                branch(prefix)
                            else:
                                new_backend.reset(prefix)
                            new_sampling = sampling_factory(segment["sampling"])
                            new_engine = EpisodeEngine(
                                new_backend,
                                initial_token_ids=target_episode["initial_token_ids"],
                                initial_text=str(target_episode["initial_text"]),
                                sampling=new_sampling,
                                stream_fingerprint=segment["stream_fingerprint"],
                                coordinate_offset=segment["coordinate_offset"],
                                max_tokens=target_episode["max_tokens"],
                                backend_positioned=True,
                                guidance_backend=cfg_backend_for(new_sampling),
                            )
                            new_engine.visible_token_ids = list(visible)
                            _inherit_budget(store, destination, new_engine, len(visible))
                            destination = _create_episode(store, new_engine, backend_provenance=new_provenance,
                                parent_episode_id=destination, fork_boundary=len(visible), mode="fork")
                            store.copy_prefix(
                                target_episode["episode_id"],
                                destination,
                                len(visible),
                                visible_text=new_backend.render(visible),
                                max_tokens=new_engine.max_tokens,
                            )
                            _record_fork_edge_state(store, destination, new_engine)
                        else:
                            visible = _visible_tokens(store, destination)
                            new_engine = _restore_engine(store, destination, new_backend,
                                max_tokens=None, sampling_override=None,
                                guidance_backend=cfg_backend_for(
                                    sampling_factory(
                                        store.sampling_segment(destination, len(visible))["sampling"]
                                    )
                                ),
                                sampling_factory=sampling_factory,
                            )
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
                        guidance_backend=cfg_backend_for(parent_engine.sampling),
                        sampling_factory=sampling_factory,
                    )
                    episode_id = _create_episode(
                        store,
                        engine,
                        backend_provenance=provenance,
                        parent_episode_id=parent_id,
                        fork_boundary=target,
                        mode="fork",
                    )
                    store.copy_prefix(
                        parent_id,
                        episode_id,
                        target,
                        visible_text=backend.render(engine.visible_token_ids),
                        max_tokens=engine.max_tokens,
                    )
                    _record_fork_edge_state(store, episode_id, engine)
                    pending_tape = None
                    continue
                if action == "spr":
                    source_id, until = value
                    recipe = build_source_replay_recipe(
                        store,
                        source_id,
                        until,
                        sampling_factory=sampling_factory,
                    )
                    pending_tape = compose_replay_plan(
                        recipe,
                        ReplayPlacement.APPEND_TO_CURRENT_BRANCH,
                        ReplayControlPolicy.PRESERVE_DESTINATION,
                    )
                    store.record_interaction(
                        episode_id, engine.boundary, "replay-start",
                        {
                            "source_episode_id": source_id,
                            "action_count": len(pending_tape),
                            "until": until,
                        },
                    )
                    # Keep the existing engine and ledger: boundary zero,
                    # prefix, sampler stream, and remaining budget do not move.
                    continue
                raise AssertionError(f"unhandled live-edge action {action!r}")
    except (EditorError, OSError, RuntimeError, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
