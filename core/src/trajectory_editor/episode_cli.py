"""Command line for the reduced Serial Policy Editor.

A live episode alternates between the ordinary teacher loop and live edges.
Token budgets, replay exhaustion, and replay divergence yield at a live edge;
only EOG or explicit ``finish`` seals the episode.
"""

from __future__ import annotations

import argparse
import copy
import math
from contextlib import ExitStack
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator

from .backend_factory import BACKEND_NAMES, create_backend
from .decoder import KV_CACHE_TYPES, LlamaCppSettings
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
    POLICY_FIELDS, _inherit_budget, _model_continuation, _visible_tokens, _restore_engine,
    _create_episode, _rewind_episode, _fork_engine, _spr_engine_from_source,
)
from .core.actions import Write
from .episode_engine import EpisodeEngine
from .episode_runner import (
    EdgeRequested,
    EpisodeRunner as CoreEpisodeRunner,
    ForkRequested,
    LiveSessionRunner,
    ReplayContext,
    ReplayOrigin,
    SeamlessEdgeRequested,
    SeamlessRewindRequested,
    TapeStep,
    ReplayPlan,
)
from .projector import project_episode, project_fork_map, project_lineage, project_procedure
from .episode_store import EpisodeStore
from .episode_session import LiveSession
from .episode_materializer import save_live_branch, save_live_family
from .episode_ui import InteractivePolicy, PolicyViewPreferences
from .transformers_backend import TransformersSettings
from .runtime_setup import RuntimePlan, effective_plan_summary, run_runtime_setup_menu
from .tui import TerminalIO
from .ui_themes import LIVE_THEME_NAMES
from .version import VERSION
from .teacher_plan import export_live_teacher_tape, export_teacher_tape, load_teacher_tape_jsonl


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
        "--setup-menu",
        action="store_true",
        help="open the pre-runtime setup menu before creating or restoring an episode",
    )
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
        default=None, help="show policy diagnostics without changing backend-rank ordering (default: automatic)",
    )
    policy_view.add_argument(
        "--no-policy-view", dest="show_policy_rank", action="store_false",
        help="hide automatic policy diagnostics; V can toggle them during the session",
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


def _backend(args: argparse.Namespace):
    args.backend = args.backend or "llama.cpp"
    if args.model is None:
        raise EditorError("--model is required to start, resume, fork, or replay")
    llama = LlamaCppSettings(
        type_k=args.type_k,
        type_v=args.type_v,
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
            provenance = dict(backend.provenance(include_model_sha256=True))
            provenance["model_path"] = str(selected.model.resolve())
            provenance["load_options"] = {
                key: value for key, value in vars(selected).items()
                if key.startswith("transformers_") or key in {
                    "n_ctx", "n_batch", "n_ubatch", "n_threads", "n_threads_batch",
                    "n_gpu_layers", "main_gpu", "no_flash_attn", "no_mmap", "use_mlock", "cache", "type_k", "type_v"
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


def _load_cfg_guidance_backend(args, provenance):
    """Load a second copy of the active model for CFG's unconditional branch."""
    selected = copy.copy(args)
    selected.model = Path(provenance["model_path"])
    selected.backend = provenance["backend"]
    for key, value in provenance.get("load_options", {}).items():
        if hasattr(selected, key):
            setattr(selected, key, value)
    return _backend(selected)


def _interactive_policy(
    args: argparse.Namespace, store: EpisodeStore, episode_id: str, io: TerminalIO,
) -> InteractivePolicy:
    preferences = getattr(args, "_policy_view_preferences", None)
    if preferences is None:
        preferences = PolicyViewPreferences(
            show=args.show_policy_rank,
            logit_view=args.logit_view,
        )
        args._policy_view_preferences = preferences
    return InteractivePolicy(
        io=io,
        menu_size=args.table_depth,
        search_radius=args.search_radius,
        default_hold_tokens=args.hold_default,
        phrase_max_tokens=getattr(args, "phrase_max_tokens", 16),
        phrase_max_shift=getattr(args, "phrase_max_shift", 6.0),
        context_characters=args.context_chars,
        manual_acceptance=args.manual_acceptance,
        view_preferences=preferences,
        store=store,
        episode_id=episode_id,
        seamless=io.supports_live_choices,
    )


def _ephemeral_policy(args: argparse.Namespace, io: TerminalIO) -> InteractivePolicy:
    """Build the normal action chooser without a workspace-backed recorder."""
    preferences = getattr(args, "_policy_view_preferences", None)
    if preferences is None:
        preferences = PolicyViewPreferences(
            show=args.show_policy_rank,
            logit_view=args.logit_view,
        )
        args._policy_view_preferences = preferences
    return InteractivePolicy(
        io=io,
        menu_size=args.table_depth,
        search_radius=args.search_radius,
        default_hold_tokens=args.hold_default,
        phrase_max_tokens=getattr(args, "phrase_max_tokens", 16),
        phrase_max_shift=getattr(args, "phrase_max_shift", 6.0),
        context_characters=args.context_chars,
        manual_acceptance=args.manual_acceptance,
        view_preferences=preferences,
        # LiveSession supplies the same token-boundary semantics without a
        # durable interaction recorder.  Keep the fullscreen review controls
        # live so Enter on a historical boundary actually rewinds the branch.
        seamless=bool(getattr(io, "supports_live_choices", False)),
    )


def _ephemeral_edge_menu(
    io: TerminalIO,
    session: LiveSession,
) -> tuple[str, Any]:
    """Small branch-oriented EDGE surface for a session with no workspace."""
    live_surface = bool(
        getattr(io, "supports_live_choices", False)
        and callable(getattr(io, "read_live_edge_command", None))
    )
    while True:
        if live_surface:
            raw = io.read_live_edge_command(  # type: ignore[attr-defined]
                episode_id=session.branch.branch_id,
                boundary=session.engine.boundary,
                current_budget=session.engine.max_tokens,
                remaining_tokens=session.engine.remaining,
                sampler_summary=_sampler_summary(session.sampler),
                mode="session",
            )
        else:
            io.write(
                f"Live branch {session.branch.branch_id} @ boundary {session.engine.boundary}"
                f" · {_sampler_summary(session.sampler)}"
            )
            raw = io.read(
                "[c]ontinue  [n N/off] budget  [s key=value] sampler  [rewind N] "
                "[f N] fork  [branches]  [switch ID]  [export FILE] "
                "[save WORKSPACE [ID]]  [save-family WORKSPACE [ROOT_ID]]  [e]nd  [q]uit > "
            )
        if raw is None:
            return "quit", None
        text = raw.strip()
        lower = text.lower()
        if lower in {"q", "quit"}:
            return "quit", None
        if lower in {"e", "end"}:
            return "end", None
        if lower in {"c", "continue", ""}:
            return "continue", "keep"
        if lower in {"branches", "ls"}:
            rows = []
            states = session.branch_states
            for branch_id, node in session.branch_tree.nodes.items():
                marker = "*" if branch_id == session.branch.branch_id else " "
                parent = node.identity.parent_id or "root"
                state = states[branch_id]
                rows.append(
                    f"{marker} {branch_id}  parent={parent}  fork={node.identity.fork_boundary} "
                    f"boundary={state.boundary}"
                )
            io.page("Live branches:\n" + "\n".join(rows))
            continue
        parts = text.split()
        if len(parts) == 2 and parts[0].lower() == "switch":
            if parts[1] not in session.branch_states:
                io.write(f"Unknown live branch {parts[1]!r}.")
                continue
            return "switch", parts[1]
        if len(parts) == 2 and parts[0].lower() == "rewind":
            try:
                return "rewind", int(parts[1])
            except ValueError:
                io.write("Rewind boundary must be an integer.")
                continue
        if len(parts) == 2 and parts[0].lower() in {"f", "fork"}:
            try:
                return "fork", int(parts[1])
            except ValueError:
                io.write("Fork boundary must be an integer.")
                continue
        if len(parts) == 2 and parts[0].lower() == "export":
            return "export", Path(parts[1])
        if parts and parts[0].lower() in {"save-family", "savefamily"}:
            if len(parts) not in {2, 3}:
                io.write("Use save-family WORKSPACE [ROOT_ID].")
                continue
            return "save-family", (Path(parts[1]), parts[2] if len(parts) == 3 else None)
        if len(parts) >= 2 and parts[0].lower() == "save" and parts[1].lower() == "family":
            if len(parts) not in {3, 4}:
                io.write("Use save family WORKSPACE [ROOT_ID].")
                continue
            return "save-family", (Path(parts[2]), parts[3] if len(parts) == 4 else None)
        if len(parts) in {2, 3} and parts[0].lower() == "save":
            return "save", (Path(parts[1]), parts[2] if len(parts) == 3 else None)
        if len(parts) == 2 and parts[0].lower() in {"n", "next"}:
            if parts[1].lower() in {"off", "none", "unlimited"}:
                return "continue", None
            try:
                budget = int(parts[1])
                if budget < 1:
                    raise ValueError
                return "continue", budget
            except ValueError:
                io.write("Budget must be a positive integer.")
                continue
        if parts and parts[0].lower() in {"s", "sampler"}:
            payload = text.split(maxsplit=1)[1] if len(parts) > 1 else ""
            if not payload:
                io.write("Use sampler key=value.")
                continue
            try:
                session.set_sampler(sampler_override(session.sampler, payload))
            except EditorError as exc:
                io.write(f"[invalid sampler change] {exc}")
            continue
        io.write("Unknown live-session command.")


def _run_ephemeral(
    args: argparse.Namespace,
    *,
    io: TerminalIO,
    teacher_tape: Any | None,
) -> int:
    """Run an explicit in-memory session without opening an episode workspace."""
    if args.new_prompt is None and args.new_prompt_file is None:
        raise EditorError("--ephemeral requires --new-prompt, --new-prompt-file, or a teacher-plan envelope prompt")
    initial_text = args.new_prompt if args.new_prompt is not None else args.new_prompt_file.read_text(encoding="utf-8")
    backend = _backend(args)
    provenance = backend.provenance()
    sampling = sampler_from_args(args)
    activation_artifact = None
    if args.activation_strength is not None and args.activation_vector is None:
        raise EditorError("--steering-strength requires --steering-vector")
    if args.activation_vector is not None:
        from .activation_vectors import SteeringVectorArtifact
        activation_artifact = SteeringVectorArtifact.from_path(args.activation_vector)
        activation_artifact.validate_against_backend(backend, provenance)
        sampling = apply_activation_artifact(sampling, activation_artifact, args)
    if not _confirm_runtime_plan(io, args, backend, provenance, sampling, activation_artifact=activation_artifact):
        return 0
    guidance_backend = None
    if sampling.cfg_unconditional_prompt is not None:
        io.write("Loading second model copy for CFG prefix guidance...")
        guidance_backend = _load_cfg_guidance_backend(args, provenance)
    engine = EpisodeEngine(
        backend,
        sampling=sampling,
        max_tokens=args.max_tokens,
        initial_text=initial_text,
        guidance_backend=guidance_backend,
    )
    session = LiveSession(
        engine,
        prompt=initial_text,
        environment_stamp={"backend": provenance, "sampler": sampling.to_dict()},
    )
    pending_tape = teacher_tape.plan if teacher_tape is not None else None
    while True:
        announced_teacher_tape = (
            teacher_tape is not None and pending_tape is teacher_tape.plan
        )
        runner = LiveSessionRunner(session, divergence_policy=args.divergence_policy)
        try:
            result = runner.run(
                tape=pending_tape,
                live_policy=_ephemeral_policy(args, io),
                stop_after_tape=True,
            )
        except EdgeRequested:
            pending_tape = None
            action, value = _ephemeral_edge_menu(io, session)
        except ForkRequested as request:
            action, value = "fork", request.boundary
        except SeamlessRewindRequested as request:
            action, value = "rewind", request.boundary
        except SeamlessEdgeRequested:
            action, value = _ephemeral_edge_menu(io, session)
        else:
            pending_tape = None
            if result.handed_off and result.handoff_reason:
                io.write(result.handoff_reason)
            if result.replay_exhausted and announced_teacher_tape:
                io.write(f"Teacher plan exhausted at boundary {session.engine.boundary}; live edge reached.")
            if session.engine.ended:
                text = session.engine.text
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text, encoding="utf-8")
                    print(f"Text: {args.output}", flush=True)
                else:
                    print("\n--- final text ---")
                    print(text)
                session.discard()
                return 0
            action, value = _ephemeral_edge_menu(io, session)
        if action == "quit":
            session.discard()
            return 0
        if action == "end":
            session.quit("menu-end")
            text = session.engine.text
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
                print(f"Text: {args.output}", flush=True)
            else:
                print("\n--- final text ---")
                print(text)
            session.discard()
            return 0
        if action == "continue":
            session.resume(max_tokens=value)
            continue
        if action == "rewind":
            try:
                session.rewind(int(value))
            except EditorError as exc:
                io.write(str(exc))
            continue
        if action == "fork":
            target = int(value)
            try:
                child = session.fork(boundary=target)
            except EditorError as exc:
                io.write(str(exc))
                continue
            session.activate(child.branch.branch_id)
            io.write(f"Forked live branch {child.branch.branch_id} at boundary {target}.")
            continue
        if action == "switch":
            session.activate(str(value))
            continue
        if action == "export":
            try:
                export_live_teacher_tape(session, value)
                io.write(f"Exported selected branch to {value}.")
            except EditorError as exc:
                io.write(str(exc))
            # Return to EDGE without asking the teacher policy for another
            # action merely because a non-mutating command completed.
            pending_tape = ReplayPlan()
            continue
        if action == "save":
            workspace, requested_id = value
            try:
                identifier = save_live_branch(session, workspace, provenance, episode_id=requested_id)
                io.write(f"Saved selected branch as {identifier} in {workspace}.")
            except (EditorError, OSError, RuntimeError) as exc:
                io.write(str(exc))
            pending_tape = ReplayPlan()
            continue
        if action == "save-family":
            workspace, requested_root_id = value
            try:
                identifiers = save_live_family(
                    session,
                    workspace,
                    provenance,
                    root_episode_id=requested_root_id,
                )
                io.write(
                    f"Saved {len(identifiers)} live branches as a family in {workspace}."
                )
            except (EditorError, OSError, RuntimeError) as exc:
                io.write(str(exc))
            pending_tape = ReplayPlan()
            continue
        raise AssertionError(f"unhandled ephemeral action {action!r}")


def _sampler_summary(config: SamplerConfig) -> str:
    summary = (
        f"temp={config.temperature:g} top_k={config.top_k} top_p={config.top_p:g} "
        f"min_p={config.min_p:g} typical_p={config.typical_p:g} tfs_z={config.tail_free_z:g} "
        f"draw={config.draw_kernel} rep={config.repeat_penalty:g}/{config.repeat_last_n} "
        f"presence={config.presence_penalty:g} frequency={config.frequency_penalty:g} "
        f"seed={config.seed}"
    )
    if config.bias_groups:
        summary += " groups=" + ",".join(
            f"{group.name}:{group.bias:g}" for group in config.bias_groups
        )
    if config.activation_vector or config.activation_vector_digest:
        norm = sum(value * value for value in config.activation_vector) ** 0.5
        summary += (
            f" steering_vector_norm={norm:g}"
            f" steering_strength={config.activation_vector_strength:g}"
            f" steering_digest={config.activation_vector_digest[:12]}"
        )
    return summary


def _confirm_runtime_plan(
    io: TerminalIO,
    args: argparse.Namespace,
    backend: Any,
    provenance: dict[str, Any],
    sampling: SamplerConfig,
    *,
    source_sampling: SamplerConfig | None = None,
    activation_artifact: SteeringVectorArtifact | None = None,
) -> bool:
    """Show the resolved plan and require an explicit final go in setup mode."""
    if not getattr(args, "_setup_menu_active", False):
        return True
    validated: list[str] = []
    if activation_artifact is not None:
        validated.append("steering vector: model and width matched")
    plan = RuntimePlan.from_args(args)
    io.page(
        effective_plan_summary(
            plan,
            sampling,
            source_sampling=source_sampling,
            provenance=provenance,
            validated_artifacts=tuple(validated),
        )
    )
    answer = io.read("Final go? [go/q] > ")
    if answer is None or answer.strip().lower() not in {"go", "g", "yes", "y"}:
        io.write("Launch cancelled.")
        return False
    return True


def _live_edge_menu(
    io: TerminalIO,
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    *,
    sampling_factory=SamplerConfig.from_record,
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
                _rewind_episode(
                    store, episode_id, engine, target,
                    sampling_factory=sampling_factory,
                )
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
                            store.replay_until(
                                source_id, until, sampling_factory=sampling_factory
                            )
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
                store.replay_until(
                    source_id, until, sampling_factory=sampling_factory
                )
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


class _SwitchableEpisodeStore:
    """Keep the setup menu's workspace choice inside one managed context."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._store: EpisodeStore | None = None

    def __enter__(self):
        self._store = EpisodeStore(self._path)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    @property
    def path(self) -> Path:
        return self._path

    def switch_workspace(self, path: Path | str) -> None:
        selected = Path(path)
        if selected == self._path:
            return
        replacement = EpisodeStore(selected)
        previous = self._store
        self._store = replacement
        self._path = selected
        if previous is not None:
            previous.close()

    def __getattr__(self, name: str):
        if self._store is None:
            raise RuntimeError("episode workspace is not open")
        return getattr(self._store, name)


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
    sampling_factory = SamplerConfig.from_record
    args._explicit_options = {
        action.dest for action in parser._actions
        if any(token.split("=", 1)[0] in action.option_strings for token in arguments)
    }
    try:
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
                args.new_prompt,
                args.new_prompt_file,
                args.replay,
                args.resume,
                args.fork_from,
                args.projector,
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
                "--resume": args.resume,
                "--replay": args.replay,
                "--fork-from": args.fork_from,
                "--projector": args.projector,
                "--export-teacher-plan": args.export_teacher_plan,
                "--list": args.list_episodes,
                "--lineage": args.lineage,
                "--setup-menu": args.setup_menu,
            }
            requested = next((flag for flag, value in incompatible.items() if value), None)
            if requested is not None:
                raise EditorError(f"--ephemeral cannot be combined with {requested}")
            if args.at is not None or args.until is not None or args.fixed_config or args.procedure:
                raise EditorError("--ephemeral accepts a new prompt and optional --teacher-plan only")
            if args.random_seed:
                args.seed = random_seed()
                print(f"Random seed: {args.seed}", flush=True)
            args._setup_menu_active = False
            io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
            open_live_session = getattr(io, "live_session", None)
            if callable(open_live_session):
                with open_live_session():
                    return _run_ephemeral(args, io=io, teacher_tape=teacher_tape)
            return _run_ephemeral(args, io=io, teacher_tape=teacher_tape)
        with _SwitchableEpisodeStore(args.workspace) as store, ExitStack() as ui_stack:
            for field in ("resume", "fork_from", "replay", "projector", "lineage"):
                value = getattr(args, field)
                if value:
                    setattr(args, field, store.resolve_id(value))
            if args.at is not None and args.fork_from is None:
                raise EditorError("--at is only valid with --fork-from")
            if args.procedure and not args.projector:
                raise EditorError("--procedure requires --projector EPISODE_ID")
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
                print(
                    project_episode(
                        store,
                        args.projector,
                        annotations=getattr(args, "annotations", "none"),
                        with_loss=getattr(args, "with_loss", False),
                        with_rank=getattr(args, "with_rank", False),
                        with_policy_rank=getattr(args, "with_policy_rank", False),
                        full_evidence=getattr(args, "full_evidence", False),
                        with_model_probs=getattr(args, "with_model_probs", False),
                        with_lineage=getattr(args, "with_lineage", False),
                    ).text
                )
                return 0
            has_episode_source = any(
                (
                    args.new_prompt is not None,
                    args.new_prompt_file is not None,
                    args.replay is not None,
                    args.resume is not None,
                    args.fork_from is not None,
                    args.teacher_plan is not None,
                )
            )
            setup_menu = bool(
                args.setup_menu
                or (
                    not has_episode_source
                    and not args.plain_ui
                    and sys.stdin.isatty()
                    and sys.stdout.isatty()
                )
            )
            args._setup_menu_active = setup_menu
            io: TerminalIO | None = None
            if setup_menu:
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    raise EditorError("--setup-menu requires an interactive terminal")
                io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
                if not run_runtime_setup_menu(io, args, store=store):
                    return 0
            elif not has_episode_source:
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    raise EditorError(
                        "no episode source supplied; use --new-prompt, "
                        "--new-prompt-file, --replay, --resume, or --fork-from"
                    )
                args.new_prompt = _read_initial_prompt()

            if args.random_seed:
                args.seed = random_seed()
                print(f"Random seed: {args.seed}", flush=True)

            if io is None:
                io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
            source_id = args.resume or args.fork_from or args.replay
            source = store.get_episode(source_id) if source_id else None
            backend, provenance, model_changed = _load_episode_backend(args, source, io)
            cfg_guidance_backend = None

            def cfg_backend_for(sampling):
                nonlocal cfg_guidance_backend
                if sampling.cfg_unconditional_prompt is None:
                    return None
                if cfg_guidance_backend is None:
                    io.write("Loading second model copy for CFG prefix guidance...")
                    cfg_guidance_backend = _load_cfg_guidance_backend(args, provenance)
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
            if args.resume is not None:
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
                if not _confirm_runtime_plan(
                    io, args, backend, provenance, sampling,
                    source_sampling=source_sampling,
                    activation_artifact=activation_artifact,
                ):
                    return 0
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
            elif args.new_prompt is not None or args.new_prompt_file is not None:
                initial_text = (
                    args.new_prompt
                    if args.new_prompt is not None
                    else args.new_prompt_file.read_text(encoding="utf-8")
                )
                sampling = sampler_from_args(args)
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                if not _confirm_runtime_plan(
                    io, args, backend, provenance, sampling,
                    activation_artifact=activation_artifact,
                ):
                    return 0
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
            elif args.replay is not None:
                source_segment = store.sampling_segment(args.replay, 0)
                source_sampling = sampling_factory(source_segment["sampling"])
                overrides = {
                    name: getattr(args, name) for name in CORE_SAMPLER_FIELDS
                    if getattr(args, name) is not None
                }
                sampling = sampler_from_args(args, source_sampling)
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                if not _confirm_runtime_plan(
                    io, args, backend, provenance, sampling,
                    source_sampling=source_sampling,
                    activation_artifact=activation_artifact,
                ):
                    return 0
                # Explicit steering imports apply to every replay segment, just
                # like explicit sampler flags. Unspecified fields follow source.
                if activation_artifact is not None:
                    for name in POLICY_FIELDS:
                        overrides[name] = getattr(sampling, name)
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
                    guidance_backend=cfg_backend_for(sampling),
                    sampling_factory=sampling_factory,
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
                source_sampling = sampling_factory(segment["sampling"])
                explicit = sampler_overrides_present(args)
                sampling = sampler_from_args(args, source_sampling)
                sampling = apply_activation_artifact(sampling, activation_artifact, args)
                if not _confirm_runtime_plan(
                    io, args, backend, provenance, sampling,
                    source_sampling=source_sampling,
                    activation_artifact=activation_artifact,
                ):
                    return 0
                engine = EpisodeEngine(
                    backend,
                    sampling=sampling,
                    max_tokens=source["max_tokens"] if args.max_tokens is None else args.max_tokens,
                    initial_text=backend.render(prefix, special=True),
                    initial_token_ids=prefix,
                    stream_fingerprint=segment["stream_fingerprint"],
                    coordinate_offset=segment["coordinate_offset"] + target,
                    guidance_backend=cfg_backend_for(sampling),
                    guidance_generated_prefix=visible[:target],
                    guidance_tokens_consumed=target,
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

            open_live_session = getattr(io, "live_session", None)
            if callable(open_live_session):
                ui_stack.enter_context(open_live_session())
            store.visit(episode_id)
            enter_edge = False
            while True:
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
                        live_policy=_interactive_policy(
                            args, store, episode_id, io
                        ),
                        stop_after_tape=True,
                    )
                except EdgeRequested:
                    pending_tape = None
                    action, value = _live_edge_menu(
                        io, store, episode_id, engine,
                        sampling_factory=sampling_factory,
                    )
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
                except SeamlessEdgeRequested as request:
                    from_boundary = engine.boundary
                    io.write(f"Restoring context at boundary {request.boundary}...")
                    details = _rewind_episode(
                        store, episode_id, engine, request.boundary,
                        sampling_factory=sampling_factory,
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
                    action, value = _live_edge_menu(
                        io, store, episode_id, engine,
                        sampling_factory=sampling_factory,
                    )
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
                    action, value = _live_edge_menu(
                        io, store, episode_id, engine,
                        sampling_factory=sampling_factory,
                    )

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
                        target_episode = store.get_episode(destination)
                        new_backend, new_provenance, changed = _load_episode_backend(args, target_episode, io, use_saved=True,
                            current_backend=backend, current_provenance=provenance)
                        if changed:
                            new_engine, destination = _model_continuation(
                                store, destination, new_backend, new_provenance,
                                guidance_backend=cfg_backend_for(
                                    sampling_factory(
                                        store.sampling_segment(destination, 0)["sampling"]
                                    )
                                ),
                                sampling_factory=sampling_factory,
                            )
                        elif sealed:
                            visible = _visible_tokens(store, destination)
                            segment = store.sampling_segment(destination, len(visible))
                            new_engine = EpisodeEngine(new_backend,
                                initial_token_ids=[*target_episode["initial_token_ids"], *visible],
                                    sampling=sampling_factory(segment["sampling"]),
                                stream_fingerprint=segment["stream_fingerprint"],
                                coordinate_offset=segment["coordinate_offset"] + len(visible),
                                max_tokens=target_episode["max_tokens"],
                                guidance_backend=cfg_backend_for(
                                    sampling_factory(segment["sampling"])
                                ),
                                guidance_generated_prefix=visible,
                                guidance_tokens_consumed=len(visible),
                            )
                            _inherit_budget(store, destination, new_engine, len(visible), rebase=True)
                            destination = _create_episode(store, new_engine, backend_provenance=new_provenance,
                                parent_episode_id=destination, fork_boundary=len(visible), mode="fork")
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
                    pending_tape = None
                    continue
                if action == "spr":
                    source_id, until = value
                    # Snapshot before appending, including self-replay. EDGE
                    # composition inserts the source prompt as literal text;
                    # CLI replay continues to use it as initial context.
                    source_prompt = store.get_episode(source_id)["initial_text"]
                    source_steps = store.replay_until(
                        source_id, until, sampling_factory=sampling_factory
                    )
                    steps = []
                    origins = []
                    if source_prompt:
                        steps.append(TapeStep(
                            Write(source_prompt, "exact"), None,
                        ))
                        origins.append(ReplayOrigin(
                            source_episode_id=source_id,
                            source_boundary=0,
                            source_part="prompt",
                        ))
                    steps.extend(
                        TapeStep(
                            step["action"], step["expectation"],
                        )
                        for step in source_steps
                    )
                    origins.extend(
                        ReplayOrigin(
                            source_episode_id=source_id,
                            source_boundary=step["boundary"],
                        )
                        for step in source_steps
                    )
                    pending_tape = ReplayPlan(
                        tuple(steps),
                        follow_source_sampling=False,
                        context=ReplayContext(origins=tuple(origins)),
                    )
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
