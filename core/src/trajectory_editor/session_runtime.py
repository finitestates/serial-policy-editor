"""Run and control persistence-free live episode sessions."""

from __future__ import annotations

import argparse
from typing import Any

from . import edge_commands, episode_backend_loader, episode_policy_setup
from .core.cli_config import (
    apply_activation_artifact,
    sampler_from_args,
    sampler_override,
)
from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .chord import Chord, ChordRequested, chord_menu
from .edge_status import sampler_summary
from .episode_engine import EpisodeEngine
from .episode_prompts import read_new_prompt, read_prompt_file
from .episode_replay_source import build_source_replay_recipe
from .run_loop import (
    EdgeRequested,
    ForkRequested,
    ReplayPlan,
    SeamlessRewindRequested,
    run_plan,
)
from .episode_session import LiveSession, LiveSessionRoster
from .projector import project_live_fork_map
from .teacher_plan import export_live_teacher_tape
from .terminal_contracts import EdgeViewState, PromptRequest, TerminalProtocol


def session_edge_menu(
    io: TerminalProtocol,
    session_or_roster: LiveSession | LiveSessionRoster,
    *,
    store: Any | None = None,
) -> tuple[str, Any]:
    """Turn EDGE input into an in-memory session operation or explicit save."""

    roster = (
        session_or_roster
        if isinstance(session_or_roster, LiveSessionRoster)
        else None
    )
    session = roster.active_session if roster is not None else session_or_roster

    def branch_aliases() -> dict[str, str]:
        return {
            str(index): branch_id
            for index, branch_id in enumerate(session.branch_tree.nodes, start=1)
        }

    def resolve_branch(reference: str) -> str:
        cleaned = reference.strip()
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]
        return branch_aliases().get(cleaned, cleaned)

    def branch_number(branch_id: str) -> str:
        if roster is not None:
            return str(roster.number_for(session, branch_id))
        return next(
            (
                number
                for number, identifier in branch_aliases().items()
                if identifier == branch_id
            ),
            branch_id,
        )

    def show_fork_map() -> tuple[str, Any] | None:
        state = session.branch_state()
        fork_map = project_live_fork_map(
            session.prompt,
            state.visible_token_ids,
            session.engine.backend,
        )
        while True:
            entered = io.prompt(PromptRequest(
                f"Fork boundary (0..{state.boundary}; blank cancels) > ",
                body=fork_map,
            ))
            if entered is None:
                return "quit", None
            value = entered.strip()
            if not value:
                return None
            try:
                target = int(value)
            except ValueError:
                io.write("Fork boundary must be an integer.")
                continue
            if not 0 <= target <= state.boundary:
                io.write(f"Fork boundary must be 0..{state.boundary}.")
                continue
            return "fork", target

    while True:
        displayed_id = branch_number(session.branch.branch_id)
        source_id = session.environment_stamp.get("source_episode_id")
        if store is not None and source_id is not None:
            try:
                displayed_id = store.label(str(source_id))
            except EditorError:
                pass
        raw = io.read_edge(EdgeViewState(
            episode_id=displayed_id,
            boundary=session.engine.boundary,
            current_budget=session.engine.max_tokens,
            remaining_tokens=session.engine.remaining,
            sampler_summary=sampler_summary(session.sampler),
            mode="session",
        ))
        if raw is None:
            return "quit", None
        try:
            command = edge_commands.parse_edge_command(raw)
        except edge_commands.EdgeCommandParseError as exc:
            io.write(str(exc))
            continue
        if isinstance(command, edge_commands.QuitCommand):
            return "quit", None
        if isinstance(command, edge_commands.EndCommand):
            return "end", None
        if isinstance(command, edge_commands.ContinueCommand):
            return "continue", "keep"
        if isinstance(command, edge_commands.NewCommand):
            prompt_text = command.prompt if command.prompt else read_new_prompt(io)
            if prompt_text is None:
                continue
            if not prompt_text:
                io.write("New prompt must not be empty.")
                continue
            return "new", prompt_text
        if isinstance(command, edge_commands.ListCommand) and store is not None:
            selected = io.prompt(PromptRequest(
                "Saved episode (Enter returns)> ",
                body=store.workspace_list(include_finished=command.include_finished),
            ))
            if selected and selected.strip():
                try:
                    return "switch-saved", store.resolve_id(selected.strip())
                except EditorError as exc:
                    io.write(str(exc))
            continue
        if (
            isinstance(command, edge_commands.BranchesCommand)
            or (
                isinstance(command, edge_commands.ListCommand)
                and not command.include_finished
            )
        ):
            rows = []
            if roster is not None:
                for entry in roster.entries():
                    identity = entry.identity
                    marker = "*" if (
                        entry.session is session
                        and entry.branch_id == session.branch.branch_id
                    ) else " "
                    source = "root"
                    if identity.parent_id is not None:
                        source = str(
                            roster.number_for(entry.session, identity.parent_id)
                        )
                    rows.append(
                        f"{marker} #{entry.number}  prompt={entry.session.prompt!r}  "
                        f"from={source}  fork={identity.fork_boundary} "
                        f"boundary={entry.state.boundary}"
                    )
            else:
                states = session.branch_states
                aliases = branch_aliases()
                reverse_aliases = {
                    branch_id: number for number, branch_id in aliases.items()
                }
                for branch_id, node in session.branch_tree.nodes.items():
                    marker = "*" if branch_id == session.branch.branch_id else " "
                    source = (
                        reverse_aliases.get(node.identity.parent_id, "root")
                        if node.identity.parent_id is not None
                        else "root"
                    )
                    state = states[branch_id]
                    rows.append(
                        f"{marker} {reverse_aliases[branch_id]}  from={source}  "
                        f"fork={node.identity.fork_boundary} "
                        f"boundary={state.boundary}"
                    )
            io.page("Live branches:\n" + "\n".join(rows))
            continue
        if isinstance(command, edge_commands.ForkMapCommand):
            result = show_fork_map()
            if result is not None:
                return result
            continue
        if isinstance(command, edge_commands.SwitchCommand):
            branch_id = resolve_branch(command.reference)
            if roster is not None:
                try:
                    roster.resolve(command.reference)
                    return "switch", command.reference
                except EditorError:
                    pass
            elif branch_id in session.branch_states:
                return "switch", branch_id
            if store is not None:
                try:
                    return "switch-saved", store.resolve_id(command.reference)
                except EditorError as exc:
                    io.write(str(exc))
                    continue
            io.write(f"Unknown live branch {command.reference!r}.")
            continue
        if isinstance(command, edge_commands.RewindCommand):
            return "rewind", command.boundary
        if isinstance(command, edge_commands.ForkCommand):
            return "fork", command.boundary
        if isinstance(command, edge_commands.ExportCommand):
            return "export", command.path
        if isinstance(command, edge_commands.ProjectCommand):
            if store is None or source_id is None:
                io.write("Save this branch before projecting its durable report.")
                continue
            from .projector import project_episode

            io.page(project_episode(store, str(source_id)).text)
            continue
        if isinstance(command, edge_commands.RenameCommand):
            if store is None or source_id is None:
                io.write("Name a saved episode after saving this branch.")
                continue
            store.rename(str(source_id), command.title)
            continue
        if isinstance(command, edge_commands.ReplayCommand):
            if store is None:
                io.write("Source replay requires a workspace.")
                continue
            try:
                source = store.resolve_id(command.source)
                if command.until is None:
                    end = sum(
                        bool(row["realized_visible"]) for row in store.tokens(source)
                    )
                else:
                    end = command.until
                return "spr", (source, end)
            except EditorError as exc:
                io.write(str(exc))
                continue
        if isinstance(command, edge_commands.ReplaySelectionCommand):
            if store is None:
                io.write("Source replay requires a workspace.")
                continue
            try:
                from .projector import project_fork_map

                source = store.resolve_id(command.source)
                fork_map = project_fork_map(store, source)
                end = sum(
                    bool(row["realized_visible"]) for row in store.tokens(source)
                )
                while True:
                    entered = io.prompt(PromptRequest(
                        f"Replay through source boundary (0..{end}; blank cancels) > ",
                        body=fork_map,
                    ))
                    if entered is None or not entered.strip():
                        break
                    try:
                        boundary = int(entered)
                        if not 0 <= boundary <= end:
                            raise ValueError
                    except ValueError:
                        io.write("Choose a valid source token boundary.")
                        continue
                    return "spr", (source, boundary)
            except EditorError as exc:
                io.write(str(exc))
                continue
            continue
        if isinstance(command, edge_commands.SaveFamilyCommand):
            return "save-family", (command.workspace, command.root_reference)
        if isinstance(command, edge_commands.SaveCommand):
            return "save", (command.workspace, command.reference)
        if isinstance(command, edge_commands.BudgetCommand):
            return "continue", command.tokens
        if isinstance(command, edge_commands.SamplerCommand):
            payload = command.text
            if payload is None:
                payload = io.read(
                    "sampler key=value changes (blank cancels; e.g. top_k=20 temperature=.8)> "
                ) or ""
            if not payload.strip():
                continue
            try:
                session.set_sampler(sampler_override(session.sampler, payload))
            except EditorError as exc:
                io.write(f"[invalid sampler change] {exc}")
            continue
        io.write("This command is not available in this session.")


def _print_final_text(session: LiveSession, output: Any | None) -> None:
    text = session.engine.text
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Text: {output}", flush=True)
    else:
        print("\n--- final text ---")
        print(text)


def run_session_roster(
    args: argparse.Namespace,
    *,
    io: Any,
    roster: LiveSessionRoster,
    backend_provenance: dict[str, Any],
    teacher_tape: Any | None = None,
    initial_tape: ReplayPlan | None = None,
    store: Any | None = None,
    load_saved_session: Any | None = None,
    default_save_id: str | None = None,
) -> int:
    """Drive all execution against in-memory sessions; storage is an EDGE action."""

    pending_tape = (
        teacher_tape.plan if teacher_tape is not None else initial_tape
    )
    while True:
        session = roster.active_session
        announced_teacher_tape = (
            teacher_tape is not None and pending_tape is teacher_tape.plan
        )
        try:
            result = run_plan(
                session,
                divergence_policy=args.divergence_policy,
                tape=pending_tape,
                live_policy=episode_policy_setup.interactive_policy(
                    args, io, session=session
                ),
            )
        except EdgeRequested:
            pending_tape = None
            action, value = session_edge_menu(io, roster, store=store)
        except ChordRequested as request:
            pending_tape = None
            chord = Chord(session.engine, request.ranks)
            try:
                chord_action, actions = chord_menu(io, chord, promote_on_select=True)
            finally:
                chord.discard()
            if chord_action == "quit":
                roster.discard()
                return 0
            if chord_action == "select":
                assert actions is not None
                session.adopt_promoted_outcomes(chord.selected_outcomes)
                result = run_plan(
                    session,
                    divergence_policy=args.divergence_policy,
                )
                if result.handed_off:
                    io.write(result.handoff_reason or "Chord selection handed off.")
            else:
                result = None
            if session.engine.ended:
                _print_final_text(session, args.output)
                action, value = session_edge_menu(io, roster, store=store)
            elif result is None:
                continue
            else:
                continue
        except ForkRequested as request:
            action, value = "fork", request.boundary
        except SeamlessRewindRequested as request:
            action, value = "rewind", request.boundary
        else:
            pending_tape = None
            if result.handed_off and result.handoff_reason:
                io.write(result.handoff_reason)
            if result.replay_exhausted and announced_teacher_tape:
                io.write(
                    "Teacher plan exhausted at boundary "
                    f"{session.engine.boundary}; live edge reached."
                )
            if session.engine.ended:
                _print_final_text(session, args.output)
            action, value = session_edge_menu(io, roster, store=store)

        session = roster.active_session
        if action == "quit":
            roster.discard()
            return 0
        if action == "end":
            if session.engine.ended:
                _print_final_text(session, args.output)
                roster.discard()
                return 0
            session.quit("menu-end")
            _print_final_text(session, args.output)
            # Ending is an in-memory terminal state. Return to EDGE once so
            # the user can explicitly save that final state before quitting.
            continue
        if action == "continue":
            try:
                session.resume(max_tokens=value)
            except EditorError as exc:
                io.write(str(exc))
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
                child = roster.fork(boundary=target)
            except EditorError as exc:
                io.write(str(exc))
                continue
            session.activate(child.branch.branch_id)
            alias = roster.number_for(session, child.branch.branch_id)
            io.write(f"Forked live branch {alias} at boundary {target}.")
            continue
        if action == "switch":
            try:
                roster.switch(str(value))
            except EditorError as exc:
                io.write(str(exc))
            continue
        if action == "switch-saved":
            if store is None or load_saved_session is None:
                io.write("Loading a saved episode is not available here.")
                continue
            current = session
            try:
                current.suspend()
                loaded = load_saved_session(str(value))
                roster.add_session(loaded)
                roster.switch(loaded.branch.branch_id)
            except (EditorError, OSError, RuntimeError) as exc:
                if current.is_detached:
                    current.activate(current.branch.branch_id)
                io.write(str(exc))
            pending_tape = None
            continue
        if action == "new":
            try:
                roster.new_root(str(value))
            except (EditorError, ValueError) as exc:
                io.write(str(exc))
            pending_tape = None
            continue
        if action == "export":
            session = roster.active_session
            try:
                export_live_teacher_tape(session, value)
                io.write(f"Exported selected branch to {value}.")
            except EditorError as exc:
                io.write(str(exc))
            pending_tape = ReplayPlan()
            continue
        if action == "save":
            session = roster.active_session
            workspace, requested_id = value
            requested_id = requested_id or default_save_id
            try:
                from .episode_materializer import save_live_branch

                provenance = dict(session.environment_stamp.get("backend", backend_provenance))
                identifier = save_live_branch(
                    session,
                    workspace,
                    provenance,
                    episode_id=requested_id,
                )
                io.write(f"Saved selected branch as {identifier} in {workspace}.")
                if requested_id == default_save_id:
                    default_save_id = None
            except (EditorError, OSError, RuntimeError) as exc:
                io.write(str(exc))
            pending_tape = ReplayPlan()
            continue
        if action == "save-family":
            session = roster.active_session
            workspace, requested_root_id = value
            requested_root_id = requested_root_id or default_save_id
            try:
                from .episode_materializer import save_live_family

                provenance = dict(session.environment_stamp.get("backend", backend_provenance))
                identifiers = save_live_family(
                    session,
                    workspace,
                    provenance,
                    root_episode_id=requested_root_id,
                )
                io.write(
                    f"Saved {len(identifiers)} live branches as a family in {workspace}."
                )
                if requested_root_id == default_save_id:
                    default_save_id = None
            except (EditorError, OSError, RuntimeError) as exc:
                io.write(str(exc))
            pending_tape = ReplayPlan()
            continue
        if action == "spr":
            if store is None:
                io.write("Source replay requires a workspace.")
                continue
            from .spr_recipe import (
                ReplayControlPolicy,
                ReplayPlacement,
                compose_replay_plan,
            )

            source_id, until = value
            try:
                recipe = build_source_replay_recipe(
                    store, source_id, until,
                    sampling_factory=SamplerConfig.from_record,
                )
                pending_tape = compose_replay_plan(
                    recipe,
                    ReplayPlacement.APPEND_TO_CURRENT_BRANCH,
                    ReplayControlPolicy.PRESERVE_DESTINATION,
                )
            except (EditorError, ValueError) as exc:
                io.write(str(exc))
                pending_tape = ReplayPlan()
                continue
            continue
        raise AssertionError(f"unhandled live-session action {action!r}")


def run_new_session(
    args: argparse.Namespace,
    *,
    io: Any,
    teacher_tape: Any | None,
) -> int:
    """Start a new in-memory session with optional EDGE saves."""

    if args.new_prompt is None and args.new_prompt_file is None:
        raise EditorError(
            "--ephemeral requires --new-prompt, --new-prompt-file, or a "
            "teacher-plan envelope prompt"
        )
    initial_text = (
        args.new_prompt
        if args.new_prompt is not None
        else read_prompt_file(args.new_prompt_file)
    )
    backend = episode_backend_loader.load_backend(args)
    provenance = backend.provenance()
    sampling = sampler_from_args(args)
    activation_artifact = None
    if args.activation_strength is not None and args.activation_vector is None:
        raise EditorError("--steering-strength requires --steering-vector")
    if args.activation_vector is not None:
        from .activation_vectors import SteeringVectorArtifact

        activation_artifact = SteeringVectorArtifact.from_path(args.activation_vector)
        activation_artifact.validate_against_backend(backend)
        sampling = apply_activation_artifact(sampling, activation_artifact, args)
    guidance_backend = None
    if episode_backend_loader.cfg_required(
        sampling, plan=teacher_tape.plan if teacher_tape is not None else None
    ):
        io.write("Loading second model copy for CFG prefix guidance...")
        guidance_backend = episode_backend_loader.load_cfg_guidance_backend(
            args,
            backend.provenance(),
        )
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
    return run_session_roster(
        args,
        io=io,
        roster=LiveSessionRoster(session),
        backend_provenance=provenance,
        teacher_tape=teacher_tape,
        default_save_id=getattr(args, "episode_id", None),
    )


__all__ = ["session_edge_menu", "run_new_session", "run_session_roster"]
