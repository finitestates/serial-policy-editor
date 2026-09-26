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
from .chord import Chord, ChordRequested, chord_menu
from .edge_status import sampler_summary
from .episode_engine import EpisodeEngine
from .episode_prompts import read_new_prompt, read_prompt_file
from .episode_runner import (
    EdgeRequested,
    ForkRequested,
    LiveSessionRunner,
    ReplayPlan,
    SeamlessRewindRequested,
)
from .episode_session import LiveSession, LiveSessionRoster
from .projector import project_live_fork_map
from .teacher_plan import export_live_teacher_tape
from .terminal_contracts import EdgeViewState, PromptRequest, TerminalProtocol


def ephemeral_edge_menu(
    io: TerminalProtocol,
    session_or_roster: LiveSession | LiveSessionRoster,
) -> tuple[str, Any]:
    """Turn ephemeral EDGE input into the next live-session operation."""

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
        raw = io.read_edge(EdgeViewState(
            episode_id=branch_number(session.branch.branch_id),
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
            try:
                if roster is not None:
                    roster.resolve(command.reference)
                    return "switch", command.reference
                branch_id = resolve_branch(command.reference)
                if branch_id not in session.branch_states:
                    raise EditorError(f"Unknown live branch {command.reference!r}.")
                return "switch", branch_id
            except EditorError as exc:
                io.write(str(exc))
                continue
        if isinstance(command, edge_commands.RewindCommand):
            return "rewind", command.boundary
        if isinstance(command, edge_commands.ForkCommand):
            return "fork", command.boundary
        if isinstance(command, edge_commands.ExportCommand):
            return "export", command.path
        if isinstance(command, edge_commands.SaveFamilyCommand):
            return "save-family", (command.workspace, command.root_reference)
        if isinstance(command, edge_commands.SaveCommand):
            return "save", (command.workspace, command.reference)
        if isinstance(command, edge_commands.BudgetCommand):
            return "continue", command.tokens
        if isinstance(command, edge_commands.SamplerCommand):
            payload = command.text or ""
            if not payload:
                io.write("Use sampler key=value.")
                continue
            try:
                session.set_sampler(sampler_override(session.sampler, payload))
            except EditorError as exc:
                io.write(f"[invalid sampler change] {exc}")
            continue
        io.write("This command is not available in an ephemeral session.")


def _print_final_text(session: LiveSession, output: Any | None) -> None:
    text = session.engine.text
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Text: {output}", flush=True)
    else:
        print("\n--- final text ---")
        print(text)


def run_ephemeral(
    args: argparse.Namespace,
    *,
    io: Any,
    teacher_tape: Any | None,
) -> int:
    """Run one CLI-requested live session without opening a workspace."""

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
    roster = LiveSessionRoster(session)
    pending_tape = teacher_tape.plan if teacher_tape is not None else None
    while True:
        session = roster.active_session
        announced_teacher_tape = (
            teacher_tape is not None and pending_tape is teacher_tape.plan
        )
        runner = LiveSessionRunner(session, divergence_policy=args.divergence_policy)
        try:
            result = runner.run(
                tape=pending_tape,
                live_policy=episode_policy_setup.ephemeral_policy(args, io),
            )
        except EdgeRequested:
            pending_tape = None
            action, value = ephemeral_edge_menu(io, roster)
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
                result = runner.run(
                    promoted_outcomes=chord.selected_outcomes,
                )
                if session.engine.ended:
                    _print_final_text(session, args.output)
                    roster.discard()
                    return 0
                if result.handed_off:
                    io.write(result.handoff_reason or "Chord selection handed off.")
                pending_tape = None
            else:
                pending_tape = None
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
                roster.discard()
                return 0
            action, value = ephemeral_edge_menu(io, roster)
        if action == "quit":
            roster.discard()
            return 0
        if action == "end":
            session.quit("menu-end")
            _print_final_text(session, args.output)
            roster.discard()
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
            # Return to EDGE without asking the teacher policy for another
            # action merely because a non-mutating command completed.
            pending_tape = ReplayPlan()
            continue
        if action == "save":
            session = roster.active_session
            workspace, requested_id = value
            try:
                from .episode_materializer import save_live_branch

                identifier = save_live_branch(
                    session,
                    workspace,
                    provenance,
                    episode_id=requested_id,
                )
                io.write(f"Saved selected branch as {identifier} in {workspace}.")
            except (EditorError, OSError, RuntimeError) as exc:
                io.write(str(exc))
            pending_tape = ReplayPlan()
            continue
        if action == "save-family":
            session = roster.active_session
            workspace, requested_root_id = value
            try:
                from .episode_materializer import save_live_family

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


__all__ = ["ephemeral_edge_menu", "run_ephemeral"]
