"""Plain terminal formatting and input for shared terminal requests."""

from __future__ import annotations

import pydoc
import sys
import termios

from .candidate_columns import CandidateColumns
from .core.candidates import Candidate
from .core.ui import ChoiceSet
from .edge_help import edge_help
from .terminal_contracts import ChoiceViewState, EdgeViewState, IO, PromptRequest
from .teacher_commands import ACTION_TEXT


def _read_line(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None


def display_choice(
    io: IO,
    choice: ChoiceSet,
    *,
    remaining_tokens: int | None = None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    display_rows: tuple[Candidate, ...] | None = None,
    target_token_id: int | None = None,
) -> None:
    io.write("\n" + "=" * 72)
    remaining = (
        f" | remaining: {remaining_tokens}"
        if remaining_tokens is not None
        else ""
    )
    io.write(
        f"Step {choice.aligned_step}{remaining} | "
        f"context tail: {choice.context_text_tail!r}"
    )
    backend = (
        f"{choice.proposal_raw_probability:.2%}"
        if choice.proposal_raw_probability is not None
        else "--"
    )
    io.write(
        f"Sampled proposal: {choice.proposal_text!r} "
        f"(id={choice.proposal_token_id}, backend={backend}, "
        f"decoder={choice.proposal_decoder_probability:.2%}"
        + (
            f", policy-rank={choice.proposal_policy_rank}"
            if policy_active and choice.proposal_policy_rank is not None
            else ""
        )
        + ")"
    )
    display_candidates(
        io,
        choice.candidates if display_rows is None else display_rows,
        heading=True,
        target_token_id=target_token_id,
        show_policy_rank=show_policy_rank,
        sort_by_policy=sort_by_policy,
        logit_view=logit_view,
        show_model_probabilities=show_model_probabilities,
        column_focus=column_focus,
        overlays=overlays,
        raw_k1_logit=choice.raw_k1_logit,
    )
    display_actions(io)


def display_candidates(
    io: IO,
    candidates: tuple,
    *,
    heading: bool = False,
    target_token_id: int | None = None,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    raw_k1_logit: float | None = None,
) -> None:
    ordered = tuple(candidates)
    if sort_by_policy:
        ordered = tuple(
            sorted(
                ordered,
                key=lambda candidate: (
                    candidate.policy_rank
                    if candidate.policy_rank is not None
                    else candidate.rank,
                    candidate.rank,
                ),
            )
        )
    if raw_k1_logit is None:
        raw_k1_logit = next(
            (candidate.raw_logit for candidate in ordered if candidate.rank == 1),
            None,
        )
    columns = CandidateColumns(
        policy=show_policy_rank,
        logit_view=logit_view,
        show_model_probabilities=show_model_probabilities,
        column_focus=column_focus,
        overlays=overlays,
        raw_k1_logit=raw_k1_logit,
    )
    if heading:
        io.write(f"\n  rank{columns.heading}  text")
    for candidate in ordered:
        suffix = " [END]" if candidate.is_eog else ""
        if candidate.bias:
            suffix += f" [bias {candidate.bias:+g}]"
        if target_token_id is not None and candidate.token_id == target_token_id:
            suffix += " [MATCH]"
        io.write(
            f"  {candidate.rank:>4}{columns.values(candidate)}  {candidate.text!r}{suffix}"
        )


def display_actions(io: IO) -> None:
    io.write(ACTION_TEXT)


def read_choice(io: IO, state: ChoiceViewState) -> str | None:
    if state.feedback is not None:
        io.write(state.feedback.title)
        for line in state.feedback.lines:
            io.write(line)
    if state.review is not None:
        review = state.review
        io.write(f"Review boundary {review.aligned_step} of {review.active_aligned_step}: {review.context_text_tail!r}")
        position = review.position
        if position.get("kind") == "inside-span":
            io.write(
                f"Position: inside {position.get('span_type', 'span')} "
                f"({position.get('offset_visible_tokens')} of "
                f"{position.get('total_visible_tokens')} tokens)"
            )
        elif position.get("kind") == "action-boundary":
            io.write(
                f"Position: {position.get('action_kind', 'action')} "
                f"{position.get('side', '')}"
            )
        if review.next_token is not None:
            io.write(f"Next token: {review.next_token['text']!r}")
    else:
        display_choice(
            io, state.choice,
            remaining_tokens=state.remaining_tokens,
            policy_active=state.policy_active,
            show_policy_rank=state.show_policy_rank,
            sort_by_policy=state.sort_by_policy,
            logit_view=state.logit_view,
            show_model_probabilities=state.show_model_probabilities,
            column_focus=state.column_focus,
            overlays=state.overlays,
            display_rows=state.display_candidates,
            target_token_id=state.target_token_id if state.search_lens_active else None,
        )
    return io.read("\nTeacher action> ")


def read_edge(io: IO, state: EdgeViewState) -> str | None:
    if state.mode == "session":
        io.write(
            f"Live branch {state.episode_id} @ boundary {state.boundary}"
            f" · {state.sampler_summary}"
        )
    else:
        io.write(state.episode_id)
        io.write(f"\nLive edge @ boundary {state.boundary} · {state.sampler_summary}")
    allowance = (
        f"default next allowance: {state.current_budget} tokens"
        if state.current_budget is not None else "no automatic checkpoint"
    )
    remaining = (
        f"{state.remaining_tokens} tokens remaining"
        if state.remaining_tokens is not None else "unlimited"
    )
    io.write(f"Budget: {remaining} · {allowance}")
    for item in edge_help(state.mode):
        io.write(f"[{item.command}] {item.description}")
    return io.read("EDGE> ")


def prompt(io: IO, request: PromptRequest) -> str | None:
    if request.page:
        pydoc.pager(request.body)
        return ""
    if request.multiline:
        from .episode_cli import _read_initial_prompt

        return _read_initial_prompt()
    if request.isolated:
        io.write(request.body)
    if not request.single_key:
        return _read_line(request.prompt)
    if not sys.stdin.isatty():
        value = _read_line(request.prompt)
        return None if value is None else "\n" if value == "" else value[:1]
    print(request.prompt, end="", flush=True)
    descriptor = sys.stdin.fileno()
    prior = termios.tcgetattr(descriptor)
    current = termios.tcgetattr(descriptor)
    current[3] &= ~(termios.ICANON | termios.ECHO)
    current[6][termios.VMIN] = 1
    current[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, current)
        value = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, prior)
        print(flush=True)
    if value == "\x03":
        raise KeyboardInterrupt
    return None if value in {"", "\x04"} else value
