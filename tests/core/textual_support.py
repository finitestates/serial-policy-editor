"""Shared prepared states and Pilot helpers for the Textual terminal tests."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from typing import Any

from trajectory_editor.core.candidates import Candidate
from trajectory_editor.core.ui import ChoiceSet
from trajectory_editor.terminal_contracts import (
    BeamViewRow,
    BeamViewState,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
)
from trajectory_editor.textual_tui import PolicyEditorApp, _RequestLifecycle


def choice_state(**changes: Any) -> ChoiceViewState:
    candidates = (
        Candidate(1, 2, " alpha", .8, False, .8),
        Candidate(2, 3, " beta", .2, False, .2),
    )
    choice = ChoiceSet(
        "choice", "prompt", 0, 0, "0" * 64, "context", 2, " alpha",
        .8, .8, False, candidates, vocabulary_size=5, proposal_raw_rank=1,
    )
    state = ChoiceViewState(
        choice,
        candidates,
        lambda text, mode: text,
        resolve_candidate=lambda rank: candidates[rank - 1],
    )
    return ChoiceViewState(**{**state.__dict__, **changes})


def edge_state(*, mode: str = "episode") -> EdgeViewState:
    return EdgeViewState("episode-1", 3, "temperature=0.7 · top_k=20", mode)


def beam_state(
    *, at_edge: bool = False, stochastic: bool = False, row_count: int = 2
) -> BeamViewState:
    rows = tuple(
        BeamViewRow(
            f"b{rank}",
            (
                "alpha continuation" if rank == 1 else
                "beta continuation" if rank == 2 else
                f"continuation for branch {rank:02d}"
            ),
            (
                "-0.45" if rank == 1 else
                "-1.25" if rank == 2 else
                f"-{0.45 + (rank - 1) * 0.4:.2f}"
            ),
            "LIVE",
            ("alpha",) if rank == 1 else ("beta",) if rank == 2 else (f"step {rank}",),
            model_rank=2 if rank == 1 else 4 if rank == 2 else rank + 1,
            step_log_probability=-0.25 * rank,
            model_log_probability=-0.8 - (rank - 1) * 0.4,
            protected=rank == 1,
            family_metadata="family A" if rank == 1 else "family B" if rank == 2 else f"family {rank}",
        )
        for rank in range(1, row_count + 1)
    )
    return BeamViewState(
        "BEAM · width 2 · depth 1",
        "shared context",
        rows,
        "b1",
        notice="beam notice",
        at_edge=at_edge,
        stochastic=stochastic,
    )


def run_pilot(scenario):
    return asyncio.run(scenario())


async def install_request(app: PolicyEditorApp, pilot, state, *, generation: int = 1):
    lifecycle = _RequestLifecycle(
        generation=generation,
        state=state,
        response=Future(),
        owner_queue=None,
    )
    mount = app.show_request(lifecycle)
    if mount is not None:
        await mount
    await pilot.pause()
    return lifecycle


def submitted(lifecycle: _RequestLifecycle):
    assert lifecycle.response.done()
    return lifecycle.response.result()


def prompt_state(prompt: str = "Input › ", **flags: bool) -> PromptRequest:
    return PromptRequest(prompt, **flags)
