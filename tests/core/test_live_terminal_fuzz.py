"""Generated input journeys: every frame stays complete whatever the user does.

Hypothesis drives random keys, pastes, clicks, wheel steps, and resizes into
each request view and checks the rendered frame after every action. A failure
shrinks to the shortest action sequence that breaks an invariant.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from trajectory_editor.core.candidates import Candidate
from trajectory_editor.terminal_contracts import BoundaryReview, ChoiceFeedback

from tests.core.term_support import (
    Harness,
    beam_state,
    choice_state,
    edge_state,
    prompt_state,
)

KEYS = [
    "enter", "tab", "shift+tab", "backspace", "up", "down", "left", "right", "pageup",
    "pagedown", "home", "end", "delete", "f1", "f2", "escape", "alt+enter", "ctrl+e", "ctrl+k",
    "ctrl+l", "ctrl+g", "ctrl+w", "ctrl+u", "a", "t", " ", "1", "2", "f", "p", "q", "[", "]",
    "?", "x", "é", "漢",
]

SIZES = [(40, 12), (80, 24), (120, 40), (160, 50), (30, 8), (100, 15)]


def _choice_many():
    state = choice_state(feedback=ChoiceFeedback("info", "NOTE", ("one", "two"), completion_commands=("t a",)))
    candidates = tuple(
        Candidate(rank, rank + 1, f" token-{rank}" * (1 + rank % 3), 1 / rank, False, 1 / rank)
        for rank in range(1, 30)
    )
    tail = "\n".join(f"context line {index} " * 3 for index in range(60))
    return replace(state, candidates=candidates, choice=replace(
        state.choice, candidates=candidates, vocabulary_size=100, context_text_tail=tail))


STATES = {
    "choice": _choice_many,
    "review": lambda: replace(choice_state(), review=BoundaryReview(
        4, 2, "historical " * 200, {"kind": "token-boundary"})),
    "edge": edge_state,
    "beam": lambda: beam_state(row_count=12),
    "beam-edge": lambda: beam_state(at_edge=True),
    "prompt": lambda: prompt_state(body="body " * 300),
    "multiline": lambda: prompt_state(multiline=True),
    "key": lambda: prompt_state(single_key=True),
    "page": lambda: prompt_state(page=True, body="\n".join(f"page {index}" for index in range(200))),
}

action = st.one_of(
    st.tuples(st.just("key"), st.sampled_from(KEYS)),
    st.tuples(st.just("text"), st.text(alphabet="ab 1t\tx漢", min_size=1, max_size=12)),
    st.tuples(st.just("paste"), st.text(alphabet="ab\n 1", min_size=1, max_size=40)),
    st.tuples(st.just("click"), st.tuples(st.integers(0, 159), st.integers(0, 49))),
    st.tuples(st.just("wheel"), st.tuples(st.integers(0, 159), st.integers(0, 49))),
    st.tuples(st.just("resize"), st.sampled_from(SIZES)),
)


def _check(ui: Harness, submissions: list) -> None:
    width, height = ui.size
    rows = ui.canvas.text_lines()
    assert len(rows) == height
    assert all(len(row) <= width for row in rows)
    if ui.lifecycle.response.done():
        submissions.append(1)
        assert ui.canvas.cursor is None or ui.app.overlay is not None
        return
    view = ui.view
    assert view.accepting
    if ui.app.overlay is None and width >= 20 and height >= 3:
        request = ui.lifecycle.state
        read_only = (getattr(request, "review", None) is not None
                     or getattr(request, "page", False) or getattr(request, "single_key", False))
        if not read_only:
            assert ui.canvas.cursor is not None, "the focused input lost its caret"
            x, y = ui.canvas.cursor
            assert 0 <= x < width and 0 <= y < height


@pytest.mark.parametrize("name", list(STATES))
@settings(max_examples=int(__import__("os").environ.get("SPE_FUZZ_EXAMPLES", "40")), deadline=None)
@given(actions=st.lists(action, min_size=1, max_size=25), size=st.sampled_from(SIZES))
def test_random_journeys_keep_every_frame_complete(name, actions, size):
    with Harness(STATES[name](), size=size) as ui:
        submissions: list = []
        for kind, value in actions:
            if ui.lifecycle.response.done():
                ui.show(STATES[name]())
            if kind == "key":
                ui.press(value)
            elif kind == "text":
                ui.type(value)
            elif kind == "paste":
                ui.paste(value)
            elif kind == "click":
                ui.click(*value)
            elif kind == "wheel":
                ui.wheel(*value)
            else:
                ui.frame(value)
            _check(ui, submissions)
