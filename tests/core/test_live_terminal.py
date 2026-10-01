"""Behavior of the live terminal views: what each key submits and what shows."""

from __future__ import annotations

from dataclasses import replace

import pytest
from trajectory_editor.core.candidates import Candidate
from trajectory_editor.edge_help import edge_help
from trajectory_editor.terminal_contracts import (
    SEAMLESS_REACTIVATE,
    BeamInput,
    BoundaryReview,
    ChoiceFeedback,
)
from trajectory_editor.ui_themes import LIVE_THEME_NAMES

from tests.core.term_support import (
    Harness,
    beam_state,
    choice_state,
    edge_state,
    prompt_state,
)


@pytest.fixture
def harness():
    created = []

    def make(state=None, **kwargs):
        instance = Harness(state, **kwargs)
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance.close()


def _wide_choice(count: int = 12):
    state = choice_state()
    candidates = tuple(
        Candidate(rank, rank + 1, f" token-{rank}", 1 / rank, False, 1 / rank)
        for rank in range(1, count + 1)
    )
    return replace(state, candidates=candidates,
                   choice=replace(state.choice, candidates=candidates, vocabulary_size=100))


def _review(**changes):
    return replace(
        choice_state(),
        review=BoundaryReview(2, 1, "historical text", {"kind": "token-boundary"}),
        **changes,
    )


# -- one request of each kind ---------------------------------------------------


def test_choice_edge_beam_and_prompt_requests_submit_their_contract_values(harness):
    ui = harness(choice_state())
    assert ui.lines[0].startswith("Step 0")
    assert "' alpha'" in ui.text and "' beta'" in ui.text
    ui.press("1", "enter", "x")
    assert ui.result() == "1"

    ui.show(edge_state())
    assert "episode-1" in ui.text
    ui.type("s top_k=2")
    ui.press("enter")
    assert ui.result() == "s top_k=2"

    ui.show(beam_state())
    assert "shared context" in ui.text
    ui.press("enter")
    assert ui.result() == BeamInput("select b1", "b1")

    ui.show(prompt_state())
    ui.type("foo")
    ui.press("enter")
    assert ui.result() == "foo"


def test_input_after_submit_is_dropped_until_the_next_request(harness):
    ui = harness(choice_state())
    ui.feed("1\rxyz")
    assert ui.result() == "1"
    assert ui.view.editor.text == "1"
    assert ui.app.stats["dropped_input"] == 3
    ui.show(prompt_state())
    assert ui.view.editor.text == ""
    ui.feed("y\r")
    assert ui.result() == "y"


def test_raw_rank_can_be_submitted_while_its_optional_preview_is_pending(harness):
    ui = harness(choice_state(resolve_candidate=lambda _rank: None))
    ui.type("5")
    assert not ui.lifecycle.owner_queue.empty()
    assert "Resolving raw rank" not in ui.text
    assert "Press Enter to select raw rank 5." in ui.text
    ui.press("enter")
    assert ui.result() == "5"


def test_submitted_view_stays_displayed_without_internal_status_until_replaced(harness):
    ui = harness(choice_state())
    ui.press("1", "enter")
    assert ui.result() == "1"
    assert "Step 0" in ui.text and "' alpha'" in ui.text
    assert "Command > 1" in ui.text
    assert "working…" not in ui.text
    assert ui.canvas.cursor is None
    ui.show(edge_state())
    assert "LIVE EDGE" in ui.text
    assert "working…" not in ui.text


def test_preview_exception_details_are_only_in_captured_output(harness):
    from trajectory_editor.core.errors import EditorError

    def fail_preview(_rank):
        raise EditorError("backend diagnostic detail")

    ui = harness(choice_state(resolve_candidate=fail_preview))
    ui.type("5")
    ui.run_previews()
    ui.frame()
    assert "preview unavailable" in ui.text.lower()
    assert "backend diagnostic detail" not in ui.text
    assert "backend diagnostic detail" in ui.app.output_history()
    ui.press("ctrl+l")
    assert "Captured output" in ui.text
    assert "[diagnostic] candidate preview failed" in ui.text
    assert "backend diagnostic detail" in ui.text


# -- choice -----------------------------------------------------------------------


@pytest.mark.parametrize(("command", "initial_command"),
                         [("m20", None), ("beam", None), ("m20", "2"), ("beam", "2")])
def test_choice_prefilled_command_is_replaced_by_typing(harness, command, initial_command):
    ui = harness(choice_state(initial_command=initial_command))
    if initial_command:
        assert ui.view.editor.text == initial_command
        assert ui.view.editor.replace_on_type
    ui.type(command)
    assert ui.view.editor.text == command
    assert not ui.done


def test_choice_prefilled_command_submits_unchanged_and_backspace_clears_it(harness):
    ui = harness(choice_state(initial_command="2"))
    ui.press("enter")
    assert ui.result() == "2"
    ui.show(choice_state(initial_command="2"))
    ui.press("backspace")
    assert ui.view.editor.text == ""


@pytest.mark.parametrize("size", [(120, 40), (160, 50), (80, 24)])
def test_choice_table_shows_every_candidate_above_the_command_row(harness, size):
    ui = harness(_wide_choice(), size=size)
    command_row = ui.row_of("Command >")
    for rank in range(1, 13):
        assert ui.row_of(f"token-{rank}'") < command_row
    assert ui.row_of("Step 0") == 0
    assert command_row < size[1] - 1  # the hint stays below it


def test_choice_menu_expansion_shows_the_new_rows(harness):
    ui = harness(choice_state())
    ui.type("m 10")
    ui.press("enter")
    assert ui.result() == "m 10"
    ui.show(_wide_choice())
    assert "token-12'" in ui.text
    assert ui.row_of("token-12'") < ui.row_of("Command >")


@pytest.mark.parametrize("theme", LIVE_THEME_NAMES)
def test_choice_invalid_submission_is_feedback_and_stays_editable(harness, theme):
    ui = harness(choice_state(), theme=theme)
    ui.press("9", "enter")
    assert not ui.done
    assert "COMMAND" in ui.text.upper()
    assert "rank must be 1..5" in ui.text.lower()
    assert ui.view.editor.text == "9"
    ui.press("backspace")
    assert ui.view.editor.text == ""
    ui.press("2", "enter")
    assert ui.result() == "2"


def test_choice_displays_feedback_title_and_detail_lines(harness):
    state = choice_state(feedback=ChoiceFeedback("error", "INVALID BIAS", ("unknown group", "rank 2 is unchanged")))
    ui = harness(state)
    title = ui.row_of("INVALID BIAS")
    assert ui.lines[title + 1].strip() == "unknown group"
    assert ui.lines[title + 2].strip() == "rank 2 is unchanged"


def test_choice_marker_follows_the_live_command_preview(harness):
    ui = harness(choice_state())
    marked = [line for line in ui.lines if line.startswith("▶")]
    assert len(marked) == 1 and "' alpha'" in marked[0]
    ui.press("2")
    assert ui.view._preview.candidate_rank == 2
    marked = [line for line in ui.lines if line.startswith("▶")]
    assert len(marked) == 1 and "' beta'" in marked[0]


def test_choice_click_on_a_candidate_stages_its_rank(harness):
    ui = harness(choice_state())
    x, y = ui.find("' beta'")
    ui.click(x, y)
    assert ui.view.editor.text == "2"
    assert not ui.done
    ui.press("enter")
    assert ui.result() == "2"


@pytest.mark.parametrize(
    ("feedback", "key", "expected"),
    [
        (ChoiceFeedback("info", "suggestions", completion_commands=("t alpha", "x beta")), "tab", "t alpha"),
        (ChoiceFeedback("info", "suggestions", completion_commands=("t alpha", "x beta")), "shift+tab", "x beta"),
        (ChoiceFeedback("search", "matches", initial_tab_command="2"), "tab", "2"),
    ],
)
def test_choice_tab_uses_feedback_completion_and_search_lens(harness, feedback, key, expected):
    state = choice_state(
        feedback=feedback,
        search_lens_active=feedback.category == "search",
        target_token_id=3 if feedback.category == "search" else None,
        display_candidates=(choice_state().candidates[1],) if feedback.category == "search" else None,
    )
    ui = harness(state)
    ui.press(key)
    assert ui.view.editor.text == expected
    ui.press("enter")
    assert ui.result() == expected


def test_choice_search_lens_falls_back_to_target_token_rank(harness):
    ui = harness(choice_state(search_lens_active=True, target_token_id=3,
                              display_candidates=(choice_state().candidates[1],)))
    ui.press("shift+tab")
    assert ui.view.editor.text == "2"
    ui.press("escape")
    assert ui.result() == "\x1b"


def test_choice_tab_and_arrows_cycle_candidates(harness):
    ui = harness(_wide_choice(4))
    ui.press("tab")
    assert ui.view.editor.text == "1"
    ui.press("tab")
    assert ui.view.editor.text == "2"
    ui.press("down")
    assert ui.view.editor.text == "3"
    ui.press("up", "shift+tab")
    assert ui.view.editor.text == "1"


@pytest.mark.parametrize("prefix", ("t ", "x "))
def test_choice_expanded_authored_editor_and_alt_enter_newline(harness, prefix):
    ui = harness(choice_state())
    ui.press("ctrl+e")
    assert not ui.view._expanded
    ui.type(prefix + "a")
    ui.press("ctrl+e", "alt+enter", "b")
    assert ui.view.editor.text == f"{prefix}a\nb"
    assert ui.view._expanded
    command = ui.row_of("Command >")
    assert ui.lines[command + 1].strip() == "b"
    ui.press("ctrl+e")
    assert not ui.view._expanded
    ui.press("enter")
    assert ui.result() == f"{prefix}a\nb"


def test_choice_tab_inserts_a_tab_while_writing(harness):
    ui = harness(choice_state())
    ui.type("t a")
    ui.press("tab")
    assert ui.view.editor.text == "t a\t"


def test_choice_ctrl_g_submits_numeric_rank_exploration(harness):
    ui = harness(choice_state())
    ui.press("2", "ctrl+g")
    assert ui.result() == "ms 2"


@pytest.mark.parametrize("key", ["[", "]"])
def test_choice_brackets_open_review_only_from_an_empty_command(harness, key):
    ui = harness(choice_state())
    ui.press(key)
    assert ui.result() == key
    ui.show(choice_state())
    ui.type("t a")
    ui.press(key)
    assert ui.view.editor.text == "t a" + key


def test_choice_ctrl_d_closes_only_an_empty_command(harness):
    ui = harness(choice_state())
    ui.press("1", "ctrl+d")
    assert not ui.done
    ui.press("backspace", "ctrl+d")
    assert ui.result() is None


def test_choice_context_paging_suspends_tail_follow(harness):
    state = choice_state()
    tail = "\n".join(f"line {index}" for index in range(80))
    ui = harness(replace(state, choice=replace(state.choice, context_text_tail=tail)))
    assert "line 79" in ui.text
    ui.press("pageup")
    assert ui.view.context_scroll.follow is False
    assert "line 79" not in ui.text
    ui.press("pagedown", "pagedown", "pagedown", "pagedown")
    assert ui.view.context_scroll.follow is True
    assert "line 79" in ui.text


def test_choice_context_keeps_long_history_and_wraps_at_any_width(harness):
    state = choice_state()
    tail = "word " * 4000
    ui = harness(replace(state, choice=replace(state.choice, context_text_tail=tail)), size=(60, 20))
    assert all(len(line) <= 60 for line in ui.lines)
    ui.frame((150, 45))
    assert "DECISION BOUNDARY" not in ui.text  # following the tail of a long context
    ui.press("pageup")
    assert "word" in ui.text


def test_choice_preview_is_cached_per_command_and_generation(harness, monkeypatch):
    from trajectory_editor.term import views

    calls = []
    original = views.action_preview
    monkeypatch.setattr(views, "action_preview", lambda *a, **k: calls.append(a[1]) or original(*a, **k))
    ui = harness(choice_state())
    ui.frame()
    ui.frame()
    assert calls == [""]
    ui.type("t")
    assert calls == ["", "t"]
    ui.show(choice_state())
    assert calls == ["", "t", ""]


def test_choice_insertion_preview_resolves_on_the_owner_thread(harness):
    seen = []

    def resolve(text, mode):
        seen.append(text)
        return f"<{text}>"

    ui = harness(choice_state(resolve_insertion=resolve))
    ui.type("x hello")
    assert seen == []  # nothing runs on the UI thread
    assert not ui.lifecycle.owner_queue.empty()
    ui.run_previews()
    assert seen[-1] == "hello"
    assert "<hello>" in ui.view._owner_preview_values.values()


def test_choice_step_heading_and_context_track_each_request(harness):
    state = choice_state()
    ui = harness(state)
    ui.type("t x")
    ui.press("enter")
    assert ui.result() == "t x"
    next_choice = replace(state.choice, aligned_step=1, context_text_tail="context x")
    ui.show(choice_state(choice=next_choice))
    assert ui.lines[0].startswith("Step 1")
    assert "context x" in ui.text


# -- historical review --------------------------------------------------------------


@pytest.mark.parametrize(
    ("seamless", "reactivate", "key", "expected"),
    [
        (True, True, "enter", SEAMLESS_REACTIVATE),
        (False, False, "enter", "\x1b"),
        (True, True, "escape", "\x1b"),
        (False, False, "[", "["),
        (False, False, "]", "]"),
        (False, False, "a", "\x1b"),
        (False, False, "tab", "\x1b"),
    ],
)
def test_review_keys_return_the_review_contract(harness, seamless, reactivate, key, expected):
    ui = harness(_review(seamless=seamless, reactivate_on_review_enter=reactivate))
    assert "HISTORICAL CONTEXT" in ui.text
    ui.press(key)
    assert ui.result() == expected


def test_review_f_stages_a_fork_and_enter_submits_it(harness):
    ui = harness(_review())
    ui.press("f")
    assert not ui.done
    assert ui.view.editor.text == "f"
    ui.press("enter")
    assert ui.result() == "f"


def test_review_ignores_editor_expansion(harness):
    ui = harness(_review())
    ui.press("ctrl+e")
    assert ui.result() == "\x1b"


# -- edge -------------------------------------------------------------------------


def test_edge_blank_enter_ctrl_d_and_ctrl_c(harness):
    ui = harness(edge_state())
    ui.press("enter")
    assert ui.result() == ""
    ui.show(edge_state())
    ui.press("ctrl+d")
    assert ui.result() is None
    ui.show(edge_state())
    ui.press("ctrl+c")
    with pytest.raises(KeyboardInterrupt):
        ui.result()


def test_edge_template_click_and_arrows_stage_commands(harness):
    ui = harness(edge_state())
    x, y = ui.find("new TEXT")
    ui.click(x, y)
    assert ui.view.editor.text == "new "
    ui.press("enter")
    assert ui.result() == "new "
    ui.show(edge_state())
    ui.press("down")
    first = edge_help("episode")[0].command
    assert ui.view.editor.text.strip() == first.split()[0]


# -- beam --------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "initial", "selected"), [("up", "b2", "b1"), ("down", "b1", "b2")])
def test_beam_arrows_move_the_selection_and_enter_commits_it(harness, key, initial, selected):
    ui = harness(replace(beam_state(), selected_label=initial))
    ui.press(key)
    assert f"SELECTED: {selected}" in ui.text
    ui.press("enter")
    assert ui.result() == BeamInput(f"select {selected}", selected)


def test_beam_empty_enter_resumes_at_the_edge(harness):
    ui = harness(beam_state(at_edge=True))
    assert "BEAM OPTIONS" in ui.lines[0]
    ui.press("enter")
    assert ui.result() == BeamInput("resume", "b1")


@pytest.mark.parametrize(
    ("key", "state", "expected", "needs_enter"),
    [
        ("backspace", beam_state(), BeamInput("kill b1", "b1"), False),
        ("p", beam_state(), BeamInput("protect", "b1"), False),
        ("p", beam_state(stochastic=True), BeamInput("p", "b1"), True),
        ("p", beam_state(at_edge=True), BeamInput("p", "b1"), True),
        ("f", beam_state(at_edge=True), BeamInput("f", "b1"), True),
        ("f", beam_state(), BeamInput("families", "b1"), False),
        ("right", beam_state(), BeamInput("advance 1", "b1"), False),
        ("left", beam_state(), BeamInput("rewind", "b1"), False),
        ("right", beam_state(at_edge=True), BeamInput("resume", "b1"), False),
        ("left", beam_state(at_edge=True), BeamInput("resume", "b1"), False),
        ("escape", beam_state(), BeamInput("return", "b1"), False),
        ("ctrl+d", beam_state(), BeamInput("return", "b1"), False),
    ],
)
def test_beam_shortcuts_submit_contract_commands(harness, key, state, expected, needs_enter):
    ui = harness(state)
    ui.press(key)
    if needs_enter:
        assert not ui.done
        assert ui.view.editor.text == expected.command
        ui.press("enter")
    assert ui.result() == expected


def test_beam_shortcut_letters_are_text_when_the_command_is_not_empty(harness):
    ui = harness(beam_state())
    ui.press("x", "p", "f", "enter")
    assert ui.result() == BeamInput("xpf", "b1")


def test_beam_stochastic_score_and_notice(harness):
    ui = harness(beam_state(stochastic=True))
    assert "G −0.45 · log-p -0.800" in ui.text
    assert "beam notice" in ui.text


@pytest.mark.parametrize("size", [(120, 40), (80, 24), (160, 50), (40, 12)])
def test_beam_every_frame_shows_heading_table_details_and_command(harness, size):
    long_continuation = "The wind howls outside, and the shutters rattle against the old stone walls."
    state = beam_state(row_count=40)
    rows = tuple(replace(row, continuation=long_continuation,
                         recent_steps=tuple(f"detail step {step}" for step in range(80)) if index == 0 else row.recent_steps)
                 for index, row in enumerate(state.rows))
    ui = harness(replace(state, rows=rows), size=size)
    for move in range(45):
        selected = ui.view.selected_label
        assert ui.lines[0].startswith("BEAM")
        assert ui.row_of("Beam >") < size[1]
        if size[1] >= 20:
            assert f"SELECTED: {selected}" in ui.text
        marked = [line for line in ui.lines if line.startswith(">")]
        assert len(marked) == 1 and f" {selected} " in marked[0]
        ui.press("down" if move < 40 else "up")


def test_beam_click_selects_a_row(harness):
    ui = harness(beam_state(row_count=5))
    x, y = ui.find(" b4 ")
    ui.click(x + 1, y)
    assert ui.view.selected_label == "b4"
    assert "SELECTED: b4" in ui.text


# -- prompts -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_", "keys", "expected"),
    [
        (prompt_state(), ("a", "enter"), "a"),
        (prompt_state(), ("ctrl+d",), None),
        (prompt_state(), ("escape",), None),
        (prompt_state(single_key=True), ("z",), "z"),
        (prompt_state(single_key=True), ("enter",), "\n"),
        (prompt_state(single_key=True), ("backspace",), "\x7f"),
        (prompt_state(single_key=True), ("escape",), "\x1b"),
        (prompt_state(single_key=True), ("ctrl+d",), None),
        (prompt_state(multiline=True), ("ctrl+d",), None),
        (prompt_state(page=True, body="line one\nline two"), ("enter",), ""),
        (prompt_state(page=True, body="line one\nline two"), ("escape",), ""),
        (prompt_state(page=True, body="line one\nline two"), ("q",), ""),
    ],
)
def test_prompt_variants_submit_their_values(harness, request_, keys, expected):
    ui = harness(request_)
    ui.press(*keys)
    assert ui.result() == expected


@pytest.mark.parametrize(("request_", "submit_keys"),
                         [(prompt_state(), ("enter",)), (prompt_state(multiline=True), ("escape", "enter"))])
def test_prompt_backspace_edits_typed_text(harness, request_, submit_keys):
    ui = harness(request_)
    ui.press("a", "b", "c", "backspace")
    assert ui.view.editor.text == "ab"
    ui.press(*submit_keys)
    assert ui.result() == "ab"


def test_page_scrolls_and_multiline_requires_escape_then_enter(harness):
    ui = harness(prompt_state(page=True, body="\n".join(f"page line {index}" for index in range(100))))
    assert "page line 0" in ui.text
    ui.press("pagedown")
    assert "page line 0" not in ui.text
    ui.press("pageup")
    assert "page line 0" in ui.text

    ui.show(prompt_state("New prompt > ", multiline=True, isolated=True))
    assert "Write the new prompt" in ui.text
    ui.press("enter")
    assert not ui.done
    assert ui.view.editor.text == "\n"
    ui.press("escape")
    assert "press Enter to submit" in ui.text
    ui.press("enter")
    assert "Write at least one character." in ui.text
    ui.press("backspace", "a", "escape", "enter")
    assert ui.result() == "a"

    ui.show(prompt_state(multiline=True))
    ui.press("a", "enter", "b", "escape", "enter")
    assert ui.result() == "a\nb"


def test_paste_keeps_newlines_only_where_text_is_multiline(harness):
    ui = harness(prompt_state(multiline=True))
    ui.paste("one\ntwo")
    assert ui.view.editor.text == "one\ntwo"
    ui.show(prompt_state())
    ui.paste("one\ntwo")
    assert ui.view.editor.text == "one two"
    ui.show(prompt_state(single_key=True))
    ui.paste("ignored")
    assert not ui.done


# -- overlays -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (choice_state(), ""),
        (edge_state(), ""),
        (beam_state(), BeamInput("select b1", "b1")),
        (prompt_state(), ""),
        (_review(), "\x1b"),
    ],
)
def test_help_opens_over_each_request_and_returns_to_it(harness, state, expected):
    ui = harness(state)
    ui.press("f1")
    assert "Help" in ui.text and "commands" in ui.text.lower()
    ui.press("enter")
    assert not ui.done
    ui.press("escape")
    assert "Help" not in ui.lines[0]
    ui.press("enter")
    assert ui.result() == expected


def test_help_scrolls_inside_its_dialog(harness):
    ui = harness(choice_state(), size=(100, 30))
    ui.press("f1")
    first = ui.text
    ui.press("pagedown")
    assert ui.text != first
    ui.press("pageup")
    assert ui.text == first


@pytest.mark.parametrize(
    ("state", "query", "expected"),
    [
        (choice_state(), "x TEXT", "x "),
        (edge_state(mode="session"), "new TEXT", "new "),
        (beam_state(), "advance N", "advance 1"),
    ],
)
def test_ctrl_k_palette_inserts_a_template(harness, state, query, expected):
    ui = harness(state)
    ui.press("ctrl+k")
    assert "Commands" in ui.text
    ui.type(query)
    ui.press("enter")
    assert "Commands" not in ui.text
    assert ui.view.editor.text == expected
    assert not ui.done


def test_palette_is_disabled_for_single_key_prompts(harness):
    ui = harness(prompt_state(single_key=True))
    ui.press("ctrl+k")
    assert ui.app.overlay is None


def test_output_overlay_shows_bounded_history_and_follows_the_tail(harness):
    from trajectory_editor.term.app import OUTPUT_HISTORY_LIMIT

    ui = harness(prompt_state(), size=(80, 24))
    ui.app.write_output("discarded" + "x" * (OUTPUT_HISTORY_LIMIT + 7))
    assert ui.app.output_history() == "x" * OUTPUT_HISTORY_LIMIT
    ui.app.write_output("\nNEWEST")
    ui.press("ctrl+l")
    assert "Captured output" in ui.text
    assert "NEWEST" in ui.text
    ui.press("pageup")
    ui.app.write_output("\nLATER")
    ui.frame()
    assert "LATER" not in ui.text
    ui.press("ctrl+l")
    assert ui.app.overlay is None
    assert len(ui.app.output_history()) == OUTPUT_HISTORY_LIMIT


def test_details_overlay_shows_the_full_choice_context(harness):
    state = choice_state()
    ui = harness(replace(state, choice=replace(state.choice, context_text_tail="the entire context")), size=(60, 24))
    ui.press("f2")
    assert "Details" in ui.text and "the entire context" in ui.text
    ui.press("f2")
    assert ui.app.overlay is None


def test_ctrl_c_interrupts_even_with_an_overlay_open(harness):
    ui = harness(choice_state())
    ui.press("f1", "ctrl+c")
    with pytest.raises(KeyboardInterrupt):
        ui.result()
    assert ui.app.overlay is None


# -- geometry ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(40, 12), (80, 24), (120, 40), (160, 50), (30, 6), (20, 3)])
@pytest.mark.parametrize("make_state", [
    choice_state, _review, edge_state, beam_state, prompt_state,
    lambda: prompt_state(multiline=True), lambda: prompt_state(single_key=True),
    lambda: prompt_state(page=True, body="x\n" * 100),
], ids=["choice", "review", "edge", "beam", "prompt", "multiline", "key", "page"])
def test_every_view_fits_and_keeps_its_input_visible_at_every_size(harness, size, make_state):
    ui = harness(make_state(), size=size)
    assert len(ui.lines) == size[1]
    assert all(len(line) <= size[0] for line in ui.canvas.text_lines())
    request = ui.lifecycle.state
    if size[0] < 30:
        return
    if getattr(request, "page", False):
        assert "returns" in ui.text
    elif getattr(request, "single_key", False):
        assert "Press a key" in ui.text
    elif getattr(request, "review", None) is not None:
        assert "Review >" in ui.text  # read-only: no caret
    else:
        assert ui.canvas.cursor is not None
        x, y = ui.canvas.cursor
        assert 0 <= x < size[0] and 0 <= y < size[1]


@pytest.mark.parametrize("make_state", [choice_state, beam_state, edge_state])
def test_resize_reflows_without_losing_typed_text(harness, make_state):
    ui = harness(make_state(), size=(120, 40))
    ui.type("t typed")
    for size in [(80, 24), (40, 12), (160, 50), (120, 40)]:
        ui.frame(size)
        assert ui.view.editor.text == "t typed"
        assert "t typed" in ui.text
        x, y = ui.canvas.cursor
        assert ui.lines[y][:x].endswith("t typed")


def test_too_small_terminal_says_so(harness):
    ui = harness(choice_state(), size=(15, 2))
    assert "Enlarge" in ui.text


@pytest.mark.parametrize("theme", LIVE_THEME_NAMES)
def test_focus_selection_and_headings_are_styled_in_every_theme(harness, theme):
    ui = harness(choice_state(), theme=theme)
    heading = ui.canvas.cells[0][0][1]
    assert heading.bold
    selected_y = next(y for y, line in enumerate(ui.lines) if line.startswith("▶"))
    assert ui.canvas.cells[selected_y][0][1].reverse
    x, y = ui.canvas.cursor
    field = ui.canvas.cells[y][x][1]
    base = ui.app.styles.base
    # A quiet field: raised where the theme paints its background, else underlined.
    assert field.underline or (field.bgcolor is not None and field.bgcolor != base.bgcolor)
    label = ui.canvas.cells[y][0][1]
    assert label.bold  # the colored prompt label marks focus
    ui.press("1", "enter")
    x = ui.lines[y].index("1")
    assert ui.canvas.cells[y][x][1] != field  # busy: the field no longer looks focused
    assert ui.canvas.cursor is None
    if theme == "monochrome":
        def gray(color):
            if color is None:
                return True
            red, green, blue = color.get_truecolor()
            return red == green == blue

        assert all(gray(style.color) and gray(style.bgcolor)
                   for row in ui.canvas.cells for _char, style in row)


def test_python_and_native_output_is_captured_and_standard_fds_are_restored():
    import subprocess
    import sys

    script = r"""
import os
import sys
from trajectory_editor.tui import _ProcessOutputCapture, _SessionOutput

stdout = _SessionOutput()
stderr = _SessionOutput()
capture = _ProcessOutputCapture(stdout, stderr)
capture.start()
try:
    os.write(1, b"native stdout")
    os.write(2, b"native stderr")
    sys.stdout.write(" python stdout")
    sys.stderr.write(" python stderr")
finally:
    capture.stop()
assert stdout.getvalue() == "native stdout python stdout"
assert stderr.getvalue() == "native stderr python stderr"
print("standard descriptors restored")
"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "core" / "src")},
    )
    assert result.stdout == "standard descriptors restored\n"
    assert result.stderr == ""
