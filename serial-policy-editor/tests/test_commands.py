from __future__ import annotations

from unittest.mock import patch

import pytest

from trajectory_editor.domain import ActionKind, EditorError, InsertMode
from trajectory_editor.tui import (
    CommandKind,
    ForkAddressKind,
    TerminalIO,
    parse_command,
)


def parse(raw: str):
    return parse_command(raw, menu_size=12, default_hold_tokens=24, vocabulary_size=1000)


def test_policy_view_toggle_is_a_view_command() -> None:
    command = parse("v")
    assert command.kind == CommandKind.POLICY_VIEW
    assert command.invoked_as == 'v'
    column = parse("V")
    assert column.kind == CommandKind.POLICY_COLUMN
    assert column.invoked_as == 'V'
    assert parse('policy-column').kind == CommandKind.POLICY_COLUMN


def test_editor_actions() -> None:
    assert parse('accept').action.kind == ActionKind.ACCEPT
    with pytest.raises(EditorError):
        parse("a")
    assert parse('7').action.selected_rank == 7
    assert parse('t continuation').action.insert_mode == InsertMode.CONTINUATION
    assert parse('x exact').action.insert_mode == InsertMode.EXACT


def test_literal_tab_survives_in_insertion_text() -> None:
    continuation = parse("t \t").action
    exact = parse("x \t").action
    assert continuation.supplied_text == '\t'
    assert continuation.insert_mode == InsertMode.CONTINUATION
    assert exact.supplied_text == '\t'
    assert exact.insert_mode == InsertMode.EXACT


def test_holds_and_notes_are_not_edit_actions() -> None:
    assert parse('h').hold_tokens == 24
    assert parse('h 3').hold_tokens == 3
    assert parse('hold 3').hold_tokens == 3
    assert parse('h .').hold_tokens == 24
    assert parse('h .').hold_boundary == 'sentence'
    assert parse('h . 9').hold_tokens == 9
    assert parse('h |').hold_boundary == 'newline'
    assert parse('hold / 7').hold_tokens == 7
    assert parse('h end').kind == CommandKind.FINISH
    assert parse('h end').invoked_as == 'h end'
    assert parse('finish').kind == CommandKind.FINISH
    assert parse('n before').kind == CommandKind.NOTE_BEFORE
    assert parse('p after').kind == CommandKind.NOTE_AFTER


@pytest.mark.parametrize(
    "compact,canonical",
    [
        ('h5', 'h 5'),
        ('h.5', 'h . 5'),
        ('h .5', 'h . 5'),
        ('h. 5', 'h . 5'),
        ('h/5', 'h / 5'),
        ('h /5', 'h / 5'),
        ('h/ 5', 'h / 5'),
        ('h.', 'h .'),
        ('h/', 'h /'),
        ('h|5', 'h | 5'),
        ('h |5', 'h | 5'),
        ('h| 5', 'h | 5'),
        ('h|', 'h |'),
        ('m10', 'm 10'),
        ('ms+10', 'ms + 10'),
        ('ms +10', 'ms + 10'),
        ('ms+ 10', 'ms + 10'),
        ('ms-10', 'ms - 10'),
        ('ms-', 'ms -'),
        ('c900', 'c 900'),
        ('H.9', 'h . 9'),
        ('MS+9', 'ms + 9'),
    ],
)
def test_compact_structured_commands_match_their_spaced_forms(compact, canonical) -> None:
    assert parse(compact) == parse(canonical)


def test_compact_command_families_do_not_collide() -> None:
    assert parse('h/5').kind == CommandKind.HOLD
    assert parse('h/5').hold_boundary == 'newline'
    assert parse('m10').kind == CommandKind.MENU_EXPAND
    assert parse('ms+10').kind == CommandKind.TOKEN_SEARCH_VIEW
    assert parse('ms+10').search_direction == '+'
    assert parse('/h5').kind == CommandKind.TOKEN_SEARCH
    assert parse('/h5').search_query == 'h5'
    assert parse('h/5').warning is not None
    assert parse('h|5').warning is None


def test_fork_addresses_are_structured_and_compact() -> None:
    current = parse('f')
    assert current.kind == CommandKind.FORK
    assert current.fork_address.kind == ForkAddressKind.CURRENT
    with pytest.raises(EditorError):
        parse('f last')


@pytest.mark.parametrize('spelling', ('f 5', 'f5', 'fork 5'))
def test_fork_addresses_are_structured_and_compact_absolute(spelling) -> None:
    address = parse(spelling).fork_address
    assert address.kind == ForkAddressKind.ABSOLUTE
    assert address.value == 5


@pytest.mark.parametrize('spelling', ('f - 5', 'f -5', 'f-5', 'fork -5'))
def test_fork_addresses_are_structured_and_compact_relative(spelling) -> None:
    address = parse(spelling).fork_address
    assert address.kind == ForkAddressKind.RELATIVE_BACKWARD
    assert address.value == 5


def test_text_bearing_commands_keep_their_delimiters_and_payloads() -> None:
    continuation = parse('t h5 ').action
    exact = parse('x ms+10 ').action
    assert continuation.supplied_text == 'h5 '
    assert exact.supplied_text == 'ms+10 '
    assert parse('n h.5 ').note == 'h.5 '
    assert parse('p m10 ').note == 'm10 '
    assert parse('/ h5').search_query == ' h5'


@pytest.mark.parametrize('raw', ('th5', 'xms+10', 'nh.5', 'pm10'))
def test_text_bearing_commands_keep_their_delimiters_and_payloads_invalid_payloads(raw) -> None:
    with pytest.raises(EditorError):
        parse(raw)


@pytest.mark.parametrize('raw', ('f - 0', 'flast', 'h+5', 'h5x', 'm+10', 'm10x', 'ms10', 'ms+-10', 'hold5', 'more10'))
def test_unimplemented_or_malformed_compact_forms_remain_invalid(raw) -> None:
    with pytest.raises(EditorError):
        parse(raw)


def test_teacher_eog_commands_are_distinct_from_finish() -> None:
    assert parse('e').kind == CommandKind.TEACHER_EOG
    assert not parse('e').force
    assert parse('eog').kind == CommandKind.TEACHER_EOG
    assert parse('e!').kind == CommandKind.TEACHER_EOG
    assert parse('e!').force
    assert parse('eog!').force
    assert parse('q').kind == CommandKind.FINISH


def test_menu_and_context_views_are_non_edit_commands() -> None:
    assert parse('m').kind == CommandKind.MAIN_MENU
    assert parse('m').additional_rows is None
    assert parse('m1').kind == CommandKind.MENU_EXPAND
    assert parse('m1').additional_rows == 1
    assert parse('m 25').additional_rows == 25
    assert parse('c').kind == CommandKind.CONTEXT
    assert parse('c').context_characters == 2000
    assert parse('c 900').context_characters == 900
    assert parse('c all').context_characters == 'all'


def test_historical_boundary_review_commands_are_distinct() -> None:
    assert parse('[').kind == CommandKind.REVIEW_BACK
    assert parse(']').kind == CommandKind.REVIEW_FORWARD
    assert parse('\x1b').kind == CommandKind.MAIN_MENU
    assert parse('\x1b').invoked_as == 'escape'


def test_exact_token_search_preserves_payload_and_has_separate_views() -> None:
    literal = parse("/ content")
    assert literal.kind == CommandKind.TOKEN_SEARCH
    assert literal.search_query == ' content'
    assert literal.invoked_as == '/ content'
    escaped = parse('/"\\n"')
    assert escaped.search_query == '\n'
    assert parse('ms').kind == CommandKind.TOKEN_SEARCH_VIEW
    assert parse('ms +').search_direction == '+'
    assert parse('ms +').search_rows == 3
    assert parse('ms - 9').search_direction == '-'
    assert parse('ms - 9').search_rows == 9


def test_runtime_bias_group_command_preserves_bare_and_quoted_members() -> None:
    command = parse('b nautical -> {anchor, " shadow"}')

    assert command.kind == CommandKind.BIAS
    assert command.bias_group_name == "nautical"
    assert command.bias_group_members == (" anchor", " shadow")
    assert command.bias_group_member_bare == (True, False)


@pytest.mark.parametrize('rank', (13, 499, 700, 1000))
def test_rank_selection_does_not_require_menu_exposure(rank) -> None:
    assert parse(str(rank)).action.selected_rank == rank


@pytest.mark.parametrize('raw', ('', '1001', '0', 'h 0', 'h . 0', 'h / nope', 'h . 2 extra', 't', 'm 0', 'c 0'))
def test_invalid_commands(raw) -> None:
    with pytest.raises(EditorError):
        parse(raw)


def test_keyboard_interrupt_is_not_converted_to_finish() -> None:
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            TerminalIO().read("> ")
