from unittest.mock import patch

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import engine
from trajectory_editor.domain import EditorError
from trajectory_editor.episode_actions import SelectRawRank
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.live_tui import action_preview, read_live_choice
from trajectory_editor.tui import parse_command


def test_rank_neighborhood_parser_preserves_existing_forms():
    def parse(text):
        return parse_command(text, menu_size=1, vocabulary_size=1000, default_hold_tokens=10)
    assert parse("ms 700").search_rank == 700
    assert parse("ms").search_rank is None
    assert parse("ms + 10").search_direction == "+"
    assert parse("ms - 10").search_rows == 10
    for text in ("ms 0", "ms 1001", "ms 700 extra"):
        with pytest.raises(EditorError):
            parse(text)


def test_direct_rank_focus_requires_no_tokenization_or_commit():
    e = engine()
    observation = e.observe()
    io = ScriptedIO(["ms 7", "ms - 2", "ms", "7"])
    with patch.object(e.backend, "tokenize", side_effect=AssertionError("not a text search")):
        action = InteractivePolicy(io=io, menu_size=1).choose(e, observation)
    assert action == SelectRawRank(7)
    assert e.boundary == 0
    assert e.backend.tokens == list(e.initial_token_ids)
    assert any("absolute raw rank=7" in line for line in io.output)


def test_numeric_preview_resolves_token_and_caches_per_decision():
    e = engine()
    observation = e.observe()

    class PreviewIO:
        supports_live_choices = True

        def read_choice(self, choice, **kwargs):
            resolve = kwargs["resolve_candidate"]
            first = resolve(7)
            assert resolve(7) is first
            preview = action_preview(choice, "7", kwargs["candidates"],
                                     kwargs["resolve_insertion"], resolve_candidate=resolve)
            assert preview.token_id == first.token_id
            assert preview.appended_text == first.text
            assert preview.raw_probability == first.raw_probability
            assert len(kwargs["candidates"]) == 1  # Preview does not expand the menu.
            return "7"

    with patch.object(e, "candidates", wraps=e.candidates) as candidates:
        action = InteractivePolicy(io=PreviewIO(), menu_size=1).choose(e, observation)
    assert action == SelectRawRank(7)
    assert candidates.call_count == 2  # Initial menu and one numeric preview.
    assert e.boundary == 0


@pytest.mark.parametrize("keys,result", [
    ("7\x07", "ms 7"),
    ("7\n", "7"),
    ("999\x07\n", "999"),  # Invalid Ctrl+G does not navigate or submit.
    ("t hello\x07\n", "t hello"),
])
def test_ctrl_g_navigates_while_enter_keeps_submission_meaning(keys, result):
    e = engine()
    observation = e.observe()
    rows = e.candidates(observation, count=1)
    choice = _choice_from_observation(e, observation, rows, context_characters=100, serial=1)
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        actual = read_live_choice(
            choice, remaining_tokens=e.remaining, candidates=rows,
            resolve_insertion=lambda text, mode: text,
            resolve_candidate=lambda rank: e.candidates(observation, start_rank=rank, count=1)[0],
            initial_command=str(observation.proposal_raw_rank),
            input_device=pipe, output_device=DummyOutput(),
        )
    assert actual == result
    assert e.boundary == 0
