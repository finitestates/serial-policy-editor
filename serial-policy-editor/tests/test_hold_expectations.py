import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Hold
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation


class SequenceBackend(ConformingFakeBackend):
    pieces = {0: "<eog>", 1: "word", 2: ".", 3: " next", 4: "\n", 5: "more", 6: "?", 7: "P"}

    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def last_logits(self):
        position = len(self.tokens) - 1
        token = self.sequence[min(position, len(self.sequence) - 1)]
        logits = np.full(8, -100.0)
        logits[token] = 100.0
        return logits


def runtime(sequence):
    return EpisodeEngine(SequenceBackend(sequence), initial_token_ids=[7],
                         sampling=SamplingConfig(temperature=0))


@pytest.mark.parametrize("expected", [
    ReplayExpectation((1,), None, "requested-length"),
    ReplayExpectation((1, 3, 5, 1, 3), None, "requested-length"),
    ReplayExpectation((5, 5, 5), None, "requested-length"),
    ReplayExpectation((1, 3, 5), None, "newline-boundary"),
])
def test_ballistic_observer_does_not_control_hold(expected):
    baseline = runtime([1, 3, 5]).apply(Hold(3))
    result = runtime([1, 3, 5]).apply(
        Hold(3), replay=True, expectation=expected, divergence_policy="ballistic")
    assert result.visible_token_ids == baseline.visible_token_ids == (1, 3, 5)
    assert result.stop_reason == baseline.stop_reason == "requested-length"
    assert result.status == "completed-with-divergence"


@pytest.mark.parametrize("expected,committed", [
    ((1,), (1,)),                 # Stop before the first extra token.
    ((1, 5, 5), (1,)),           # Stop before the first changed token.
    ((1, 3, 5, 1), (1, 3, 5)),  # Detect a shorter result at completion.
])
def test_handoff_interrupts_without_redefining_limit(expected, committed):
    result = runtime([1, 3, 5]).apply(
        Hold(3), replay=True, expectation=ReplayExpectation(expected),
        divergence_policy="handoff")
    assert result.visible_token_ids == committed
    assert result.status == "handed-off"
    assert result.divergence is not None


@pytest.mark.parametrize("boundary,sequence,committed", [
    ("newline", [1, 4, 5], (1, 4)),
    ("sentence", [1, 2, 3], (1, 2)),
])
def test_current_text_controls_conditional_stops(boundary, sequence, committed):
    baseline = runtime(sequence).apply(Hold(6, boundary))
    result = runtime(sequence).apply(
        Hold(6, boundary), replay=True,
        expectation=ReplayExpectation((1,), None, "requested-length"),
        divergence_policy="ballistic")
    assert result.visible_token_ids == baseline.visible_token_ids == committed
    assert result.stop_reason == baseline.stop_reason == boundary + "-boundary"


def test_sentence_stop_does_not_consume_a_diverging_move():
    result = runtime([1, 2, 3]).apply(
        Hold(6, "sentence"), replay=True,
        expectation=ReplayExpectation((1, 2), None, "sentence-boundary"),
        divergence_policy="handoff")
    assert result.visible_token_ids == (1, 2)
    assert result.status == "completed"
    assert result.divergence is None


@pytest.mark.parametrize("mode", ["handoff", "ballistic"])
def test_stop_reason_is_observed_not_copied(mode):
    result = runtime([1, 3]).apply(
        Hold(2), replay=True, expectation=ReplayExpectation((1, 3), None, "checkpoint"),
        divergence_policy=mode)
    assert result.stop_reason == "requested-length"
    assert result.divergence.reason == "stop-condition-changed"


@pytest.mark.parametrize("count", [1, 40])
@pytest.mark.parametrize("finish", [False, True])
def test_counted_generation_only_decodes_final_span(count, finish):
    from trajectory_editor.episode_actions import Finish

    e = runtime([1, 3, 5])
    if finish:
        e.resume(max_tokens=count)
    render = e.backend.render
    span_calls = []

    def record_render(tokens, *, special=False):
        if not special:
            span_calls.append(tuple(tokens))
        return render(tokens, special=special)

    e.backend.render = record_render
    result = e.apply(Finish() if finish else Hold(count))
    assert len(result.visible_token_ids) == count
    assert span_calls == [result.visible_token_ids]
    assert result.resolved_text == render(result.visible_token_ids)
    assert result.stop_reason == ("checkpoint" if finish else "requested-length")


@pytest.mark.parametrize("boundary,piece", [
    ("sentence", "."), ("sentence", "!"), ("sentence", "?"),
    ("sentence", '.”'), ("sentence", ".\n\n"),
    ("sentence", "word.Next"),
    ("newline", "\n"), ("newline", "\n\n"),
    ("newline", '"\n\n'), ("newline", ".\n\n"),
    ("newline", "\nNext"),
])
@pytest.mark.parametrize("penalties", [False, True])
def test_hold_commits_entire_matching_token_without_lookahead(boundary, piece, penalties):
    from dataclasses import replace
    from unittest.mock import patch

    e = runtime([1, 2, 3])
    e.backend.pieces = {**e.backend.pieces, 2: piece, 3: '”'}
    if penalties:
        e.sampling = replace(e.sampling, repeat_penalty=1.2, presence_penalty=0.5)
    with patch.object(e.backend, "last_logits", wraps=e.backend.last_logits) as observe:
        result = e.apply(Hold(6, boundary))
    assert result.visible_token_ids == (1, 2)
    assert result.resolved_text == "word" + piece
    assert result.stop_reason == boundary + "-boundary"
    assert e.backend.tokens == [7, 1, 2]
    assert observe.call_count == 2
    assert e.observe().proposal_token_id == 3  # Separate closer stays available.


@pytest.mark.parametrize("boundary", ["sentence", "newline"])
def test_conditional_hold_reuses_token_flags_without_extra_decoding(boundary):
    from dataclasses import replace
    from unittest.mock import patch
    from trajectory_editor.boundaries import token_boundaries

    e = runtime([1])
    render = e.backend.render
    span_calls = []

    def record_render(tokens, *, special=False):
        if not special:
            span_calls.append(tuple(tokens))
        return render(tokens, special=special)

    e.backend.render = record_render
    with patch("trajectory_editor.episode_engine.token_boundaries", wraps=token_boundaries) as classify:
        first = e.apply(Hold(40, boundary))
        e.rewind_to(0)
        e.sampling = replace(e.sampling, seed=999)
        second = e.apply(Hold(40, boundary))
    assert classify.call_count == 1
    assert span_calls == [(1,) * 40, (1,) * 40]
    assert first == second
    assert first.stop_reason == "requested-length"


def test_boundary_cache_does_not_leak_between_tokenizers():
    first = runtime([1])
    first.apply(Hold(2, "sentence"))  # Token 1 has no terminal here.
    second = runtime([1])
    second.backend.pieces = {**second.backend.pieces, 1: "!"}
    result = second.apply(Hold(2, "sentence"))
    assert result.visible_token_ids == (1,)
    assert result.stop_reason == "sentence-boundary"


@pytest.mark.parametrize("mode", ["handoff", "ballistic"])
def test_old_sentence_tail_expectation_reports_divergence(mode):
    e = runtime([1, 2, 3])
    e.backend.pieces = {**e.backend.pieces, 3: '”'}
    result = e.apply(
        Hold(6, "sentence"), replay=True, divergence_policy=mode,
        expectation=ReplayExpectation((1, 2, 3), None, "sentence-boundary"),
    )
    assert result.visible_token_ids == (1, 2)
    assert result.divergence is not None
    assert result.status == ("handed-off" if mode == "handoff" else "completed-with-divergence")
