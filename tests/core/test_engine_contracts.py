from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import (
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    SelectRawRank,
    Write,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine, InstructionRejected

pytestmark = pytest.mark.invariant

class PhraseBackend(ConformingFakeBackend):
    def tokenize(self, text, *, add_bos=False, special=False):
        if not add_bos and text == "C!":
            return [3, 5]
        return super().tokenize(text, add_bos=add_bos, special=special)


class SequenceBackend(ConformingFakeBackend):
    pieces = {
        0: "<eog>",
        1: "word",
        2: ".",
        3: " next",
        4: "\n",
        5: "more",
        6: "?",
        7: "P",
    }

    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def last_logits(self):
        position = len(self.tokens) - 1
        token = self.sequence[min(position, len(self.sequence) - 1)]
        logits = np.full(8, -100.0)
        logits[token] = 100.0
        return logits


def engine(backend=None, *, max_tokens=10):
    return EpisodeEngine(
        backend or ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        max_tokens=max_tokens,
    )


@pytest.mark.parametrize(
    "action, expected",
    [
        (Accept(), (1,)),
        (SelectRawRank(1), (1,)),
        (SelectRawRank(2), (2,)),
        (SelectRawRank(3), (3,)),
    ],
)
def test_e01_accept_and_rank_selection_commit_exactly_one_selected_token(action, expected):
    runtime = engine()

    outcome = runtime.apply(action)

    assert outcome.resolved_token_ids == expected
    assert outcome.visible_token_ids == expected
    assert runtime.visible_token_ids == list(expected)
    assert runtime.boundary == len(expected)


def test_e02_raw_rank_selection_addresses_the_full_vocabulary():
    runtime = engine()
    observation = runtime.observe()

    candidate = runtime.candidates(observation, start_rank=8, count=1)[0]
    outcome = runtime.apply(SelectRawRank(8))

    assert candidate.rank == 8
    assert outcome.resolved_token_ids == (candidate.token_id,)
    assert runtime.boundary == 1


@pytest.mark.parametrize(
    "prior, text, mode, expected",
    [
        ("a", "nother", "continuation", " nother"),
        ("a", "word", "continuation", " word"),
        ("a ", "word", "continuation", "word"),
        ("(", "word", "continuation", "word"),
        ("a", " word", "continuation", " word"),
        ("a", ",", "continuation", ","),
        ("a", "\nword", "continuation", "\nword"),
        ("a", "nother", "exact", "nother"),
        ("a ", "word", "exact", "word"),
    ],
)
def test_e03_exact_and_ordinary_writes_produce_the_expected_visible_span(
    prior, text, mode, expected
):
    runtime = engine()
    with patch.object(runtime.backend, "render", return_value=prior), patch.object(
        runtime.backend, "tokenize", return_value=[4]
    ) as tokenize:
        outcome = runtime.apply(Write(text, mode))

    tokenize.assert_called_once_with(expected, add_bos=False, special=False)
    assert outcome.resolved_text == expected
    assert outcome.visible_token_ids == (4,)


@pytest.mark.parametrize(
    "action, expected_ids, rejected",
    [
        (Phrase("A B", mode="continuation", max_shift=0.0), (1, 2), False),
        (Phrase("C!", mode="exact", max_shift=0.5), (), True),
    ],
)
def test_e04_phrase_check_is_atomic(action, expected_ids, rejected):
    runtime = engine(PhraseBackend())
    original_ids = list(runtime.token_ids)

    if rejected:
        with pytest.raises(InstructionRejected, match="check phrase rejected"):
            runtime.apply(action)
        assert runtime.token_ids == original_ids
        assert runtime.boundary == 0
    else:
        outcome = runtime.apply(action)
        assert outcome.resolved_token_ids == expected_ids
        assert runtime.visible_token_ids == list(expected_ids)
        assert runtime.boundary == len(expected_ids)


def test_e05_force_phrase_uses_temporary_bias_without_persistent_residue():
    runtime = engine(PhraseBackend())

    outcome = runtime.apply(Phrase("C!", mode="exact", force=True, max_shift=0.5))

    assert outcome.resolved_token_ids == (3, 5)
    assert runtime._ephemeral_logit_biases == {}
    assert runtime.observe().statistics.ephemeral_logit_biases == {}


@pytest.mark.parametrize(
    "limit, boundary, piece, expected, stop_reason",
    [
        (6, "sentence", ".", (1, 2), "sentence-boundary"),
        (6, "sentence", "!", (1, 2), "sentence-boundary"),
        (6, "newline", "\n", (1, 2), "newline-boundary"),
        (6, "newline", "\nNext", (1, 2), "newline-boundary"),
        (2, None, ".", (1, 2), "requested-length"),
    ],
)
def test_e06_holds_respect_count_boundary_and_tokenizer_stop(
    limit, boundary, piece, expected, stop_reason
):
    backend = SequenceBackend([1, 2, 3])
    backend.pieces = {**backend.pieces, 2: piece, 3: '”'}
    runtime = engine(backend)

    outcome = runtime.apply(Hold(limit, boundary))

    assert outcome.visible_token_ids == expected
    assert outcome.resolved_text == "word" + piece
    assert outcome.stop_reason == stop_reason
    assert runtime.backend.tokens == [7, 1, 2]


@pytest.mark.parametrize("force", [False, True])
def test_e07_check_and_force_are_durable_write_actions(force):
    runtime = engine(PhraseBackend())
    action = Phrase("C!", mode="exact", force=force, max_shift=0.5 if force else 100.0)

    outcome = runtime.apply(action)

    assert outcome.action == action
    assert outcome.visible_token_ids == (3, 5)
    assert outcome.diagnostics["operation"] == ("force-phrase" if force else "check-phrase")


@pytest.mark.parametrize(
    "actions, stop_reason, terminal",
    [
        ([EndGeneration()], "eog", "teacher-eog"),
        ([Accept(), Accept(), Hold(3)], "eog", "model-eog"),
        ([Hold(1)], "requested-length", None),
    ],
)
def test_e08_eog_hold_and_menu_end_have_distinct_terminal_semantics(
    actions, stop_reason, terminal
):
    runtime = engine()

    outcome = None
    for action in actions:
        outcome = runtime.apply(action)

    assert outcome.stop_reason == stop_reason
    assert runtime.terminal_reason == terminal
    assert runtime.ended is (terminal is not None)

    if terminal is None:
        runtime.terminate("menu-end")
        assert runtime.ended
        assert runtime.terminal_reason == "menu-end"


def test_e09_budget_is_a_checkpoint_that_can_be_explicitly_resumed():
    runtime = engine(max_tokens=1)

    runtime.apply(Accept())

    assert runtime.checkpointed
    assert not runtime.ended
    assert runtime.terminal_reason is None

    runtime.resume(max_tokens=2)
    assert not runtime.checkpointed
    assert runtime.remaining == 2


def test_observe_rejects_after_a_terminal_event():
    runtime = engine()

    runtime.apply(EndGeneration())

    with pytest.raises(EditorError, match="no live decision boundary"):
        runtime.observe()


def test_observe_rejects_at_a_budget_checkpoint():
    runtime = engine(max_tokens=1)
    runtime.apply(Accept())

    assert runtime.checkpointed
    with pytest.raises(EditorError, match="no live decision boundary"):
        runtime.observe()


def test_phrase_accepts_a_token_count_equal_to_its_limit():
    runtime = engine(PhraseBackend())

    outcome = runtime.apply(
        Phrase("C!", mode="exact", max_tokens=2, max_shift=100.0)
    )

    assert outcome.resolved_token_ids == (3, 5)
    assert runtime.visible_token_ids == [3, 5]
    assert runtime.boundary == 2


def test_phrase_rejects_over_its_token_limit_without_changing_token_state():
    runtime = engine(PhraseBackend())
    original = (list(runtime.token_ids), runtime.boundary)

    with pytest.raises(InstructionRejected, match="has 2 tokens; max is 1"):
        runtime.apply(
            Phrase("C!", mode="exact", max_tokens=1, max_shift=100.0)
        )

    assert (runtime.token_ids, runtime.boundary) == original


@pytest.mark.parametrize(
    "action",
    [
        SelectRawRank(999),
        Phrase("C!", mode="exact", max_shift=0.5),
    ],
)
def test_e10_rejected_actions_leave_token_state_unchanged(action):
    runtime = engine(PhraseBackend())
    original = (list(runtime.token_ids), runtime.boundary)

    with pytest.raises(InstructionRejected):
        runtime.apply(action)

    assert (runtime.token_ids, runtime.boundary) == original


class CountingBackend(ConformingFakeBackend):
    def __init__(self):
        super().__init__()
        self.eval_calls: list[list[int]] = []

    def eval(self, token_ids):
        self.eval_calls.append([int(value) for value in token_ids])
        super().eval(token_ids)


class EogWriteBackend(CountingBackend):
    def tokenize(self, text, *, add_bos=False, special=False):
        if not add_bos and text == "STOP":
            return [1, 0, 2]
        return super().tokenize(text, add_bos=add_bos, special=special)


def test_e11_write_batch_commits_all_tokens_with_a_single_backend_eval():
    backend = CountingBackend()
    runtime = engine(backend)
    backend.eval_calls.clear()

    outcome = runtime.apply(Write(" A B", mode="exact"))

    assert outcome.resolved_token_ids == (1, 2)
    assert outcome.visible_token_ids == (1, 2)
    assert runtime.boundary == 2
    assert backend.eval_calls == [[1, 2]]


def test_e12_write_batch_evidence_has_one_record_per_token_with_exact_boundaries():
    runtime = engine()

    outcome = runtime.apply(Write(" A B", mode="exact"))

    assert outcome.stop_reason == "completed"
    assert len(outcome.evidence) == 2
    assert [item.boundary for item in outcome.evidence] == [0, 1]
    assert [item.sampling_boundary for item in outcome.evidence] == [0, 1]
    assert [item.token_id for item in outcome.evidence] == [1, 2]
    assert [item.text for item in outcome.evidence] == [" A", " B"]
    assert all(item.realized_visible for item in outcome.evidence)
    assert not any(item.is_eog for item in outcome.evidence)
    # Inserts deliberately record no per-token model probabilities; the
    # history adapter already treats these as the "unknown" defaults.
    assert all(item.decoder_probability == 0.0 for item in outcome.evidence)
    assert all(not item.proposal_agreement for item in outcome.evidence)
    assert all(item.raw_rank is None for item in outcome.evidence)


def test_e13_write_batch_stops_at_first_eog_token():
    backend = EogWriteBackend()
    runtime = engine(backend)
    backend.eval_calls.clear()

    outcome = runtime.apply(Write("STOP", mode="exact"))

    assert outcome.resolved_token_ids == (1, 0)
    assert outcome.visible_token_ids == (1,)
    assert runtime.boundary == 1
    assert outcome.stop_reason == "eog"
    assert outcome.terminal_token_id == 0
    assert runtime.terminal_reason == "teacher-eog"
    assert len(outcome.evidence) == 2
    assert outcome.evidence[-1].is_eog
    assert not outcome.evidence[-1].realized_visible
    # The EOG token rides the same single eval; it is resolved but never visible.
    assert backend.eval_calls == [[1, 0]]


def test_e14_write_batch_matches_replay_path_outcome():
    live = engine(EogWriteBackend())
    live_outcome = live.apply(Write(" A B", mode="exact"))
    replayed = engine(EogWriteBackend())
    replay_outcome = replayed.apply(Write(" A B", mode="exact"), replay=True)

    assert live_outcome.resolved_token_ids == replay_outcome.resolved_token_ids
    assert live_outcome.visible_token_ids == replay_outcome.visible_token_ids
    assert live_outcome.stop_reason == replay_outcome.stop_reason
    assert len(live_outcome.evidence) == len(replay_outcome.evidence)
    for live_item, replay_item in zip(live_outcome.evidence, replay_outcome.evidence):
        assert (
            live_item.boundary,
            live_item.sampling_boundary,
            live_item.token_id,
            live_item.text,
            live_item.is_eog,
            live_item.realized_visible,
        ) == (
            replay_item.boundary,
            replay_item.sampling_boundary,
            replay_item.token_id,
            replay_item.text,
            replay_item.is_eog,
            replay_item.realized_visible,
        )
