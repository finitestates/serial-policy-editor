"""Policy requests carry the same decision data to both terminal backends."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO, SnapshotFakeBackend
from tests.core.runtime_helpers import LiveScriptedIO
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.edge_help import edge_help
from trajectory_editor.edge_tui import _edge_header
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.plain_tui import read_edge
from trajectory_editor.terminal_contracts import EdgeViewState
from trajectory_editor.teacher_commands import HELP_TEXT

pytestmark = pytest.mark.current_workflow

class CountingBackend(ConformingFakeBackend):
    def __init__(self):
        super().__init__()
        self.positions = 0

    def reset(self, prefix_token_ids):
        self.positions += 1
        super().reset(prefix_token_ids)

    def eval(self, token_ids):
        self.positions += 1
        super().eval(token_ids)


class PlainCapture(ScriptedIO):
    def __init__(self, responses):
        super().__init__(responses)
        self.states = []

    def read_choice(self, state):
        self.states.append(state)
        return super().read_choice(state)


class LiveCapture(LiveScriptedIO):
    def __init__(self, responses):
        super().__init__(responses)
        self.states = []

    def read_choice(self, state):
        self.states.append(state)
        return super().read_choice(state)


def test_help_note_and_eog_confirmation_return_to_the_choice():
    class RequestCapture(PlainCapture):
        def __init__(self):
            super().__init__(["?", "n", "memo", "e", "x", "1"])
            self.pages = []
            self.prompts = []

        def page(self, text):
            self.pages.append(text)

        def prompt(self, request):
            self.prompts.append(request)
            return super().prompt(request)

    terminal = RequestCapture()
    engine = EpisodeEngine(
        CountingBackend(), initial_text="P", initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    action = InteractivePolicy(io=terminal, menu_size=1).choose(
        engine, engine.observe(),
    )
    assert action.rank == 1
    assert terminal.pages == [HELP_TEXT]
    assert [request.prompt for request in terminal.prompts] == [
        "Note> ", "[e/Enter] confirm EOG  [Backspace/Esc] cancel > ",
    ]
    assert terminal.prompts[1].single_key
    assert len(terminal.states) == 4


def test_choice_requests_preserve_actions_feedback_and_lazy_statistics():
    requests = []
    interactions = []
    for terminal in (PlainCapture(["/P", "nonsense", "l", "1"]),
                     LiveCapture(["/P", "nonsense", "l", "1"])):
        class Recorder:
            def __init__(self):
                self.rows = []

            def record_interaction(self, episode_id, boundary, kind, payload):
                self.rows.append((episode_id, boundary, kind, payload))

        recorder = Recorder()
        backend = CountingBackend()
        engine = EpisodeEngine(
            backend, initial_text="P", initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        )
        observation = engine.observe()
        positions_before = backend.positions
        action = InteractivePolicy(
            io=terminal, menu_size=1, store=recorder, episode_id="same",
        ).choose(engine, observation)
        assert action.rank == 1
        assert backend.positions == positions_before
        assert not observation.statistics._raw_logsumexp_ready
        assert not observation.statistics._policy_logsumexp_ready
        requests.append(terminal.states)
        interactions.append(recorder.rows)

    assert len(requests[0]) == len(requests[1]) == 4
    for plain, live in zip(*requests):
        plain_choice = replace(
            plain.choice, context_text_tail=str(plain.choice.context_text_tail)
        )
        live_choice = replace(
            live.choice, context_text_tail=str(live.choice.context_text_tail)
        )
        assert plain_choice == live_choice
        assert plain.display_candidates == live.display_candidates
        assert plain.feedback == live.feedback
        assert plain.target_token_id == live.target_token_id
        assert plain.logit_view == live.logit_view
        assert plain.default_hold_tokens == live.default_hold_tokens
        assert plain.default_search_radius == live.default_search_radius
    assert requests[0][1].feedback.category == "search"
    assert requests[0][1].search_lens_active
    assert requests[0][2].feedback.category == "error"
    assert requests[0][3].logit_view == "raw"
    assert interactions[0] == interactions[1]
    assert [row[2] for row in interactions[0]] == [
        "vocabulary-search-view",
    ]


def test_candidate_columns_do_not_depend_on_terminal_width():
    class SizedIO(ScriptedIO):
        def __init__(self, width):
            super().__init__([])
            self.width = width

        def terminal_size(self):
            return self.width, 24

    engine = EpisodeEngine(
        CountingBackend(), initial_text="P", initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    narrow = InteractivePolicy(
        io=SizedIO(36), menu_size=1, show_model_probabilities=True,
    )._view_plan(engine)
    wide = InteractivePolicy(
        io=SizedIO(160), menu_size=1, show_model_probabilities=True,
    )._view_plan(engine)

    assert narrow == wide
    assert {label for label, _ in narrow.columns} == {
        "raw-p", "decode-p", "token-id",
    }


def test_review_request_does_not_position_backend_or_commit_action():
    terminal = PlainCapture(["[", "]", "1", "1"])
    backend = CountingBackend()
    engine = EpisodeEngine(
        backend, initial_text="P", initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    observation = engine.observe()
    positions_before = backend.positions
    action = InteractivePolicy(io=terminal, menu_size=1).choose(engine, observation)
    assert action.rank == 1
    assert terminal.states[1].review is not None
    assert terminal.states[1].review.aligned_step == 0
    assert backend.positions == positions_before
    assert engine.boundary == 0


def test_bias_feedback_is_in_next_choice_request():
    terminal = PlainCapture(["2+100", "1"])
    engine = EpisodeEngine(
        CountingBackend(), initial_text="P", initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    action = InteractivePolicy(io=terminal, menu_size=2).choose(
        engine, engine.observe(),
    )
    assert action.rank == 1
    assert terminal.states[1].feedback.title == "STEERING UPDATED"
    assert terminal.states[1].choice != terminal.states[0].choice


def test_edge_help_is_shared_by_plain_and_live_with_mode_specific_actions():
    for mode in ("episode", "session"):
        state = EdgeViewState("one", 2, 10, 8, "temp=1", mode=mode)
        plain = ScriptedIO(["q"])
        assert read_edge(plain, state) == "q"
        plain_text = "".join(plain.output)
        live_text = "".join(fragment for _, fragment in _edge_header(
            episode_id=state.episode_id, boundary=state.boundary,
            current_budget=state.current_budget,
            remaining_tokens=state.remaining_tokens,
            sampler_summary=state.sampler_summary, mode=mode,
        ))
        for item in edge_help(mode):
            assert f"[{item.command}] {item.description}" in plain_text
            assert item.command in live_text and item.description in live_text
    assert "save WORKSPACE [ID]" not in "".join(
        item.command for item in edge_help("episode")
    )


@pytest.mark.parametrize("submitted_rank", (1, 2))
def test_choice_warm_callback_resolves_selected_rank_through_policy(submitted_rank):
    class WarmCapture(LiveScriptedIO):
        def __init__(self):
            super().__init__([])
            self.target = None

        def read_choice(self, state):
            candidate = state.resolve_candidate(2)
            self.target = (candidate.rank, candidate.token_id)
            assert state.warm_selection is not None
            assert state.warm_selection(
                candidate.rank, candidate.token_id, 1, lambda: False,
            )
            return str(submitted_rank)

    terminal = WarmCapture()
    backend = SnapshotFakeBackend()
    engine = EpisodeEngine(
        backend, initial_text="P", initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    observation = engine.observe()
    action = InteractivePolicy(io=terminal, menu_size=1).choose(engine, observation)
    assert terminal.target == (2, 2)
    assert action.rank == submitted_rank
    assert backend.eval_calls == [(2,)]

    engine.apply(action)
    assert backend.tokens == [7, submitted_rank]
    assert backend.eval_calls == ([(2,)] if submitted_rank == 2 else [(2,), (1,)])
