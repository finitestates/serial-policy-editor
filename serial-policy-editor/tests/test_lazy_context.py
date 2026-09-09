"""Context rendering stays exact without decoding hidden decision boundaries."""
from unittest.mock import patch

import pytest

from tests.fakes import ConformingFakeBackend
from tests.test_episode_runtime import engine
from trajectory_editor.episode_actions import Accept, Hold, Write
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_ui import _choice_from_observation


class ContextBackend(ConformingFakeBackend):
    def render(self, token_ids, *, special=False):
        # Deliberately non-compositional, like context-sensitive token decoding.
        text = super().render(token_ids, special=special)
        return ('<special>' if special else '') + text.replace(' A B', '世界')


def test_context_is_lazy_cached_and_bound_to_original_prefix():
    runtime = engine(ContextBackend(), max_tokens=8)
    with patch.object(runtime.backend, 'render', wraps=runtime.backend.render) as render:
        initial = runtime.observe()
        runtime.candidates(initial)
        render.assert_not_called()
        runtime.apply(Write(' A B', mode='exact'))
        later = runtime.observe()
        runtime.rewind_to(0)
        runtime.apply(Accept())
        render.reset_mock()
        assert later.context_text == '<special>P世界'
        assert later.context_text == '<special>P世界'
        render.assert_called_once_with([7, 1, 2], special=True)
        assert initial.context_text == '<special>P'
        assert runtime.observe().context_text == '<special>P A'


@pytest.mark.parametrize('action', [Hold(2), Write(' A B', mode='exact')])
def test_hidden_tokens_do_not_decode_full_context(action):
    runtime = engine(ContextBackend(), max_tokens=8)
    with patch.object(runtime.backend, 'render', wraps=runtime.backend.render) as render:
        outcome = runtime.apply(action)
        assert outcome.visible_token_ids == (1, 2)
        # Rendering the action's output is still required; full context is not.
        assert not any(call.kwargs.get('special') for call in render.call_args_list)
        observation = runtime.observe()
        render.reset_mock()
        choice = _choice_from_observation(
            runtime, observation, (), context_characters=0, serial=1,
        )
        assert choice.context_text_tail == '<special>P世界'
        assert observation.context_text == choice.context_text_tail
        render.assert_called_once_with([7, 1, 2], special=True)


@pytest.mark.parametrize('action,budget,replay,expectation', [
    (Hold(2), 8, False, None),
    (Hold(2), 2, False, None),
    (Hold(4), 8, False, None),  # EOG
    (Hold(4), 8, True, None),  # Replay EOG handoff
    (Hold(2), 8, True, ReplayExpectation((3,))),  # Divergence
    (Write(' A B', mode='exact'), 8, False, None),
    (Write(' A B'), 8, True, ReplayExpectation((1, 2))),
])
def test_outcomes_match_eager_context_rendering(action, budget, replay, expectation):
    lazy = engine(ContextBackend(), max_tokens=budget)
    eager = engine(ContextBackend(), max_tokens=budget)
    original_observe = eager.observe

    def eager_observe():
        observation = original_observe()
        _ = observation.context_text
        return observation

    with patch.object(eager, 'observe', side_effect=eager_observe):
        expected = eager.apply(action, replay=replay, expectation=expectation)
    actual = lazy.apply(action, replay=replay, expectation=expectation)
    assert actual == expected
    assert lazy.token_ids == eager.token_ids
    assert lazy.text == eager.text


def test_forked_engine_context_is_independent():
    parent = engine(ContextBackend(), max_tokens=8)
    parent.apply(Accept())
    snapshot = parent.observe()
    fork = EpisodeEngine(
        ContextBackend(), sampling=parent.sampling, max_tokens=8,
        initial_token_ids=parent.token_ids,
    )
    fork.apply(Write('!', mode='exact'))
    parent.rewind_to(0)
    assert snapshot.context_text == '<special>P A'
    assert fork.observe().context_text == '<special>P A!'
    assert parent.observe().context_text == '<special>P'


@pytest.mark.parametrize('boundary,piece', [('sentence', '!'), ('newline', '\n')])
def test_conditional_hold_stops_without_context_decode(boundary, piece):
    backend = ContextBackend()
    backend.pieces = {**backend.pieces, 1: piece}
    runtime = engine(backend, max_tokens=8)
    with patch.object(backend, 'render', wraps=backend.render) as render:
        outcome = runtime.apply(Hold(4, boundary=boundary))
        assert outcome.stop_reason == f'{boundary}-boundary'
        assert outcome.visible_token_ids == (1,)
        assert not any(call.kwargs.get('special') for call in render.call_args_list)
        assert runtime.observe().context_text == '<special>P' + piece
