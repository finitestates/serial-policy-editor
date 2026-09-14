"""Policy views expose existing evidence without changing sampling or rank addresses."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_menu_expansion import LargeBackend
from tests.test_latent_preference import LatentBackend
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.candidate_columns import CandidateColumns
from trajectory_editor.domain import Candidate, EditorError, SamplingConfig
from trajectory_editor.episode_actions import SelectRawRank
from trajectory_editor.episode_cli import _interactive_policy, build_parser
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.live_tui import _render_choice
from trajectory_editor.sampling import ObservationStatistics
from trajectory_editor.tui import display_candidates


class ViewIO(ScriptedIO):
    supports_live_choices = True

    def __init__(self, commands):
        super().__init__(commands)
        self.views = []

    def read_choice(self, choice, **kwargs):
        self.views.append(kwargs)
        return self.responses.pop(0)


def promoted_engine():
    return EpisodeEngine(LargeBackend(), initial_token_ids=[7], sampling=SamplingConfig(
        temperature=0, bias_rules=(BiasRule(routes=((500,),), bias=30),),
    ))


def test_policy_top_n_includes_promoted_tokens_and_keeps_raw_rank_addresses():
    runtime = promoted_engine()
    observation = runtime.observe()
    assert 500 not in [c.token_id for c in runtime.candidates(observation)]
    rows = runtime.policy_candidates(observation, count=12)
    assert rows[0].token_id == 500
    assert rows[0].rank == 501
    assert rows[0].policy_rank == 1
    assert rows[0].policy_logit_adjustment == 30
    with patch('trajectory_editor.sampling._top_ids', side_effect=AssertionError('cached')):
        assert runtime.policy_candidates(observation, count=3) == rows[:3]
    io = ViewIO(['v', '501'])
    with patch('trajectory_editor.episode_engine.draw_token', side_effect=AssertionError('must not resample')):
        action = InteractivePolicy(io=io).choose(runtime, observation)
    assert action == SelectRawRank(501)
    assert io.views[0]['sort_by_policy'] is False
    assert io.views[1]['display_candidates'][0] == rows[0]
    assert rows[0] in io.views[1]['candidates']
    assert runtime.boundary == 0
    assert runtime.observe() is observation
    assert runtime.apply(action).resolved_token_ids == (500,)


def test_policy_order_ties_share_rank_tiebreak_and_expansion_is_stable():
    stats = ObservationStatistics(np.array([3., 2., 1., 0.]), SamplingConfig(
        bias_rules=(BiasRule(routes=((1,),), bias=1), BiasRule(routes=((3,),), bias=3)),
    ), [])
    assert stats.top_policy_ids(2) == [0, 1]
    assert stats.top_policy_ids(4) == [0, 1, 3, 2]
    assert [stats.policy_rank(t) for t in stats.top_policy_ids(4)] == [1, 2, 3, 4]
    raw = ObservationStatistics(np.array([2., 2., 1.]), SamplingConfig(), [])
    assert raw.top_policy_ids(3) == raw.top_raw_ids(3)
    with pytest.raises(EditorError):
        promoted_engine().policy_candidates(promoted_engine().observe(), count=0)


@pytest.mark.parametrize('learning,state,expected', [
    (False, SamplingConfig(), False),
    (True, SamplingConfig(), True),
    (False, SamplingConfig(bias_rules=(BiasRule(routes=((1,),), bias=1),)), True),
    (False, SamplingConfig(bias_groups=(BiasGroup('zero', (BiasRule(routes=((1,),), bias=0),)),)), True),
    (False, SamplingConfig(repeat_penalty=1.1), True),
    (False, SamplingConfig(latent_preference_fast_z=(.1, .2), latent_fast_strength=.5), True),
])
def test_automatic_view_uses_learning_and_restored_policy(learning, state, expected):
    runtime = EpisodeEngine(LatentBackend(), initial_token_ids=[7], sampling=state)
    io = ViewIO(['1'])
    InteractivePolicy(io=io, learning_enabled=learning).choose(runtime, runtime.observe())
    assert io.views[0]['show_policy_rank'] is expected
    assert not io.views[0]['sort_by_policy']


@pytest.mark.parametrize('flags,expected', [([], None), (['--policy-view'], True),
                                         (['--show-policy-rank'], True), (['--no-policy-view'], False)])
def test_cli_explicit_visibility_and_session_preferences_survive_new_adapters(flags, expected):
    args = build_parser().parse_args(flags + ['--latent-preference'])
    runtime = promoted_engine()
    first_io = ViewIO(['V', 'v', '1'])
    first = _interactive_policy(args, None, 'first', first_io)
    assert first.view_preferences.show is expected
    first.choose(runtime, runtime.observe())
    expected_show = first_io.views[-1]['show_policy_rank']
    second_io = ViewIO(['1'])
    second = _interactive_policy(args, None, 'second', second_io)
    assert second.view_preferences is first.view_preferences
    second.choose(runtime, runtime.observe())
    assert second_io.views[0]['show_policy_rank'] is expected_show
    assert second_io.views[0]['sort_by_policy'] is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--policy-view', '--no-policy-view'])


def test_explicit_off_stays_off_and_explicit_on_works_without_policy():
    io = ViewIO(['1'])
    runtime = promoted_engine()
    policy = InteractivePolicy(io=io, show_policy_rank=False, learning_enabled=True)
    policy.choose(runtime, runtime.observe())
    assert not io.views[0]['show_policy_rank']
    io = ViewIO(['1'])
    runtime = EpisodeEngine(LargeBackend(), initial_token_ids=[7], sampling=SamplingConfig())
    InteractivePolicy(io=io, show_policy_rank=True).choose(runtime, runtime.observe())
    assert io.views[0]['show_policy_rank']


def test_plain_ui_redraws_visibility_and_policy_candidate_changes():
    runtime = promoted_engine()
    io = ScriptedIO(['V', 'v', 'V', '501'])
    InteractivePolicy(io=io).choose(runtime, runtime.observe())
    headers = [line for line in io.output if line.startswith('\n  rank')]
    assert len(headers) == 4
    assert ['Δlogit' in header for header in headers] == [True, False, False, True]
    assert any('500' in line and '+30.000' in line and '+500' in line for line in io.output)


def test_search_neighborhood_keeps_raw_order_in_policy_mode():
    runtime = promoted_engine()
    io = ViewIO(['v', 'ms 500', '500'])
    InteractivePolicy(io=io).choose(runtime, runtime.observe())
    lens = io.views[-1]
    assert lens['search_lens_active']
    assert not lens['sort_by_policy']
    ranks = [c.rank for c in lens['display_candidates']]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize('width', [36, 50, 60, 80, 100, 120, 140])
def test_responsive_live_table_preserves_deltas_and_token_text(width):
    runtime = promoted_engine()
    observation = runtime.observe()
    rows = runtime.policy_candidates(observation, count=3)
    choice = _choice_from_observation(runtime, observation, rows, context_characters=100, serial=1)
    fragments = _render_choice(choice, rows, '', None, lambda text, mode: text, None,
                               show_policy_rank=True, sort_by_policy=True, terminal_size=(width, 30))
    table = [text for style, text in fragments if style in ('class:table-header', 'class:table-row')]
    assert 'Δrank' in table[0] and 'Δlogit' in table[0]
    assert len(table) > 1
    for row in table[1:]:
        assert len(row.rstrip('\n')) <= width
        assert "'" in row
    if width < 134:
        assert 'token-id' not in table[0]
    if width < 92:
        assert 'pol-rank' not in table[0]


def test_shared_diagnostic_signs_missing_values_and_plain_live_parity():
    candidate = Candidate(rank=20, token_id=3, text=' token', raw_probability=.01,
                          decoder_probability=0, is_eog=False, policy_rank=2,
                          policy_probability=.2, policy_logit_adjustment=.125)
    columns = CandidateColumns(policy=True)
    values = columns.values(candidate)
    assert '+18' in values and '+0.125' in values
    demoted = replace(candidate, policy_rank=30, policy_logit_adjustment=-.25)
    assert '-10' in columns.values(demoted) and '-0.250' in columns.values(demoted)
    missing = replace(candidate, policy_rank=None, policy_logit_adjustment=None, policy_probability=None)
    assert columns.values(missing).count('--') == 5  # deltas, policy rank/p, decoder p
    io = ScriptedIO([])
    display_candidates(io, (candidate,), heading=True, show_policy_rank=True)
    assert values in io.output[-1]
    assert columns.heading in io.output[0]


def test_plain_return_from_search_restores_current_policy_table():
    runtime = promoted_engine()
    io = ScriptedIO(['v', 'ms 500', 'm', '501'])
    InteractivePolicy(io=io).choose(runtime, runtime.observe())
    headers = [index for index, line in enumerate(io.output) if line.startswith('\n  rank')]
    final_rows = io.output[headers[-1] + 1:]
    assert '+30.000' in final_rows[0] and '+500' in final_rows[0]
