"""Learning explanations must distinguish teacher evidence from state movement."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.fakes import ScriptedIO
from tests.test_token_preference import FEATURES
from tests.test_sampler_learning_gate import GateBackend, engine
from trajectory_editor.episode_cli import main, _create_episode, _live_edge_menu
from trajectory_editor.episode_policy import _WriteLearningAccumulator
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.token_preference import TokenPreferenceLearner
from trajectory_editor.online_learning import OnlineLearner
from trajectory_editor.learning_readout import selection_notice, write_notice, show_learning_details


def display(learner, runtime, chosen=1):
    obs = replace(runtime.observe(), proposal_token_id=1)
    result = learner.update(obs, chosen, runtime.sampling)
    io = ScriptedIO([])
    selection_notice(io, result, preference=True, token_text=runtime.backend.token_text)
    assert len(io.output) == 1  # No automatic blocking page.
    show_learning_details(io)
    return io.output[0], io.output[-1], result


def test_rank_one_selection_is_full_severity_and_explains_memory_change():
    runtime = engine(token_preference_vector=(.06, .08), token_preference_strength=2.)
    summary, details, _ = display(TokenPreferenceLearner(FEATURES, enabled=True,
        dimension=2, token_preference_strength=2.), runtime)
    assert len(summary.strip()) <= 80
    assert 'severity 1' in summary and 'learned' in summary
    assert "Chosen: ' A' (id 1)" in details and 'teacher matched proposal' in details
    assert 'policy probability' in details and 'sampler probability' in details
    assert 'Memory size (z norm): 0.1 → 0.1044' in details
    assert 'Token bias bound: ±0.2089 logits' in details
    assert 'not confidence' in details


@pytest.mark.parametrize('mode,word', [('update', 'decayed'), ('evidence', 'unchanged')])
def test_gate_skip_can_decay_without_learning(mode, word):
    runtime = engine(token_preference_vector=(.06, .08))
    summary, details, _ = display(TokenPreferenceLearner(FEATURES, enabled=True,
        dimension=2, learning_gate='sampler', decay=.1, decay_on=mode), runtime)
    assert 'already eligible' in summary and word in summary
    assert 'New learning: 0' in details
    assert ('Decay: 0.01 removed' if mode == 'update' else 'Decay: 0 removed') in details


def test_clipped_learning_and_memory_bound_are_reported_separately():
    runtime = engine(top_k=2, token_preference_vector=(.06, .08))
    summary, details, result = display(TokenPreferenceLearner(FEATURES, enabled=True,
        dimension=2, learning_gate='sampler', learning_rate=10, max_step=.02, max_norm=.01), runtime, 3)
    assert 'sampler excluded' in summary and 'learned' in summary
    assert 'New learning: 0.02' in details and 'step clipping applied' in details
    assert 'Memory bounds: no adjustment' not in details
    assert result.z_norm == pytest.approx(.01)


def test_zero_memory_bound_is_not_misreported_as_no_clipping():
    runtime = engine(top_k=2)
    _, details, result = display(TokenPreferenceLearner(FEATURES, enabled=True,
        dimension=2, learning_gate='sampler', max_norm=0), runtime, 3)
    assert result.new_z == ()
    assert 'Memory bounds: no adjustment' not in details


def test_zero_learning_rate_is_distinct_from_a_gate_skip():
    summary, details, _ = display(TokenPreferenceLearner(FEATURES, enabled=True,
        dimension=2, learning_gate='sampler', learning_rate=0), engine(top_k=2), 3)
    assert 'sampler excluded' in summary and 'unchanged' in summary
    assert 'evidence admitted' in details and 'New learning: 0' in details


def test_fast_only_update_is_visible_when_slow_vector_does_not_move():
    runtime = engine(top_k=2)
    summary, details, result = display(TokenPreferenceLearner(FEATURES, enabled=True,
        dimension=2, learning_gate='sampler', fast_slow=True,
        learning_rate=0, fast_learning_rate=.1), runtime, 3)
    assert result.update_norm == 0 and result.fast_update_norm > 0
    assert 'learned' in summary and 'unchanged' not in summary
    assert 'Slow memory (2 dimensions)' in details and 'Fast memory (2 dimensions)' in details


def test_frozen_groups_are_explained_and_both_learners_share_one_report():
    runtime = engine()
    runtime.sampling = replace(runtime.sampling, bias_groups=(replace(runtime.sampling.bias_groups[0], learnable=False),))
    obs = replace(runtime.observe(), proposal_token_id=1)
    io = ScriptedIO([])
    group = OnlineLearner(enabled=True).update(obs, 3, runtime.sampling)
    preference = TokenPreferenceLearner(FEATURES, enabled=True, dimension=2).update(obs, 3, runtime.sampling)
    selection_notice(io, group, preference=False, episode_id='one')
    assert 'no eligible groups' in io.output[-1]
    selection_notice(io, preference, preference=True, episode_id='one')
    show_learning_details(io, episode_id='one')
    assert 'Groups' in io.output[-1] and 'token preference' in io.output[-1]
    assert 'skipped: disabled or frozen' in io.output[-1]
    show_learning_details(io, episode_id='two')
    assert 'No teaching report' in io.output[-1]


def test_write_reports_each_learners_gate_without_using_averaged_ranks():
    runtime = engine(top_k=2)
    group = OnlineLearner(enabled=True, dead_zone_rank=1)
    preference = TokenPreferenceLearner(FEATURES, enabled=True, dimension=2,
                                    learning_gate='sampler', write_reduction='mean')
    acc = _WriteLearningAccumulator(runtime.backend, runtime.sampling, group, preference)
    obs = replace(runtime.observe(), proposal_token_id=1)
    for token in (3, 2, 1):
        acc.add(obs, token)
    result = acc.finish(3)
    io = ScriptedIO([])
    write_notice(io, result, token_text=runtime.backend.token_text)
    assert 'groups 1/3 evidence' in io.output[-1] and 'preference 1/3 evidence' in io.output[-1]
    show_learning_details(io)
    details = io.output[-1]
    assert 'Proposal agreement: 1/3 tokens' in details
    assert "' C' (id 3): rank 3" in details and "' A' (id 1): rank 1" in details
    assert details.count('Each written token received its own bounded update') == 2


def test_learning_command_at_choice_and_edge_is_read_only(tmp_path):
    runtime = engine()
    io = ScriptedIO(['learning', '3', 'learning', 'quit'])
    with EpisodeStore(tmp_path / 'report.db') as store:
        eid = _create_episode(store, runtime, backend_provenance=runtime.backend.provenance())
        original = runtime.sampling
        policy = InteractivePolicy(io=io, store=store, episode_id=eid)
        policy.choose(runtime, runtime.observe())
        assert runtime.boundary == 0 and runtime.sampling == original
        assert _live_edge_menu(io, store, eid, runtime) == ('quit', None)
        assert runtime.boundary == 0 and runtime.sampling == original
    assert sum('No teaching report' in t for t in io.output) == 2


def test_real_cli_callbacks_supply_text_and_keep_report_available_at_edge(tmp_path):
    io = ScriptedIO(['3', 'learning', 'q', 'learning', 'quit'])
    backend = GateBackend()
    backend.token_preference_features = lambda **kwargs: FEATURES
    with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), \
         patch('trajectory_editor.episode_cli._backend', return_value=backend):
        assert main(['--model', str(tmp_path / 'model.gguf'), '--workspace', str(tmp_path / 'run.db'),
                     '--new-prompt', 'P', '--token-preference', '--token-preference-dimension', '2']) == 0
    reports = [t for t in io.output if t.startswith('Learning details')]
    assert len(reports) == 2 and reports[0] == reports[1]
    assert "Chosen: ' C' (id 3)" in reports[0]
    assert 'Teaching selection @ boundary 1' in reports[0]
    assert sum(t.startswith('token preference [learning]') for t in io.output) == 1


def test_live_notice_and_details_fit_existing_surface_and_return_to_same_choice():
    from prompt_toolkit.input import create_pipe_input
    from tests.test_persistent_tui import DrivenSession, RecordingOutput, choice_state
    from trajectory_editor.tui import TerminalIO

    runtime = engine(token_preference_vector=(.06, .08))
    observation = replace(runtime.observe(), proposal_token_id=1)
    result = TokenPreferenceLearner(FEATURES, enabled=True, dimension=2).update(
        observation, 1, runtime.sampling)
    io = TerminalIO(live_choices=False)
    output = RecordingOutput()
    with create_pipe_input() as pipe:
        with DrivenSession(pipe, output, ['learning\r', '\r', '3\r']) as session:
            io._live_session = session
            selection_notice(io, result, preference=True, token_text=runtime.backend.token_text)
            state = choice_state(runtime)
            assert session.read_choice(state) == 'learning'
            view = session.choice_view
            show_learning_details(io)
            assert 'Memory size (z norm): 0.1 → 0.1' in session.views[1].body
            assert session.read_choice(state) == '3'
            assert session.choice_view is view
            assert all(events == ('enter', 'erase') for events in session.snapshots)
            io._live_session = None
    assert runtime.boundary == 0
