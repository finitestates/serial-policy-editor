import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend, LiveScriptedIO
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.episode_policy import EpisodeRunner, EdgeRequested
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_lifecycle import _create_episode, _fork_engine, _rewind_episode, _restore_engine, _spr_engine_from_source
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.episode_cli import main
from trajectory_editor.sampling import ObservationStatistics
from trajectory_editor.tui import parse_command, CommandKind


def parse(text):
    return parse_command(text, menu_size=3, vocabulary_size=8, default_hold_tokens=10)


@pytest.mark.parametrize('text,phrase,last,prefix,rank', [
    ('b " A B" +0.5', ' A B', None, None, None),
    ('bl 1 -', None, 1, None, None),
    ('bl 3 =', None, 3, None, None),
    ('2+0.5 ... " A"', None, None, ' A', 2),
    ('2= ... " A"', None, None, ' A', 2),
    (r'b "\n\"quoted\"" -', '\n"quoted"', None, None, None),
])
def test_sequence_syntax(text, phrase, last, prefix, rank):
    command = parse(text)
    assert command.kind == CommandKind.BIAS
    assert (command.bias_text, command.bias_last, command.bias_prefix, command.search_rank) == (phrase, last, prefix, rank)


@pytest.mark.parametrize('text', ['b "" +', 'bl 0 -', 'bl -1 +', '2+ ... ""', 'b " A" =1', 'bl 2 +nan', 'b "unterminated +', 'b "x" +0', '9+ ... " A"'])
def test_invalid_sequence_syntax(text):
    with pytest.raises(EditorError):
        parse(text)


def test_matching_sums_overlaps_only_at_context_tail():
    config = SamplingConfig(temperature=1, top_k=8, top_p=1, min_p=0,
        logit_bias=((2, .5),), sequence_bias=(((1, 2), 2), ((7, 1, 2), -1), ((3, 2), 20)))
    assert config.active_biases([7, 1]) == {2: 1.5}
    assert config.active_biases([7, 1, 4]) == {2: .5}
    assert config.active_biases([1]) == {2: 2.5}
    assert config.active_biases([]) == {2: .5}
    logits = [3., 2., 1., 0., -1., -2., -3., -4.]
    stats = ObservationStatistics(logits, config, [7, 1])
    np.testing.assert_allclose(stats.adjusted, [3, 2, 2.5, 0, -1, -2, -3, -4])
    assert stats.top_raw_ids(3) == [0, 1, 2]
    assert stats.policy_rank(2) == 2
    with pytest.raises(EditorError, match='exact context'):
        config.active_biases(None)


@pytest.mark.parametrize('io_type', [ScriptedIO, LiveScriptedIO])
def test_entry_forms_canonicalize_and_clear_same_rules(io_type, tmp_path):
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_token_ids=[7, 1, 2], sampling=SamplingConfig())
    original = runtime.observe()
    with EpisodeStore(tmp_path/'a.db') as store:
        source = _create_episode(store, runtime, backend_provenance=backend.provenance())
        # Phrase and history forms refer to identical sequences, and bl 1 is ordinary bias.
        policy = InteractivePolicy(io=io_type(['b " A B" +', 'bl 2 +', 'bl 1 -', 'b " B" +', 'q']), store=store, episode_id=source)
        with patch.object(backend, 'eval', side_effect=AssertionError('Unexpected evaluation')):
            with pytest.raises(EdgeRequested):
                policy.choose(runtime, original)
        assert runtime.boundary == 0
        assert runtime.sampling.sequence_bias == (((1, 2), 1.),)
        assert runtime.sampling.logit_bias == ()
        assert len(store.interactions(source)) == 4
        # Resolve B's current raw rank, then clear that same sequence via a conditional rank.
        observation = runtime.observe()
        rank = observation.statistics.raw_rank(2)
        with pytest.raises(EdgeRequested):
            InteractivePolicy(io=io_type([f'{rank}= ... " A"', 'q']), store=store, episode_id=source).choose(runtime, observation)
        assert runtime.sampling.sequence_bias == ()
        assert runtime.observe().proposal_token_id == original.proposal_token_id
        assert runtime.observe().sampling_coordinate == original.sampling_coordinate
        np.testing.assert_array_equal(runtime.observe().logits, original.logits)


def test_condition_uses_selected_token_id_without_retokenizing_combination():
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    with patch.object(backend, 'tokenize', wraps=backend.tokenize) as tokenize:
        with pytest.raises(EdgeRequested):
            InteractivePolicy(io=ScriptedIO(['2+ ... " A"', 'q'])).choose(runtime, runtime.observe())
    tokenize.assert_called_once_with(' A', add_bos=False, special=False)
    assert runtime.sampling.sequence_bias == (((1, 2), .5),)


def test_excessive_history_reports_error_without_mutation():
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    io = ScriptedIO(['bl 100 +', 'q'])
    with pytest.raises(EdgeRequested):
        InteractivePolicy(io=io).choose(runtime, runtime.observe())
    assert runtime.sampling.sequence_bias == () and runtime.sampling.logit_bias == ()


def test_projection_and_v1_v2_presets(tmp_path):
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    with EpisodeStore(tmp_path/'a.db') as store:
        source = _create_episode(store, runtime, backend_provenance=backend.provenance())
        with pytest.raises(EdgeRequested):
            InteractivePolicy(io=ScriptedIO(['b " A B" +', '1-', 'q']), store=store, episode_id=source).choose(runtime, runtime.observe())
        path = tmp_path/'preset.json'
        path.write_text(project_biases(store, source))
        exported = json.loads(path.read_text())
        assert exported['format'] == 'spe-logit-bias-v2'
        assert {tuple(row['token_ids']) for row in exported['biases']} == {(1,), (1, 2)}
        assert load_bias_preset(path, backend, backend.provenance()) == runtime.sampling
        exported['biases'][0]['texts'] = ['wrong']
        path.write_text(json.dumps(exported))
        with pytest.raises(EditorError, match='text mismatch'):
            load_bias_preset(path, backend, backend.provenance())
    path.write_text(json.dumps({'format': 'spe-logit-bias-v1', 'model': {'vocabulary_size': 8},
        'biases': [{'token_id': 1, 'bias': -.5}]}))
    config = load_bias_preset(path, backend, backend.provenance())
    assert config.logit_bias == ((1, -.5),) and config.sequence_bias == ()


def test_fork_and_rewind_restore_same_rules_and_active_matches(tmp_path):
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    with EpisodeStore(tmp_path/'a.db') as store:
        source = _create_episode(store, runtime, backend_provenance=backend.provenance())
        runner = EpisodeRunner(runtime, store, source)
        runner.run(live_policy=InteractivePolicy(io=ScriptedIO(['b " A B" +2', 'x  A B']), store=store, episode_id=source), max_live_actions=1)
        runner.run(live_policy=InteractivePolicy(io=ScriptedIO(['bl 2 -', '1']), store=store, episode_id=source), max_live_actions=1)
        resumed = _restore_engine(store, source, backend, max_tokens=None, sampling_override=None)
        assert resumed.sampling == runtime.sampling
        replay, plan = _spr_engine_from_source(store, source, backend, sampling=SamplingConfig(), max_tokens=None)
        child = _create_episode(store, replay, backend_provenance=backend.provenance())
        result = EpisodeRunner(replay, store, child).run(tape=plan)
        assert result.replay_exhausted and replay.visible_token_ids == runtime.visible_token_ids
        assert replay.sampling == runtime.sampling
        for target in (2, 1, 0):
            fork = _fork_engine(store, source, runtime, target, backend=backend, max_tokens=None)
            expected_biases = fork.sampling.active_biases(fork.token_ids)
            expected_adjusted = fork.observe().statistics.adjusted.copy()
            _rewind_episode(store, source, runtime, target)
            assert runtime.sampling == fork.sampling
            assert runtime.sampling.active_biases(runtime.token_ids) == expected_biases
            np.testing.assert_allclose(runtime.observe().statistics.adjusted, expected_adjusted)
            assert store.final_sampling(source) == runtime.sampling
        assert runtime.sampling.sequence_bias == (((1, 2), 2.),)


def test_cli_preset_replaces_both_kinds_through_replay(tmp_path):
    backend = NoEogBackend()
    workspace = str(tmp_path/'cli.db')
    def run(args, commands):
        with patch('trajectory_editor.episode_cli._backend', return_value=backend), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(commands)):
            assert main(['--workspace', workspace, '--plain-ui', *args]) == 0
    run(['--model', '/fake', '--new-prompt', 'P'], ['b " A B" +', '1-', '1', 'bl 2 +', '1', 'q', 'quit'])
    path = tmp_path/'preset.json'
    with EpisodeStore(workspace) as store:
        path.write_text(project_biases(store, store.resolve_id('#1')))
        original = store.final_sampling(store.resolve_id('#1'))
    run(['--model', '/fake', '--new-prompt', 'P', '--biases', str(path)], ['q', 'quit'])
    with EpisodeStore(workspace) as store:
        assert store.final_sampling(store.resolve_id('#2')) == original
    path.write_text(json.dumps({'format': 'spe-logit-bias-v1', 'model': {'vocabulary_size': 8}, 'biases': []}))
    run(['--replay', '#1', '--biases', str(path), '--divergence-policy', 'ballistic'], ['quit'])
    with EpisodeStore(workspace) as store:
        for step in store.replay_procedure(store.resolve_id('#3')):
            assert step['sampling'].sequence_bias == () and step['sampling'].logit_bias == ()
        assert store.final_sampling(store.resolve_id('#3')).sequence_bias == ()


@pytest.mark.parametrize('rules', [(((), 1),), (((-1, 2), 1),), (((True, 2), 1),), (((1, 2), float('nan')),), (((1, 2), 1), ((1, 2), 2)), None])
def test_invalid_sequence_rules(rules):
    with pytest.raises(EditorError):
        SamplingConfig(sequence_bias=rules)


def test_singleton_normalization_and_old_saved_records():
    config = SamplingConfig(sequence_bias=(((1,), .5), ((1, 2), 2)))
    assert config.logit_bias == ((1, .5),)
    assert config.sequence_bias == (((1, 2), 2.),)
    assert SamplingConfig.from_record(json.loads(json.dumps(config.to_dict()))) == config
    assert SamplingConfig.from_record(SamplingConfig().to_dict()).sequence_bias == ()
    with pytest.raises(EditorError, match='duplicate'):
        SamplingConfig(logit_bias=((1, 2),), sequence_bias=(((1,), 1),))


@pytest.mark.llama_smoke
def test_real_phrase_rule_changes_only_completion_and_preserves_baseline():
    import os
    from pathlib import Path
    model = os.environ.get('SPE_LLAMA_SMOKE_MODEL')
    if not model:
        pytest.skip('Set SPE_LLAMA_SMOKE_MODEL for real sequence integration')
    from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
    backend = LlamaCppDecoder(Path(model), LlamaCppSettings(n_ctx=128, n_gpu_layers=0, n_threads=2))
    try:
        phrase = backend.tokenize(' New York', add_bos=False, special=False)
        assert len(phrase) > 1
        runtime = EpisodeEngine(backend, initial_token_ids=phrase[:-1], sampling=SamplingConfig(temperature=0))
        original = runtime.observe()
        count = backend._model.n_tokens
        with patch.object(backend, 'eval', side_effect=AssertionError('Bias must not evaluate new tokens')):
            with pytest.raises(EdgeRequested):
                InteractivePolicy(io=ScriptedIO(['b " New York" +100', 'q'])).choose(runtime, original)
            assert runtime.observe().proposal_token_id == phrase[-1]
            np.testing.assert_array_equal(runtime.observe().logits, original.logits)
            assert backend._model.n_tokens == count
            with pytest.raises(EdgeRequested):
                InteractivePolicy(io=ScriptedIO(['b " New York" =', 'q'])).choose(runtime, runtime.observe())
            assert runtime.observe().proposal_token_id == original.proposal_token_id
            assert runtime.sampling.sequence_bias == ()
    finally:
        backend._model.close()
