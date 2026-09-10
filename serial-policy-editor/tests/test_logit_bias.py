import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend, LiveScriptedIO
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_actions import SelectRawRank, Write
from trajectory_editor.episode_policy import EpisodeRunner
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.episode_lifecycle import _create_episode, _fork_engine, _rewind_episode, _restore_engine, _spr_engine_from_source
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.episode_cli import main
from trajectory_editor.sampling import ObservationStatistics
from trajectory_editor.tui import parse_command, CommandKind


@pytest.mark.parametrize('text,op,amount', [('12-', '-', None), ('7+', '+', None), ('8-0.5', '-', .5), ('8+.25', '+', .25), ('12=', '=', None)])
def test_bias_commands(text, op, amount):
    c = parse_command(text, menu_size=12, vocabulary_size=100, default_hold_tokens=100)
    assert c.kind == CommandKind.BIAS
    assert c.bias_operator == op and c.bias_amount == amount


@pytest.mark.parametrize('text', ['0+', '101-', '1=2', '1+-2', '1+nan', '1+0'])
def test_invalid_bias_commands(text):
    with pytest.raises(EditorError):
        parse_command(text, menu_size=12, vocabulary_size=100, default_hold_tokens=100)


def test_bias_before_temperature_and_filtering_preserves_raw_surface():
    config = SamplingConfig(temperature=2, top_k=2, top_p=1, min_p=0,
                            logit_bias=((2, 5),), repeat_penalty=2)
    stats = ObservationStatistics([3., 2., 1.], config, [0])
    np.testing.assert_allclose(stats.adjusted, [1.5, 2., 6.])
    assert stats.top_raw_ids(3) == [0, 1, 2]
    assert set(stats.distribution.ids) == {1, 2}
    expected = np.exp(np.array([2., 6.]) / 2)
    expected /= expected.sum()
    assert stats.distribution.probability(2) == pytest.approx(expected[1])


@pytest.mark.parametrize('io_type', [ScriptedIO, LiveScriptedIO])
def test_multiple_bias_edits_do_not_advance_and_clear_restores_proposal(tmp_path, io_type):
    backend = NoEogBackend()
    engine = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    original = engine.observe()
    prefix = list(backend.tokens)
    with EpisodeStore(tmp_path/'a.db') as store:
        identifier = _create_episode(store, engine, backend_provenance=backend.provenance())
        policy = InteractivePolicy(io=io_type(['1-', '2+', '1-0.5', '1=', '2=', '3']), store=store, episode_id=identifier)
        action = policy.choose(engine, original)
        assert action == SelectRawRank(3)
        assert engine.boundary == 0 and backend.tokens == prefix
        assert engine.sampling.logit_bias == ()
        assert engine.observe().proposal_token_id == original.proposal_token_id
        assert engine.observe().sampling_coordinate == original.sampling_coordinate
        np.testing.assert_array_equal(engine.observe().logits, original.logits)
        assert len(store.interactions(identifier)) == 5
        assert store.final_sampling(identifier).logit_bias == ()


def test_fork_rewind_resume_replay_and_projection_agree(tmp_path):
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    with EpisodeStore(tmp_path/'a.db') as store:
        source = _create_episode(store, runtime, backend_provenance=backend.provenance())
        # Bias at boundary zero, followed by a two-token atomic write.
        runner = EpisodeRunner(runtime, store, source)
        runner.run(live_policy=InteractivePolicy(io=ScriptedIO(['1-', 'x  A B']), store=store, episode_id=source), max_live_actions=1)
        initial_bias = runtime.sampling.logit_bias
        runner.run(live_policy=InteractivePolicy(io=ScriptedIO(['2+3', '1']), store=store, episode_id=source), max_live_actions=1)
        final_bias = runtime.sampling.logit_bias
        assert initial_bias != final_bias
        preset = tmp_path/'bias.json'
        preset.write_text(project_biases(store, source))
        assert load_bias_preset(preset, backend, backend.provenance()).logit_bias == final_bias
        resumed = _restore_engine(store, source, backend, max_tokens=None, sampling_override=None)
        assert resumed.sampling.logit_bias == final_bias
        replay, plan = _spr_engine_from_source(store, source, backend,
            sampling=SamplingConfig(), max_tokens=None)
        replay_id = _create_episode(store, replay, backend_provenance=backend.provenance())
        result = EpisodeRunner(replay, store, replay_id).run(tape=plan)
        assert result.replay_exhausted
        assert replay.sampling.logit_bias == final_bias
        assert replay.visible_token_ids == runtime.visible_token_ids
        # Fork and rewind at the same boundary, including inside the write.
        for target in (2, 1, 0):
            expected = SamplingConfig.from_record(store.sampling_segment(source, target)['sampling'])
            fork = _fork_engine(store, source, runtime, target, backend=backend, max_tokens=None)
            assert fork.sampling == expected
            _rewind_episode(store, source, runtime, target)
            assert runtime.sampling == fork.sampling
            assert store.final_sampling(source) == expected
        assert runtime.sampling.logit_bias == initial_bias


def test_cli_load_project_resume_and_override(tmp_path, capsys):
    backend = NoEogBackend()
    preset = tmp_path/'preset.json'
    preset.write_text(json.dumps({'format': 'spe-logit-bias-v1',
        'model': {'vocabulary_size': 8}, 'biases': [{'token_id': 1, 'bias': -2, 'text': ' A'}]}))
    workspace = str(tmp_path/'cli.db')
    def run(args, commands):
        with patch('trajectory_editor.episode_cli._backend', return_value=backend), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(commands)):
            assert main(['--workspace', workspace, '--plain-ui', *args]) == 0
    run(['--model', '/fake', '--new-prompt', 'P', '--biases', str(preset)], ['2+', 'q', 'quit'])
    with EpisodeStore(workspace) as store:
        saved = store.final_sampling(store.resolve_id('#1'))
        assert dict(saved.logit_bias) == {1: -2, 2: .5}
    run(['--resume', '#1'], ['q', 'quit'])
    capsys.readouterr()
    assert main(['--workspace', workspace, '--project', '#1', '--biases-only']) == 0
    exported = json.loads(capsys.readouterr().out)
    assert len(exported['biases']) == 2
    run(['--resume', '#1', '--biases', str(preset)], ['q', 'quit'])
    with EpisodeStore(workspace) as store:
        assert dict(store.final_sampling(store.resolve_id('#1')).logit_bias) == {1: -2}


@pytest.mark.parametrize('pairs', [[(1, float('nan'))], [(1, float('inf'))], [(True, 1)], [(1, 1), (1, 2)], [(1,)], None])
def test_invalid_bias_config(pairs):
    with pytest.raises(EditorError):
        SamplingConfig(logit_bias=pairs)


def test_old_records_and_preset_mismatch(tmp_path):
    old = SamplingConfig().to_dict()
    assert 'logit_bias' not in old
    assert SamplingConfig.from_record(old).logit_bias == ()
    new = SamplingConfig(logit_bias=((2, -.25),), bias_step=1)
    assert SamplingConfig.from_record(json.loads(json.dumps(new.to_dict()))) == new
    path = tmp_path/'wrong.json'
    path.write_text(json.dumps({'format': 'spe-logit-bias-v1', 'model': {'vocabulary_size': 8},
        'biases': [{'token_id': 1, 'bias': 2, 'text': 'wrong'}]}))
    backend = NoEogBackend()
    with pytest.raises(EditorError, match='text mismatch'):
        load_bias_preset(path, backend, backend.provenance())


def test_cli_replay_preset_overrides_later_source_biases(tmp_path):
    backend = NoEogBackend()
    workspace = str(tmp_path/'replay.db')
    def run(args, commands):
        with patch('trajectory_editor.episode_cli._backend', return_value=backend), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(commands)):
            assert main(['--workspace', workspace, '--plain-ui', *args]) == 0
    run(['--model', '/fake', '--new-prompt', 'P'], ['1-', '1', '2+3', '1', 'q', 'quit'])
    preset = tmp_path/'clear.json'
    preset.write_text(json.dumps({'format': 'spe-logit-bias-v1', 'model': {'vocabulary_size': 8}, 'biases': []}))
    run(['--replay', '#1', '--biases', str(preset), '--divergence-policy', 'ballistic'], ['quit'])
    with EpisodeStore(workspace) as store:
        replay_id = store.resolve_id('#2')
        assert store.final_sampling(replay_id).logit_bias == ()
        for step in store.replay_procedure(replay_id):
            assert step['sampling'].logit_bias == ()
    run(['--fork-from', '#1', '--at', '1'], ['q', 'quit'])
    with EpisodeStore(workspace) as store:
        parent = SamplingConfig.from_record(store.sampling_segment(store.resolve_id('#1'), 1)['sampling'])
        assert store.final_sampling(store.resolve_id('#3')) == parent


def test_trailing_biases_replay_even_without_a_token_move(tmp_path):
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    from trajectory_editor.episode_policy import EdgeRequested
    with EpisodeStore(tmp_path/'trailing.db') as store:
        source = _create_episode(store, runtime, backend_provenance=backend.provenance())
        with pytest.raises(EdgeRequested):
            InteractivePolicy(io=ScriptedIO(['2+', 'q']), store=store, episode_id=source).choose(runtime, runtime.observe())
        replay, plan = _spr_engine_from_source(store, source, backend, sampling=SamplingConfig(), max_tokens=None)
        child = _create_episode(store, replay, backend_provenance=backend.provenance())
        result = EpisodeRunner(replay, store, child).run(tape=plan)
        assert result.replay_exhausted
        assert replay.boundary == 0 and replay.sampling.logit_bias == runtime.sampling.logit_bias


@pytest.mark.llama_smoke
def test_real_llama_bias_changes_proposal_without_evaluation():
    import os
    from pathlib import Path
    model = os.environ.get('SPE_LLAMA_SMOKE_MODEL')
    if not model:
        pytest.skip('Set SPE_LLAMA_SMOKE_MODEL for real bias integration')
    from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
    backend = LlamaCppDecoder(Path(model), LlamaCppSettings(n_ctx=128, n_gpu_layers=0, n_threads=2))
    try:
        runtime = EpisodeEngine(backend, initial_text='Once upon a time', sampling=SamplingConfig(temperature=0))
        before = runtime.observe()
        second = runtime.candidates(before, count=2)[1]
        tokens = backend._model.n_tokens
        with patch.object(backend, 'eval', side_effect=AssertionError('Bias must not evaluate tokens')):
            runtime.sampling = replace(runtime.sampling, logit_bias=((second.token_id, 100.),))
            after = runtime.observe()
            assert after.proposal_token_id == second.token_id
            assert after.proposal_raw_rank == 2
            np.testing.assert_array_equal(before.logits, after.logits)
            runtime.sampling = replace(runtime.sampling, logit_bias=())
            assert runtime.observe().proposal_token_id == before.proposal_token_id
        assert backend._model.n_tokens == tokens
    finally:
        backend._model.close()


def test_live_preview_treats_bias_as_effect_not_insertion():
    from trajectory_editor.live_tui import action_preview
    from trajectory_editor.episode_ui import _choice_from_observation
    backend = NoEogBackend()
    runtime = EpisodeEngine(backend, initial_text='P', sampling=SamplingConfig())
    observation = runtime.observe()
    candidates = runtime.candidates(observation)
    choice = _choice_from_observation(runtime, observation, candidates, context_characters=100, serial=1)
    preview = action_preview(choice, '2-0.5', candidates, lambda text, mode: text)
    assert preview.valid and preview.kind == 'effect'
    assert not preview.appended_text
    assert not action_preview(choice, '2=1', candidates, lambda text, mode: text).valid
