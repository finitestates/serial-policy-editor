import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend, LiveScriptedIO
from trajectory_editor.boundaries import token_boundaries
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.scoped_bias import ScopedBias
from trajectory_editor.sampling import ObservationStatistics
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.episode_policy import EpisodeRunner, EdgeRequested
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_lifecycle import _create_episode, _fork_engine, _rewind_episode, _restore_engine, _spr_engine_from_source
from trajectory_editor.episode_cli import main
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.tui import parse_command


class Backend(NoEogBackend):
    pieces = {**NoEogBackend.pieces, 6: '\n'}


backend = Backend()
def boundaries(token):
    return token_boundaries(backend.token_text(token))


def rule(target=(2,), triggers=((1,),), until='sentence', bias=1):
    return ScopedBias(triggers, target, until, bias)


@pytest.mark.parametrize('text,triggers,until', [
    ('b " B" + after " A" until .', (' A',), 'sentence'),
    ('2-0.25 after [" A", " hello"] until |', (' A', ' hello'), 'newline'),
    ('b " A B" = after [" A B"] until .', (' A B',), 'sentence'),
    (r'b " B" + after "\"A\"" until .', ('"A"',), 'sentence'),
])
def test_scoped_syntax(text, triggers, until):
    command = parse_command(text, menu_size=3, vocabulary_size=8, default_hold_tokens=10)
    assert command.bias_triggers == triggers and command.bias_until == until


@pytest.mark.parametrize('text', [
    'b " B" + after " A"', '2+ until .', '2+ after [] until .',
    '2+ after [""] until .', '2+ after " A" until paragraph',
    '2=2 after " A" until .', '2+ after [" A",] until .',
    '2+ ... " A" after " B" until .', 'bl 2 + after " A" until .',
])
def test_invalid_scoped_syntax(text):
    with pytest.raises(EditorError):
        parse_command(text, menu_size=3, vocabulary_size=8, default_hold_tokens=10)


def test_alternatives_retrigger_and_span_boundaries():
    config = SamplingConfig(scoped_bias=(rule(triggers=((1,), (7, 4))),))
    assert config.active_biases([7], boundaries) == {}
    assert config.active_biases([7, 1, 3], boundaries) == {2: 1}
    assert config.active_biases([1, 1, 1], boundaries) == {2: 1}
    assert config.active_biases([7, 4, 3], boundaries) == {2: 1}
    assert config.active_biases([1, 3, 5], boundaries) == {}
    assert config.active_biases([1, 5, 1], boundaries) == {2: 1}
    assert config.active_biases([7, 5, 4], boundaries) == {}
    # Newline isn't a sentence terminator; sentence punctuation isn't a newline.
    assert config.active_biases([1, 6, 3], boundaries) == {2: 1}
    line = SamplingConfig(scoped_bias=(rule(until='newline'),))
    assert line.active_biases([1, 5, 3], boundaries) == {2: 1}
    assert line.active_biases([1, 6, 3], boundaries) == {}
    # A delimiter-containing trigger expires in its own token.
    assert SamplingConfig(scoped_bias=(rule(triggers=((5,),)),)).active_biases([5], boundaries) == {}


def test_boundary_token_is_biased_before_expiry_and_raw_ranks_unchanged():
    config = SamplingConfig(temperature=1, top_k=8, top_p=1, min_p=0,
        logit_bias=((5, .5),), sequence_bias=(((1, 5), .25),),
        scoped_bias=(rule(target=(5,), bias=4), rule(target=(5,), until='newline', bias=-1)))
    logits = np.arange(8, dtype=float)
    active = ObservationStatistics(logits, config, [1], boundaries)
    assert active.active_biases[5] == 3.75
    np.testing.assert_array_equal(active.logits, logits)
    assert active.raw_rank(5) == 3 and active.policy_rank(5) == 1
    expired = ObservationStatistics(logits, config, [1, 5], boundaries)
    assert expired.active_biases[5] == -.5  # ordinary plus still-active newline rule


def test_sequence_target_requires_tail_match_while_trigger_active():
    config = SamplingConfig(scoped_bias=(rule(target=(3, 2)),))
    assert config.active_biases([1, 4], boundaries) == {}
    assert config.active_biases([1, 4, 3], boundaries) == {2: 1}
    assert config.active_biases([1, 5, 3], boundaries) == {}
    assert config.active_biases([3], boundaries) == {}
    with pytest.raises(EditorError, match='boundary classification'):
        config.active_biases([1, 3])


@pytest.mark.parametrize('io_type', [ScriptedIO, LiveScriptedIO])
def test_edit_reordered_alternatives_clear_exact_rule_without_advancing(io_type, tmp_path):
    model = Backend()
    runtime = EpisodeEngine(model, initial_token_ids=[7, 1], sampling=SamplingConfig())
    before = runtime.observe()
    with EpisodeStore(tmp_path/'ui.db') as store:
        source = _create_episode(store, runtime, backend_provenance=model.provenance())
        with patch.object(model, 'eval', side_effect=AssertionError('Unexpected evaluation')):
            with pytest.raises(EdgeRequested):
                InteractivePolicy(io=io_type([
                    'b " B" + after [" A", " hello"] until .',
                    'b " B" + after [" hello", " A", " A"] until .', 'q']),
                    store=store, episode_id=source).choose(runtime, before)
            assert len(runtime.sampling.scoped_bias) == 1
            assert runtime.sampling.scoped_bias[0].bias == 1
            rank = runtime.observe().statistics.raw_rank(2)
            with pytest.raises(EdgeRequested):
                InteractivePolicy(io=io_type([f'{rank}= after [" A", " hello"] until .', 'q']),
                    store=store, episode_id=source).choose(runtime, runtime.observe())
        assert runtime.sampling.scoped_bias == () and runtime.boundary == 0
        assert runtime.observe().proposal_token_id == before.proposal_token_id
        np.testing.assert_array_equal(runtime.observe().logits, before.logits)
        assert len(store.interactions(source)) == 3


def test_fork_rewind_replay_and_resume_reconstruct_activation(tmp_path):
    model = Backend()
    runtime = EpisodeEngine(model, initial_text='P', sampling=SamplingConfig())
    with EpisodeStore(tmp_path/'lifecycle.db') as store:
        source = _create_episode(store, runtime, backend_provenance=model.provenance())
        runner = EpisodeRunner(runtime, store, source)
        runner.run(live_policy=InteractivePolicy(io=ScriptedIO([
            'b " B" +2 after " A" until .', 'x  A B']), store=store, episode_id=source), max_live_actions=1)
        runner.run(live_policy=InteractivePolicy(io=ScriptedIO([
            'b " B" + after " A" until .', 'x !']), store=store, episode_id=source), max_live_actions=1)
        resumed = _restore_engine(store, source, model, max_tokens=None, sampling_override=None)
        assert resumed.sampling == runtime.sampling
        assert resumed.observe().statistics.active_biases == {}
        replay, plan = _spr_engine_from_source(store, source, model, sampling=SamplingConfig(), max_tokens=None)
        child = _create_episode(store, replay, backend_provenance=model.provenance())
        assert EpisodeRunner(replay, store, child).run(tape=plan).replay_exhausted
        assert replay.sampling == runtime.sampling and replay.visible_token_ids == runtime.visible_token_ids
        for target in (3, 2, 1, 0):
            fork = _fork_engine(store, source, runtime, target, backend=model, max_tokens=None)
            expected = fork.observe().statistics.adjusted.copy()
            active = dict(fork.observe().statistics.active_biases)
            _rewind_episode(store, source, runtime, target)
            assert runtime.sampling == fork.sampling
            np.testing.assert_array_equal(runtime.observe().statistics.adjusted, expected)
            assert runtime.observe().statistics.active_biases == active
            assert bool(active) == (target in (1, 2))
        assert runtime.sampling.scoped_bias[0].bias == 2


def test_cli_v3_roundtrip_and_old_preset_clears_scoped_rules(tmp_path):
    model = Backend()
    workspace = str(tmp_path/'cli.db')
    def run(args, commands):
        with patch('trajectory_editor.episode_cli._backend', return_value=model), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(commands)):
            assert main(['--workspace', workspace, '--plain-ui', *args]) == 0
    run(['--model', '/fake', '--new-prompt', 'P'], ['b " B" + after [" A", " hello"] until .', 'x  A B', 'q', 'quit'])
    path = tmp_path/'bias.json'
    with EpisodeStore(workspace) as store:
        source = store.resolve_id('#1')
        path.write_text(project_biases(store, source))
        original = store.final_sampling(source)
    assert json.loads(path.read_text())['format'] == 'spe-logit-bias-v3'
    assert load_bias_preset(path, model, model.provenance()) == original
    run(['--model', '/fake', '--new-prompt', 'P', '--biases', str(path)], ['q', 'quit'])
    with EpisodeStore(workspace) as store:
        assert store.final_sampling(store.resolve_id('#2')) == original
    path.write_text(json.dumps({'format':'spe-logit-bias-v1','model':{'vocabulary_size':8},'biases':[]}))
    run(['--replay', '#1', '--biases', str(path), '--divergence-policy', 'ballistic'], ['quit'])
    with EpisodeStore(workspace) as store:
        assert store.final_sampling(store.resolve_id('#3')).scoped_bias == ()
        assert all(not step['sampling'].scoped_bias for step in store.replay_procedure(store.resolve_id('#3')))


@pytest.mark.parametrize('kwargs', [dict(triggers=()), dict(target=()), dict(target=(-1,)),
    dict(triggers=((True,),)), dict(until='paragraph'), dict(bias=float('nan'))])
def test_invalid_rules(kwargs):
    with pytest.raises(EditorError):
        rule(**kwargs)


def test_records_are_immutable_and_legacy_compatible():
    config = SamplingConfig(scoped_bias=[rule().to_dict()])
    assert SamplingConfig.from_record(json.loads(json.dumps(config.to_dict()))) == config
    assert SamplingConfig.from_record(SamplingConfig().to_dict()).scoped_bias == ()
    with pytest.raises(EditorError, match='duplicate'):
        SamplingConfig(scoped_bias=[rule(), rule()])


@pytest.mark.llama_smoke
def test_real_trigger_changes_proposal_then_expires():
    import os
    from pathlib import Path
    model = os.environ.get('SPE_LLAMA_SMOKE_MODEL')
    if not model:
        pytest.skip('Set SPE_LLAMA_SMOKE_MODEL for real scoped-bias integration')
    from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
    from trajectory_editor.episode_actions import Write
    decoder = LlamaCppDecoder(Path(model), LlamaCppSettings(n_ctx=128, n_gpu_layers=0, n_threads=2))
    try:
        runtime = EpisodeEngine(decoder, initial_text='Once upon a time', sampling=SamplingConfig(temperature=0))
        before = runtime.observe()
        target = runtime.candidates(before, count=2)[1].token_id
        runtime.sampling = replace(runtime.sampling, scoped_bias=(
            rule(target=(target,), triggers=((runtime.token_ids[-1],),), bias=100),))
        active = runtime.observe()
        assert active.proposal_token_id == target
        np.testing.assert_array_equal(active.logits, before.logits)
        runtime.apply(Write('!', 'exact'))
        assert runtime.observe().statistics.active_biases == {}
    finally:
        decoder._model.close()
