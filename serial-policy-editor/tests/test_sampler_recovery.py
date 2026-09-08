import hashlib
import json
from unittest.mock import patch

import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import engine
from tests.test_replay_eog import create
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_hash import token_prefix_sha256
from trajectory_editor.episode_recovery import inspect_sampler_record, recover_sampler_record
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import main, _live_edge_menu
from trajectory_editor.sampling import position_uniform


@pytest.mark.parametrize('bad', [True, '2', 3.9, -1, 1 << 63, None])
def test_hash_rejects_malformed_token_ids(bad):
    with pytest.raises(EditorError):
        token_prefix_sha256([bad])


@pytest.mark.parametrize("seed", [-(1 << 63), -1, 0, 77, (1 << 63) - 1])
def test_valid_hash_and_draw_are_unchanged(seed):
    fingerprint = token_prefix_sha256([7, 1, 2])
    assert fingerprint == hashlib.sha256(b''.join(x.to_bytes(8, 'little', signed=True) for x in [7, 1, 2])).hexdigest()
    payload = f'blake2b64-token-prefix-quantile-v2:{seed}:{fingerprint}:12'.encode()
    expected = (int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') + 0.5) / float(1 << 64)
    assert position_uniform(seed, fingerprint, 12) == expected


@pytest.mark.parametrize('bad', ['', 123, False, 'garbage', 'A' * 64])
def test_supplied_invalid_stream_never_regenerates(bad):
    runtime = engine()
    with pytest.raises(EditorError, match='stream_fingerprint'):
        EpisodeEngine(runtime.backend, sampling=runtime.sampling, initial_token_ids=[7], stream_fingerprint=bad)


@pytest.mark.parametrize('seed,offset', [('77', 0), (True, 0), (1 << 63, 0), (77, -1), (77, 1.5), (77, True)])
def test_draw_rejects_invalid_coordinates(seed, offset):
    with pytest.raises(EditorError):
        position_uniform(seed, 'a' * 64, offset)


def test_defaults_only_apply_to_new_configuration():
    assert SamplingConfig.from_mapping({'seed': 77}).seed == 77
    with pytest.raises(EditorError, match='missing'):
        SamplingConfig.from_record({'seed': 77})
    record = SamplingConfig().to_dict()
    record['rng_scheme'] = 'different'
    with pytest.raises(EditorError, match='unsupported rng_scheme'):
        SamplingConfig.from_record(record)


def damage(store, identifier):
    values = SamplingConfig(seed=77).to_dict()
    del values['seed']
    store.connection.execute('UPDATE sampler_segments SET sampling_json=?, stream_fingerprint=?, coordinate_offset=? WHERE episode_id=?',
                             (json.dumps(values), '', 1.5, identifier))
    store.connection.commit()


def snapshot(store):
    return {table: [tuple(row) for row in store.connection.execute(f'SELECT * FROM {table}')]
            for table in ('episodes', 'actions', 'tokens', 'sampler_segments', 'interactions', 'episode_names')}


@pytest.mark.parametrize('reply', ['n', None])
def test_cancel_recovery_changes_no_records(tmp_path, reply):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        identifier = create(store, engine())
        damage(store, identifier)
        before = snapshot(store)
        io = ScriptedIO([reply])
        with pytest.raises(EditorError, match='cancelled'):
            recover_sampler_record(store, identifier, io)
        assert snapshot(store) == before
        message = '\n'.join(io.output)
        assert 'missing seed would become 12345' in message
        assert 'coordinate offset would become 0' in message
        assert 'may change subsequent draws and replay results' in message


@pytest.mark.parametrize('reply', ['y', ''])
def test_confirm_recovers_copy_preserving_source_and_evidence(tmp_path, reply):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        runtime = engine(max_tokens=10)
        identifier = create(store, runtime)
        store.record_action(identifier, 0, runtime.apply(Write(' A', 'exact')))
        store.update_episode(identifier, visible_text=' A', max_tokens=None)
        damage(store, identifier)
        before = snapshot(store)
        copied = recover_sampler_record(store, identifier, ScriptedIO([reply]))
        assert copied != identifier
        assert store.get_episode(identifier)['visible_text'] == ' A'
        for table, rows in before.items():
            after = snapshot(store)[table]
            assert all(row in after for row in rows)
        segment = store.sampling_segment(copied)
        assert segment['sampling']['seed'] == 12345
        assert segment['coordinate_offset'] == 0
        assert segment['stream_fingerprint'] == token_prefix_sha256([7])
        assert store.get_episode(copied)['metadata']['sampler_recovery']['source'] == identifier
        assert store.tokens(copied)[0]['token_id'] == 1
    with EpisodeStore(path) as reopened:
        assert not inspect_sampler_record(reopened, copied).changes
        assert reopened.replay_tape(copied)[0][0] == Write(' A', 'exact')


def test_unsupported_scheme_never_offers_default_recovery(tmp_path):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        identifier = create(store, engine())
        values = SamplingConfig().to_dict()
        values['rng_scheme'] = 'future-rng'
        store.connection.execute('UPDATE sampler_segments SET sampling_json=?', (json.dumps(values),))
        store.connection.commit()
        before = snapshot(store)
        with pytest.raises(EditorError, match='unsupported rng_scheme'):
            recover_sampler_record(store, identifier, ScriptedIO([]))
        assert snapshot(store) == before


@pytest.mark.parametrize('mode', ['--resume', '--replay', '--fork-from'])
def test_cli_recovery_cancel_precedes_model_loading(tmp_path, mode):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        identifier = create(store, engine())
        damage(store, identifier)
        before = snapshot(store)
    with patch('trajectory_editor.episode_cli._backend') as load, patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['n'])):
        assert main(['--workspace', str(path), mode, identifier, '--model', 'fake', '--plain-ui']) == 2
        load.assert_not_called()
    with EpisodeStore(path) as store:
        assert snapshot(store) == before


def test_edge_replay_recovery_cancel_returns_to_menu(tmp_path):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        source = create(store, engine(), 'source')
        destination = engine()
        target = create(store, destination, 'target')
        damage(store, source)
        before = snapshot(store)
        assert _live_edge_menu(ScriptedIO(['spr #1', 'n', 'q']), store, target, destination) == ('quit', None)
        assert snapshot(store) == before


@pytest.mark.parametrize('value', [True, 1.5, -1, '2'])
def test_store_rejects_invalid_offsets_before_write(tmp_path, value):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        runtime = engine()
        identifier = create(store, runtime)
        before = snapshot(store)
        with pytest.raises(EditorError):
            store.record_sampling_segment(identifier, start_boundary=0, sampling=runtime.sampling,
                stream_fingerprint=runtime.stream_fingerprint, coordinate_offset=value)
        assert snapshot(store) == before


def test_missing_entire_sampler_segment_requires_recovery(tmp_path):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        identifier = create(store, engine())
        store.connection.execute('DELETE FROM sampler_segments WHERE episode_id=?', (identifier,))
        store.connection.commit()
        with pytest.raises(EditorError, match='no sampler segment'):
            store.sampling_segment(identifier)
        recovered = recover_sampler_record(store, identifier, ScriptedIO(['y']))
        assert store.sampling_segment(recovered)['sampling'] == SamplingConfig().to_dict()
        assert store.connection.execute('SELECT COUNT(*) FROM sampler_segments WHERE episode_id=?', (identifier,)).fetchone()[0] == 0


def test_valid_record_never_prompts_or_copies(tmp_path):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        identifier = create(store, engine())
        before = snapshot(store)
        assert recover_sampler_record(store, identifier, ScriptedIO([])) == identifier
        assert snapshot(store) == before


@pytest.mark.parametrize('bad_json', ['{', 'null', '[]'])
def test_malformed_sampler_object_is_reported_before_recovery(tmp_path, bad_json):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        identifier = create(store, engine())
        store.connection.execute('UPDATE sampler_segments SET sampling_json=?', (bad_json,))
        store.connection.commit()
        plan = inspect_sampler_record(store, identifier)
        assert any('malformed sampler object' in change for change in plan.changes)
        assert plan.segments[0]['sampling'] == SamplingConfig().to_dict()


def test_resume_recovered_copy_continues_without_touching_source(tmp_path):
    from tests.test_episode_runtime import NoEogBackend

    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        identifier = create(store, engine())
        damage(store, identifier)
        source_row = tuple(store.connection.execute('SELECT * FROM sampler_segments WHERE episode_id=?', (identifier,)).fetchone())
    io = ScriptedIO(['y', 't hello', 'q', 'quit'])
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch('trajectory_editor.episode_cli.TerminalIO', return_value=io):
        assert main(['--workspace', str(path), '--resume', identifier, '--model', 'fake', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        copied = store.resolve_id('#2')
        assert store.get_episode(copied)['visible_text'] == ' hello'
        assert store.tokens(identifier) == []
        assert tuple(store.connection.execute('SELECT * FROM sampler_segments WHERE episode_id=?', (identifier,)).fetchone()) == source_row


def test_llama_adapter_does_not_forward_editor_seed(tmp_path):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings

    model = tmp_path / 'fake.gguf'
    model.touch()
    constructor = Mock(return_value=SimpleNamespace(n_vocab=lambda: 8))
    module = SimpleNamespace(Llama=constructor)
    with patch.dict(sys.modules, {'llama_cpp': module}):
        LlamaCppDecoder(model, LlamaCppSettings())
    assert 'seed' not in constructor.call_args.kwargs
