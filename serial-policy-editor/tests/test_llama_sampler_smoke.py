"""Opt-in real llama.cpp tests: SPE_LLAMA_SMOKE_MODEL=/path/model.gguf pytest -m llama_smoke."""
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor import episode_engine

pytestmark = pytest.mark.llama_smoke
INITIAL = SamplingConfig(temperature=.83, top_k=31, top_p=.91, min_p=.07,
    repeat_penalty=1.12, repeat_last_n=24, presence_penalty=.13, frequency_penalty=.09, seed=71)
EDITED = SamplingConfig(temperature=1.17, top_k=19, top_p=.87, min_p=.11,
    repeat_penalty=1.21, repeat_last_n=16, presence_penalty=.23, frequency_penalty=.17, seed=97)
EDIT = 's temperature=1.17 top_k=19 top_p=.87 min_p=.11 repeat_penalty=1.21 repeat_last_n=16 presence_penalty=.23 frequency_penalty=.17 seed=97'


@pytest.fixture(scope='module')
def model():
    value = os.environ.get('SPE_LLAMA_SMOKE_MODEL')
    if not value:
        pytest.skip('Set SPE_LLAMA_SMOKE_MODEL to a local GGUF to run real sampling checks')
    path = Path(value).resolve()
    assert path.is_file(), f'Configured smoke model does not exist: {path}'
    import llama_cpp
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    print(f'\nSmoke model: {path}; sha256={digest.hexdigest()}; llama-cpp-python {llama_cpp.__version__}')
    return path


def reference(logits, config, history):
    """Full-sort reference, independent of production sampling helpers."""
    values = np.array(logits, dtype=np.float64, copy=True)
    recent = history[-config.repeat_last_n:] if config.repeat_last_n else []
    for token in set(recent):
        values[token] = (values[token] * config.repeat_penalty if values[token] < 0
                         else values[token] / config.repeat_penalty)
        values[token] -= config.presence_penalty + config.frequency_penalty * recent.count(token)
    values /= config.temperature
    ids = sorted(range(len(values)), key=lambda token: (-values[token], token))[:config.top_k]
    def probabilities(ids):
        weights = np.exp(values[ids] - max(values[ids]))
        return weights / sum(weights)
    probs = probabilities(ids)
    retained, mass = [], 0.
    for token, probability in zip(ids, probs):
        retained.append(token)
        mass += probability
        if mass >= config.top_p:
            break
    probs = probabilities(retained)
    ids = [token for token, probability in zip(retained, probs)
           if probability >= config.min_p * max(probs)]
    return np.array(ids), probabilities(ids)


class CheckedIO(ScriptedIO):
    def __init__(self, commands, expected, first_edge=None):
        super().__init__(commands)
        self.expected = expected
        self.headers = 0
        self.first_edge = first_edge

    def write(self, text='', **kwargs):
        if 'Live edge @ boundary' in text:
            if self.headers == 0 and self.first_edge is not None:
                self.expected = self.first_edge
            fields = dict(item.split('=') for item in text.split(' · ', 1)[1].split())
            c = self.expected
            expected = dict(temp=f'{c.temperature:g}', top_k=str(c.top_k), top_p=f'{c.top_p:g}',
                min_p=f'{c.min_p:g}', rep=f'{c.repeat_penalty:g}/{c.repeat_last_n}',
                presence=f'{c.presence_penalty:g}', frequency=f'{c.frequency_penalty:g}', seed=str(c.seed))
            assert fields == expected
            self.headers += 1
        super().write(text, **kwargs)

    def read(self, prompt):
        command = super().read(prompt)
        if command == EDIT:
            self.expected = EDITED
        elif command in ('rewind 1', 'f 1'):
            self.expected = INITIAL
        return command


def run(path, model, commands, expected, *flags, first_edge=None):
    io = CheckedIO(commands, expected, first_edge)
    original_statistics = episode_engine.ObservationStatistics
    original_draw = episode_engine.draw_token
    calls = []
    pending = {}

    def statistics(logits, config, history):
        assert config == io.expected, 'Actual sampler configuration disagrees with scenario/UI'
        assert len(logits) > 10000 and np.isfinite(logits).all()
        result = original_statistics(logits, config, history)
        ids, probabilities = reference(logits, io.expected, list(history))
        np.testing.assert_array_equal(result.distribution.ids, ids)
        np.testing.assert_allclose(result.distribution.probabilities, probabilities, rtol=1e-11, atol=1e-14)
        pending.update(ids=ids, probabilities=probabilities)
        return result

    def draw(distribution, **kwargs):
        assert kwargs['seed'] == io.expected.seed
        payload = f"blake2b64-token-prefix-quantile-v2:{io.expected.seed}:{kwargs['stream_fingerprint']}:{kwargs['aligned_step']}".encode()
        uniform = (int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') + .5) / 2**64
        index = np.searchsorted(np.cumsum(pending['probabilities']), uniform, side='right')
        expected_token = int(pending['ids'][min(index, len(pending['ids']) - 1)])
        token = original_draw(distribution, **kwargs)
        assert token == expected_token
        calls.append((io.expected, token))
        return token

    with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), patch.object(
        episode_engine, 'ObservationStatistics', statistics
    ), patch.object(episode_engine, 'draw_token', draw):
        status = main(['--workspace', str(path), '--model', str(model), '--plain-ui',
            '--n-gpu-layers', '0', '--n-threads', '2', '--n-threads-batch', '2', '--n-ctx', '256',
            '--n-batch', '64', *flags])
    assert status == 0
    assert not io.responses, 'The intended user journey did not finish'
    assert io.headers > 0 and calls
    return calls


def initial_flags():
    flags = ['--new-prompt', 'Continue this numbered list of animals: 1. cat 2. dog 3.', '--episode-id', 'source']
    for field in INITIAL.__dataclass_fields__:
        flags.extend(['--' + field.replace('_', '-'), str(getattr(INITIAL, field))])
    return flags


@pytest.mark.parametrize('navigation', ['rewind 1', 'f 1', 'resume'])
def test_real_sampler_display_edit_and_navigation(tmp_path, model, navigation):
    path = tmp_path / 'workspace'
    commands = ['q', 'c', 'h 2', 'q', EDIT, 'c', 'h 2', 'q']
    if navigation != 'resume':
        commands += [navigation, *(['q'] if navigation == 'f 1' else []), 'c', 'h 1', 'q']
    commands += ['quit']
    calls = run(path, model, commands, INITIAL, *initial_flags())
    assert INITIAL in [c for c, _ in calls] and EDITED in [c for c, _ in calls]
    if navigation == 'resume':
        calls = run(path, model, ['q', 'c', 'h 1', 'q', 'quit'], EDITED, '--resume', 'source')
        assert all(c == EDITED for c, _ in calls)
    with EpisodeStore(path) as store:
        episodes = store.list_episodes()
        assert len(episodes) == (2 if navigation == 'f 1' else 1)
        source_config = store.final_sampling('source')
        assert source_config == (INITIAL if navigation == 'rewind 1' else EDITED)


@pytest.mark.parametrize('fixed', [False, True])
def test_real_replay_override_yields_to_edge_edits(tmp_path, model, fixed):
    path = tmp_path / 'workspace'
    run(path, model, ['h 2', 'q', EDIT, 'c', 'h 2', 'q', 'quit'], INITIAL, *initial_flags())
    replay = replace(INITIAL, seed=777)
    edge = replace(INITIAL if fixed else EDITED, seed=777)
    calls = run(path, model, [EDIT, 'c', 'h 1', 'q', 'quit'], replay,
        '--replay', 'source', '--episode-id', 'replay', '--until', '2', '--seed', '777',
        '--divergence-policy', 'ballistic', *(['--fixed-config'] if fixed else []), first_edge=edge)
    assert calls[0][0] == replay
    assert calls[-1][0] == EDITED
    with EpisodeStore(path) as store:
        assert store.final_sampling('replay') == EDITED
        assert len([t for t in store.tokens('replay') if t['realized_visible']]) == 3
