"""Opt-in real Transformers tests: SPE_TRANSFORMERS_SMOKE_MODEL=/path/model pytest -m transformers_smoke."""
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
from trajectory_editor.transformers_backend import TransformersBackend

pytestmark = pytest.mark.transformers_smoke
INITIAL = SamplingConfig(temperature=.83, top_k=31, top_p=.91, min_p=.07,
    repeat_penalty=1.12, repeat_last_n=24, presence_penalty=.13, frequency_penalty=.09, seed=71)
EDITED = SamplingConfig(temperature=1.17, top_k=19, top_p=.87, min_p=.11,
    repeat_penalty=1.21, repeat_last_n=16, presence_penalty=.23, frequency_penalty=.17, seed=97)
EDIT = 's temperature=1.17 top_k=19 top_p=.87 min_p=.11 repeat_penalty=1.21 repeat_last_n=16 presence_penalty=.23 frequency_penalty=.17 seed=97'


@pytest.fixture(scope='module')
def model():
    value = os.environ.get('SPE_TRANSFORMERS_SMOKE_MODEL')
    if not value:
        pytest.skip('Set SPE_TRANSFORMERS_SMOKE_MODEL to a local Hugging Face model directory')
    path = Path(value).resolve()
    assert path.is_dir(), f'Configured smoke model directory does not exist: {path}'
    import torch
    import transformers
    print(f'\nSmoke model: {path}; transformers {transformers.__version__}; torch {torch.__version__}')
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


def top_raw_ids(logits, count=64):
    """Independent raw-logit ordering with SPE's token-id tie break."""
    values = np.asarray(logits, dtype=np.float64)
    token_ids = np.arange(len(values), dtype=np.int64)
    return token_ids[np.lexsort((token_ids, -values))[:min(count, len(values))]]


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


def run(path, model, commands, expected, *flags, first_edge=None, cache='auto'):
    io = CheckedIO(commands, expected, first_edge)
    original_statistics = episode_engine.ObservationStatistics
    original_draw = episode_engine.draw_token
    original_last_logits = TransformersBackend.last_logits
    original_observe = episode_engine.EpisodeEngine.observe
    calls = []
    pending = {}

    def checked_observe(engine):
        # Capture the intended prefix from the episode, independently of the
        # backend's ledger. Check even when observe reuses a cached observation.
        prefix = tuple(engine.token_ids)
        assert tuple(engine.backend._tokens) == prefix, (
            "Transformers token prefix disagrees with the episode's intended prefix"
        )
        pending.clear()
        pending.update(engine_backend=engine.backend, engine_prefix=prefix)
        try:
            return original_observe(engine)
        finally:
            pending.clear()

    def checked_last_logits(backend):
        """Compare SPE's cached result with a direct full-prefix HF forward pass."""
        assert pending.get('engine_backend') is backend, 'Logit read outside the observed episode'
        prefix = pending['engine_prefix']
        assert tuple(backend._tokens) == prefix, 'Backend prefix changed before logit evaluation'
        actual = original_last_logits(backend)
        torch = backend._torch
        input_ids = torch.tensor([prefix], dtype=torch.long, device=backend._input_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = backend._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        oracle = (
            outputs.logits[0, -1, :backend.vocabulary_size()]
            .detach().to(dtype=torch.float32, device='cpu').numpy().copy()
        )
        np.testing.assert_allclose(actual, oracle, rtol=2e-4, atol=2e-5)
        np.testing.assert_array_equal(top_raw_ids(actual), top_raw_ids(oracle))
        pending['oracle_logits'] = oracle
        return actual

    def statistics(logits, config, history):
        assert config == io.expected, 'Actual sampler configuration disagrees with scenario/UI'
        assert len(logits) > 100 and np.isfinite(logits).all()
        assert tuple(history) == pending['engine_prefix'], 'Sampler history disagrees with episode prefix'
        oracle = pending.get('oracle_logits')
        assert oracle is not None, 'No direct Transformers logit oracle was captured'

        # First prove that backend/cache differences do not change sampler consequences.
        oracle_ids, oracle_probabilities = reference(oracle, io.expected, list(history))
        actual_ids, actual_probabilities = reference(logits, io.expected, list(history))
        np.testing.assert_array_equal(actual_ids, oracle_ids)
        np.testing.assert_allclose(actual_probabilities, oracle_probabilities, rtol=2e-4, atol=2e-6)

        # Then prove SPE's sampler agrees with the independent reference.
        result = original_statistics(logits, config, history)
        np.testing.assert_array_equal(result.distribution.ids, actual_ids)
        np.testing.assert_allclose(result.distribution.probabilities, actual_probabilities, rtol=1e-11, atol=1e-14)
        pending.update(ids=actual_ids, probabilities=actual_probabilities,
                       oracle_ids=oracle_ids, oracle_probabilities=oracle_probabilities)
        return result

    def draw(distribution, **kwargs):
        assert kwargs['seed'] == io.expected.seed
        payload = f"blake2b64-token-prefix-quantile-v2:{io.expected.seed}:{kwargs['stream_fingerprint']}:{kwargs['aligned_step']}".encode()
        uniform = (int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') + .5) / 2**64
        index = np.searchsorted(np.cumsum(pending['probabilities']), uniform, side='right')
        expected_token = int(pending['ids'][min(index, len(pending['ids']) - 1)])
        oracle_index = np.searchsorted(np.cumsum(pending['oracle_probabilities']), uniform, side='right')
        oracle_token = int(pending['oracle_ids'][min(oracle_index, len(pending['oracle_ids']) - 1)])
        assert expected_token == oracle_token, (
            f"Cached and fresh full-prefix sampling disagree at coordinate {kwargs['aligned_step']}: "
            f"cached={expected_token}, fresh={oracle_token}, uniform={uniform!r}"
        )
        token = original_draw(distribution, **kwargs)
        assert token == expected_token
        calls.append((io.expected, token))
        return token

    with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), patch.object(
        episode_engine.EpisodeEngine, 'observe', checked_observe
    ), patch.object(
        TransformersBackend, 'last_logits', checked_last_logits
    ), patch.object(
        episode_engine, 'ObservationStatistics', statistics
    ), patch.object(episode_engine, 'draw_token', draw):
        status = main(['--workspace', str(path), '--model', str(model), '--backend', 'transformers',
            '--transformers-device', 'cpu', '--transformers-dtype', 'float32', '--cache', cache,
            '--plain-ui', *flags])
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


def test_real_transformers_cache_matches_full_prefix(tmp_path, model):
    """One short journey with cache disabled, to cover both public cache modes."""
    path = tmp_path / 'workspace'
    calls = run(path, model, ['h 2', 'q', 'quit'], INITIAL, *initial_flags(), cache='off')
    assert calls
