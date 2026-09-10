from unittest.mock import patch

import pytest

from trajectory_editor.decoder import LlamaCppSettings
from trajectory_editor.domain import EditorError
from trajectory_editor.episode_cli import build_parser, _backend, _load_episode_backend
from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend


def test_cache_flags_reach_backend():
    args = build_parser().parse_args(['--model', '/model.gguf', '--cache-type-k', 'q8_0', '--cache-type-v', 'q4_0'])
    with patch('trajectory_editor.episode_cli.create_backend') as create:
        _backend(args)
    settings = create.call_args.kwargs['llama_settings']
    assert (settings.type_k, settings.type_v) == ('q8_0', 'q4_0')


def test_saved_precision_and_explicit_override():
    args = build_parser().parse_args(['--cache-type-k', 'f16'])
    args._explicit_options = {'type_k'}
    source = {'backend': {'backend': 'llama.cpp', 'model_path': '/model.gguf',
                         'load_options': {'type_k': 'q8_0', 'type_v': 'q4_0'}}}
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()) as load:
        _, provenance, _ = _load_episode_backend(args, source, ScriptedIO([]))
    selected = load.call_args.args[0]
    assert (selected.type_k, selected.type_v) == ('f16', 'q4_0')
    assert provenance['load_options']['type_v'] == 'q4_0'
    assert provenance['load_options']['type_k'] == 'f16'


@pytest.mark.parametrize('value', ['q8_0', 'q4_0'])
def test_quantized_v_requires_flash_attention(value):
    with pytest.raises(EditorError, match='requires Flash Attention'):
        LlamaCppSettings(type_v=value, flash_attn=False)
    LlamaCppSettings(type_k=value, type_v='f16', flash_attn=False)


def test_defaults_and_invalid_precision():
    args = build_parser().parse_args([])
    assert args.type_k is None and args.type_v is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--cache-type-k', 'q7'])
    with pytest.raises(EditorError):
        LlamaCppSettings(type_k='q7')


@pytest.mark.llama_smoke
@pytest.mark.parametrize('precision', ['f16', 'q8_0', 'q4_0'])
def test_real_cache_generation_and_rewind(precision):
    import os
    from pathlib import Path
    import numpy as np
    from trajectory_editor.decoder import LlamaCppDecoder

    model = os.environ.get('SPE_LLAMA_SMOKE_MODEL')
    if not model:
        pytest.skip('Set SPE_LLAMA_SMOKE_MODEL for real cache tests')
    import llama_cpp
    decoder = LlamaCppDecoder(Path(model), LlamaCppSettings(
        n_ctx=128, n_batch=32, n_threads=2, n_gpu_layers=0,
        type_k=precision, type_v=precision))
    try:
        expected = getattr(llama_cpp, 'GGML_TYPE_' + precision.upper())
        assert decoder._model.context_params.type_k == expected
        assert decoder._model.context_params.type_v == expected
        prefix = decoder.tokenize('Once upon a time', add_bos=True)
        decoder.reset(prefix)
        initial = decoder.last_logits()
        assert np.isfinite(initial).all()
        decoder.eval([int(initial.argmax())])
        assert np.isfinite(decoder.last_logits()).all()
        decoder.branch_to_prefix(prefix)
        np.testing.assert_allclose(decoder.last_logits(), initial, atol=0.002, rtol=0.002)
        assert decoder.provenance()['runtime_configuration']['type_k'] == precision
    finally:
        decoder._model.close()
