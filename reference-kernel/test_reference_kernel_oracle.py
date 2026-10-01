import json
import pytest
from reference_kernel import Policy
from reference_kernel_oracle import records, parser, text_record


class Backend:
    def __init__(self, eog=None):
        self.prefixes = []
        self.eog = eog

    def logits(self, prefix):
        self.prefixes.append(tuple(prefix))
        return (0., 1., 2., 3.)

    def render(self, ids):
        return ''.join(str(i) for i in ids)

    def is_eog(self, token):
        return token == self.eog


def test_atomic_proposal_always_in_bounded_menu():
    for seed in range(20):
        backend = Backend()
        result, = records(backend, (1,), Policy(), seed=seed, menu_size=1)
        assert len(result['candidates']) == 1
        assert result['candidates'][0]['proposal']
        assert backend.prefixes == [(1,)]
        assert json.loads(json.dumps(result)) == result
        assert '*' in text_record(result)


def test_pattern_full_prefix_and_eog():
    backend = Backend()
    result = list(records(backend, (1,), Policy(temperature=0), mode='pattern', count=3))
    assert backend.prefixes == [(1,), (1, 3), (1, 3, 3)]
    assert result[-1]['token_ids'] == [3, 3, 3]
    backend = Backend(eog=3)
    result = list(records(backend, (1,), Policy(temperature=0), mode='pattern', count=3))
    assert result[-1]['token_ids'] == []
    assert result[-1]['terminal_token_id'] == 3
    assert result[-1]['stop_reason'] == 'eog'


def test_zero_length_and_details():
    backend = Backend()
    assert list(records(backend, (1,), Policy(), mode='pattern', count=0))[-1]['token_ids'] == []
    assert not backend.prefixes
    result, = records(backend, (1,), Policy(), menu_size=0, detail='tokens')
    assert len(result['candidates']) == 4
    assert set(result['candidates'][0]) == {'token_id', 'text', 'proposal'}


def test_cli_sampler_options():
    args = parser().parse_args(['pattern', '--model', 'x', '--prefix', 'hello', '--draw-kernel', 'student-t-max', '--student-t-df', '7', '--top-k', 'none'])
    assert args.student_t_df == 7
    assert args.top_k is None


def test_real_llama_full_prefix_matches_editor_hold():
    import os
    from pathlib import Path
    model = os.environ.get('SPE_ORACLE_LLAMA_MODEL')
    if not model:
        pytest.skip('set SPE_ORACLE_LLAMA_MODEL to a local GGUF')
    from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
    from trajectory_editor.episode_engine import EpisodeEngine
    from trajectory_editor.core.sampler_config import SamplerConfig
    from trajectory_editor.core.actions import Hold
    from reference_kernel_oracle import LlamaBackend
    args = parser().parse_args(['pattern', '--model', model, '--prefix', 'The capital of France is', '--threads', '2', '--context-size', '128', '--top-k', '20'])
    oracle = LlamaBackend(args)
    production = None
    try:
        prefix = oracle.tokenize(args.prefix, True, False)
        expected = list(records(oracle, prefix, Policy(top_k=20), mode='pattern', count=3))[-1]
        production = LlamaCppDecoder(Path(model), LlamaCppSettings(n_ctx=128, n_batch=512, n_threads=2, n_gpu_layers=0))
        engine = EpisodeEngine(production, initial_token_ids=list(prefix), sampling=SamplerConfig(seed=12345, top_k=20, top_p=1., min_p=0.))
        actual = engine.apply(Hold(3))
        assert list(actual.visible_token_ids) == expected['token_ids']
        assert actual.stop_reason == expected['stop_reason']
    finally:
        oracle.close()
        if production is not None:
            production._model.close()


def test_transformers_model_cli_both_modes(tmp_path):
    """Exercise real HF serialization/loading/inference without downloading weights."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from reference_kernel_oracle import main
    model_path = tmp_path / 'model'
    tokenizer = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, 'hello': 2, 'world': 3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>', eos_token='<eos>').save_pretrained(model_path)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=16, n_embd=8, n_layer=1, n_head=1, eos_token_id=1)).save_pretrained(model_path)
    common = ['--backend', 'transformers', '--model', str(model_path), '--prefix', 'hello world', '--temperature', '0', '--biases', '[[3,100]]']
    output = tmp_path / 'pattern.jsonl'
    main(['pattern', *common, '--count', '2', '--format', 'jsonl', '--output', str(output)])
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert result[0]['prefix_token_ids'] == [2, 3]
    assert result[-1]['token_ids'] == [3, 3]
    assert result[-1]['text'] == 'world world'
    output = tmp_path / 'atomic.txt'
    main(['atomic', *common, '--menu-size', '1', '--output', str(output)])
    assert 'proposal=3' in output.read_text()
    assert '*      3' in output.read_text()
