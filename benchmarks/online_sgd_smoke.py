"""Offline, reproducible tiny-model CLI experiment (not a natural-language benchmark).

Run: .venv/bin/python benchmarks/online_sgd_smoke.py --output /tmp/online-sgd-smoke
The generated checkpoint is random GPT-2, not a pretrained writing model.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from trajectory_editor.core.actions import Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.online_learning import LearningConfig
from trajectory_editor.transformers_backend import TransformersBackend, TransformersSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint'
    torch.set_num_threads(1)
    torch.manual_seed(19)
    vocabulary = {'<unk>': 0, '<eos>': 1, 'P': 2, 'red': 3, 'blue': 4,
                  'green': 5, 'house': 6, 'ghost': 7}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
        eos_token='<eos>').save_pretrained(checkpoint)
    GPT2LMHeadModel(GPT2Config(vocab_size=8, n_embd=16, n_layer=1, n_head=2,
        n_positions=64, bos_token_id=2, eos_token_id=1)).save_pretrained(checkpoint)
    settings = TransformersSettings(device='cpu', dtype='float32', torch_num_threads=1)
    comparison = {}
    for online in [False, True]:
        backend = TransformersBackend(checkpoint, settings)
        if online:
            backend.enable_online_learning(LearningConfig(lr=0.05, steps=5))
        e = EpisodeEngine(backend, initial_text='P', initial_token_ids=[2],
            sampling=SamplerConfig(temperature=0.0, top_k=8, min_p=0.0))
        before = e.observe()
        original = before.logits.copy()
        e.apply(Write('red blue', mode='exact'))
        continuation = e.observe().proposal_token_id
        backend.reset([2])
        logits = backend.last_logits()
        probabilities = torch.softmax(torch.tensor(logits), 0)
        comparison['online' if online else 'frozen'] = {
            'original_proposal': before.proposal_token_id,
            'corrected_prefix_red_probability': probabilities[3].item(),
            'original_prefix_logits_changed': not np.array_equal(original, logits),
            'continuation_proposal': continuation,
            'learning_events': e.learner.events if online else [],
        }
    artifact = args.output.resolve() / 'cli-experiment.json'
    commands = '\n'.join(['learn steps=3', 'x red blue', 'learn lr=0.05',
        'teach last steps=5', 'learn status', 'q', f'export {artifact}',
        'c', 'learn reset', 'learn status', 'q', 'quit']) + '\n'
    command = [str(Path(sys.executable).with_name('policy-editor')), '--plain-ui',
        '--backend', 'transformers', '--model', str(checkpoint.resolve()),
        '--transformers-device', 'cpu', '--transformers-dtype', 'float32',
        '--transformers-torch-threads', '1', '--new-prompt', 'P',
        '--temperature', '0', '--online-sgd', '--online-lr', '0.05']
    result = subprocess.run(command, input=commands, text=True, capture_output=True, timeout=60)
    (args.output / 'cli-transcript.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    exported = json.loads(artifact.read_text())
    assert [event['steps'] for event in exported['learning_events']] == [3, 5]
    assert exported['token_ids'] == [2, 3, 4]
    assert '[online SGD]' in result.stdout
    assert 'Original episode weights restored' in result.stdout
    summary = {'seed': 19, 'model': 'random GPT-2, 4016 parameters, float32 CPU',
        'torch': torch.__version__, 'comparison': comparison,
        'cli_exit_code': result.returncode, 'cli_export': str(artifact),
        'interpretation': 'Tests local update mechanics only; no evidence of fewer future interventions.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
