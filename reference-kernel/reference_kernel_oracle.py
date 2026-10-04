"""Minimal full-prefix model oracle. No editor engine or production sampler imports."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys

from reference_kernel import Branch, DRAW_KERNELS, Policy, State, World, draw


class LlamaBackend:
    def __init__(self, args):
        import llama_cpp
        self.binding = llama_cpp
        self.model = llama_cpp.Llama(
            model_path=args.model, n_ctx=args.context_size,
            n_batch=args.batch_size, n_threads=args.threads,
            n_gpu_layers=args.gpu_layers, seed=args.seed,
            logits_all=False, verbose=False,
        )

    def tokenize(self, text, add_bos, special):
        return tuple(self.model.tokenize(text.encode('utf-8'), add_bos=add_bos, special=special))

    def logits(self, prefix):
        if len(prefix) > self.model.n_ctx():
            raise ValueError('prefix exceeds model context size')
        self.model.reset()
        self.model.eval(list(prefix))
        pointer = self.binding.llama_get_logits_ith(self.model._ctx.ctx, -1)
        if not pointer:
            raise RuntimeError('llama.cpp returned no final-position logits')
        return tuple(float(pointer[i]) for i in range(self.model.n_vocab()))

    def render(self, ids):
        return self.model.detokenize(list(ids), special=True).decode('utf-8', errors='replace')

    def is_eog(self, token_id):
        return bool(self.binding.llama_vocab_is_eog(self.model._model.vocab, token_id))

    def close(self):
        self.model.close()


class TransformersBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        options = {'trust_remote_code': args.trust_remote_code}
        if args.dtype != 'auto':
            options['torch_dtype'] = getattr(torch, args.dtype)
        else:
            options['torch_dtype'] = 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **options).to(args.device).eval()
        self.context_size = args.context_size
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def tokenize(self, text, add_bos, special):
        # HF's add_special_tokens follows the tokenizer's own BOS/EOS template.
        return tuple(self.tokenizer.encode(text, add_special_tokens=add_bos))

    def logits(self, prefix):
        if len(prefix) > self.context_size:
            raise ValueError('prefix exceeds configured context size')
        tokens = self.torch.tensor([prefix], dtype=self.torch.long, device=self.model.device)
        with self.torch.inference_mode():
            output = self.model(input_ids=tokens, use_cache=False)
        return tuple(output.logits[0, -1].double().cpu().tolist())

    def render(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def is_eog(self, token_id):
        return token_id in self.eos

    def close(self):
        pass


def records(backend, prefix, policy, *, seed=12345, mode='atomic', count=1,
            menu_size=12, detail='full', stream_fingerprint=None, boundary=0):
    """Yield menu/selection records, then a terminal summary for pattern mode.

    Token IDs and raw token bytes (where available) are authoritative; decoded
    fragments can contain replacement characters at UTF-8 token boundaries.
    """
    branch = Branch(State(tuple(prefix)), policy, World(seed, stream_fingerprint) if stream_fingerprint else World.for_prefix(seed, prefix))
    generated = []
    terminal = None
    for step in range(1 if mode == 'atomic' else count):
        # Full-prefix evaluation; draw coordinates can start at a replay boundary.
        logits = tuple(backend.logits(branch.state.token_ids))
        distribution = policy.distribution(logits, branch.state.token_ids)
        raw_order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
        ranks = {i: rank for rank, i in enumerate(raw_order, 1)}
        proposal = draw(distribution, branch.world, boundary+step, policy.draw_kernel, policy=policy, model_ranks=ranks)
        ordered = sorted(range(len(distribution.ids)), key=lambda j: (-distribution.scores[j], distribution.ids[j]))
        selected = ordered if menu_size == 0 else ordered[:menu_size]
        proposal_index = distribution.ids.index(proposal)
        if proposal_index not in selected:
            selected = selected[:-1] + [proposal_index] if selected else [proposal_index]
        candidates = []
        for j in selected:
            token = distribution.ids[j]
            row = {'token_id': token, 'text': backend.render([token]), 'proposal': token == proposal}
            if detail != 'tokens':
                row['eligible_softmax'] = distribution.probabilities[j]
            if detail == 'full':
                row.update(score=distribution.scores[j], model_rank=ranks[token], logit=logits[token], eog=backend.is_eog(token))
            candidates.append(row)
        yield {'kind': 'choice', 'boundary': boundary+step, 'proposal_token_id': proposal,
               'support_size': len(distribution.ids), 'candidates': candidates}
        if mode == 'atomic':
            return
        if backend.is_eog(proposal):
            terminal = proposal
            break
        generated.append(proposal)
        branch = Branch(State(tuple(prefix), tuple(generated)), policy, branch.world)
    yield {'kind': 'result', 'token_ids': generated, 'text': backend.render(generated),
           'terminal_token_id': terminal, 'stop_reason': 'eog' if terminal is not None else 'requested-length'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('atomic', 'pattern'))
    p.add_argument('--backend', choices=('llama', 'transformers'), default='llama')
    p.add_argument('--model', required=True, help='GGUF path or Transformers model path/Hub ID')
    prefix = p.add_mutually_exclusive_group(required=True)
    prefix.add_argument('--prefix')
    prefix.add_argument('--prefix-file', type=Path)
    prefix.add_argument('--prefix-token-ids', help='JSON array; bypass tokenizer for exact parity')
    p.add_argument('--count', type=int, default=1, help='pattern proposal limit')
    p.add_argument('--menu-size', type=int, default=12, help='total entries, including proposal; 0 = all support')
    p.add_argument('--detail', choices=('tokens', 'probabilities', 'full'), default='full')
    p.add_argument('--format', choices=('text', 'jsonl'), default='text')
    p.add_argument('--output', type=Path)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--stream-fingerprint', help='root token SHA-256 for continuing at an existing coordinate')
    p.add_argument('--boundary', type=int, default=0)
    p.add_argument('--context-size', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--threads', type=int)
    p.add_argument('--gpu-layers', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--no-bos', action='store_true')
    p.add_argument('--special', action='store_true', help='recognize llama special token spellings in prefix')
    for f in fields(Policy):
        option = '--'+f.name.replace('_', '-')
        if f.name in ('biases', 'excluded_token_ids'):
            p.add_argument(option, type=json.loads, default=f.default, help='JSON array')
        elif f.name == 'draw_kernel':
            p.add_argument(option, choices=DRAW_KERNELS, default=f.default)
        elif f.name == 'gumbel_noise_address':
            p.add_argument(option, choices=('token-id', 'model-rank'), default=f.default)
        elif f.name in ('top_k', 'selective_noise_k'):
            p.add_argument(option, type=lambda v: None if v == 'none' else int(v), default=f.default)
        else:
            p.add_argument(option, type=type(f.default), default=f.default)
    return p


def text_record(record):
    if record['kind'] == 'choice':
        lines = [f"boundary {record['boundary']} proposal={record['proposal_token_id']} support={record['support_size']}"]
        for row in record['candidates']:
            extra = ' '.join(f'{k}={v}' for k,v in row.items() if k not in ('token_id', 'text', 'proposal'))
            lines.append(f"{'*' if row['proposal'] else ' '} {row['token_id']:>6} {json.dumps(row['text'], ensure_ascii=False)} {extra}".rstrip())
        return '\n'.join(lines)
    return f"{record['stop_reason']} tokens={json.dumps(record['token_ids'])} terminal={record['terminal_token_id']}\n{record['text']}"


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    backend = None
    try:
        policy = Policy(**{f.name: getattr(args, f.name) for f in fields(Policy)})
        World(args.seed, args.stream_fingerprint or '0'*64)
        if min(args.count, args.menu_size, args.boundary) < 0 or min(args.context_size, args.batch_size) < 1:
            raise ValueError('count, menu-size and boundary must be nonnegative; context and batch sizes must be positive')
        if args.boundary and not args.stream_fingerprint:
            raise ValueError('--boundary requires --stream-fingerprint')
        prefix = json.loads(args.prefix_token_ids) if args.prefix_token_ids is not None else None
        if prefix is not None:
            State(tuple(prefix))
        backend = LlamaBackend(args) if args.backend == 'llama' else TransformersBackend(args)
        if prefix is None:
            text = args.prefix_file.read_text(encoding='utf-8') if args.prefix_file else args.prefix
            prefix = backend.tokenize(text, not args.no_bos, args.special)
        State(tuple(prefix))
        def write(stream):
            metadata = {'kind': 'config', 'backend': args.backend, 'model': args.model,
                        'prefix_token_ids': list(prefix), 'seed': args.seed,
                        'stream_fingerprint': args.stream_fingerprint or World.for_prefix(args.seed, prefix).stream_fingerprint,
                        'boundary': args.boundary, 'policy': asdict(policy),
                        'model_settings': {k: getattr(args, k) for k in ('context_size', 'batch_size', 'threads', 'gpu_layers', 'device', 'dtype', 'no_bos', 'special', 'trust_remote_code')}}
            if args.format == 'jsonl':
                print(json.dumps(metadata, ensure_ascii=False, allow_nan=False), file=stream)
            for record in records(backend, prefix, policy, seed=args.seed, mode=args.mode,
                                  count=args.count, menu_size=args.menu_size, detail=args.detail,
                                  stream_fingerprint=args.stream_fingerprint, boundary=args.boundary):
                print(json.dumps(record, ensure_ascii=False, allow_nan=False) if args.format == 'jsonl' else text_record(record), file=stream, flush=True)
        if args.output:
            with args.output.open('w', encoding='utf-8') as stream:
                write(stream)
        else:
            write(sys.stdout)
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as exc:
        p.exit(2, f'{p.prog}: {exc}\n')
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    main()
