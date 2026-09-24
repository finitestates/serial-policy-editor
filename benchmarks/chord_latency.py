"""Separate ephemeral chord restoration, commit, and next-menu preparation.

Requires an explicitly selected local GGUF. No downloads or database writes.
Use the same interpreter/model/settings with --package-root pointing at source
copies of each revision. Terminal painting is measured by tui_transitions.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--rounds', type=int, nargs='+', default=[0, 3])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--cache', choices=['auto', 'off'], default='auto')
    parser.add_argument('--one-live-path', action='store_true', help='pair rank 1 with an already-ended EOG path')
    parser.add_argument('--prompt-repeats', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.rounds) < 0 or min(args.repeats, args.threads, args.prompt_repeats) < 1:
        parser.error('rounds must be nonnegative; repeats, threads and prompt-repeats must be positive')
    root = args.package_root.resolve()
    sys.path.insert(0, str(root / 'core' / 'src'))
    from trajectory_editor import chord as chord_module, live_tui
    from trajectory_editor.chord import Chord, ActionSequencePolicy
    from trajectory_editor.core.sampler_config import SamplerConfig
    from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
    from trajectory_editor.episode_engine import EpisodeEngine
    from trajectory_editor.episode_runner import LiveSessionRunner
    from trajectory_editor.episode_session import LiveSession
    from trajectory_editor.episode_ui import InteractivePolicy

    if Path(chord_module.__file__).resolve() != root / 'core/src/trajectory_editor/chord.py':
        raise RuntimeError('loaded source from the wrong checkout')

    class MenuReady(Exception):
        pass

    class PreparedMenu:
        # Support the request contract on either side of TUI consolidation.
        supports_live_choices = True
        capabilities = SimpleNamespace(live_views=True, seamless_review=True)

        def terminal_size(self):
            return 120, 40

        def read_choice(self, *args, **kwargs):
            raise MenuReady

    backend = LlamaCppDecoder(args.model, LlamaCppSettings(
        n_ctx=2048, n_batch=128, n_threads=args.threads, n_gpu_layers=args.gpu_layers,
    ), cache_mode=args.cache)
    samples = []
    prompt = 'The story begins with a curious traveler. ' * args.prompt_repeats
    calls = []
    native_eval = backend._model.eval

    def counted_eval(values):
        calls.append(len(values))
        return native_eval(values)

    @contextmanager
    def phase(record, name):
        start_call = len(calls)
        start = perf_counter()
        try:
            yield
        finally:
            record[name] = {
                'ms': (perf_counter() - start) * 1000,
                'model_calls': len(calls) - start_call,
                'input_positions': sum(calls[start_call:]),
            }

    try:
        # Historical menu preparation queried the live module for its width.
        # Preimport it and pin the same size so neither lazy imports nor this
        # harness process's terminal affect the before/after comparison.
        with patch.object(backend._model, 'eval', counted_eval), patch.object(
            live_tui, '_terminal_size', return_value=(120, 40),
        ):
            # One untimed warmup. Every sample then starts from the same prompt.
            backend.reset(backend.tokenize(prompt, add_bos=True, special=True))
            for repeat in range(args.repeats):
                for rounds in args.rounds:
                    engine = EpisodeEngine(backend, initial_text=prompt,
                                           sampling=SamplerConfig(temperature=0.0))
                    session = LiveSession(engine, prompt=prompt)
                    observation = engine.observe()
                    ranks, selected_index = (1, 2, 3), 0
                    if args.one_live_path:
                        eog_rank = observation.statistics.raw_rank(backend.eog_token_ids()[0])
                        if eog_rank == 1:
                            raise RuntimeError('rank 1 is EOG; choose another prompt fixture')
                        ranks, selected_index = (eog_rank, 1), 1
                    phases = {}
                    with phase(phases, 'construct'):
                        chord = Chord(engine, ranks)
                    try:
                        with phase(phases, 'advance'):
                            for _ in range(rounds):
                                if not chord.advance():
                                    raise RuntimeError('all paths ended before the requested number of rounds')
                        selected_path = chord.paths[selected_index]
                        preview = list(selected_path.engine.visible_token_ids)
                        with phase(phases, 'restore'):
                            actions = chord.select(selected_path.label)
                        with phase(phases, 'commit'):
                            result = LiveSessionRunner(session).run(
                                live_policy=ActionSequencePolicy(actions), max_live_actions=len(actions),
                            )
                        if len(result.outcomes) != len(actions) or engine.visible_token_ids != preview:
                            raise RuntimeError('selected continuation did not match the preview')
                        if engine.ended or engine.checkpointed:
                            raise RuntimeError('selected path has no next menu; choose another fixture')
                        with phase(phases, 'next_menu'):
                            try:
                                InteractivePolicy(io=PreparedMenu(), seamless=True).choose(engine, engine.observe())
                            except MenuReady:
                                pass
                            else:
                                raise RuntimeError('menu preparation did not reach read_choice')
                        samples.append(dict(repeat=repeat, rounds=rounds, phases=phases,
                                            prompt_tokens=len(engine.initial_token_ids),
                                            visible_token_ids=list(engine.visible_token_ids)))
                    finally:
                        chord.discard()
    finally:
        backend.close()
    report = dict(
        package_root=str(root), model=str(args.model.resolve()), threads=args.threads,
        gpu_layers=args.gpu_layers, cache=args.cache, prompt_repeats=args.prompt_repeats,
        one_live_path=args.one_live_path,
        python=sys.version, llama_cpp_version=backend._llama_cpp.__version__,
        scope='Ephemeral, greedy, no CFG; excludes load, initial prompt, input waits and terminal painting. Input positions count model eval arguments, not attention work.',
        samples=samples,
        median_ms={str(rounds): {
            phase: round(statistics.median(sample['phases'][phase]['ms'] for sample in samples
                                           if sample['rounds'] == rounds), 3)
            for phase in ('construct', 'advance', 'restore', 'commit', 'next_menu')
        } for rounds in args.rounds},
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
