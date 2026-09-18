"""Measure identical prepared-view transitions through temporary/persistent TUIs.

No inference, database work or terminal-emulator painting is included. Input is
sent after each view is painted; latency runs from that submission to the next
view's first paint. Use --package-root to compare another checkout with the same
script and interpreter. ANSI bytes/erase counts cover the same intervals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from io import TextIOBase
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--implementation', choices=['persistent', 'temporary'], default='persistent')
parser.add_argument('--iterations', type=int, default=160)
parser.add_argument('--rows', type=int, default=40)
parser.add_argument('--columns', type=int, default=120)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.package_root.resolve()))
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from trajectory_editor.domain import Candidate, ChoiceSet
from trajectory_editor.tui import BoundaryReview, ChoiceFeedback


class AnsiSink(TextIOBase):
    def __init__(self):
        self.bytes = 0

    def write(self, text):
        self.bytes += len(text.encode('utf-8'))
        return len(text)


class Output(Vt100_Output):
    def __init__(self):
        self.sink = AnsiSink()
        self.clears = 0
        super().__init__(self.sink, lambda: Size(rows=args.rows, columns=args.columns),
                         term='xterm-256color', enable_cpr=False)

    def erase_down(self):
        self.clears += 1
        super().erase_down()


candidates = tuple(Candidate(i, i, f' candidate {i}', .01, .01, False)
                   for i in range(1, 41))
context = '\n'.join(f'History line {i}: previously generated text.' for i in range(200))
base = ChoiceSet('bench', 'bench', 200, 200, '0'*64, context,
                 1, ' candidate 1', .01, .01, False, candidates[:12],
                 vocabulary_size=128000, proposal_raw_rank=1)
variants = [
    ('menu', base, {}),
    ('expanded menu', replace(base, candidates=candidates), {}),
    ('rank neighborhood', base, dict(display_candidates=candidates[20:27], search_lens_active=True,
                                  feedback=ChoiceFeedback('search', 'SEARCH · rank 24'))),
    ('history review', base, dict(review=BoundaryReview(200, 199, context, '0'*64, {'kind':'token-boundary'}))),
]
output = Output()
samples = defaultdict(list)
byte_samples = []
erases = []
previous = None
iteration = 0


def painted(pipe):
    global previous
    now = time.perf_counter()
    if previous is not None and iteration >= 8:
        samples[variants[iteration % len(variants)][0]].append((now-previous[0])*1000)
        byte_samples.append(output.sink.bytes-previous[1])
        erases.append(output.clears-previous[2])
    previous = (time.perf_counter(), output.sink.bytes, output.clears)
    pipe.send_text('\r')


with create_pipe_input() as pipe:
    if args.implementation == 'persistent':
        from trajectory_editor.live_tui import ChoiceViewState
        from trajectory_editor.persistent_tui import PersistentTerminalSession

        class MeasuredSession(PersistentTerminalSession):
            seen = None
            def _rendered(self, app):
                super()._rendered(app)
                if not app.is_done and self.accepting_input and self._current is not self.seen:
                    self.seen = self._current
                    painted(pipe)

        with MeasuredSession(input_device=pipe, output_device=output) as session:
            for iteration in range(args.iterations + 8):
                label, choice, options = variants[iteration % len(variants)]
                session.read_choice(ChoiceViewState(choice, 100, choice.candidates,
                                                    lambda text, mode: text, **options))
    else:
        from trajectory_editor.live_tui import PersistentFullscreenSession, read_live_choice
        original_run = Application.run

        def run(app):
            first = True
            def after_render(app):
                nonlocal first
                if first and not app.is_done:
                    first = False
                    painted(pipe)
            app.after_render += after_render
            return original_run(app)

        with PersistentFullscreenSession(input_device=pipe, output_device=output) as session, \
             patch.object(Application, 'run', run):
            for iteration in range(args.iterations + 8):
                label, choice, options = variants[iteration % len(variants)]
                read_live_choice(choice, remaining_tokens=100, candidates=choice.candidates,
                                 resolve_insertion=lambda text, mode: text,
                                 input_device=session.input_device, output_device=session.output_device,
                                 **options)


def summary(values):
    return dict(median_ms=round(statistics.median(values), 3),
                p95_ms=round(sorted(values)[min(len(values)-1, int(len(values)*.95))], 3))

report = dict(implementation=args.implementation, package_root=str(args.package_root.resolve()),
              terminal=dict(rows=args.rows, columns=args.columns), samples=len(byte_samples),
              transitions={label:summary(values) for label,values in samples.items()},
              all_transitions=summary([value for values in samples.values() for value in values]),
              median_ansi_bytes=statistics.median(byte_samples), erase_commands=sum(erases),
              scope='Prepared views; excludes inference, database work, and emulator painting.')
rendered = json.dumps(report, indent=2)
if args.output:
    args.output.write_text(rendered+'\n')
print(rendered)
