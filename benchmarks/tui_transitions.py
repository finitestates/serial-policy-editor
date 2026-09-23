"""Measure prepared-view transitions through a persistent live terminal.

No inference, database work or terminal-emulator painting is included. Input is
sent after each view is painted; latency runs from that submission to the next
view's first paint. Run this script with --package-root for each checkout being
compared; it loads that checkout's core/src with the same harness and interpreter.
ANSI bytes and erase counts cover the same intervals.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--iterations', type=int, default=160)
parser.add_argument('--rows', type=int, default=40)
parser.add_argument('--columns', type=int, default=120)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
package_root = args.package_root.resolve()
source_root = package_root / 'core' / 'src'
if not (source_root / 'trajectory_editor' / 'persistent_tui.py').is_file():
    parser.error(f'no persistent terminal implementation under {source_root}')
sys.path.insert(0, str(source_root))
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from trajectory_editor.core.candidates import Candidate
from trajectory_editor.core.ui import ChoiceSet
from trajectory_editor.terminal_contracts import (
    BoundaryReview, ChoiceFeedback, ChoiceViewState, EdgeViewState, PromptRequest,
)
from trajectory_editor.persistent_tui import PersistentTerminalSession
import trajectory_editor.persistent_tui as loaded_terminal

if Path(loaded_terminal.__file__).resolve() != (source_root / 'trajectory_editor' / 'persistent_tui.py'):
    raise RuntimeError('benchmark loaded a terminal implementation from the wrong checkout')


class AnsiSink(TextIOBase):
    def __init__(self):
        self.bytes = 0
        self.alt_enters = 0
        self.alt_exits = 0

    def write(self, text):
        self.bytes += len(text.encode('utf-8'))
        self.alt_enters += text.count('\x1b[?1049h')
        self.alt_exits += text.count('\x1b[?1049l')
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


candidates = tuple(Candidate(i, i, f' candidate {i}', .01, False, .01)
                   for i in range(1, 41))
context = '\n'.join(f'History line {i}: previously generated text.' for i in range(200))
base = ChoiceSet('bench', 'bench', 200, 200, '0'*64, context,
                 1, ' candidate 1', .01, .01, False, candidates[:12],
                 vocabulary_size=128000, proposal_raw_rank=1)
variants = [
    ('menu', 'choice', base, {}, '\r'),
    ('expanded menu', 'choice', replace(base, candidates=candidates), {}, '\r'),
    ('rank neighborhood', 'choice', base,
     dict(display_candidates=candidates[20:27], search_lens_active=True,
          feedback=ChoiceFeedback('search', 'SEARCH · rank 24')), '\r'),
    ('history review', 'choice', base,
     dict(review=BoundaryReview(200, 199, context, '0'*64, {'kind': 'token-boundary'})), '\r'),
    ('edge', 'edge', EdgeViewState('bench', 200, 100, 100, 'temperature 1'), {}, 'q\r'),
    ('prompt', 'prompt', PromptRequest('Name> '), {}, 'name\r'),
    ('page', 'prompt', PromptRequest('', body=context, page=True), {}, '\r'),
]
output = Output()
samples = defaultdict(list)
byte_samples = []
bytes_by_surface = defaultdict(list)
redraw_samples = []
erases = []
previous = None
iteration = 0


def painted(pipe, render_count):
    global previous
    now = time.perf_counter()
    if previous is not None and iteration >= 8:
        label = variants[iteration % len(variants)][0]
        samples[label].append((now-previous[0])*1000)
        byte_samples.append(output.sink.bytes-previous[1])
        bytes_by_surface[label].append(output.sink.bytes-previous[1])
        erases.append(output.clears-previous[2])
        redraw_samples.append(render_count-previous[3])
    previous = (time.perf_counter(), output.sink.bytes, output.clears, render_count)
    pipe.send_text(variants[iteration % len(variants)][4])


with create_pipe_input() as pipe:
    class MeasuredSession(PersistentTerminalSession):
        seen = None
        render_count = 0
        applications = None

        def _rendered(self, app):
            self.render_count += 1
            if self.applications is None:
                self.applications = {id(app)}
            else:
                self.applications.add(id(app))
            super()._rendered(app)
            if not app.is_done and self.accepting_input and self._current is not self.seen:
                self.seen = self._current
                painted(pipe, self.render_count)

    with MeasuredSession(input_device=pipe, output_device=output) as session:
        for iteration in range(args.iterations + 8):
            label, kind, state, options, response = variants[iteration % len(variants)]
            if kind == 'choice':
                session.read_choice(ChoiceViewState(
                    state, 100, state.candidates, lambda text, mode: text, **options))
            elif kind == 'edge':
                session.read_edge(state)
            else:
                session.prompt(state)


def summary(values):
    return dict(median_ms=round(statistics.median(values), 3),
                p95_ms=round(sorted(values)[min(len(values)-1, int(len(values)*.95))], 3))

report = dict(implementation='persistent', package_root=str(package_root),
              terminal=dict(rows=args.rows, columns=args.columns), samples=len(byte_samples),
              transitions={label:summary(values) for label,values in samples.items()},
              all_transitions=summary([value for values in samples.values() for value in values]),
              median_ansi_bytes_by_surface={label:statistics.median(values)
                                            for label,values in bytes_by_surface.items()},
              median_ansi_bytes=statistics.median(byte_samples), erase_commands=sum(erases),
              median_redraws=statistics.median(redraw_samples),
              applications=len(session.applications), renders=session.render_count,
              fullscreen_enters=output.sink.alt_enters,
              fullscreen_exits=output.sink.alt_exits,
              scope='Prepared views; excludes inference, database work, and emulator painting.')
rendered = json.dumps(report, indent=2)
if args.output:
    args.output.write_text(rendered+'\n')
print(rendered)
