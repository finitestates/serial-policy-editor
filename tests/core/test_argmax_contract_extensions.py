"""Focused negative controls for the new numeric, evidence and refill contracts."""
from dataclasses import replace
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.beam import BeamSearch
from trajectory_editor.candidate_columns import CandidateColumns
from trajectory_editor.core.actions import Accept, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.sampling import EligibleScores
from trajectory_editor.episode_ui import PolicyViewPreferences
from trajectory_editor.episode_cli import build_parser
from trajectory_editor.controller_profiles import load_controller_profile, profile_arguments
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.projector import project_episode


def runtime(backend=None):
    return EpisodeEngine(backend or ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplerConfig())


def test_normal_observe_accept_write_and_render_never_request_softmax(monkeypatch):
    def forbidden(_):
        pytest.fail('unrequested eligible softmax was materialized')
    monkeypatch.setattr(EligibleScores, 'softmax', property(forbidden))
    engine = runtime()
    policy = InteractivePolicy(io=ScriptedIO(['1']), menu_size=3)
    policy.choose(engine, engine.observe())
    assert engine.apply(Accept()).evidence[0].eligible_softmax is None
    outcome = engine.apply(Write(' A', mode='exact'))
    assert all(item.eligible_softmax is None for item in outcome.evidence)


def test_softmax_underflow_does_not_change_membership_and_zero_is_visible():
    class Extreme(ConformingFakeBackend):
        def last_logits(self):
            return np.asarray([0., 1000., -1000., -1000., -1000., -1000., -1000., -1000.])
    engine = runtime(Extreme())
    observation = engine.observe()
    rows = engine.candidates(observation, count=8, metrics=frozenset({'eligible_softmax'}))
    row = next(row for row in rows if row.token_id == 2)
    assert row.eligible and row.eligible_softmax == 0.
    columns = CandidateColumns(overlays=frozenset({'probability'}))
    assert columns.values(row) != columns.values(replace(row, eligible_softmax=None))
    engine.sampling = replace(engine.sampling, top_k=1)
    excluded = next(row for row in engine.candidates(engine.observe(), count=8) if row.token_id == 2)
    assert not excluded.eligible and excluded.eligible_softmax is None


@pytest.mark.parametrize('version', [1, 2])
def test_previous_schema_rejection_leaves_database_bytes_unchanged(tmp_path, version):
    path = tmp_path / 'old.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE schema_info (version INTEGER NOT NULL)')
        db.execute('INSERT INTO schema_info VALUES (?)', (version,))
        db.execute('CREATE TABLE retained_content (value TEXT)')
        db.execute("INSERT INTO retained_content VALUES ('keep me')")
    before = path.read_bytes()
    with pytest.raises(EditorError, match='left untouched'):
        EpisodeStore(path)
    assert path.read_bytes() == before


def test_schema3_nullable_softmax_and_explicit_projector_reconstruction(tmp_path):
    engine = runtime()
    with EpisodeStore(tmp_path / 'fresh.sqlite3') as store:
        identifier = store.create_episode(initial_text='', initial_token_ids=[7],
            sampling=engine.sampling, stream_fingerprint=engine.stream_fingerprint,
            backend=engine.backend.provenance())
        store.record_action(identifier, 0, engine.apply(Accept()))
        assert store.connection.execute('SELECT version FROM schema_info').fetchone()[0] == 3
        assert store.tokens(identifier)[0]['eligible_softmax'] is None
        columns = {row['name']: row for row in store.connection.execute('PRAGMA table_info(tokens)')}
        assert 'decoder_probability' not in columns
        assert columns['eligible_softmax']['notnull'] == 0
        report = project_episode(store, identifier, with_model_probs=True, backend=ConformingFakeBackend()).text
        assert 'eligible-softmax=' in report and 'model-softmax=' in report
        assert store.tokens(identifier)[0]['eligible_softmax'] is None


def test_exact_overlay_selection_on_off_and_clear_preserves_order():
    engine = runtime()
    prefs = PolicyViewPreferences()
    io = ScriptedIO(['columns logit diff noise', 'overlay noise off', 'overlay probability on', '1'])
    policy = InteractivePolicy(io=io, view_preferences=prefs, menu_size=2)
    policy.choose(engine, engine.observe())
    assert prefs.overlays == frozenset({'logit', 'diff', 'probability'})
    order = (prefs.sort_by_policy, prefs.sort_by_gumbel)
    policy.io = ScriptedIO(['C', '1'])
    policy.choose(engine, engine.observe())
    assert prefs.overlays == frozenset() and (prefs.sort_by_policy, prefs.sort_by_gumbel) == order


@pytest.mark.parametrize('name', ['argmax', 'gumbel-control', 'eligible-selective', 'gaussian-robustness'])
def test_documented_controller_profiles_parse(name):
    parser = build_parser(include_vector=False)
    path = Path(__file__).resolve().parents[2] / 'examples' / (name + '.yaml')
    values, _ = load_controller_profile(path, parser)
    tokens, _ = profile_arguments(parser, values)
    args = parser.parse_args(tokens)
    assert args.temperature == 1.


class FakeBatch:
    def __init__(self, prefixes):
        self.lanes = [ConformingFakeBackend() for _ in prefixes]
        self.closed = False
        for lane, prefix in zip(self.lanes, prefixes):
            lane.reset(prefix)
    def lane(self, index):
        return self.lanes[index]
    def fork(self, mapping):
        snapshots = {index: list(lane.tokens) for index, lane in enumerate(self.lanes)}
        for destination, source in mapping.items():
            self.lanes[destination].reset(snapshots[source])
    def retire(self, indices):
        for index in indices:
            self.lanes[index].reset([])
    def rebuild(self, indices):
        for index in indices:
            assert self.lanes[index].tokens
    def flush(self, indices):
        for index in indices:
            assert self.lanes[index].tokens
    def close(self):
        self.closed = True


class BatchBackend(ConformingFakeBackend):
    def create_batch(self, prefixes):
        self.batch = FakeBatch(prefixes)
        return self.batch


def test_batched_beam_refill_matches_serial_frontier_and_can_commit():
    serial_engine, batched_engine = runtime(), runtime(BatchBackend())
    serial, batched = BeamSearch(serial_engine, width=3), BeamSearch(batched_engine, width=3)
    def frontier(beam):
        return [(beam._path_token_ids(path), path.score) for path in beam.active]
    for _ in range(3):
        assert frontier(serial) == frontier(batched)
        serial.kill(serial.active[0].label)
        batched.kill(batched.active[0].label)
    assert frontier(serial) == frontier(batched)
    serial.advance(2)
    batched.advance(2)
    assert frontier(serial) == frontier(batched)
    serial.rewind()
    batched.rewind()
    serial.kill(serial.active[0].label)
    batched.kill(batched.active[0].label)
    assert frontier(serial) == frontier(batched)
    serial.discard()
    selected = batched.active[0]
    expected = tuple(selected.engine.visible_token_ids)
    batched.select(selected.label, promote=True)
    assert tuple(batched_engine.visible_token_ids) == expected
    assert batched_engine.backend.batch.closed


def test_beam_pruning_exhausts_candidates_without_resurrecting_killed_paths():
    beam = BeamSearch(runtime(), width=2)
    seen = set()
    while beam.active or beam.finished:
        victim = beam.ordered_paths()[0]
        tokens = beam._path_token_ids(victim)
        assert tokens not in seen
        seen.add(tokens)
        live = beam.kill(victim.label)
        assert seen.isdisjoint({beam._path_token_ids(path) for path in (*beam.active, *beam.finished)})
        if not live:
            break
    assert len(seen) == 8
    assert beam.closed


def test_batched_cfg_backfill_matches_serial_with_protected_lineage():
    def guided(primary, guidance):
        return EpisodeEngine(primary, guidance_backend=guidance, initial_token_ids=[7],
            sampling=SamplerConfig(cfg_unconditional_prompt=' A', cfg_scale=1.5))
    serial = BeamSearch(guided(ConformingFakeBackend(), ConformingFakeBackend()), width=3)
    batched = BeamSearch(guided(BatchBackend(), BatchBackend()), width=3)
    for beam in (serial, batched):
        beam.toggle_protection(beam.active[-1].label)
        beam.kill(beam.active[0].label)
        beam.advance(2)
        beam.kill(beam.active[0].label)
    assert [(serial._path_token_ids(p), p.score) for p in serial.active] == [
        (batched._path_token_ids(p), p.score) for p in batched.active]
    assert all(path.lane_id is not None for path in batched.active)
    serial.discard()
    batched.discard()


def test_beam_backfill_reuses_parent_logits(monkeypatch):
    backend = ConformingFakeBackend()
    beam = BeamSearch(runtime(backend), width=2)
    def forbidden():
        pytest.fail('backfill rescored parent logits instead of reusing observations')
    monkeypatch.setattr(backend, 'last_logits', forbidden)
    assert beam.kill(beam.active[0].label)
    assert len(beam.active) == 2
    beam.discard()


@pytest.mark.parametrize('view,expected', [('none', set()), ('raw', {'logit'}), ('gap', {'diff'}), ('both', {'logit', 'diff'})])
def test_launch_overlay_flags_normalize_once(view, expected):
    from trajectory_editor.episode_policy_setup import _preferences
    args = build_parser(include_vector=False).parse_args(['--logit-view', view, '--show-model-probabilities'])
    preferences = _preferences(args)
    assert preferences.overlays == frozenset(expected | {'probability'})
    preferences.set_overlays(frozenset())
    assert _preferences(args) is preferences
    assert _preferences(args).overlays == frozenset()


@pytest.mark.parametrize('option', ['top-k', 'eligible-k', 'selective-noise-k', 'gumbel-top-k'])
def test_controller_profiles_accept_explicit_none_limit(tmp_path, option):
    parser = build_parser(include_vector=False)
    path = tmp_path / 'profile.yaml'
    path.write_text(f'{option}: none\ndraw-kernel: gumbel-max\n')
    values, _ = load_controller_profile(path, parser)
    tokens, _ = profile_arguments(parser, values)
    args = parser.parse_args(tokens)
    dest = {'eligible-k': 'top_k'}.get(option, option.replace('-', '_'))
    assert getattr(args, dest) is None


def test_beam_menu_backfill_preserves_selected_row_and_unselected_highlight():
    from trajectory_editor.beam import beam_menu
    from trajectory_editor.terminal_contracts import BeamInput
    beam = BeamSearch(runtime(), width=3)
    class Terminal:
        def __init__(self):
            self.states = []
        def read_beam(self, state):
            self.states.append(state)
            if len(self.states) == 1:
                return BeamInput('kill ' + state.rows[1].label, state.rows[1].label)
            return BeamInput('return', state.selected_label)
    terminal = Terminal()
    beam_menu(terminal, beam)
    before, after = terminal.states
    assert len(before.rows) == len(after.rows)
    assert before.rows[1].label not in {row.label for row in after.rows}
    assert after.selected_label == after.rows[1].label
    selected = beam.active[-1].label
    beam.set_selection(selected)
    assert beam.kill(beam.active[0].label)
    assert beam.selected_label == selected
    beam.discard()


def test_beam_rewind_checkpoints_do_not_retain_dense_refill_observations():
    beam = BeamSearch(runtime(), width=2)
    beam.advance(3)
    assert all(observation is not None for _, observation in beam._expansion.parents)
    assert all(observation is None for checkpoint in beam._history
               if checkpoint.expansion is not None
               for _, observation in checkpoint.expansion.parents)
    beam.discard()
    assert beam._expansion is None
