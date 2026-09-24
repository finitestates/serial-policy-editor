"""Runtime budgets for interactive operations.

Every operation runs inside `guard.interactive(...)`. Banned calls raise
`Tripwire` (a BaseException) and are recorded; the fixture fails the test if
anything was recorded, even if production code swallowed it.
"""

from __future__ import annotations

import threading

import pytest

from trajectory_editor.chord import Chord
from trajectory_editor.core.actions import Accept, SelectRawRank
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import _ContextRenderCursor

from .tripwire import CountingBackend, Guard


@pytest.fixture
def guard():
    g = Guard()
    yield g
    assert not g.violations, "banned calls on interactive paths:\n" + "\n".join(g.violations)


def _engine(guard):
    backend = CountingBackend(guard)
    engine = EpisodeEngine(backend, initial_text="P", sampling=SamplerConfig(temperature=0.0))
    return engine, backend


def _chord(guard, rounds=12):
    engine, backend = _engine(guard)
    engine.observe()
    with guard.interactive("chord open"):
        chord = Chord(engine, (1, 2, 3))
    with guard.interactive("chord advance"):
        for _ in range(rounds):
            chord.advance()
    return engine, backend, chord


def test_observe_budget(guard):
    engine, backend = _engine(guard)
    with guard.interactive("first observe"):
        mark = backend.mark()
        engine.observe()
        cost = backend.since(mark)
    assert cost["model_calls"] == 0 and cost["calls"]["last_logits"] == 1, cost
    with guard.interactive("repeat observe"):
        mark = backend.mark()
        engine.observe()
        assert backend.since(mark)["calls"] == {}, "observe() of an unchanged decision must be free"


def test_chord_scenarios_never_full_prefill(guard):
    engine, backend, chord = _chord(guard)
    with guard.interactive("chord mixed"):
        mark = backend.mark()
        chord.rewind(); chord.advance(); chord.rewind(); chord.rewind()
        path_b = list(chord.paths[1].token_ids)
        chord.promote("b")
        cost = backend.since(mark)
    assert cost["full_prefills"] == 0, cost
    # Cheap must also be correct: path b equals a plain run of rank 2 + Accepts.
    reference, _ = _engine(Guard())
    reference.observe()
    reference.apply(SelectRawRank(2))
    for _ in range(len(path_b) - 1):
        reference.observe()
        reference.apply(Accept())
    assert path_b == reference.visible_token_ids, (path_b, reference.visible_token_ids)


def test_chord_rewind_reads_nothing_and_does_at_most_one_model_call(guard):
    engine, backend, chord = _chord(guard)
    with guard.interactive("chord rewind"):
        mark = backend.mark()
        chord.rewind()
        cost = backend.since(mark)
    assert cost["full_prefills"] == 0, cost
    assert cost["model_calls"] <= 1, cost
    assert cost["calls"]["last_logits"] == 0, f"rewind must not read logits: {cost}"


def test_chord_advance_round_budget(guard):
    engine, backend, chord = _chord(guard, rounds=20)
    suffixes = [len(p.token_ids) for p in chord.paths if p.state == "live"]
    with guard.interactive("chord advance round"):
        mark = backend.mark()
        chord.advance()
        cost = backend.since(mark)
    # Phase 1 bound: replay each suffix once, plus one position per Accept.
    # No refresh evals, no prefills, one logits read per live path.
    assert cost["full_prefills"] == 0, cost
    assert cost["calls"]["branch_refresh"] == 0, cost
    assert cost["positions"] <= sum(s + 1 for s in suffixes), cost
    assert cost["calls"]["last_logits"] <= len(suffixes), cost


def test_speculation_hit_and_miss_budgets(guard):
    engine, backend = _engine(guard)
    observation = engine.observe()
    with guard.interactive("warm"):
        mark = backend.mark()
        engine.speculate_accept(observation, raw_rank=2)
        warm = backend.since(mark)
    assert warm["positions"] == 1, f"a warm is exactly one position: {warm}"
    with guard.interactive("hit"):
        mark = backend.mark()
        engine.apply(SelectRawRank(2))
        hit = backend.since(mark)
    # Zero, not "at most one": a declined or discarded warm would cost 1 here,
    # so this also proves speculation actually happened.
    assert hit["positions"] == 0 and hit["full_prefills"] == 0, hit

    observation = engine.observe()
    with guard.interactive("warm then miss"):
        engine.speculate_accept(observation, raw_rank=2)
        mark = backend.mark()
        engine.apply(SelectRawRank(3))
        miss = backend.since(mark)
    assert miss["positions"] == 1 and miss["full_prefills"] == 0, miss

    # Speculation must be invisible: same tokens and same next decision.
    reference, _ = _engine(Guard())
    reference.observe(); reference.apply(SelectRawRank(2))
    reference.observe(); reference.apply(SelectRawRank(3))
    assert engine.visible_token_ids == reference.visible_token_ids
    assert (engine.observe().logits == reference.observe().logits).all()


def test_review_navigation_budget(guard):
    engine, backend = _engine(guard)
    cursor = _ContextRenderCursor()
    threads_before = threading.active_count()
    with guard.interactive("live decisions"):
        for _ in range(200):
            cursor.update(engine, engine.observe())
            engine.apply(Accept())
        cursor.update(engine, engine.observe())
    for boundary in range(199, 150, -1):
        with guard.interactive(f"review step to {boundary}"):
            mark = backend.mark()
            cursor.prewarm(engine, boundary)
            engine.rewind_to(boundary)
            cursor.update(engine, engine.observe())
            cost = backend.since(mark)
        assert cost["calls"]["stream.append"] <= 8, f"step to {boundary}: {cost}"
        assert cost["calls"]["render"] == 0, f"full re-render instead of a cursor step: {cost}"
        assert cost["full_prefills"] == 0, cost
    assert threading.active_count() == threads_before, "review started background threads"
