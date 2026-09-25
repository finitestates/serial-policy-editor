"""Algebraic contracts for the small kernel; no production runtime is imported."""

from dataclasses import replace
import json
import secrets

import pytest

from trajectory_editor.reference_kernel import (
    Accept, Branch, Distribution, End, Hold, Policy, Reroll, SelectToken,
    SetPolicy, State, Tape, World, WriteTokens, apply, draw, fork, observe,
    position_uniform, position_uniform_token, record, replay, reroll, rewind,
    token_prefix_sha256,
)


class ScriptedBackend:
    """An explicit pure function of the prefix, with a ten-token vocabulary."""

    def logits(self, prefix_token_ids):
        last, length, total = prefix_token_ids[-1], len(prefix_token_ids), sum(prefix_token_ids)
        return tuple(
            -20.0 if i == 0 else ((last * 3 + i * 7 + length * 5 + total) % 17) / 4.0
            for i in range(10)
        )

    def is_eog(self, token_id):
        return False


def start(seed=12345, *, policy=None, offset=0):
    prefix = (1, 2, 3)
    return Branch(State(prefix), policy or Policy(), World.for_prefix(seed, prefix, offset))


@pytest.mark.parametrize("kernel", ["categorical", "gumbel-max"])
def test_observe_is_pure_and_directly_addressed(kernel):
    backend = ScriptedBackend()
    branch = start(policy=Policy(draw_kernel=kernel), offset=41)
    first = observe(backend, branch)
    for _ in range(1000):
        assert observe(backend, branch) == first
    assert first.sampling_coordinate == 41
    later = position_uniform(branch.world, 500)
    for coordinate in range(500):
        position_uniform(branch.world, coordinate)
    assert position_uniform(branch.world, 500) == later
    advanced, _ = apply(backend, branch, Accept())
    assert observe(backend, advanced).sampling_coordinate == 42


@pytest.mark.parametrize("kernel", ["categorical", "gumbel-max"])
@pytest.mark.parametrize("seed", [-11, 1, 12345, 67890])
def test_immutable_continuation_rewind_and_fork(seed, kernel):
    backend = ScriptedBackend()
    original = start(seed, policy=Policy(top_k=6, draw_kernel=kernel), offset=13)
    first_observation = observe(backend, original)
    continuation, first_result = apply(backend, original, Hold(20))
    restored = rewind(continuation, 0)
    assert observe(backend, restored) == first_observation
    repeated, second_result = apply(backend, restored, Hold(20))
    forked, fork_result = apply(backend, fork(original), Hold(20))
    assert (repeated, second_result) == (continuation, first_result)
    assert (forked, fork_result) == (continuation, first_result)
    assert first_result.stop_reason == "requested-length"
    assert rewind(continuation, 7).world == original.world
    assert observe(backend, rewind(continuation, 7)).sampling_coordinate == 20


def test_interventions_and_conditional_hold_keep_the_same_world():
    backend = ScriptedBackend()
    original = start()
    natural, result = apply(backend, fork(original), Hold(12))
    first_token = result.visible_token_ids[0]
    alternative = next(i for i in range(1, 10) if i != first_token)
    altered, _ = apply(backend, fork(original), SelectToken(alternative))
    altered, _ = apply(backend, altered, Hold(11))
    assert altered.state.token_ids != natural.state.token_ids
    assert altered.world == natural.world
    for coordinate in range(12):
        assert position_uniform(altered.world, coordinate) == position_uniform(natural.world, coordinate)

    # A conditional hold has a fixed stopping token and boundary as well.
    stop = result.visible_token_ids[3]
    conditional, stopped = apply(backend, original, Hold(12, (stop,)))
    assert stopped.stop_reason == "stop-token"
    assert stopped.visible_token_ids == result.visible_token_ids[:conditional.state.boundary]
    assert apply(backend, rewind(conditional, 0), Hold(12, (stop,))) == (conditional, stopped)


def test_policy_truncation_remaps_one_fixed_categorical_quantile():
    class Flat:
        def logits(self, prefix):
            return (0.0, 0.0, 0.0)

        def is_eog(self, token_id):
            return False

    backend = Flat()
    branch = start(policy=Policy(top_k=2))
    u = position_uniform(branch.world, 0)
    assert 1 / 3 < u < 1 / 2
    a = observe(backend, branch)
    branch, _ = apply(backend, branch, SetPolicy(Policy(top_k=3)))
    b = observe(backend, branch)
    branch, _ = apply(backend, branch, SetPolicy(Policy(top_k=2)))
    assert (a.proposal_token_id, b.proposal_token_id, observe(backend, branch).proposal_token_id) == (0, 1, 0)
    assert a.sampling_coordinate == b.sampling_coordinate == 0
    assert a.distribution.ids != b.distribution.ids


def test_gumbel_candidate_order_and_exclusions_preserve_noise():
    world = start().world
    original = Distribution((5, 3, 8), (0.4, 0.3, 0.3), (1.0, 0.5, 0.0))
    reordered = Distribution((8, 5, 3), (0.3, 0.4, 0.3), (0.0, 1.0, 0.5))
    for coordinate in range(40):
        assert draw(original, world, coordinate, "gumbel-max") == draw(reordered, world, coordinate, "gumbel-max")

    class Constant:
        def logits(self, prefix):
            return (2.0, 1.0, 0.0)

        def is_eog(self, token_id):
            return False

    backend = Constant()
    branch = start(policy=Policy(draw_kernel="gumbel-max"))
    winner = observe(backend, branch).proposal_token_id
    nonwinner = next(i for i in range(3) if i != winner)
    without_other = replace(branch, policy=replace(branch.policy, excluded_token_ids=(nonwinner,)))
    without_winner = replace(branch, policy=replace(branch.policy, excluded_token_ids=(winner,)))
    assert observe(backend, without_other).proposal_token_id == winner
    assert observe(backend, without_winner).proposal_token_id != winner
    assert observe(backend, branch).proposal_token_id == winner
    for token_id in range(3):
        assert position_uniform_token(branch.world, 0, token_id) == position_uniform_token(without_winner.world, 0, token_id)
    assert dict(zip(observe(backend, branch).distribution.ids,
                    observe(backend, branch).distribution.scores))[nonwinner] == \
           dict(zip(observe(backend, without_winner).distribution.ids,
                    observe(backend, without_winner).distribution.scores))[nonwinner]


def test_reroll_round_trip_and_seed_is_replay_data(monkeypatch):
    backend = ScriptedBackend()
    original = start()
    before, first = apply(backend, original, Hold(10))
    another_world = reroll(original, 67890)
    another, second = apply(backend, another_world, Hold(10))
    restored, third = apply(backend, reroll(another_world, original.world.seed), Hold(10))
    assert (restored, third) == (before, first)
    assert another.state.token_ids != before.state.token_ids
    assert another_world.policy == original.policy and another_world.state == original.state
    assert another_world.world.coordinate_offset == original.world.coordinate_offset

    events = (Hold(2), Reroll(67890), Hold(3), SetPolicy(Policy(top_k=4,
              biases=((7, 0.5),), draw_kernel="gumbel-max")),
              SelectToken(6), WriteTokens((2, 3)), Hold(2), End())
    tape = record(original, events)
    assert tape.events[1].seed == 67890
    restored_tape = Tape.from_dict(json.loads(json.dumps(tape.to_dict())))
    assert restored_tape == tape
    monkeypatch.setattr(secrets, "randbits", lambda bits: pytest.fail("replay requested entropy"))
    assert replay(backend, restored_tape) == replay(backend, tape)
    terminal, results = replay(backend, tape)
    assert terminal.state.ended and results[-1].stop_reason == "end"
    assert terminal.world.seed == 67890
    assert terminal.policy == events[3].policy


def test_entropy_is_resolved_before_a_reroll_enters_the_tape(monkeypatch):
    monkeypatch.setattr(secrets, "randbits", lambda bits: 987654321)
    selected_seed = secrets.randbits(63)
    tape = record(start(), (Reroll(selected_seed), Hold(5)))
    assert tape.to_dict()["events"][0] == {"kind": "Reroll", "seed": 987654321}
    restored = Tape.from_dict(json.loads(json.dumps(tape.to_dict())))
    monkeypatch.setattr(secrets, "randbits", lambda bits: pytest.fail("replay requested entropy"))
    assert replay(ScriptedBackend(), restored) == replay(ScriptedBackend(), tape)


def test_eog_is_terminal_evidence_and_rewind_restores_the_coordinate():
    class Finite(ScriptedBackend):
        def logits(self, prefix):
            if len(prefix) >= 5:
                return (100.0,) + (-100.0,) * 9
            return super().logits(prefix)

        def is_eog(self, token_id):
            return token_id == 0

    backend = Finite()
    initial = start(policy=Policy(top_k=1))
    ended, result = apply(backend, initial, Hold(20))
    assert ended.state.boundary == 2
    assert ended.state.terminal_token_id == 0
    assert result.stop_reason == "eog"
    assert result.resolved_token_ids == result.visible_token_ids + (0,)
    assert apply(backend, rewind(ended, 0), Hold(20)) == (ended, result)
    assert observe(backend, rewind(ended, 0)) == observe(backend, initial)
    with pytest.raises(ValueError, match="live boundary"):
        observe(backend, ended)


def test_small_counterfactual_example():
    """Hold, rewind, fork+intervene, reroll, restore: no special chord action."""
    backend = ScriptedBackend()
    root = start()
    original, held = apply(backend, root, Hold(5))
    assert held.visible_token_ids == (5, 5, 6, 4, 8)
    assert apply(backend, rewind(original, 0), Hold(5)) == (original, held)
    altered, _ = apply(backend, fork(root), SelectToken(7))
    altered, branch_hold = apply(backend, altered, Hold(4))
    assert altered.world == root.world
    assert (7,) + branch_hold.visible_token_ids != held.visible_token_ids
    rerolled, _ = apply(backend, reroll(root, 67890), Hold(5))
    assert rerolled.state.visible_token_ids != held.visible_token_ids
    assert apply(backend, reroll(root, 12345), Hold(5)) == (original, held)
    # A third independent branch uses a different policy in the same world.
    third, _ = apply(backend, fork(root), SetPolicy(Policy(top_k=1)))
    third, _ = apply(backend, third, Hold(5))
    assert third.world == altered.world == original.world


def test_world_prefix_validation_and_exact_serialization():
    assert World.for_prefix(4, (1, 2)).stream_fingerprint == token_prefix_sha256([1, 2])
    for invalid in (1 << 63, -(1 << 63) - 1, True):
        with pytest.raises(ValueError, match="seed"):
            World.for_prefix(invalid, (1,))
    with pytest.raises(ValueError, match="fingerprint"):
        World(1, "ABC")
    with pytest.raises(ValueError, match="token IDs"):
        token_prefix_sha256((1 << 63,))
