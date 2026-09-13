import math

import pytest

from trajectory_editor.domain import SamplingConfig
from trajectory_editor.sampling import ReferencePriorTrie, reference_prior_snapshot


def _routes(*items):
    return tuple(((tuple(route)), float(weight)) for route, weight in items)


def test_global_reference_state_drops_unrelated_root_branches_after_entry():
    routes = _routes(((1, 2), 10), ((1, 3), 1), ((4, 5), 100))

    root = reference_prior_snapshot(
        routes, [], strength=1.0, attraction=0.0, scope="global"
    )
    inside = reference_prior_snapshot(
        routes, [1], strength=1.0, attraction=0.0, scope="global"
    )

    assert {row[0] for row in root.outgoing} == {1, 4}
    assert {row[0] for row in inside.outgoing} == {2, 3}
    assert 4 not in inside.biases


def test_stateful_reference_branches_preserve_relative_weight_preference():
    snapshot = reference_prior_snapshot(
        _routes(((1, 2), 10), ((1, 3), 1)),
        [1],
        strength=1.0,
        attraction=0.0,
    )

    assert snapshot.biases[2] > snapshot.biases[3]
    assert snapshot.biases[2] - snapshot.biases[3] == pytest.approx(
        math.log(10.0)
    )


def test_singleton_continuation_gets_attraction_but_not_branch_bias():
    routes = _routes(((1, 2), 10))
    contrastive = reference_prior_snapshot(
        routes, [1], strength=1.0, attraction=0.0
    )
    attracted = reference_prior_snapshot(
        routes, [1], strength=1.0, attraction=0.5
    )

    assert contrastive.biases == {2: pytest.approx(0.0)}
    assert attracted.biases[2] > 0.0
    assert attracted.outgoing[0][2] == pytest.approx(0.0)
    assert attracted.outgoing[0][3] == pytest.approx(0.5)


def test_ballistic_global_applies_attraction_at_the_root():
    routes = _routes(((1,), 100))
    contrastive = reference_prior_snapshot(
        routes, [], strength=1.0, attraction=0.5, scope="global"
    )
    ballistic = reference_prior_snapshot(
        routes, [], strength=1.0, attraction=0.5, scope="ballistic-global"
    )

    assert contrastive.biases == {1: pytest.approx(0.0)}
    assert ballistic.biases == {1: pytest.approx(0.5)}


def test_sampling_config_preserves_ballistic_global_scope():
    config = SamplingConfig(
        reference_prior_routes=(((1,), 100.0),),
        reference_prior_scope="ballistic-global",
        reference_prior_strength=1.0,
        reference_prior_attraction=0.5,
    )

    snapshot = config.active_reference_prior_snapshot([])

    assert snapshot.scope == "ballistic-global"
    assert snapshot.biases == {1: pytest.approx(0.5)}


def test_terminal_mass_dampens_longer_continuation_attraction():
    snapshot = reference_prior_snapshot(
        _routes(((1, 2), 100), ((1, 2, 3), 1)),
        [1, 2],
        strength=1.0,
        attraction=1.0,
    )

    assert snapshot.terminal_mass == pytest.approx(100.0)
    assert snapshot.outgoing[0][3] == pytest.approx(1.0 / 101.0)
    assert snapshot.biases[3] == pytest.approx(1.0 / 101.0)


def test_reference_weight_scaling_is_invariant():
    first = reference_prior_snapshot(
        _routes(((1, 2), 10), ((1, 3), 1)),
        [1], strength=1.0, attraction=0.25,
    )
    scaled = reference_prior_snapshot(
        _routes(((1, 2), 1000), ((1, 3), 100)),
        [1], strength=1.0, attraction=0.25,
    )

    assert scaled.biases == pytest.approx(first.biases)
    assert scaled.state_mass == pytest.approx(first.state_mass * 100)


def test_failure_transitions_preserve_overlapping_suffix_prefixes():
    trie = ReferencePriorTrie(_routes(((1, 2), 10), ((2, 3), 5)))

    suffix = reference_prior_snapshot(
        _routes(((1, 2), 10), ((2, 3), 5)),
        [1, 2], strength=1.0, attraction=0.0,
    )
    reset = reference_prior_snapshot(
        _routes(((1, 2), 10), ((2, 3), 5)),
        [1, 99, 2], strength=1.0, attraction=0.0,
    )

    assert trie.nodes[trie.state_for_history([1, 2])].prefix == (2,)
    assert {row[0] for row in suffix.outgoing} == {3}
    assert {row[0] for row in reset.outgoing} == {3}


def test_terminal_prefix_keeps_terminal_and_continuation_mass_distinct():
    snapshot = reference_prior_snapshot(
        _routes(((1, 2), 10), ((1, 2, 3), 5)),
        [1, 2], strength=1.0, attraction=0.0,
    )

    assert snapshot.state_prefix == (1, 2)
    assert snapshot.terminal_mass == pytest.approx(10.0)
    assert snapshot.outgoing == ((3, 5.0, 0.0, 0.0, 0.0),)


def test_reference_state_is_reconstructed_deterministically():
    routes = _routes(((1, 2), 10), ((2, 3), 5))
    histories = ((), (1,), (1, 2), (1, 99, 2), (2, 3, 1))

    first = [reference_prior_snapshot(
        routes, history, strength=0.75, attraction=0.2
    ) for history in histories]
    second = [reference_prior_snapshot(
        routes, history, strength=0.75, attraction=0.2
    ) for history in histories]

    assert first == second
