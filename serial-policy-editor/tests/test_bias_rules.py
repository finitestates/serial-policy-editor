import json

import pytest

from trajectory_editor.bias_catalog import BiasCatalog, CatalogEntry, CompiledRoute
from trajectory_editor.bias_presets import FORMAT, load_bias_preset, project_biases
from trajectory_editor.bias_rules import BiasGroup, BiasMatcher, BiasRule, routes_for_catalog_entry
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_lifecycle import _create_episode
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_policy import EdgeRequested
from trajectory_editor.episode_ui import InteractivePolicy
from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend


def test_path_rule_telescopes_from_head_to_continuation():
    rule = BiasRule(routes=((10, 11, 12),), bias=1.5, mode="path")
    matcher = BiasMatcher((rule,))

    assert matcher.active_biases([]) == {10: 1.5}
    assert matcher.active_biases([4, 10]) == {11: 1.5}
    assert matcher.active_biases([4, 10, 11]) == {12: 1.5}
    # Once a route has completed, the term can begin again at a later edge.
    assert matcher.active_biases([10, 11, 12]) == {10: 1.5}


def test_alternate_routes_share_one_logical_bias_amount():
    matcher = BiasMatcher((BiasRule(
        routes=((10, 11), (10, 12), (20, 21)), bias=2, mode="path"),))

    assert matcher.active_biases([]) == {10: 2, 20: 2}
    # A fresh route may also begin at the next position; the active
    # continuation edges are added without multiplying the shared rule.
    assert matcher.active_biases([10]) == {11: 2, 12: 2, 20: 2}


def test_tail_rule_only_biases_completion():
    matcher = BiasMatcher((BiasRule(routes=((10, 11),), bias=-2, mode="tail"),))

    assert matcher.active_biases([]) == {}
    assert matcher.active_biases([10]) == {11: -2}
    assert matcher.active_biases([10, 99]) == {}


def test_scoped_path_rule_telescopes_inside_trigger_scope():
    matcher = BiasMatcher((BiasRule(
        routes=((10, 11),), bias=1, mode="path",
        triggers=((7,),), until="sentence"),))

    boundaries = lambda token: {"sentence"} if token == 9 else set()
    assert matcher.active_biases([7, 10], boundaries) == {11: 1}
    assert matcher.active_biases([7], boundaries) == {10: 1}
    assert matcher.active_biases([7, 9], boundaries) == {}


def test_catalog_routes_group_by_mode_and_deduplicate():
    entry = CatalogEntry(
        name="velociraptor",
        kind="term",
        routes=(
            CompiledRoute((1, 2), (" velociraptor",), (" velo", "ciraptor"), "path", ("canonical",)),
            CompiledRoute((1, 2), ("velociraptor",), (" velo", "ciraptor"), "path", ("alternate",)),
            CompiledRoute((3, 4), (" velociraptor",), (" velo", "ciraptor"), "tail", ("alternate",)),
        ),
    )
    rules = routes_for_catalog_entry(entry, 0.75)

    assert len(rules) == 2
    assert rules[0].mode == "path"
    assert rules[0].routes == ((1, 2),)
    assert rules[1].mode == "tail"


def test_sampling_config_round_trips_logical_rules():
    config = SamplingConfig(bias_rules=(BiasRule(
        routes=((1, 2), (1, 3)), bias=-1.25, mode="path"),))
    restored = SamplingConfig.from_record(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert config.active_biases([]) == {1: -1.25}
    assert config.active_biases([1]) == {2: -1.25, 3: -1.25}


def test_sampling_config_round_trips_named_groups_and_flattens_effective_rules():
    config = SamplingConfig(bias_groups=(BiasGroup(
        name="nautical",
        members=("anchor", "steamship"),
        rules=(
            BiasRule(routes=((14,),), bias=0, mode="path"),
            BiasRule(routes=((15, 16),), bias=0, mode="tail"),
        ),
        bias=1.5,
    ),))
    restored = SamplingConfig.from_record(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert config.active_biases([]) == {14: 1.5}
    assert config.active_biases([15]) == {14: 1.5, 16: 1.5}
    assert len(config.effective_bias_rules) == 2


def test_logical_preset_round_trips_rules(tmp_path):
    path = tmp_path / "logical.json"
    path.write_text(json.dumps({
        "format": FORMAT,
        "model": {"vocabulary_size": 8},
        "bias_rules": [{
            "routes": [[1, 2], [1, 3]], "mode": "path", "bias": 0.75,
        }],
    }))

    config = load_bias_preset(path, NoEogBackend(), {"vocabulary_size": 8})
    assert config.bias_rules[0].routes == ((1, 2), (1, 3))
    assert config.active_biases([1]) == {2: 0.75, 3: 0.75}


def test_projected_logical_rules_use_v2_preset(tmp_path):
    backend = NoEogBackend()
    config = SamplingConfig(bias_rules=(BiasRule(
        routes=((1, 2),), bias=1.25, mode="path"),))
    runtime = EpisodeEngine(backend, initial_text="P", sampling=config)
    with EpisodeStore(tmp_path / "episodes.db") as store:
        episode = _create_episode(
            store, runtime, backend_provenance=backend.provenance()
        )
        exported = json.loads(project_biases(store, episode))

    assert exported["format"] == FORMAT
    assert exported["bias_rules"][0]["routes"] == [[1, 2]]


def test_projected_named_groups_can_be_exported_full_or_flattened(tmp_path):
    backend = NoEogBackend()
    config = SamplingConfig(bias_groups=(BiasGroup(
        name="nautical",
        members=("anchor",),
        rules=(BiasRule(routes=((1, 2),), bias=0, mode="path"),),
        bias=2.0,
    ),))
    runtime = EpisodeEngine(backend, initial_text="P", sampling=config)
    with EpisodeStore(tmp_path / "episodes.db") as store:
        episode = _create_episode(
            store, runtime, backend_provenance=backend.provenance()
        )
        full = json.loads(project_biases(store, episode))
        flat = json.loads(project_biases(store, episode, rules_only=True))

    assert full["bias_groups"][0]["name"] == "nautical"
    assert full["bias_groups"][0]["bias"] == 2.0
    assert "bias_groups" not in flat
    assert flat["bias_rules"] == [{
        "routes": [[1, 2]], "mode": "path", "bias": 2.0,
    }]


class CatalogBackend(NoEogBackend):
    pieces = {
        **NoEogBackend.pieces,
        8: " velo",
        9: "ciraptor",
    }

    def tokenize(self, text, *, add_bos=False, special=False):
        if not add_bos and text in {" velociraptor", "velociraptor", " unknownword"}:
            return [8, 9]
        return super().tokenize(text, add_bos=add_bos, special=special)


def _catalog(*routes):
    return BiasCatalog(
        model={},
        compiler={},
        entries={
            "velociraptor": CatalogEntry(
                name="velociraptor",
                kind="term",
                routes=tuple(routes),
                mode="path",
            )
        },
    )


def test_policy_prefers_catalog_for_bare_names_and_falls_back_for_unknown_names():
    backend = CatalogBackend()
    catalog = _catalog(
        CompiledRoute((1, 2), ("velociraptor",), (" A", " B"), "path", ("canonical",)),
        CompiledRoute((1, 3), ("velociraptor",), (" A", " C"), "path", ("alternate",)),
    )
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplingConfig())
    with pytest.raises(EdgeRequested):
        InteractivePolicy(
            io=ScriptedIO(["b velociraptor +2", "q"]), catalog=catalog
        ).choose(runtime, runtime.observe())

    assert runtime.sampling.bias_rules[0].routes == ((1, 2), (1, 3))
    assert runtime.sampling.active_biases([]) == {1: 2}
    assert runtime.sampling.active_biases([1]) == {2: 2, 3: 2}

    fallback = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplingConfig())
    with pytest.raises(EdgeRequested):
        InteractivePolicy(
            io=ScriptedIO(["b unknownword +2", "q"]), catalog=catalog
        ).choose(fallback, fallback.observe())
    assert fallback.sampling.bias_rules[0].routes == ((8, 9),)


def test_at_reference_requires_catalog_entry_and_quotes_bypass_resolution():
    backend = CatalogBackend()
    catalog = _catalog(
        CompiledRoute((1, 2), ("velociraptor",), (" A", " B"), "path", ("canonical",)),
    )
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplingConfig())
    with pytest.raises(EdgeRequested):
        InteractivePolicy(
            io=ScriptedIO(["b @missing +", "q"]), catalog=catalog
        ).choose(runtime, runtime.observe())
    assert runtime.sampling.bias_rules == ()

    quoted = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplingConfig())
    with pytest.raises(EdgeRequested):
        InteractivePolicy(
            io=ScriptedIO(['b "velociraptor" +2', "q"]), catalog=catalog
        ).choose(quoted, quoted.observe())
    assert quoted.sampling.bias_rules[0].routes == ((8, 9),)


def test_runtime_groups_are_append_only_and_share_one_bias():
    backend = CatalogBackend()
    catalog = BiasCatalog(
        model={},
        compiler={},
        entries={
            "anchor": CatalogEntry(
                name="anchor", kind="term",
                routes=(CompiledRoute((1,), ("anchor",), ("anchor",), "path", ("canonical",)),),
            ),
            "steamship": CatalogEntry(
                name="steamship", kind="term",
                routes=(CompiledRoute((2, 3), (" steamship",), (" steam", "ship"), "tail", ("canonical",)),),
            ),
        },
    )
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplingConfig())
    with pytest.raises(EdgeRequested):
        InteractivePolicy(
            io=ScriptedIO([
                "b nautical -> {anchor}",
                "b nautical +2",
                "b nautical -> {steamship}",
                "q",
            ]),
            catalog=catalog,
        ).choose(runtime, runtime.observe())

    group = runtime.sampling.bias_groups[0]
    assert group.name == "nautical"
    assert group.members == ("anchor", "steamship")
    assert group.bias == 2.0
    assert runtime.sampling.active_biases([]) == {1: 2.0}
    assert runtime.sampling.active_biases([2]) == {1: 2.0, 3: 2.0}


@pytest.mark.parametrize("value", [
    {"routes": [[1, 2]], "bias": 1, "mode": "unknown"},
    {"routes": [[1, 2]], "bias": float("nan")},
    {"routes": [[1, 2]], "bias": 1, "triggers": [[3]]},
    {"routes": [[1, 2]], "bias": 1, "until": "."},
])
def test_invalid_logical_rules(value):
    with pytest.raises(EditorError):
        BiasRule.from_record(value)
