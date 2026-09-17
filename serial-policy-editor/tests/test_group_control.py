from dataclasses import replace
import json

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from tests.test_bias_catalog import CatalogBackend, RouteRankingBackend
from trajectory_editor.bias_catalog import CompileOptions, compile_term, compile_catalog
from trajectory_editor.bias_commands import apply_bias_command
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.bias_rules import BiasGroup, BiasRule, routes_for_catalog_entry
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Hold, SelectRawRank
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_lifecycle import _create_episode
from trajectory_editor.episode_policy import EpisodeRunner, _WriteLearningAccumulator
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.group_control import GroupControl, appearances, control_adjustments
from trajectory_editor.lexical_reference import compile_reference, load_reference
from trajectory_editor.online_learning import OnlineLearningConfig, OnlineLearner
from trajectory_editor.sampling import ObservationStatistics
from trajectory_editor.tui import parse_bias_command


def group(name="concrete", route=(3,), **kwargs):
    return BiasGroup(name, (BiasRule(routes=(route,), bias=0),), **kwargs)


def edit_group(engine, text, catalog=None):
    command = parse_bias_command(text, vocabulary_size=engine.backend.vocabulary_size())
    sampling, _ = apply_bias_command(command, engine.backend, engine.sampling,
                                     engine.observe(), lambda n: None, catalog)
    engine.sampling = sampling
    return sampling


@pytest.mark.parametrize('target', ('concrete', '@concrete'))
def test_learning_toggle_resolves_yaml_group_and_freezes_learned_amount(tmp_path, target):
    backend = ConformingFakeBackend()
    path = tmp_path / 'groups.yaml'
    path.write_text('defaults:\n  cases: [original]\n  plural: false\n  leading_space: false\n'
                    'groups:\n  concrete: [C]\n')
    from trajectory_editor.bias_catalog import load_yaml_source
    catalog = compile_catalog(load_yaml_source(path), backend)
    engine = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplingConfig())
    sampling = edit_group(engine, f'b {target} learn on', catalog)
    assert sampling.bias_groups[0].learnable and sampling.bias_groups[0].enabled
    learner = OnlineLearner(enabled=True, no_severity_attenuation=True)
    result = learner.update(engine.observe(), 3, sampling)
    assert result.group_deltas['concrete'] > 0
    engine.sampling = result.sampling
    amount = result.new_group_weights['concrete']
    frozen = edit_group(engine, 'b concrete learn off')
    assert frozen.bias_groups[0].bias == amount
    assert frozen.bias_groups[0].enabled and not frozen.bias_groups[0].learnable
    assert frozen.bias_groups[0].effective_rules()
    assert learner.update(engine.observe(), 3, frozen).sampling == frozen


def test_learning_choice_survives_membership_and_numeric_edits():
    engine = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplingConfig())
    assert not edit_group(engine, 'b concrete -> {C}').bias_groups[0].learnable
    assert not edit_group(engine, 'b concrete +0.5').bias_groups[0].learnable
    edit_group(engine, 'b concrete learn on')
    assert edit_group(engine, 'b concrete -> {B}').bias_groups[0].learnable
    current = edit_group(engine, 'b concrete +0.5').bias_groups[0]
    assert current.learnable and current.bias == 1
    edit_group(engine, 'b concrete learn off')
    assert not edit_group(engine, 'b concrete -> {A}').bias_groups[0].learnable
    assert not edit_group(engine, 'b concrete +0.5').bias_groups[0].learnable


@pytest.mark.parametrize('suffix', ('', ' after "P" until "!"'))
def test_learning_toggle_rejects_objectives_without_mutating_them(suffix):
    engine = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7],
                           sampling=SamplingConfig(bias_groups=(group(learnable=False),)))
    controlled = edit_group(engine, 'b concrete +' + suffix)
    with pytest.raises(EditorError, match='appearance objective'):
        edit_group(engine, 'b concrete learn on')
    assert engine.sampling == controlled
    edit_group(engine, 'b concrete off' + suffix)
    assert edit_group(engine, 'b concrete learn on').bias_groups[0].learnable
    assert not edit_group(engine, 'b concrete +' + suffix).bias_groups[0].learnable


def test_learning_toggle_does_not_enable_session_or_bypass_group_filter():
    engine = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7],
                           sampling=SamplingConfig(bias_groups=(group(enabled=False, learnable=False),)))
    sampling = edit_group(engine, 'b concrete learn on')
    assert sampling.bias_groups[0].enabled
    for learner in (OnlineLearner(enabled=False), OnlineLearner(enabled=True, learnable_groups=())):
        assert learner.update(engine.observe(), 3, sampling).sampling == sampling
    with pytest.raises(EditorError, match='Unknown group'):
        edit_group(engine, 'b typo learn on')
    assert engine.sampling == sampling


def test_live_learning_toggle_updates_before_selection_and_persists(tmp_path):
    engine = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7],
                           sampling=SamplingConfig(bias_groups=(group(learnable=False),)))
    io = ScriptedIO(['b concrete learn on', 'b', '3'])
    results = []
    with EpisodeStore(tmp_path / 'store.db') as store:
        eid = _create_episode(store, engine, backend_provenance=engine.backend.provenance())
        EpisodeRunner(engine, store, eid, learner=OnlineLearner(enabled=True),
                      on_learning_update=results.append).run(
            live_policy=InteractivePolicy(io=io, store=store, episode_id=eid), max_live_actions=1)
        assert results[0].group_deltas['concrete'] > 0
        path = tmp_path / 'biases.json'
        path.write_text(project_biases(store, eid))
        restored = load_bias_preset(path, engine.backend, engine.backend.provenance())
        assert restored.bias_groups == engine.sampling.bias_groups
    assert any('learnable on' in text for text in io.output)


def test_headless_learning_toggle_survives_fork_and_reopen(tmp_path):
    from tests.test_headless import send
    from tests.test_episode_runtime import NoEogBackend
    from trajectory_editor.headless import Session
    backend = NoEogBackend()
    with EpisodeStore(tmp_path / 'api.db') as store:
        session = Session(backend, store, backend.provenance(), sampling=SamplingConfig())
        send(session, 'open', prompt='P')
        send(session, 'steering', command='b concrete -> {C}')
        send(session, 'steering', command='b concrete learn on')
        send(session, 'actions', action={'kind': 'hold', 'limit': 4})
        send(session, 'fork', boundary=4)
        assert session.engine.sampling.bias_groups[0].learnable
        identifier = session.episode_id
        send(session, 'close')
        send(session, 'open', episode_id=identifier)
        assert session.engine.sampling.bias_groups[0].learnable
        send(session, 'steering', command='b concrete learn off')
        send(session, 'close')
        send(session, 'open', episode_id=identifier)
        assert not session.engine.sampling.bias_groups[0].learnable


def test_control_increases_and_reduces_appearance_during_hold():
    class SteadyBackend(ConformingFakeBackend):
        def last_logits(self):
            return np.asarray([-30., 2., 1., 0., -20., -20., -20., -20.])
    counts = {}
    for direction in (None, "promote", "suppress"):
        controls = () if direction is None else (GroupControl("concrete", direction, .09),)
        runtime = EpisodeEngine(SteadyBackend(), initial_token_ids=[7], sampling=SamplingConfig(
            temperature=1., top_k=8, top_p=1., min_p=0., bias_groups=(group(),), group_controls=controls))
        runtime.apply(Hold(256))
        counts[direction] = runtime.visible_token_ids.count(3)
    assert counts["promote"] > counts[None] > counts["suppress"]


def test_feedback_relaxes_after_appearances_and_never_reverses_direction():
    logits = np.zeros(8)
    g = group()
    for direction, sign in (("promote", 1), ("suppress", -1)):
        c = GroupControl(g.name, direction, .1)
        absent, _ = control_adjustments((c,), (g,), [1] * 100, logits)
        frequent, _ = control_adjustments((c,), (g,), [3] * 100, logits)
        assert absent[3] >= frequent[3]
        assert 0 <= sign * absent[3] <= c.max_bias
        assert 0 <= sign * frequent[3] <= c.max_bias


def test_maintain_can_correct_in_either_direction():
    c = GroupControl("concrete", "maintain", .1)
    low, _ = control_adjustments((c,), (group(),), [1] * 100, np.zeros(8))
    high, _ = control_adjustments((c,), (group(),), [3] * 100, np.zeros(8))
    assert low[3] > 0 > high[3]


def test_monitor_counts_phrases_independently_of_tokenization_and_word_stems():
    b = CatalogBackend()
    g = BiasGroup("nautical", (BiasRule(routes=((17, 18, 19),), bias=0),), surfaces=("port of call",))
    assert appearances(g, [17, 18, 19], b.render) == appearances(g, [21], b.render) == 1
    shadow = group(surfaces=("shadow",))
    assert appearances(shadow, [1, 8], b.render) == 0


def test_scope_does_not_create_unconditional_learning_and_stops():
    g = group(learnable=False)
    s = SamplingConfig(bias_groups=(g,))
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=s)
    cmd = parse_bias_command('b concrete + after "P" until "!"', vocabulary_size=8)
    s, _ = apply_bias_command(cmd, e.backend, s, e.observe(), lambda n: None)
    c = s.group_controls[0]
    assert c.triggers == ((7,),) and c.until == 5
    assert s.bias_groups[0].bias == 0 and not s.bias_rules
    for history, expected in (([1], False), ([7], True), ([7, 1], True), ([7, 5], False)):
        adjustments, diagnostics = control_adjustments(s.group_controls, s.bias_groups, history, np.zeros(8))
        assert diagnostics[0]["active"] is expected
        assert bool(adjustments) is expected
    e.sampling = s
    assert OnlineLearner(enabled=True).update(e.observe(), 3, s).sampling == s


def test_off_is_durable_and_numeric_bias_remains_manual():
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplingConfig(bias_groups=(group(),)))
    for text in ('b concrete +', 'b concrete off'):
        s, _ = apply_bias_command(parse_bias_command(text, vocabulary_size=8), e.backend, e.sampling, e.observe(), lambda n: None)
        e.sampling = s
    assert not e.sampling.group_controls and e.sampling.bias_groups[0].bias == 0
    assert OnlineLearner(enabled=True).update(e.observe(), 3, e.sampling).sampling == e.sampling
    s, _ = apply_bias_command(parse_bias_command('b concrete +2', vocabulary_size=8), e.backend, e.sampling, e.observe(), lambda n: None)
    assert s.bias_groups[0].bias == 2 and not s.group_controls


def test_same_history_and_saved_configuration_reconstruct_same_control():
    sampling = SamplingConfig(bias_groups=(group(),), group_controls=(GroupControl("concrete", "promote", .1, history_start=1),))
    restored = SamplingConfig.from_record(json.loads(json.dumps(sampling.to_dict())))
    for history in ([7], [7, 3], [7, 1, 2, 3] * 10):
        a = ObservationStatistics(np.zeros(8), sampling, history)
        b = ObservationStatistics(np.zeros(8), restored, history)
        np.testing.assert_array_equal(a.policy_probabilities, b.policy_probabilities)
        assert a.group_control_diagnostics == b.group_control_diagnostics


def test_compiler_runtime_uses_canonical_routes_even_when_exploring():
    b = RouteRankingBackend()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('trajectory_editor.bias_catalog._enumerate_routes', lambda *a, **k: pytest.fail("normal compilation must not enumerate"))
        ordinary = compile_term("another", "another", b, options=CompileOptions(cases=("original",), plural=False, leading_space=False))
    assert [r.token_ids for r in ordinary.routes] == [(1, 2)]
    exploration = compile_term("another", "another", b, options=CompileOptions(route_policy="all", cases=("original",), plural=False, leading_space=False))
    assert {r.token_ids for r in exploration.routes} == {(1, 2), (3,)}
    assert {route for rule in routes_for_catalog_entry(exploration, 1) for route in rule.routes} == {(1, 2)}


def test_reference_works_without_groups_and_is_scale_invariant(tmp_path):
    b = CatalogBackend()
    path = tmp_path / 'reference.yaml'
    path.write_text('shadow: 100\nshadowing: 1\n')
    routes = load_reference(path, b)
    assert routes == compile_reference({'shadow': 10, 'shadowing': .1}, b)
    config = SamplingConfig(reference_prior_routes=routes, reference_prior_scope="global", reference_prior_mode="lexical")
    plain = ObservationStatistics(np.zeros(b.vocabulary_size()), SamplingConfig(), [])
    weighted = ObservationStatistics(np.zeros(b.vocabulary_size()), config, [])
    assert weighted.policy_probabilities[1] > plain.policy_probabilities[1]
    assert weighted.policy_probabilities[6] < plain.policy_probabilities[6]
    assert weighted.policy_probabilities[14] > 0
    assert max(abs(v) for v in weighted.reference_prior_biases.values()) <= 2
    for amount in (-1, 0, 1):
        varied = replace(config, bias_groups=(BiasGroup("shadow", (BiasRule(routes=((1,),), bias=0),), bias=amount),))
        assert varied.active_reference_prior([]) == config.active_reference_prior([])


def test_complete_steering_preset_roundtrip_and_portable_history(tmp_path):
    b = ConformingFakeBackend()
    state = SamplingConfig(bias_groups=(group(),), group_controls=(GroupControl("concrete", "maintain", .1, history_start=17),),
                           reference_prior_routes=(((3,), 1.), ((2,), .5)), reference_prior_scope="global", reference_prior_mode="lexical",
                           token_preference_vector=(.1, .2), token_preference_fast_vector=(.2, .1), token_preference_fast_strength=.4)
    # Persist a complete sampler without requiring model embeddings for this serialization test.
    with EpisodeStore(tmp_path / 'store.db') as store:
        eid = store.create_episode(initial_text='P', initial_token_ids=[7], sampling=state, stream_fingerprint='0' * 64,
                                   coordinate_offset=0, max_tokens=None, backend=b.provenance())
        path = tmp_path / 'biases.json'
        path.write_text(project_biases(store, eid))
        restored = load_bias_preset(path, b, b.provenance())
    assert restored == replace(state, group_controls=(replace(state.group_controls[0], history_start=None),))
    restored = replace(restored, token_preference_vector=(), token_preference_fast_vector=())
    e = EpisodeEngine(b, initial_token_ids=[7, 1], sampling=restored)
    assert e.sampling.group_controls[0].history_start == 2


def test_bias_edit_selection_uses_authoritative_observation(tmp_path):
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplingConfig(bias_groups=(group(),)))
    with EpisodeStore(tmp_path / 'store.db') as store:
        eid = _create_episode(store, e, backend_provenance=e.backend.provenance())
        results = []
        EpisodeRunner(e, store, eid, learner=OnlineLearner(enabled=True), on_learning_update=results.append).run(
            live_policy=InteractivePolicy(io=ScriptedIO(['b concrete +4', '3'])), max_live_actions=1)
        assert results[0].old_policy_rank == 1 and results[0].severity == 1


def test_outside_bounds_and_excluded_groups_do_not_change():
    s = SamplingConfig(bias_groups=(group("fixed", (4, 3), bias=8), group()))
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=s)
    learner = OnlineLearner(enabled=True, learnable_groups=("concrete",))
    accumulator = _WriteLearningAccumulator(e.backend, s, learner, None)
    accumulator.add(e.observe(), 3)
    assert accumulator.finish(1).sampling.bias_groups[1].bias == 8
    result = OnlineLearner(enabled=True).update(e.observe(), 1, s)
    assert result.group_deltas["fixed"] == 0


def test_group_feature_gradient_matches_counterfactual_and_controls():
    s = SamplingConfig(bias_groups=(group(bias=.2),))
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=s)
    learner = OnlineLearner(enabled=True, no_severity_attenuation=True)
    o = e.observe()
    r = learner.update(o, 3, s)
    epsilon = 1e-4
    plus = learner._counterfactual(o, learner._with_group_bias(s, "concrete", .2 + epsilon))
    minus = learner._counterfactual(o, learner._with_group_bias(s, "concrete", .2 - epsilon))
    numeric = (learner._loss(plus, 3) - learner._loss(minus, 3)) / (2 * epsilon)
    assert r.gradients["concrete"] == pytest.approx(numeric, abs=1e-8)
    assert r.severity == 1
    assert OnlineLearner(
        enabled=True, dead_zone_rank=3, no_severity_attenuation=False
    ).update(o, 3, s).update_norm == 0


def test_zero_dead_zone_rank_is_valid_for_learnable_groups():
    config = OnlineLearningConfig(dead_zone_rank=0, no_severity_attenuation=False)
    assert config.dead_zone_rank == 0
    assert OnlineLearner(config=config)._severity(1) > 0.0


def test_write_applies_group_updates_sequentially():
    s = SamplingConfig(bias_groups=(group(bias=.5),))
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=s)
    learner = OnlineLearner(enabled=True, decay=.2, no_severity_attenuation=True)
    accumulator = _WriteLearningAccumulator(e.backend, s, learner, None)
    expected_sampling = s
    for _ in range(2):
        expected_sampling = learner.update(e.observe(), 3, expected_sampling).sampling
        accumulator.add(e.observe(), 3)
    result = accumulator.finish(2).group_result
    assert len(accumulator.group_results) == 2
    assert result.sampling == accumulator.sampling
    assert result.sampling == expected_sampling


def test_headless_reference_and_group_control_survive_fork_and_reopen(tmp_path):
    from tests.test_headless import send
    from tests.test_episode_runtime import NoEogBackend
    from trajectory_editor.headless import Session
    b = NoEogBackend()
    initial = SamplingConfig(reference_prior_routes=(((3,), 10.), ((2,), 1.)),
                             reference_prior_mode="lexical", reference_prior_scope="global")
    with EpisodeStore(tmp_path / 'api.db') as store:
        session = Session(b, store, b.provenance(), sampling=initial)
        send(session, 'open', prompt='P')
        send(session, 'steering', command='b concrete -> {C}')
        send(session, 'steering', command='b concrete +')
        send(session, 'actions', action={'kind': 'hold', 'limit': 4})
        old = session.engine.observe().statistics
        send(session, 'fork', boundary=4)
        forked = session.engine.observe().statistics
        np.testing.assert_array_equal(old.policy_probabilities, forked.policy_probabilities)
        assert old.group_control_diagnostics == forked.group_control_diagnostics
        identifier = session.episode_id
        send(session, 'close')
        send(session, 'open', episode_id=identifier)
        np.testing.assert_array_equal(session.engine.observe().statistics.policy_probabilities, forked.policy_probabilities)


def test_cli_standalone_yaml_and_preset_export_import(tmp_path):
    from contextlib import redirect_stdout
    from io import StringIO
    from unittest.mock import patch
    from tests.test_episode_runtime import NoEogBackend
    from trajectory_editor.episode_cli import main
    reference = tmp_path / 'reference.yaml'
    reference.write_text('C: 10\nB: 1\n')
    groups = tmp_path / 'groups.yaml'
    groups.write_text('defaults:\n  cases: [original]\n  plural: false\n  leading_space: false\ngroups:\n  concrete: [C]\n')
    workspace = tmp_path / 'cli.db'
    b = NoEogBackend()
    with patch('trajectory_editor.episode_cli._backend', return_value=b), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['b concrete +', 'b', 'h 2', 'q'])):
        assert main(['--workspace', str(workspace), '--model', 'fake.gguf', '--new-prompt', 'P',
                     '--episode-id', 'first', '--max-tokens', '2', '--plain-ui',
                     '--groups', str(groups), '--reference', str(reference)]) == 0
    with EpisodeStore(workspace) as store:
        state = store.final_sampling('first')
        assert state.reference_prior_mode == 'lexical' and state.reference_prior_scope == 'global'
        assert state.group_controls[0].direction == 'promote'
        exported = tmp_path / 'biases.json'
        exported.write_text(project_biases(store, 'first'))
    b = NoEogBackend()
    with patch('trajectory_editor.episode_cli._backend', return_value=b), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['h 2', 'q'])):
        assert main(['--workspace', str(workspace), '--model', 'fake.gguf', '--new-prompt', 'P',
                     '--episode-id', 'second', '--max-tokens', '2', '--plain-ui', '--biases', str(exported)]) == 0
    with EpisodeStore(workspace) as store:
        restored = store.final_sampling('second')
        assert restored.reference_prior_routes == state.reference_prior_routes
        assert restored.group_controls == state.group_controls


def test_reference_validation_and_minimal_preference_array(tmp_path):
    from trajectory_editor.domain import EditorError
    b = ConformingFakeBackend()
    for bad in ({'C': -1}, {'C': float('nan')}, {'': 1}, {'C': True}, {}):
        with pytest.raises(EditorError):
            compile_reference(bad, b)
    path = tmp_path / 'weights.json'
    path.write_text(json.dumps(dict(format='spe-bias-rules-v4', model=b.provenance(), token_preference_vector=[.1, .2], token_preference_projection_seed=42)))
    loaded = load_bias_preset(path, b, b.provenance())
    assert loaded.token_preference_vector == (.1, .2) and loaded.token_preference_projection_seed == 42


def test_control_rewind_and_replay_restore_exact_policy(tmp_path):
    from tests.test_episode_runtime import NoEogBackend
    from trajectory_editor.episode_lifecycle import _rewind_episode, _spr_engine_from_source
    s = SamplingConfig(temperature=.8, top_k=8, top_p=1, min_p=0,
        bias_groups=(group(),), group_controls=(GroupControl('concrete', 'promote', .1),),
        reference_prior_routes=(((3,), 2.), ((2,), 1.)), reference_prior_scope='global', reference_prior_mode='lexical')
    e = EpisodeEngine(NoEogBackend(), initial_token_ids=[7], sampling=s)
    with EpisodeStore(tmp_path / 'replay.db') as store:
        eid = _create_episode(store, e, backend_provenance=e.backend.provenance())
        store.record_action(eid, 0, e.apply(Hold(4)))
        expected = e.observe().statistics
        prefix = list(e.visible_token_ids)
        # A later objective must disappear when rewinding to the earlier segment.
        e.sampling = replace(e.sampling, group_controls=(replace(e.sampling.group_controls[0], direction='suppress'),))
        store.record_sampling_segment(eid, start_boundary=4, sampling=e.sampling,
                                      stream_fingerprint=e.stream_fingerprint, coordinate_offset=0)
        store.record_action(eid, 1, e.apply(Hold(4)))
        _rewind_episode(store, eid, e, 3)
        assert e.sampling.group_controls[0].direction == 'promote'
        store.record_action(eid, 1, e.apply(Hold(1)))
        assert e.visible_token_ids == prefix
        np.testing.assert_array_equal(e.observe().statistics.policy_probabilities, expected.policy_probabilities)
        replay, plan = _spr_engine_from_source(store, eid, NoEogBackend(), sampling=s, max_tokens=None)
        rid = _create_episode(store, replay, backend_provenance=replay.backend.provenance())
        result = EpisodeRunner(replay, store, rid).run(tape=plan)
        assert result.replay_exhausted
        assert replay.visible_token_ids == prefix
        np.testing.assert_array_equal(replay.observe().statistics.policy_probabilities, expected.policy_probabilities)
        assert replay.observe().statistics.group_control_diagnostics == expected.group_control_diagnostics


@pytest.mark.parametrize('use_preset', [False, True])
def test_cli_steering_override_survives_all_replay_segments(tmp_path, use_preset):
    from unittest.mock import patch
    from tests.test_episode_runtime import NoEogBackend
    from trajectory_editor.episode_cli import main
    workspace = tmp_path / 'replay.db'
    s = SamplingConfig(bias_groups=(group('old'),), group_controls=(GroupControl('old', 'promote', .1),))
    source = EpisodeEngine(NoEogBackend(), initial_token_ids=[7], sampling=s)
    with EpisodeStore(workspace) as store:
        _create_episode(store, source, backend_provenance=source.backend.provenance(), requested_id='source')
        store.record_action('source', 0, source.apply(Hold(1)))
        source.sampling = replace(source.sampling, temperature=.5)
        store.record_sampling_segment('source', start_boundary=1, sampling=source.sampling,
                                      stream_fingerprint=source.stream_fingerprint, coordinate_offset=0)
        store.record_action('source', 1, source.apply(Hold(1)))
        target = replace(s, bias_groups=(group('new'),), group_controls=(GroupControl('new', 'maintain', .2),),
                         reference_prior_routes=(((3,), 10.), ((2,), 1.)), reference_prior_mode='lexical', reference_prior_scope='global')
        e = EpisodeEngine(NoEogBackend(), initial_token_ids=[7], sampling=target)
        _create_episode(store, e, backend_provenance=e.backend.provenance(), requested_id='preset')
        preset = tmp_path / 'biases.json'
        preset.write_text(project_biases(store, 'preset'))
    reference = tmp_path / 'ref.yaml'
    reference.write_text('C: 10\nB: 1\n')
    flags = ['--biases', str(preset)] if use_preset else ['--reference', str(reference)]
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch(
        'trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['quit'])):
        assert main(['--workspace', str(workspace), '--model', 'fake', '--replay', 'source',
                     '--episode-id', 'target', '--plain-ui', '--divergence-policy', 'ballistic', *flags]) == 0
    with EpisodeStore(workspace) as store:
        for boundary in (0, 1, 2):
            state = SamplingConfig.from_record(store.sampling_segment('target', boundary)['sampling'])
            assert state.reference_prior_routes and state.reference_prior_mode == 'lexical'
            assert state.group_controls[0].group == ('new' if use_preset else 'old')
            assert state.group_controls[0].history_start == 1
        assert store.final_sampling('target').temperature == .5


def test_changing_models_clears_token_based_steering(tmp_path):
    from trajectory_editor.episode_lifecycle import _model_continuation
    s = SamplingConfig(bias_groups=(group(),), group_controls=(GroupControl('concrete', 'promote', .1),),
                       reference_prior_routes=(((3,), 2.),), reference_prior_mode='lexical', reference_prior_scope='global')
    e = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=s)
    with EpisodeStore(tmp_path / 'model.db') as store:
        eid = _create_episode(store, e, backend_provenance=e.backend.provenance())
        changed, _ = _model_continuation(store, eid, ConformingFakeBackend(), e.backend.provenance())
        assert not changed.sampling.group_controls
        assert not changed.sampling.bias_groups
        assert not changed.sampling.reference_prior_routes


def test_phrase_generation_responds_to_completed_appearance_objective():
    class PhraseBackend(ConformingFakeBackend):
        pieces = {**ConformingFakeBackend.pieces, 1: ' gathering', 2: ' storm', 3: ' sunlight'}

        def last_logits(self):
            return np.asarray([-30., -1., 0. if self.tokens[-1] == 1 else -5., 1., -20., -20., -20., -20.])

    g = group(route=(1, 2), surfaces=('gathering storm',))
    counts = {}
    for direction in (None, 'promote', 'suppress'):
        controls = () if direction is None else (GroupControl(g.name, direction, .025),)
        e = EpisodeEngine(PhraseBackend(), initial_token_ids=[7], sampling=SamplingConfig(
            temperature=1, top_k=8, top_p=1, min_p=0, bias_groups=(g,), group_controls=controls))
        e.apply(Hold(512))
        counts[direction] = appearances(g, e.visible_token_ids, e.backend.render)
        assert 3 in e.visible_token_ids
    assert counts['promote'] > counts[None] > counts['suppress']


def test_phrase_prefix_at_prompt_boundary_receives_continuation_support():
    g = group(route=(1, 2))
    c = GroupControl(g.name, 'promote', .1, history_start=2)
    biases, diagnostic = control_adjustments((c,), (g,), [7, 1], np.zeros(8))
    assert biases[2] > 0 and 1 not in biases
    assert diagnostic[0]['tokens'] == 0 and diagnostic[0]['continuing']
