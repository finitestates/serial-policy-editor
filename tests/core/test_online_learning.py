"""Actual local float32 GPT-2 training through the production backend."""
import json

import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from trajectory_editor.core.actions import Accept, SelectRawRank, Write, Phrase
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import build_parser, main
from trajectory_editor.episode_engine import EpisodeEngine, InstructionRejected
from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
from trajectory_editor.online_learning import LearningConfig
from trajectory_editor.transformers_backend import TransformersBackend, TransformersSettings
from trajectory_editor.teacher_commands import parse_command, CommandKind


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    path = tmp_path_factory.mktemp('tiny-online-lm')
    vocabulary = {'<unk>': 0, '<eos>': 1, 'P': 2, 'red': 3, 'blue': 4,
                  'green': 5, 'house': 6, 'ghost': 7}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
        eos_token='<eos>').save_pretrained(path)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_embd=16, n_layer=1,
        n_head=2, n_positions=64, bos_token_id=2, eos_token_id=1,
        resid_pdrop=0.5, embd_pdrop=0.5, attn_pdrop=0.5))
    model.save_pretrained(path)
    return path


@pytest.fixture
def backend(checkpoint):
    result = TransformersBackend(checkpoint, TransformersSettings(device='cpu', dtype='float32'))
    result.enable_online_learning(LearningConfig(lr=0.05, steps=1))
    return result


def engine(backend):
    return EpisodeEngine(backend, initial_token_ids=[2], initial_text='P',
        sampling=SamplerConfig(temperature=0.0, top_k=8, min_p=0.0))


def weights(backend):
    return {k: v.clone() for k, v in backend._model.state_dict().items()}


def test_correction_changes_actual_weights_and_likelihood(backend):
    e = engine(backend)
    old = weights(backend)
    observation = e.observe()
    prefix = e.token_ids.copy()
    target = int(observation.policy_calculations.top_raw_ids(2)[-1])
    before = torch.softmax(torch.tensor(observation.logits), 0)[target].item()
    e.apply(SelectRawRank(2))
    event = e.learner.events[-1]
    assert event['prefix_token_ids'] == prefix
    assert event['corrected_token_ids'] == [target]
    assert event['proposal_token_id'] == observation.proposal_token_id
    assert event['gradient_norms'][0] > 0
    assert any(not torch.equal(old[k], v) for k, v in backend._model.state_dict().items())
    backend.reset(prefix)
    after = torch.softmax(torch.tensor(backend.last_logits()), 0)[target].item()
    assert after > before
    assert event['loss_after'] < event['loss_before']
    assert not backend._model.training


def test_teacher_forcing_alignment_and_steps(backend):
    e = engine(backend)
    e.learner.command('learn steps=3')
    with torch.no_grad():
        logits = backend._model(input_ids=torch.tensor([[2, 3]]), use_cache=False).logits
        expected = torch.nn.functional.cross_entropy(logits[0], torch.tensor([3, 4])).item()
    before = backend.weight_revision
    outcome = e.apply(Write('red blue', mode='exact'))
    event = e.learner.events[-1]
    assert outcome.resolved_token_ids == (3, 4)
    assert event['loss_before'] == pytest.approx(expected, abs=1e-6)
    assert len(event['losses']) == 3
    assert backend.weight_revision == before + 3
    assert json.loads(e.learner.command('learn status'))['optimizer_steps'] == 3


def test_teach_uses_original_context_and_preserves_defaults(backend):
    e = engine(backend)
    e.apply(Write('red', mode='exact'))
    example = e.learner.last
    e.apply(Accept())
    e.learner.command('teach last steps=5')
    event = e.learner.events[-1]
    assert event['prefix_token_ids'] == list(example[0]) == [2]
    assert event['corrected_token_ids'] == [3]
    assert event['steps'] == 5
    assert e.learner.config.steps == 1
    assert e.learner.corrections == 1


def test_reset_same_prefix_and_trajectory_retained(backend):
    e = engine(backend)
    original = e.observe().logits.copy()
    e.apply(Write('red blue', mode='exact'))
    trajectory = e.token_ids.copy()
    revision = backend.weight_revision
    e.learner.command('learn reset')
    assert e.token_ids == trajectory
    assert backend.weight_revision > revision
    assert not e.learner.events and e.learner.last is None
    backend.reset([2])
    np.testing.assert_array_equal(backend.last_logits(), original)


def test_cache_and_observation_invalidated_at_same_prefix(backend):
    e = engine(backend)
    e.apply(Write('red', mode='exact'))
    old = e.observe()
    old_cache = backend._past_key_values
    e.learner.command('teach last steps=2')
    new = e.observe()
    assert new is not old
    assert backend._past_key_values is not old_cache
    cached = new.logits.copy()
    backend.reset(e.token_ids)
    np.testing.assert_allclose(backend.last_logits(), cached, atol=1e-7)
    with pytest.raises(EditorError, match='batch'):
        backend.create_batch([[2]])


def test_separate_episodes_restore_original_and_old_owner_blocked(backend):
    first = engine(backend)
    original = first.observe().logits.copy()
    first.apply(Write('red', mode='exact'))
    second = engine(backend)
    np.testing.assert_array_equal(second.observe().logits, original)
    assert second.learner.corrections == 0
    with pytest.raises(EditorError, match='no longer active'):
        first.observe()


def test_pause_rejected_diagnostic_and_accept_do_not_train(backend):
    e = engine(backend)
    e.apply(Accept())
    revision = backend.weight_revision
    with pytest.raises(InstructionRejected):
        e.apply(SelectRawRank(100))
    e.apply(Phrase('blue'))
    assert e.learner.corrections == 0
    assert backend.weight_revision == revision
    e.learner.command('learn steps=0')
    e.apply(Write('red', mode='exact'))
    assert e.learner.corrections == 1 and e.learner.last is not None
    assert backend.weight_revision == revision
    e.learner.command('teach last steps=2')
    assert backend.weight_revision == revision + 2
    assert e.learner.config.steps == 0


def test_unsupported_operations(backend):
    from trajectory_editor.beam import BeamSearch
    from trajectory_editor.chord import Chord
    e = engine(backend)
    session = LiveSession(e)
    with pytest.raises(EditorError, match='replay'):
        session.generate(Write('red'), replay=True)
    with pytest.raises(EditorError, match='rewind'):
        session.rewind(0)
    with pytest.raises(EditorError, match='fork'):
        session.fork()
    with pytest.raises(EditorError, match='unsupported'):
        BeamSearch(e, width=2)
    with pytest.raises(EditorError, match='unsupported'):
        Chord(e, (1, 2))
    roster = LiveSessionRoster(session)
    with pytest.raises(EditorError, match='switching'):
        roster.switch(1)
    assert session.engine.boundary == 0


def test_cli_and_runtime_controls(checkpoint):
    args = build_parser().parse_args(['--online-sgd', '--online-steps', '7', '--online-lr', '0.02'])
    assert args.online_steps == 7 and args.online_lr == 0.02
    from trajectory_editor.episode_backend_loader import load_backend
    args.backend = 'transformers'
    args.model = checkpoint
    loaded = load_backend(args)
    e = engine(loaded)
    e.apply(Write('red', mode='exact'))
    assert e.learner.events[-1]['steps'] == 7
    for text in ['learn status', 'learn reset', 'learn lr=0.1', 'learn steps=0', 'teach last steps=5']:
        assert parse_command(text, menu_size=8, default_hold_tokens=1).kind == CommandKind.LEARN
    for text in ['learn lr=nan', 'learn lr=0', 'learn steps=-1', 'teach last steps=0']:
        with pytest.raises(EditorError):
            e.learner.command(text)
    assert main(['--online-sgd', '--backend', 'llama.cpp', '--new-prompt', 'P']) == 2


def test_dtype_rejected(checkpoint):
    b = TransformersBackend(checkpoint, TransformersSettings(device='cpu', dtype='float16'))
    with pytest.raises(EditorError, match='float32'):
        b.enable_online_learning(LearningConfig())


def test_training_failure_blocks_inference_until_reset(backend, monkeypatch):
    e = engine(backend)
    original = weights(backend)
    actual_step = torch.optim.SGD.step
    def fail(optimizer, *args, **kwargs):
        actual_step(optimizer, *args, **kwargs)
        raise RuntimeError('injected failure after parameter mutation')
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.SGD, 'step', fail)
        with pytest.raises(EditorError, match='training failed'):
            e.apply(Write('red', mode='exact'))
    with pytest.raises(EditorError, match='training failed'):
        e.observe()
    e.learner.command('learn reset')
    assert all(torch.equal(original[k], v) for k, v in backend._model.state_dict().items())
    e.observe()


def test_runtime_menu_refreshes_predictions_after_teach_and_reset(backend):
    from tests.fakes import ScriptedIO
    from trajectory_editor.episode_ui import InteractivePolicy
    e = engine(backend)
    e.apply(Write('red', mode='exact'))
    before = e.observe()
    io = ScriptedIO(['learn lr=0.1', 'learn steps=4', 'teach last steps=20',
                     'learn status', 'learn reset', '2'])
    policy = InteractivePolicy(io=io, menu_size=8)
    action = policy.choose(e, before)
    assert action.kind == 'select-raw-rank'
    assert e.learner.config.steps == 4
    requests = io.choice_requests
    assert len(requests) == 6
    backend.reset(e.token_ids)
    np.testing.assert_array_equal(e.observe().logits, backend.last_logits())
    assert requests[-1].choice.proposal_token_id == e.observe().proposal_token_id
    assert any('Original episode weights restored' in line for line in io.output)


def test_same_proposal_selection_is_not_supervision(backend):
    e = engine(backend)
    observation = e.observe()
    revision = backend.weight_revision
    e.apply(SelectRawRank(observation.proposal_raw_rank))
    assert e.learner.corrections == 0
    assert backend.weight_revision == revision


def test_discard_restores_weights_and_failure_keeps_action_history(backend, monkeypatch):
    e = engine(backend)
    session = LiveSession(e)
    original = weights(backend)
    actual_step = torch.optim.SGD.step
    def fail(optimizer, *args, **kwargs):
        actual_step(optimizer, *args, **kwargs)
        raise RuntimeError('injected failure')
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.SGD, 'step', fail)
        with pytest.raises(EditorError, match='training failed'):
            session.generate(Write('red', mode='exact'))
    assert session.history_tape[-1].action.kind == 'write'
    assert session.history_outcomes[-1].resolved_token_ids == (3,)
    session.discard()
    assert all(torch.equal(original[k], v) for k, v in backend._model.state_dict().items())


def test_prior_batch_cache_closed_when_learning_enabled(checkpoint):
    b = TransformersBackend(checkpoint, TransformersSettings(device='cpu', dtype='float32'))
    batch = b.create_batch([[2], [2, 3]])
    lane = batch.lane(0)
    lane.last_logits()
    b.enable_online_learning(LearningConfig())
    with pytest.raises(RuntimeError, match='closed'):
        lane.last_logits()


def test_cli_recovers_from_partial_training_failure(backend, monkeypatch):
    from tests.fakes import ScriptedIO
    io = ScriptedIO(['x red', 'learn reset', 'learn status', 'q', 'quit'])
    monkeypatch.setattr('trajectory_editor.episode_cli.TerminalIO', lambda **kwargs: io)
    monkeypatch.setattr('trajectory_editor.episode_backend_loader.load_backend', lambda args: backend)
    actual_step = torch.optim.SGD.step
    def fail(optimizer, *args, **kwargs):
        actual_step(optimizer, *args, **kwargs)
        raise RuntimeError('injected failure after mutation')
    monkeypatch.setattr(torch.optim.SGD, 'step', fail)
    assert main(['--backend', 'transformers', '--model', 'unused', '--new-prompt', 'P',
                 '--online-sgd', '--plain-ui', '--temperature', '0']) == 0
    assert any('online training failed' in line for line in io.output)
    assert any('Original episode weights restored' in line for line in io.output)


def test_suspended_episode_cannot_silently_reset_learning(backend):
    session = LiveSession(engine(backend))
    session.generate(Write('red', mode='exact'))
    learned = weights(backend)
    branch_id = session.branch.branch_id
    session.suspend()
    with pytest.raises(EditorError, match='reactivation'):
        session.activate(branch_id)
    assert all(torch.equal(learned[k], v) for k, v in backend._model.state_dict().items())


@pytest.mark.parametrize('text', ['2!', '2 !', '2?', '2 ?'])
def test_rank_training_parser(text):
    command = parse_command(text, menu_size=8, default_hold_tokens=1)
    assert command.learning_rank == 2
    assert command.learning_mode == text[-1]


def test_train_until_argmax_pins_original_target(backend):
    e = engine(backend)
    target = int(e.observe().policy_calculations.top_raw_ids(8)[-1])
    e.apply(SelectRawRank(8, learning_mode='!'))
    event = e.learner.events[-1]
    assert event['corrected_token_ids'] == [target]
    assert event['stop_reason'] == 'argmax' and event['steps'] > 0
    backend.reset([2])
    assert int(np.argmax(backend.last_logits())) == target
    assert e.token_ids == [2, target]


def test_bounded_training_obeys_ceiling_and_ignores_fixed_steps(backend):
    e = engine(backend)
    e.learner.command('learn lr=0.000000000001')
    e.learner.command('learn steps=0')
    e.learner.command('max_step=2')
    messages = []
    e.learner.progress = messages.append
    e.apply(SelectRawRank(8, learning_mode='?'))
    event = e.learner.events[-1]
    assert event['steps'] == 2 and event['stop_reason'] == 'ceiling'
    assert len(messages) == 2 and 'step 2/2' in messages[-1]


def test_already_argmax_needs_no_updates(backend):
    e = engine(backend)
    revision = backend.weight_revision
    e.apply(SelectRawRank(1, learning_mode='!'))
    event = e.learner.events[-1]
    assert event['steps'] == 0 and event['stop_reason'] == 'argmax'
    assert backend.weight_revision == revision


def test_max_step_cli_and_validation():
    assert build_parser().parse_args([]).max_step == 5
    assert build_parser().parse_args(['--max-step', '9']).max_step == 9
    with pytest.raises(EditorError):
        LearningConfig(max_step=0).validate()


def test_rank_training_runtime_dispatch_and_disabled_guard(backend):
    from tests.fakes import ScriptedIO
    from trajectory_editor.episode_ui import InteractivePolicy
    from trajectory_editor.core.actions import action_from_dict
    e = engine(backend)
    policy = InteractivePolicy(io=ScriptedIO(['max_step=2', '8 ?']), menu_size=8)
    action = policy.choose(e, e.observe())
    assert action.rank == 8 and action.learning_mode == '?'
    assert action_from_dict(action.to_dict()) == action
    assert e.learner.config.max_step == 2
    e.apply(action)
    assert e.learner.events[-1]['steps'] <= 2
    e.learner = None
    with pytest.raises(EditorError, match='requires --online-sgd'):
        e.apply(action)


def test_fixed_step_training_reports_every_completed_update(backend):
    e = engine(backend)
    e.learner.command('learn steps=3')
    messages = []
    e.learner.progress = messages.append
    e.apply(Write('red', mode='exact'))
    assert len(messages) == 3
    assert all(f'step {i}/3' in text for i, text in enumerate(messages, 1))


@pytest.mark.parametrize('text', ['2! 5.0', '2 ! 5.0', '2? 5.0', '2 ? 5.0'])
def test_margin_parser_and_roundtrip(text):
    from trajectory_editor.core.actions import action_from_dict
    command = parse_command(text, menu_size=8, default_hold_tokens=1)
    assert command.learning_margin == 5.0
    action = SelectRawRank(command.learning_rank, learning_mode=command.learning_mode,
                           learning_margin=command.learning_margin)
    assert action_from_dict(action.to_dict()) == action


@pytest.mark.parametrize('text', ['2 ! -1', '2 ? nan', '2 ! inf', '2 ? nope', '2*'])
def test_invalid_or_retired_training_syntax(text):
    with pytest.raises(EditorError):
        parse_command(text, menu_size=8, default_hold_tokens=1)


def test_training_requires_margin_even_if_already_argmax(backend):
    e = engine(backend)
    target = int(np.argmax(e.observe().logits))
    logits = e.observe().logits
    initial_margin = float(logits[target] - np.max(np.delete(logits, target)))
    threshold = initial_margin + 1.0
    messages = []
    e.learner.progress = messages.append
    outcome = e.apply(SelectRawRank(1, learning_mode='!', learning_margin=threshold))
    event = e.learner.events[-1]
    assert event['steps'] > 0 and event['stop_reason'] == 'margin'
    assert event['requested_margin'] == threshold
    backend.reset([2])
    row = backend.last_logits()
    actual = float(row[target] - np.max(np.delete(row, target)))
    assert actual >= threshold
    assert event['final_margin'] == pytest.approx(actual, abs=1e-6)
    assert outcome.resolved_token_ids == (target,)
    assert 'logits; reached' in messages[-1]


def test_margin_ceiling_stops_before_requested_victory(backend):
    e = engine(backend)
    e.learner.command('max_step=2')
    e.apply(SelectRawRank(1, learning_mode='?', learning_margin=1000.0))
    event = e.learner.events[-1]
    assert event['steps'] == 2 and event['stop_reason'] == 'ceiling'
    assert event['final_margin'] < 1000.0


def test_satisfied_margin_skips_updates(backend):
    e = engine(backend)
    logits = e.observe().logits
    top = np.sort(logits)[-2:]
    threshold = float(top[1] - top[0]) / 2
    e.apply(SelectRawRank(1, learning_mode='?', learning_margin=threshold))
    event = e.learner.events[-1]
    assert event['steps'] == 0 and event['stop_reason'] == 'margin'


def test_old_plus_syntax_is_not_training():
    command = parse_command('2+', menu_size=8, default_hold_tokens=1)
    assert command.kind != CommandKind.LEARN


def test_margin_runtime_dispatch(backend):
    from tests.fakes import ScriptedIO
    from trajectory_editor.episode_ui import InteractivePolicy
    e = engine(backend)
    io = ScriptedIO(['max_step=2', '1 ? 5.0'])
    action = InteractivePolicy(io=io, menu_size=8).choose(e, e.observe())
    assert action.learning_mode == '?' and action.learning_margin == 5.0
    e.apply(action)
    assert e.learner.events[-1]['requested_margin'] == 5.0
    assert any('logits;' in text for text in io.output)


def test_live_diagnostics_match_actual_post_update_logits_and_rival(backend):
    e = engine(backend)
    snapshots = []
    def record(snapshot):
        snapshots.append(snapshot)
        if snapshot.current is None:
            return
        point = snapshot.current.comparison
        row = backend.learning_probe_logits([2])
        rivals = row.copy()
        rivals[point.target_id] = -np.inf
        assert point.target_logit == pytest.approx(float(row[point.target_id]), abs=1e-6)
        assert point.rival_id == int(np.argmax(rivals))
        assert point.rival_logit == pytest.approx(float(rivals.max()), abs=1e-6)
        assert point.margin == pytest.approx(float(row[point.target_id] - rivals.max()), abs=1e-6)
    e.learner.display = record
    e.learner.command('learn steps=3')
    snapshots.clear()
    e.apply(SelectRawRank(8))
    training = [s for s in snapshots if s.current.status == 'training']
    assert [s.current.step for s in training] == [0, 1, 2, 3]
    assert [s.optimizer_steps for s in training] == [0, 1, 2, 3]
    for before, after in zip(training, training[1:]):
        assert after.current.target_delta == pytest.approx(after.current.comparison.target_logit - before.current.comparison.target_logit)
        assert after.current.margin_delta == pytest.approx(after.current.comparison.margin - before.current.comparison.margin)
    assert snapshots[-1].current.status == 'fixed-steps'
    assert len(snapshots[-1].history) == 4


def test_overview_is_read_only_and_rechecks_earlier_corrections(backend):
    e = engine(backend)
    e.apply(Write('red', mode='exact'))
    first = e.learner.snapshot().probes[-1]
    assert first.baseline_kind == 'first seen'
    taught_margin = first.taught_margin
    old_snapshot = e.learner.snapshot()
    e.apply(Write('blue', mode='exact'))
    assert e.learner.snapshot().probes[-2].revision < backend.weight_revision
    observation = e.observe()
    cache = backend._past_key_values
    original_weights = weights(backend)
    revision = backend.weight_revision
    events = list(e.learner.events)
    trajectory = e.token_ids.copy()
    result = e.learner.overview()
    assert all(p.revision == revision for p in result.probes)
    assert backend.weight_revision == revision and e.learner.events == events
    assert e.token_ids == trajectory
    assert e.observe() is observation and backend._past_key_values is cache
    assert all(torch.equal(original_weights[k], v) for k, v in backend._model.state_dict().items())
    first_now = result.probes[-2]
    row = backend.learning_probe_logits([2])
    rivals = row.copy()
    rivals[3] = -np.inf
    assert first_now.current.margin == pytest.approx(float(row[3] - rivals.max()))
    assert first_now.taught_margin == taught_margin
    assert old_snapshot.probes[-1].revision < result.probes[-2].revision


def test_root_probes_use_original_predictions_and_reset_clears_panel(backend):
    e = engine(backend)
    original = e.learner.snapshot()
    assert original.probes and all(p.baseline_kind == 'episode start' for p in original.probes)
    original_sample = original.probes[0].baseline
    e.apply(Write('red blue', mode='exact'))
    result = e.learner.overview()
    assert result.probes[0].baseline == original_sample
    assert result.probes[0].drift > 0
    e.learner.command('learn reset')
    reset = e.learner.snapshot()
    assert reset.current is None and reset.history == () and reset.optimizer_steps == 0
    assert reset.corrections == 0
    assert len(reset.probes) == len(original.probes)
    assert reset.probes[0].baseline == original_sample
    assert reset.probes[0].drift == 0


def test_probe_history_is_bounded_and_diagnostics_do_not_change_inference(backend):
    from trajectory_editor.learning_diagnostics import CORRECTION_PROBE_LIMIT, HISTORY_LIMIT
    e = engine(backend)
    observation = e.observe()
    row = backend.learning_probe_logits([2])
    np.testing.assert_allclose(row, observation.logits, atol=1e-7)
    cache = backend._past_key_values
    e.learner.overview()
    assert backend._past_key_values is cache and e.observe() is observation
    for _ in range(HISTORY_LIMIT + 2):
        e.learner.overview()
    assert len(e.learner.snapshot().probes[0].history) == HISTORY_LIMIT
    for index in range(CORRECTION_PROBE_LIMIT + 2):
        e.learner.last = ((2,) * (index + 1), (3,), 4, index)
        e.learner.train(e.learner.last, 1)
    assert sum(p.baseline_kind == 'first seen' for p in e.learner.snapshot().probes) == CORRECTION_PROBE_LIMIT


def test_reteaching_moves_probe_to_recent_end_without_replacing_baseline(backend):
    e = engine(backend)
    e.apply(Write('red', mode='exact'))
    first_example = e.learner.last
    first_baseline = e.learner.snapshot().probes[-1].baseline
    e.apply(Write('blue', mode='exact'))
    e.learner.train(first_example, 1)
    recent = e.learner.snapshot().probes[-1]
    assert recent.label == 'correction · boundary 0'
    assert recent.baseline == first_baseline
    assert sum(p.baseline_kind == 'first seen' for p in e.learner.snapshot().probes) == 2


def test_paused_correction_is_watched_without_training_or_replacing_trace(backend):
    e = engine(backend)
    e.apply(Write('red', mode='exact'))
    previous_trace = e.learner.snapshot().current
    e.learner.command('learn steps=0')
    revision = backend.weight_revision
    e.apply(Write('blue', mode='exact'))
    snapshot = e.learner.snapshot()
    assert snapshot.corrections == 2 and snapshot.optimizer_steps == 1
    assert snapshot.current == previous_trace
    assert backend.weight_revision == revision
    probe = snapshot.probes[-1]
    assert probe.current.target_id == 4 and not probe.trained
    assert probe.baseline == probe.current


def test_phase_two_inventory_and_tied_head_rejection(backend):
    e = engine(backend)
    inventory = json.loads(e.learner.command('learn targets'))
    assert inventory['layers'] == [0]
    tied = next(p for p in inventory['parameters'] if 'lm_head.weight' in p['names'])
    assert set(tied['blocks']) == {'lm_head', 'embeddings'}
    before = [p.requires_grad for p in backend._model.parameters()]
    with pytest.raises(EditorError, match='tied aliases'):
        e.learner.command('learn thaw im_head')
    assert before == [p.requires_grad for p in backend._model.parameters()]
    e.learner.command('learn freeze im_head')
    assert not backend._model.transformer.wte.weight.requires_grad


def test_phase_two_selected_weights_and_atomic_errors(backend):
    e = engine(backend)
    e.learner.command('learn thaw mlp')
    e.learner.command('learn layers 0+')
    before = weights(backend)
    e.apply(SelectRawRank(2))
    changed = {n for n, v in backend._model.state_dict().items() if not torch.equal(before[n], v)}
    assert changed and all('.mlp.' in n for n in changed)
    assert not backend.learning_selection['preserves_kv']
    for command in ('learn thaw nonexistent', 'learn thaw attention_qk', 'learn layers 1',
                    'learn layers 0-3', 'learn layers 1-0', 'learn layers -1', 'learn style invalid'):
        old = e.learner.command('learn status')
        with pytest.raises(EditorError):
            e.learner.command(command)
        assert e.learner.command('learn status') == old
    e.learner.command('learn layers all')
    e.learner.command('learn thaw all')
    assert all(p.requires_grad for p in backend._model.parameters())


def test_phase_two_combined_gradient_is_teacher_minus_rival(backend):
    e = engine(backend)
    e.learner.command('learn style sgd-and-sga')
    e.learner.command('learn thaw mlp')
    e.learner.command('learn lr=0.01')
    model = backend._model
    inputs = torch.tensor([[2, 3]])
    targets = torch.tensor([3, 4])
    logits = model(input_ids=inputs, use_cache=False).logits[0]
    competitors = logits.detach().clone()
    competitors.scatter_(1, targets[:, None], -torch.inf)
    rivals = competitors.argmax(-1)
    assert (rivals != targets).all()
    objective = (torch.nn.functional.cross_entropy(logits, targets)
                 - torch.nn.functional.cross_entropy(logits, rivals))
    objective.backward()
    expected = {n: p.detach().clone() - 0.01 * p.grad if p.grad is not None else p.detach().clone()
                for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    event = e.learner.train(([2], [3, 4], None, 0), 1)
    assert event['objectives'] == pytest.approx([float(objective.detach())])
    assert event['rival_token_ids'] == [rivals.tolist()]
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, expected[n], atol=1e-7, rtol=1e-6)
    assert event['learning_config']['style'] == 'sgd-and-sga'


def test_phase_two_untied_head_preserves_kv_and_matches_full_prefill(checkpoint):
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    model.config.tie_word_embeddings = False
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    b = TransformersBackend(checkpoint, TransformersSettings(device='cpu', dtype='float32'))
    b._model = model.eval()
    b.enable_online_learning(LearningConfig(lr=0.05, thaw=('lm_head',)))
    e = engine(b)
    b.reset([2])
    cache = b._past_key_values
    assert cache is not None
    baseline = weights(b)
    b.supervised_update([2], [3], 0.05, 1)
    assert b._past_key_values is cache
    assert b._last_logits is None
    b.refresh_learning_prefix([2, 3])
    np.testing.assert_allclose(b.last_logits(), b.learning_probe_logits([2, 3]), atol=1e-6)
    changed = {n for n, v in b._model.state_dict().items() if not torch.equal(baseline[n], v)}
    assert changed == {'lm_head.weight'}
    b.restore_learning_base()
    assert b._past_key_values is None


def test_phase_two_cli_discovery_without_prompt(checkpoint, capsys):
    assert main(['--model', str(checkpoint), '--target-list']) == 0
    output = capsys.readouterr().out
    assert '"layers": [' in output and 'lm_head.weight' in output
    args = build_parser().parse_args(['--freeze', 'im_head', 'attention', '--layers', '0', '2-4', '7+'])
    assert args.freeze == ['im_head', 'attention']
    assert args.layers == ['0', '2-4', '7+']
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--freeze', 'mlp', '--thaw', 'lm_head'])
    for text in ['learn targets', 'learn style sgd-and-sga', 'learn freeze mlp attention',
                 'learn thaw all', 'learn layers 0 2-4 7+']:
        assert parse_command(text, vocabulary_size=8, menu_size=8, default_hold_tokens=1).kind is CommandKind.LEARN


def test_phase_two_llama_projection_groups_and_layer_ranges():
    from transformers import LlamaConfig, LlamaForCausalLM
    from trajectory_editor.learning_targets import LearningTargets
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16,
        intermediate_size=24, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=False))
    targets = LearningTargets(model)
    selected = targets.select(LearningConfig(thaw=('attention_qk',), layers=('1-2',)))
    names = [n for t in selected for n in t.names]
    assert set(names) == {f'model.layers.{i}.self_attn.{p}.weight'
                          for i in (1, 2) for p in ('q_proj', 'k_proj')}
    selected = targets.select(LearningConfig(thaw=('attention_vo',), layers=('2+',)))
    assert {n for t in selected for n in t.names} == {
        f'model.layers.2.self_attn.{p}.weight' for p in ('v_proj', 'o_proj')}
    assert targets.apply(LearningConfig(thaw=('im_head',)))['preserves_kv']
    assert not targets.apply(LearningConfig(thaw=('mlp',)))['preserves_kv']
    selected = targets.select(LearningConfig(layers=('0', '2')))
    assert all(t.layers <= {0, 2} for t in selected)


def test_phase_two_runtime_cache_invalidation_for_mlp(backend):
    e = engine(backend)
    e.learner.command('learn thaw mlp')
    backend.reset([2])
    assert backend._past_key_values is not None
    backend.supervised_update([2], [3], 0.05, 1)
    assert backend._past_key_values is None
    assert not backend._cache_active
