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
