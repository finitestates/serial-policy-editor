"""S03/S08, L01-L08, R01/R08: CFG contexts and evaluation work."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.actions import Accept, SelectRawRank, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_lifecycle import _restore_engine
from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
from trajectory_editor.run_loop import ReplayContext, ReplayPlan, TapeStep, run_plan
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.fresh_episode import fresh_root_from


class PrefixBackend(ConformingFakeBackend):
    """BOS and non-round-tripping IDs; logits depend on ordered full context."""

    def tokenizer_id(self):
        return "prefix-destination" if self.destination else "prefix-source"
    pieces = {0: '<EOG>', 1: '<BOS>', 2: 'c', 3: 'o', 4: 'n', 5: 'u',
              6: 'v', 7: 'x', 8: 'x', 9: 'y', 10: 'z', 11: 'q'}

    def __init__(self, *, destination=False, tokenless=False):
        super().__init__()
        self.destination = destination
        self.tokenless = tokenless
        self.work = []
        self.tokenizations = []
        self.controls = []

    def tokenize(self, text, *, add_bos=False, special=False):
        self.tokenizations.append((text, add_bos, special))
        if self.tokenless and text == '':
            return []
        ids = {'conditional': [2, 3, 4], 'other': [4, 3], 'U': [5],
               'B': [6, 5], '': [], 'x': [8], 'y': [9], 'z': [10],
               'xy': [8, 9], '<BOS>U': [1, 5]}[text]
        if self.destination:
            ids = [11 if t == 5 else 10 if t == 8 else t for t in ids]
        return ([1] if add_bos else []) + ids

    def reset(self, prefix_token_ids):
        self.work.append(('reset', tuple(prefix_token_ids)))
        super().reset(prefix_token_ids)

    def eval(self, token_ids):
        self.work.append(('eval', tuple(token_ids)))
        super().eval(token_ids)

    def last_logits(self):
        state = 0
        for token in self.tokens:
            state = (state * 17 + token) % 997
        logits = np.array([(state * (j + 3) % 101) / 17 for j in range(12)])
        logits[0] = -99
        return logits

    def activation_control_vector_width(self):
        return 1

    def activation_control_vector_layer_count(self):
        return 1

    def set_activation_control_vector(self, vector, **kwargs):
        self.controls.append((vector, kwargs))


def reference(prefix):
    # Independent complete-prefix evaluation; no engine helper or fake logits.
    state = sum(t * 17 ** (len(prefix) - i - 1) for i, t in enumerate(prefix)) % 997
    result = np.remainder(state * np.arange(3, 15), 101).astype(float) / 17
    result[0] = -99
    return result


def config(**kwargs):
    return replace(SamplerConfig(temperature=.8, top_k=12, top_p=1, min_p=0,
                                cfg_unconditional_prompt='U', cfg_scale=1.7,
                                cfg_prefix_tokens=5), **kwargs)


def engine(**kwargs):
    options = dict(initial_text='conditional', sampling=config(),
                   guidance_backend=PrefixBackend())
    options.update(kwargs)
    return EpisodeEngine(PrefixBackend(), **options)


def assert_context(runtime, unconditional=(1, 5)):
    visible = runtime.visible_token_ids
    primary = [*runtime.initial_token_ids, *visible]
    assert runtime.backend.tokens == primary
    expected = reference(primary)
    limit = runtime.sampling.cfg_prefix_tokens
    active = (
    runtime.sampling.cfg_unconditional_prompt is not None
    and (limit == 0 or len(visible) < limit)
    )
    observation = runtime.observe()
    if active:
        guidance = [*unconditional, *visible]
        assert runtime.guidance_backend.tokens == guidance
        uncond = reference(guidance)
        expected = uncond + runtime.sampling.cfg_scale * (expected - uncond)
    np.testing.assert_allclose(observation.logits, expected, rtol=0, atol=1e-14)
    probabilities = np.exp((expected - expected.max()) / runtime.sampling.temperature)
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(
        [observation.distribution.probability(i) for i in range(12)], probabilities,
        rtol=1e-12, atol=1e-14,
    )
    return observation


def raw_token(runtime, token=7):
    observation = runtime.observe()
    return runtime.apply(SelectRawRank(observation.statistics.raw_rank(token)))


def save(store, runtime, name='source'):
    return store.create_episode(
        episode_id=name,
        initial_text=runtime.initial_text,
        initial_token_ids=list(runtime.initial_token_ids),
        sampling=runtime.sampling,
        stream_fingerprint=runtime.stream_fingerprint,
        max_tokens=runtime.max_tokens,
        backend=runtime.backend.provenance(),
        checkpoint_boundary=runtime.checkpoint_boundary,
    )


@pytest.mark.parametrize('prompt,expected', [('U', (1, 5)), ('', (1,)), ('<BOS>U', (1, 1, 5))])
@pytest.mark.parametrize('boundary', [0, 1])
@pytest.mark.invariant
def test_s03_l01_prompt_entrances_and_real_resume(tmp_path, prompt, expected, boundary):
    sampling = config(cfg_unconditional_prompt=prompt)
    text = engine(sampling=sampling)
    exact = engine(sampling=sampling, initial_token_ids=text.initial_token_ids,
                   add_bos=False, special=False)
    np.testing.assert_array_equal(assert_context(text, expected).logits,
                                  assert_context(exact, expected).logits)
    with EpisodeStore(tmp_path / 'episodes.db') as store:
        identifier = save(store, text)
        if boundary:
            outcome = raw_token(text)
            store.record_action(identifier, 0, outcome)
            assert text.visible_token_ids == [7]
            assert text.backend.tokenize(text.backend.render([7])) == [8]
        original = assert_context(text, expected)
        restored = _restore_engine(store, identifier, PrefixBackend(), max_tokens=None,
                                   sampling_override=None, guidance_backend=PrefixBackend())
        np.testing.assert_array_equal(original.logits, assert_context(restored, expected).logits)
        assert restored.guidance_backend.work == [('reset', (*expected, *text.visible_token_ids))]


@pytest.mark.invariant
def test_s03_tokenless_guidance_and_missing_backend_fail_clearly(tmp_path):
    sampling = config(cfg_unconditional_prompt='')
    with EpisodeStore(tmp_path / 'episodes.db') as store:
        source = engine(sampling=sampling)
        identifier = save(store, source)
        for exact in (False, True):
            with pytest.raises(EditorError, match='produced no tokens'):
                engine(sampling=sampling, guidance_backend=PrefixBackend(tokenless=True),
                       initial_token_ids=source.initial_token_ids if exact else None)
        with pytest.raises(EditorError, match='produced no tokens'):
            _restore_engine(store, identifier, PrefixBackend(), max_tokens=None,
                            sampling_override=None, guidance_backend=PrefixBackend(tokenless=True))
    runtime = engine(sampling=config(cfg_unconditional_prompt=None), guidance_backend=None)
    runtime.sampling = config()
    with pytest.raises(EditorError, match='no unconditional guidance backend'):
        runtime.observe()


@pytest.mark.parametrize('boundary', [0, 1, 2])
@pytest.mark.invariant
def test_l04_live_forks_keep_the_cfg_window_at_each_boundary(boundary):
    live = LiveSession(engine(sampling=config(cfg_prefix_tokens=2)), prompt='conditional')
    for token in (7, 9):
        observation = live.engine.observe()
        live.generate(SelectRawRank(observation.statistics.raw_rank(token)))

    child = live.fork(boundary=boundary)

    child_engine = child.engine
    assert_context(child_engine)
    assert child_engine.visible_token_ids == [7, 9][:boundary]
    assert child_engine._cfg_active() == (boundary < 2)


@pytest.mark.current_workflow
def test_s03_l02_l07_cutoff_rewind_and_lazy_catchup():
    runtime = engine(sampling=config(cfg_prefix_tokens=2), max_tokens=2)
    guidance = runtime.guidance_backend
    assert guidance.work == []
    first = assert_context(runtime)
    assert runtime.observe() is first
    raw_token(runtime)
    assert_context(runtime)  # N-1
    runtime.apply(Write('y', mode='exact'))
    runtime.resume(max_tokens=4)
    assert_context(runtime)  # N, renewal does not restart CFG
    runtime.apply(Write('z', mode='exact'))
    assert_context(runtime)  # N+1
    assert guidance.work == [('reset', (1, 5)), ('eval', (7,))]
    runtime.sampling = replace(runtime.sampling, cfg_prefix_tokens=6)
    assert_context(runtime)
    assert guidance.work[-1] == ('eval', (9, 10))
    runtime.rewind_to(1)
    assert_context(runtime)
    runtime.apply(Write('x', mode='exact'))
    assert_context(runtime)
    assert guidance.work[-2:] == [('reset', (1, 5, 7)), ('eval', (8,))]


@pytest.mark.current_workflow
def test_s03_append_only_evaluation_and_sampler_changes():
    runtime = engine()
    guidance = runtime.guidance_backend
    assert_context(runtime)
    for name, value in [('temperature', 1.2), ('seed', 999), ('cfg_scale', .4), ('cfg_prefix_tokens', 8)]:
        runtime.sampling = replace(runtime.sampling, **{name: value})
        assert_context(runtime)
    assert guidance.work == [('reset', (1, 5))]
    raw_token(runtime)
    assert_context(runtime)
    runtime.apply(Write('xy', mode='exact'))
    assert_context(runtime)
    assert guidance.work[0] == ('reset', (1, 5))
    assert all(kind == 'eval' for kind, _ in guidance.work[1:])
    assert [t for _, ids in guidance.work[1:] for t in ids] == [7, 8, 9]
    evaluated_calls = len(guidance.work)
    runtime.sampling = replace(runtime.sampling, cfg_unconditional_prompt=None)
    runtime.apply(Write('z', mode='exact'))
    assert_context(runtime)
    assert len(guidance.work) == evaluated_calls
    runtime.sampling = replace(runtime.sampling, cfg_unconditional_prompt='U')
    assert_context(runtime)
    assert guidance.work[-1] == ('eval', (10,))
    runtime.sampling = replace(runtime.sampling, cfg_unconditional_prompt='B')
    assert_context(runtime, (1, 6, 5))
    assert guidance.work[-1] == ('reset', (1, 6, 5, 7, 8, 9, 10))


@pytest.mark.current_workflow
def test_s03_shared_guidance_invalidates_cached_observation():
    # Independent primary backends isolate guidance ownership itself. A's
    # ledger and cached observation are unchanged while B uses shared guidance.
    shared = PrefixBackend()
    a = engine(guidance_backend=shared)
    first = assert_context(a)
    b = engine(guidance_backend=shared, sampling=config(cfg_unconditional_prompt='B'))
    assert_context(b, (1, 6, 5))
    assert_context(a)
    assert a.observe() is not first
    assert shared.work == [('reset', (1, 5)), ('reset', (1, 6, 5)), ('reset', (1, 5))]


@pytest.mark.invariant
def test_l04_l06_root_and_sibling_switches():
    session = LiveSession(engine(), prompt='conditional')
    session.generate(Write('x', mode='exact'))
    root = session.branch.branch_id
    cached = assert_context(session.engine)
    roster = LiveSessionRoster(session)
    roster.new_root('other')
    other = roster.active_session
    other.set_sampler(config(cfg_unconditional_prompt='B'))
    other.generate(Write('y', mode='exact'))
    assert_context(other.engine, (1, 6, 5))
    roster.switch('#1')
    assert_context(session.engine)
    assert session.engine.observe() is not cached
    sibling = session.fork(boundary=0)
    session.activate(sibling.branch.branch_id)
    session.set_sampler(config(cfg_unconditional_prompt='B'))
    session.generate(Write('z', mode='exact'))
    assert_context(session.engine, (1, 6, 5))
    session.activate(root)
    assert session.engine.visible_token_ids == [8]
    assert_context(session.engine)


@pytest.mark.parametrize('resumed', [False, True])
@pytest.mark.invariant
def test_l01_fresh_root_never_inherits_guidance_continuation(tmp_path, resumed):
    source = engine()
    with EpisodeStore(tmp_path / 'episodes.db') as store:
        identifier = save(store, source)
        outcome = raw_token(source)
        store.record_action(identifier, 0, outcome)
        if resumed:
            source = _restore_engine(store, identifier, PrefixBackend(), max_tokens=None,
                                     sampling_override=None, guidance_backend=PrefixBackend())
        assert_context(source)
        fresh = fresh_root_from(source, 'other')
        assert fresh.boundary == 0
        assert fresh.visible_token_ids == []
        assert_context(fresh)


@pytest.mark.invariant
def test_r01_r08_l06_source_controls_follow_live_replay_and_rewind():
    unguided = config(cfg_unconditional_prompt=None)
    session = LiveSession(engine(sampling=unguided), prompt='conditional')
    a, b = config(), config(cfg_unconditional_prompt='B')
    plan = ReplayPlan(
        tuple(TapeStep(Write(text, mode='exact'), None) for text in ('x', 'y', 'z')),
        context=ReplayContext(sampling=(a, b, unguided)), final_sampling=b,
    )
    result = run_plan(session, divergence_policy="handoff", tape=plan)
    assert result.replay_exhausted
    assert_context(session.engine, (1, 6, 5))
    session.rewind(1)
    assert session.sampler == b
    assert_context(session.engine, (1, 6, 5))
    session.rewind(0)
    assert session.sampler == a
    assert_context(session.engine)


@pytest.mark.invariant
def test_l08_destination_tokenizer_owns_both_contexts(tmp_path):
    from trajectory_editor.episode_live_restore import model_change_session

    with EpisodeStore(tmp_path / 'episodes.db') as store:
        source = engine()
        identifier = save(store, source)
        store.record_action(identifier, 0, raw_token(source))
        destination = PrefixBackend(destination=True)
        child = model_change_session(
            store,
            identifier,
            destination,
            destination.provenance(),
            boundary=1,
            sampling=source.sampling,
            max_tokens=None,
            guidance_backend=PrefixBackend(destination=True),
        )
        assert child.engine.visible_token_ids == [10]
        assert_context(child.engine, (1, 11))
        fresh = EpisodeEngine(PrefixBackend(destination=True), initial_text='conditional',
                              sampling=config(), guidance_backend=PrefixBackend(destination=True))
        fresh.apply(Write('x', mode='exact'))
        np.testing.assert_array_equal(child.engine.observe().logits, fresh.observe().logits)


@pytest.mark.parametrize('scale', [0, 1, 1.7])
@pytest.mark.invariant
def test_s03_formula_and_conditional_only_hidden_controls(scale):
    runtime = engine(sampling=config(cfg_scale=scale, activation_vector=(.25,),
                                    activation_vector_strength=1,
                                    activation_vector_layer='control-vector',
                                    activation_vector_position='layers',
                                    activation_vector_digest='a' * 64,
                                    activation_vector_layer_start=1,
                                    activation_vector_layer_end=1))
    assert_context(runtime)
    assert runtime.backend.controls
    assert runtime.guidance_backend.controls == []


@pytest.mark.parametrize('final_only', [False, True])
@pytest.mark.invariant
def test_r08_ephemeral_setup_provisions_future_cfg(final_only):
    from trajectory_editor.episode_cli import build_parser
    from trajectory_editor.session_runtime import run_new_session
    plan = ReplayPlan(
        (TapeStep(Write('x', mode='exact'), None),),
        context=ReplayContext(sampling=(None if final_only else config(),)),
        final_sampling=config(),
    )
    args = build_parser().parse_args(['--ephemeral', '--model', 'fake', '--new-prompt',
                                     'conditional', '--plain-ui'])
    guidance = PrefixBackend()
    with patch('trajectory_editor.episode_backend_loader.load_backend', return_value=PrefixBackend()), \
         patch('trajectory_editor.episode_backend_loader.load_cfg_guidance_backend', return_value=guidance) as load:
        assert run_new_session(args, io=ScriptedIO(['q']), teacher_tape=SimpleNamespace(plan=plan)) == 0
    load.assert_called_once()
    if not final_only:
        assert guidance.work == [('reset', (1, 5))]


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.invariant
def test_r01_cli_replay_provisions_cfg_after_unguided_root(tmp_path, fixed):
    from trajectory_editor.episode_cli import main
    path = tmp_path / 'episodes.db'
    source = engine(sampling=config(cfg_unconditional_prompt=None))
    with EpisodeStore(path) as store:
        identifier = save(store, source)
        store.record_action(identifier, 0, source.apply(Write('x', mode='exact')))
        source.sampling = config(cfg_unconditional_prompt='B')
        store.record_sampling_segment(identifier, start_boundary=1, sampling=source.sampling,
                                      stream_fingerprint=source.stream_fingerprint)
        store.record_action(identifier, 1, source.apply(Write('y', mode='exact')))
    backend, guidance = PrefixBackend(), PrefixBackend()
    with patch('trajectory_editor.episode_backend_loader.load_episode_backend',
               return_value=(backend, backend.provenance(), False)), \
         patch('trajectory_editor.episode_backend_loader.load_cfg_guidance_backend', return_value=guidance) as load, \
         patch(
             'trajectory_editor.episode_cli.TerminalIO',
             return_value=ScriptedIO([f'save {path} replayed', 'q']),
         ):
        assert main(['--workspace', str(path), '--replay', identifier, '--episode-id', 'replayed',
                     '--plain-ui', *(['--fixed-config'] if fixed else [])]) == 0
    assert load.call_count == (0 if fixed else 1)
    if not fixed:
        assert guidance.tokens == [1, 6, 5, 8]
        np.testing.assert_array_equal(guidance.last_logits(), reference([1, 6, 5, 8]))
    with EpisodeStore(path) as store:
        assert [row['token_id'] for row in store.tokens('replayed')] == [8, 9]


@pytest.mark.current_workflow
@pytest.mark.parametrize('kind', ['llama', 'transformers'])
@pytest.mark.parametrize('cache', [True, False])
def test_s08_adapter_cfg_evaluation_preserves_cache_mode(kind, cache):
    """Real adapter reset/eval/tokenize/logits with instrumented model doubles."""
    import ctypes
    from trajectory_editor.decoder import LlamaCppDecoder
    from trajectory_editor.transformers_backend import TransformersBackend

    fake = PrefixBackend()
    submitted = []
    if kind == 'llama':
        class Model:
            def __init__(self):
                self._ctx = SimpleNamespace(ctx=1)
                self.tokens = []

            def reset(self):
                self.tokens = []

            def eval(self, ids):
                submitted.append(tuple(ids))
                self.tokens.extend(ids)
                self.logits = reference(self.tokens).astype(np.float32)

            def tokenize(self, text, **kwargs):
                return fake.tokenize(text.decode(), **kwargs)

            def detokenize(self, ids, **kwargs):
                return fake.render(ids, **kwargs).encode()

        guidance = object.__new__(LlamaCppDecoder)
        guidance._model = Model()
        guidance._llama_cpp = SimpleNamespace(llama_get_logits=lambda _: guidance._model.logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        guidance._fallback_eog_ids = {0}
    else:
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)
                self.ndim, self.shape = self.values.ndim, self.values.shape

            def __getitem__(self, key):
                return Tensor(self.values[key])

            def tolist(self):
                return self.values.tolist()

            def detach(self):
                return self

            def to(self, **kwargs):
                return Tensor(self.values.astype(kwargs.get('dtype', self.values.dtype)))

            def numpy(self):
                return self.values

        torch = SimpleNamespace(
            tensor=lambda values, **kw: Tensor(values),
            ones_like=lambda value: Tensor(np.ones(value.shape)),
            ones=lambda shape, **kw: Tensor(np.ones(shape)),
            arange=lambda start, end, **kw: Tensor(np.arange(start, end)),
            inference_mode=nullcontext, long=np.int64, float32=np.float32,
        )

        class Model:
            def __call__(self, input_ids, past_key_values=(), use_cache=False, **kwargs):
                ids = input_ids[0].tolist()
                submitted.append(tuple(ids))
                prefix = [*past_key_values, *ids]
                return SimpleNamespace(
                    logits=Tensor(reference(prefix).astype(np.float32).reshape(1, 1, -1)),
                    past_key_values=tuple(prefix) if use_cache else None,
                )

        guidance = object.__new__(TransformersBackend)
        guidance._model = Model()
        guidance._torch = torch
        guidance._input_device = 'cpu'
        guidance._context_limit = 100
        guidance._supports_logits_to_keep = False
        guidance._cache_active = False
        guidance._past_key_values = None
        guidance._eog_ids = {0}
        guidance._tokenizer = SimpleNamespace(
            bos_token_id=1, all_special_ids=[0, 1],
            encode=lambda text, **kw: fake.tokenize(text, special=True),
            decode=lambda ids, **kw: fake.render(ids),
        )
    guidance._tokens = []
    guidance._speculation_prefix = None
    guidance._speculation_logits = None
    guidance._vocabulary_size = 12
    guidance._cache_enabled = cache
    runtime = engine(guidance_backend=guidance)
    for boundary in range(3):
        observation = runtime.observe()
        expected_u = reference([1, 5, *runtime.visible_token_ids]).astype(np.float32).astype(float)
        expected_c = reference(runtime.token_ids)
        np.testing.assert_allclose(observation.logits, expected_u + 1.7 * (expected_c - expected_u))
        assert runtime.observe() is observation
        if boundary < 2:
            raw_token(runtime, 7 + boundary)
    assert submitted == ([(1, 5), (7,), (8,)] if cache else
                         [(1, 5), (1, 5, 7), (1, 5, 7, 8)])
