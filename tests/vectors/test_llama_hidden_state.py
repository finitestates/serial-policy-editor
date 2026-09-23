from __future__ import annotations

import ctypes

import numpy as np
import pytest

from trajectory_editor.decoder import LlamaCppDecoder

pytestmark = pytest.mark.optional

class _FakeBatch:
    def __init__(self):
        self.tokens = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.tokens = []

    def add_sequence(self, tokens, sequence_id, logits):
        del sequence_id, logits
        self.tokens = list(tokens)


class _FakeContext:
    ctx = ctypes.c_void_p(17)

    def __init__(self):
        self.clear_calls = 0
        self.decode_calls = 0

    def kv_cache_clear(self):
        self.clear_calls += 1

    def decode(self, batch):
        assert batch.tokens
        self.decode_calls += 1


class _FakeEmbeddingModel:
    n_batch = 8

    def __init__(self):
        self._ctx = _FakeContext()
        self._batch = _FakeBatch()
        self.reset_calls = 0

    def tokenize(self, text, *, add_bos, special):
        assert text == b"prompt"
        assert add_bos is True
        assert special is False
        return [101, 102, 103]

    def n_embd(self):
        return 2

    def reset(self):
        self.reset_calls += 1


def _decoder_with_capture_buffers(buffers):
    decoder = object.__new__(LlamaCppDecoder)
    model = _FakeEmbeddingModel()
    enabled = []
    arrays = {
        layer: np.asarray(values, dtype=np.float32)
        for layer, values in buffers.items()
    }
    pointers = {
        layer: values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for layer, values in arrays.items()
    }

    def setter(context, layer, value):
        assert context is model._ctx.ctx
        enabled.append((int(layer), bool(value)))

    def getter(context, layer):
        assert context is model._ctx.ctx
        return pointers.get(int(layer))

    decoder._activation_model = model
    decoder._activation_embedding_model = lambda: model
    decoder._hidden_state_capture_symbols = lambda: (setter, getter)
    decoder._model_layer_count = lambda: 4
    decoder.activation_width = lambda: 2
    return decoder, model, enabled


def test_llama_hidden_state_snapshot_captures_positions_and_cleans_up():
    decoder, model, enabled = _decoder_with_capture_buffers({
        2: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    })

    first = decoder.hidden_state_snapshot("prompt", layer=2, position="first")
    last = decoder.hidden_state_snapshot("prompt", layer=2, position="current")
    all_rows = decoder.hidden_state_snapshot("prompt", layer=2, position="all")

    assert first == pytest.approx([1.0, 2.0])
    assert last == pytest.approx([5.0, 6.0])
    assert all_rows.shape == (3, 2)
    assert np.isfinite(all_rows).all()
    assert enabled == [(2, True), (2, False), (2, True), (2, False), (2, True), (2, False)]
    assert model._ctx.decode_calls == 3
    assert model.reset_calls == 3


def test_llama_hidden_state_range_captures_all_selected_layers_in_one_eval():
    decoder, model, enabled = _decoder_with_capture_buffers({
        1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        2: [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
    })

    captured = decoder.hidden_state_snapshots(
        "prompt", layer_start=1, layer_end=2, position="last"
    )

    assert captured[1] == pytest.approx([5.0, 6.0])
    assert captured[2] == pytest.approx([50.0, 60.0])
    assert model._ctx.decode_calls == 1
    assert enabled == [(1, True), (2, True), (1, False), (2, False)]


def test_llama_hidden_state_capture_rejects_missing_layer_data_and_cleans_up():
    decoder, model, enabled = _decoder_with_capture_buffers({
        1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    })

    with pytest.raises(RuntimeError, match="no hidden-state capture for layer 2"):
        decoder.hidden_state_snapshots(
            "prompt", layer_start=1, layer_end=2, position="last"
        )

    assert enabled == [(1, True), (2, True), (1, False), (2, False)]
    assert model.reset_calls == 1


def test_llama_hidden_state_coordinate_excludes_unsteerable_final_layer():
    decoder, _, _ = _decoder_with_capture_buffers({1: [[1.0, 2.0]]})

    capabilities = decoder.hidden_state_capabilities()

    assert capabilities["site"] == "decoder-block-output-residual"
    assert capabilities["layer_numbering"] == "one-based"
    assert capabilities["layer_count"] == 4
    assert capabilities["capture_coordinate"] == "canonical block-output N <- native input tap N"
    assert capabilities["injection_coordinate"] == "canonical block-output N -> native cvector slot N-1"
    assert capabilities["capture_layer_range"] == [1, 3]
    assert capabilities["runtime_layer_range"] == [2, 4]
    assert capabilities["final_layer_policy"] == "worker-graph-output-callback"


def test_llama_control_vector_maps_canonical_layers_to_native_slots():
    decoder, model, _ = _decoder_with_capture_buffers({})
    calls = []

    class Binding:
        def llama_set_adapter_cvec(self, context, pointer, size, width, start, end):
            values = np.ctypeslib.as_array(pointer, shape=(int(size),)).copy()
            calls.append((context, values, int(width), int(start), int(end)))
            return 0

    decoder._model = model
    decoder._llama_cpp = Binding()

    decoder.set_activation_control_vector(
        np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        layer_start=2,
        layer_end=3,
        strength=0.5,
    )

    assert len(calls) == 1
    context, values, width, native_start, native_end = calls[0]
    assert context is model._ctx.ctx
    assert width == 2
    assert native_start == 1
    assert native_end == 2
    np.testing.assert_allclose(values, [1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def test_llama_control_vector_rejects_unaddressable_first_block_output():
    decoder, _, _ = _decoder_with_capture_buffers({})
    decoder._model = _FakeEmbeddingModel()

    class Binding:
        def llama_set_adapter_cvec(self, *args):
            raise AssertionError("setter must not be called")

    decoder._llama_cpp = Binding()
    with pytest.raises(RuntimeError, match="canonical layer range"):
        decoder.set_activation_control_vector(
            np.zeros(8, dtype=np.float32),
            layer_start=1,
            layer_end=2,
            strength=1.0,
        )
