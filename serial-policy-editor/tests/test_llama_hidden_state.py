from __future__ import annotations

import ctypes

import numpy as np
import pytest

from trajectory_editor.decoder import LlamaCppDecoder


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
    assert capabilities["layer_count"] == 3
    assert capabilities["final_layer_policy"] == "excluded-from-control-vector-runtime"
