from __future__ import annotations

import ctypes

import numpy as np
import pytest

from trajectory_editor.decoder import LlamaCppDecoder

pytestmark = pytest.mark.optional


class _FakeContext:
    ctx = ctypes.c_void_p(17)


class _FakeModel:
    _ctx = _FakeContext()


class _Binding:
    def __init__(self):
        self.calls = []

    def llama_set_adapter_cvec(self, context, pointer, size, width, start, end):
        values = None if pointer is None else np.ctypeslib.as_array(
            pointer, shape=(int(size),)
        ).copy()
        self.calls.append((context, values, int(width), int(start), int(end)))
        return 0


def _decoder():
    decoder = object.__new__(LlamaCppDecoder)
    decoder._model = _FakeModel()
    decoder._llama_cpp = _Binding()
    decoder.activation_control_vector_width = lambda: 2
    decoder.activation_control_vector_layer_count = lambda: 4
    return decoder


def test_control_vector_maps_canonical_layers_to_native_slots():
    decoder = _decoder()

    decoder.set_activation_control_vector(
        np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        layer_start=2,
        layer_end=3,
        strength=0.5,
    )

    assert len(decoder._llama_cpp.calls) == 1
    context, values, width, native_start, native_end = decoder._llama_cpp.calls[0]
    assert context is decoder._model._ctx.ctx
    assert width == 2
    assert native_start == 1
    assert native_end == 2
    np.testing.assert_allclose(values, [1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def test_control_vector_rejects_unaddressable_first_block_output():
    decoder = _decoder()
    with pytest.raises(RuntimeError, match="canonical layer range"):
        decoder.set_activation_control_vector(
            np.zeros(8, dtype=np.float32),
            layer_start=1,
            layer_end=2,
            strength=1.0,
        )

    assert decoder._llama_cpp.calls == []
