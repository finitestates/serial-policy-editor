from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np

from trajectory_editor.decoder import LlamaCppDecoder


class FakeContext:
    def __init__(self) -> None:
        self.logits_buffer = (ctypes.c_float * 3)(10.0, 5.0, 0.0)

    def set_logits(self, values: tuple[float, float, float]) -> None:
        self.logits_buffer = (ctypes.c_float * 3)(*values)


class FakeLlamaModel:
    def __init__(self, context: FakeContext) -> None:
        self._ctx = SimpleNamespace(ctx=context)
        self.tokens = [7]

    def save_state(self) -> tuple[int, ...]:
        return tuple(self.tokens)

    def load_state(self, state: tuple[int, ...]) -> None:
        # Model the binding behavior under test: KV/token state is restored,
        # while the low-level output buffer still contains the last eval.
        self.tokens = list(state)

    def eval(self, token_ids: list[int]) -> None:
        self.tokens.extend(token_ids)
        if self.tokens[-1] == 1:
            self._ctx.ctx.set_logits((0.0, 10.0, 5.0))
        else:
            self._ctx.ctx.set_logits((10.0, 5.0, 0.0))


def test_restore_state_restores_logits_used_for_rank_resolution() -> None:
    context = FakeContext()
    model = FakeLlamaModel(context)
    decoder = object.__new__(LlamaCppDecoder)
    decoder._model = model
    decoder._llama_cpp = SimpleNamespace(
        llama_get_logits_ith=lambda ctx, index: ctx.logits_buffer
    )
    decoder._tokens = [7]
    decoder._snapshot_token = object()
    decoder._vocabulary_size = 3
    decoder._cache_enabled = True
    decoder._real_model_probe = None
    decoder._restored_logits = None

    prefix_logits = decoder.last_logits()
    snapshot = decoder.snapshot_state()
    assert snapshot is not None

    decoder.eval([1])
    assert int(np.argmax(decoder.last_logits())) == 1

    assert decoder.restore_state(snapshot)
    decoder.branch_to_prefix([7])  # Chord's same-prefix activation is a no-op.
    assert decoder.last_logits().tolist() == prefix_logits.tolist()
    assert int(np.argmax(decoder.last_logits())) == 0

    decoder.eval([1])
    assert int(np.argmax(decoder.last_logits())) == 1
