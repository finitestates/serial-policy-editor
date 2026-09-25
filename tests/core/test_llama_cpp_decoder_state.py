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
        self._ctx = SimpleNamespace(ctx=context, kv_cache_seq_rm=self.kv_cache_seq_rm)
        self.tokens = [7]
        self.n_tokens = len(self.tokens)
        self._requires_eval = False

    def kv_cache_seq_rm(self, sequence: int, start: int, end: int) -> bool:
        if sequence != -1 or start < 0 or end != -1:
            return False
        self.tokens = self.tokens[:start]
        self.n_tokens = len(self.tokens)
        return True

    def eval(self, token_ids: list[int]) -> None:
        self.tokens.extend(token_ids)
        self.n_tokens = len(self.tokens)
        if self.tokens[-1] == 1:
            self._ctx.ctx.set_logits((0.0, 10.0, 5.0))
        else:
            self._ctx.ctx.set_logits((10.0, 5.0, 0.0))


def decoder_for(model: FakeLlamaModel) -> LlamaCppDecoder:
    decoder = object.__new__(LlamaCppDecoder)
    decoder._model = model
    decoder._llama_cpp = SimpleNamespace(
        llama_get_logits_ith=lambda ctx, index: ctx.logits_buffer
    )
    decoder._tokens = [7]
    decoder._vocabulary_size = 3
    decoder._cache_enabled = True
    decoder._real_model_probe = None
    decoder._restored_logits = None
    decoder._last_logits_cache = None
    decoder._speculation_prefix = None
    decoder._speculation_logits = None
    return decoder


def test_rollback_speculation_restores_logits_used_for_rank_resolution() -> None:
    model = FakeLlamaModel(FakeContext())
    decoder = decoder_for(model)

    prefix_logits = decoder.last_logits()
    assert decoder.speculate(1)
    assert decoder._tokens == [7, 1]
    assert model.tokens == [7, 1]

    decoder.rollback_speculation()
    assert decoder._tokens == [7]
    assert model.tokens == [7]
    assert decoder.last_logits().tolist() == prefix_logits.tolist()
    assert int(np.argmax(decoder.last_logits())) == 0

    assert decoder.speculate(1)
    decoder.commit_speculation()
    assert decoder._tokens == [7, 1]
    assert int(np.argmax(decoder.last_logits())) == 1
