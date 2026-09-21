"""Bounded real-backend release checks.

Set ``SPE_LLAMA_SMOKE_MODEL`` to a local GGUF to run this file. It deliberately
keeps one scenario and closes every decoder so it can run serially with the
other llama smoke tests.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trajectory_editor.activation_vectors import SteeringVectorArtifact
from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Accept, Phrase
from trajectory_editor.episode_engine import EpisodeEngine, InstructionRejected


pytestmark = pytest.mark.llama_smoke


@pytest.fixture(scope="module")
def model() -> Path:
    value = os.environ.get("SPE_LLAMA_SMOKE_MODEL")
    if not value:
        pytest.skip("Set SPE_LLAMA_SMOKE_MODEL to a local GGUF to run release checks")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.skip(f"Configured smoke model does not exist: {path}")
    pytest.importorskip("llama_cpp")
    return path


def _backend(model: Path) -> LlamaCppDecoder:
    return LlamaCppDecoder(
        model,
        LlamaCppSettings(
            n_ctx=128,
            n_batch=32,
            n_gpu_layers=0,
            n_threads=2,
            n_threads_batch=2,
        ),
    )


def _text_token(engine: EpisodeEngine, token_id: int) -> str:
    text = engine.backend.token_text(token_id)
    return text if text and not engine.backend.is_eog(token_id) else ""


def test_real_llama_p0_state_paths_and_sampling_invariants(model: Path):
    primary = _backend(model)
    guidance = _backend(model)
    try:
        prefix = primary.tokenize(
            "A short factual answer:", add_bos=True, special=False
        )
        sampling = SamplingConfig(
            temperature=0.9,
            top_k=64,
            top_p=0.96,
            min_p=0.0,
            typical_p=0.95,
            tail_free_z=0.95,
            draw_kernel="gumbel-max",
            seed=41,
            cfg_unconditional_prompt="A short answer:",
            cfg_scale=1.25,
            cfg_prefix_tokens=2,
        )
        engine = EpisodeEngine(
            primary,
            guidance_backend=guidance,
            initial_token_ids=prefix,
            sampling=sampling,
        )

        for consumed in (0, 1):
            observation = engine.observe()
            assert observation.distribution.ids.size > 0
            assert observation.statistics.candidate_filter_diagnostics["draw_kernel"] == "gumbel-max"
            engine.apply(Accept())
            assert engine.boundary == consumed + 1
        assert not engine._cfg_active()
        post_cfg = engine.observe()
        assert post_cfg.distribution.ids.size > 0

        # The branch path must restore the exact intended prefix.
        engine.apply(Accept())
        engine.rewind_to(1)
        assert tuple(primary._tokens) == tuple(engine.token_ids)
        engine.resume(max_tokens=None)
        resumed = engine.observe()
        assert resumed.prefix_token_ids == tuple(engine.token_ids)
        engine.apply(Accept())

        phrase_engine = EpisodeEngine(
            primary,
            initial_token_ids=prefix,
            sampling=replace_sampling_without_cfg(sampling),
        )
        first_phrase_observation = phrase_engine.observe()
        top_ids = first_phrase_observation.statistics.top_raw_ids(64)
        top_text = next(
            (_text_token(phrase_engine, token_id) for token_id in top_ids), ""
        )
        assert top_text
        checked = phrase_engine.apply(
            Phrase(top_text, mode="exact", max_shift=0.0)
        )
        assert checked.resolved_token_ids
        assert checked.diagnostics["tokens"]
        assert phrase_engine._ephemeral_logit_biases == {}

        # A low-ranked token must fail the dry-run without leaving partial state;
        # the force path then commits the same phrase through a temporary bias.
        phrase_engine.rewind_to(0)
        low_text = next(
            (
                _text_token(phrase_engine, token_id)
                for token_id in reversed(top_ids)
                if _text_token(phrase_engine, token_id) != top_text
            ),
            "",
        )
        assert low_text
        before_failure = tuple(phrase_engine.token_ids)
        with pytest.raises(InstructionRejected, match="check phrase rejected"):
            phrase_engine.apply(Phrase(low_text, mode="exact", max_shift=0.0))
        assert tuple(phrase_engine.token_ids) == before_failure
        assert phrase_engine._ephemeral_logit_biases == {}
        forced = phrase_engine.apply(
            Phrase(low_text, mode="exact", force=True, max_shift=0.0)
        )
        assert forced.resolved_token_ids
        assert phrase_engine._ephemeral_logit_biases == {}

        # Capture a portable vector, install it, then verify clear + reset
        # returns the same prefix logits. The selected range is runtime-valid.
        artifact = SteeringVectorArtifact.from_hidden_state_prompt_pair(
            primary,
            primary.provenance(),
            "A calm lake.",
            "A crowded city.",
            layer_start=2,
            layer_end=2,
            capture_position="last",
        )
        artifact.validate_against_backend(primary, primary.provenance())
        primary.clear_activation_control_vector()
        primary.reset(prefix)
        baseline = primary.last_logits().copy()
        primary.set_activation_control_vector(
            artifact.vector,
            layer_start=artifact.layer_start,
            layer_end=artifact.layer_end,
            strength=artifact.strength,
        )
        primary.reset(prefix)
        steered = primary.last_logits().copy()
        assert np.isfinite(steered).all()
        assert not np.array_equal(steered, baseline)
        primary.clear_activation_control_vector()
        primary.reset(prefix)
        restored = primary.last_logits().copy()
        np.testing.assert_allclose(restored, baseline, rtol=1e-5, atol=1e-5)
        with pytest.raises(RuntimeError, match="outside the loaded model runtime range"):
            primary.set_activation_control_vector(
                artifact.vector, layer_start=1, layer_end=2, strength=1.0
            )
    finally:
        primary.close()
        guidance.close()


def replace_sampling_without_cfg(sampling: SamplingConfig) -> SamplingConfig:
    """Preserve advanced sampler settings while disabling CFG for phrase checks."""

    return replace(
        sampling,
        cfg_unconditional_prompt=None,
        cfg_scale=1.0,
        cfg_prefix_tokens=0,
    )
