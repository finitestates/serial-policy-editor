"""The ``s`` command edits the complete scalar sampler surface."""

from __future__ import annotations

import pytest

from trajectory_editor.core.cli_config import sampler_override
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig

pytestmark = pytest.mark.current_workflow


def test_sampler_override_includes_seed_in_the_sampler_record():
    config = SamplerConfig(seed=7)
    updated = sampler_override(config, "seed=42")
    assert updated.seed == 42
    assert updated.temperature == config.temperature


def test_sampler_override_rejects_bare_random_seed_forms():
    config = SamplerConfig(seed=7)
    for raw in ("random-seed", "random", "RANDOM-SEED"):
        with pytest.raises(EditorError):
            sampler_override(config, raw)


def test_sampler_override_still_edits_other_fields():
    config = SamplerConfig(seed=7)
    updated = sampler_override(config, "temperature=0.5 top_k=20")
    assert updated.temperature == 0.5
    assert updated.top_k == 20
    assert updated.seed == 7
