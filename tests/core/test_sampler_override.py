"""The ``s`` command edits the complete scalar sampler surface."""

from __future__ import annotations

import argparse

import pytest

from trajectory_editor.core.cli_config import (
    add_core_sampler_arguments,
    sampler_from_args,
    sampler_override,
    sampler_overrides_from_args,
    sampler_overrides_present,
)
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


def test_student_t_df_is_available_from_cli_and_live_sampler_edits():
    parser = argparse.ArgumentParser()
    add_core_sampler_arguments(parser, include_vector=False)
    args = parser.parse_args(
        ["--draw-kernel", "student-t-max", "--student-t-df", "1"]
    )
    assert sampler_overrides_present(args)
    assert sampler_overrides_from_args(args) == {
        "draw_kernel": "student-t-max",
        "gumbel_top_k": None,
        "student_t_df": 1.0,
    }
    assert sampler_from_args(args).student_t_df == 1.0

    live = sampler_override(
        SamplerConfig(draw_kernel="student-t-max"), "student_t_df=1.5"
    )
    assert live.student_t_df == 1.5
