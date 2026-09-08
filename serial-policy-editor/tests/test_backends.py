from __future__ import annotations

import pytest

from trajectory_editor.decoder import LlamaCppSettings
from trajectory_editor.domain import EditorError
from trajectory_editor.transformers_backend import TransformersSettings


def test_llama_settings_do_not_own_sampler_seed() -> None:
    from dataclasses import asdict

    assert "seed" not in asdict(LlamaCppSettings())


def test_transformers_full_prefix_settings_validate() -> None:
    value = TransformersSettings(device="cpu", dtype="float32")
    assert value.device == 'cpu'
    with pytest.raises(EditorError):
        TransformersSettings(dtype="float128")
