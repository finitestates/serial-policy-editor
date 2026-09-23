from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fakes import ScriptedIO
from trajectory_editor.episode_backend_loader import (
    load_backend,
    load_episode_backend,
)
from trajectory_editor.episode_cli import build_parser

pytestmark = pytest.mark.current_workflow

class _LoadedBackend:
    def provenance(self, *, include_model_sha256: bool):
        assert include_model_sha256 is True
        return {"backend": "transformers", "model_family": "fake"}


def test_load_backend_maps_launch_arguments_to_backend_factory_settings():
    args = build_parser(include_vector=False).parse_args(
        [
            "--model",
            "fake-model",
            "--backend",
            "transformers",
            "--cache",
            "off",
            "--transformers-device",
            "cpu",
            "--n-ctx",
            "4096",
        ]
    )
    sentinel = object()

    with patch(
        "trajectory_editor.episode_backend_loader.create_backend",
        return_value=sentinel,
    ) as create:
        assert load_backend(args) is sentinel

    assert create.call_args.args == ("transformers", Path("fake-model"))
    assert create.call_args.kwargs["cache_mode"] == "off"
    assert create.call_args.kwargs["llama_settings"].n_ctx == 4096
    assert create.call_args.kwargs["transformers_settings"].device == "cpu"


def test_load_episode_backend_restores_saved_launch_options_before_loading(tmp_path):
    args = build_parser(include_vector=False).parse_args([])
    args._explicit_options = set()
    model = tmp_path / "saved-model"
    source = {
        "backend": {
            "backend": "transformers",
            "model_path": str(model),
            "load_options": {
                "transformers_device": "cpu",
                "transformers_dtype": "float32",
                "cache": "off",
            },
        }
    }

    with patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        return_value=_LoadedBackend(),
    ) as load:
        backend, provenance, changed = load_episode_backend(
            args,
            source,
            ScriptedIO([]),
        )

    selected = load.call_args.args[0]
    assert isinstance(backend, _LoadedBackend)
    assert changed is False
    assert selected.model == model
    assert selected.backend == "transformers"
    assert selected.transformers_device == "cpu"
    assert selected.transformers_dtype == "float32"
    assert selected.cache == "off"
    assert provenance["model_path"] == str(model.resolve())
    assert provenance["load_options"]["transformers_device"] == "cpu"
