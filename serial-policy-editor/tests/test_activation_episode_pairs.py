from __future__ import annotations

import json

import numpy as np

from tests.fakes import ConformingFakeBackend
from trajectory_editor.activation_vectors import ActivationVectorArtifact
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.vector_cli import main


class EpisodeActivationBackend(ConformingFakeBackend):
    snapshots = {
        "POS-1": np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        "NEG-1": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "POS-2": np.asarray([4.0, 1.0, 0.0], dtype=np.float32),
        "NEG-2": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "POS\nline\\tail\t": np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        "NEG": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    }

    def activation_width(self) -> int:
        return 3

    def activation_snapshot(self, text, *, layer="output", position="last"):
        assert layer == "output"
        assert position == "last"
        return self.snapshots[text]

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {
            "backend": "fake",
            "adapter": "episode-pair-test",
            "vocabulary_size": 8,
        }


def _episode(store: EpisodeStore, episode_id: str, text: str) -> None:
    store.create_episode(
        episode_id=episode_id,
        initial_text=text,
        initial_token_ids=[7],
        sampling=SamplingConfig(temperature=1.0),
        stream_fingerprint="0" * 64,
        coordinate_offset=0,
        max_tokens=None,
        backend={"backend": "fake", "vocabulary_size": 8},
    )


def test_prompt_pairs_average_raw_differences_before_normalizing():
    backend = EpisodeActivationBackend()
    artifact = ActivationVectorArtifact.from_prompt_pairs(
        backend,
        backend.provenance(),
        [("POS-1", "NEG-1"), ("POS-2", "NEG-2")],
    )

    assert artifact.vector == (1.0, 0.0, 0.0)
    assert artifact.method == "prompt-pairs-mean-v1"
    assert artifact.source["pair_count"] == 2
    assert artifact.source["pair_delta_norms"] == [1.0, 3.0]


def test_activation_derive_uses_episode_provenance(tmp_path, monkeypatch):
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        _episode(store, "positive-1", "POS-1")
        _episode(store, "negative-1", "NEG-1")
        _episode(store, "positive-2", "POS-2")
        _episode(store, "negative-2", "NEG-2")

    output = tmp_path / "derived.json"
    monkeypatch.setattr(
        "trajectory_editor.vector_cli.create_backend",
        lambda *args, **kwargs: EpisodeActivationBackend(),
    )
    assert main([
        "activation", "derive",
        "--workspace", str(workspace),
        "--positive", "positive-1", "positive-2",
        "--negative", "negative-1", "negative-2",
        "--model", "fake.model",
        "--output", str(output),
    ]) == 0

    artifact = ActivationVectorArtifact.from_path(output)
    assert artifact.vector == (1.0, 0.0, 0.0)
    assert artifact.source["positive"][0]["episode_id"] == "positive-1"
    assert artifact.source["negative"][1]["episode_id"] == "negative-2"
    assert artifact.source["positive"][0]["text_sha256"]


def test_activation_export_pairs_escapes_cvector_prompt_lines(tmp_path, capsys):
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        _episode(store, "positive", "POS\nline\\tail\t")
        _episode(store, "negative", "NEG")

    output_dir = tmp_path / "cvector-input"
    assert main([
        "activation", "export-pairs",
        "--workspace", str(workspace),
        "--positive", "positive",
        "--negative", "negative",
        "--output-dir", str(output_dir),
    ]) == 0

    assert (output_dir / "positive.txt").read_text() == "POS\\nline\\\\tail\\t\n"
    assert (output_dir / "negative.txt").read_text() == "NEG\n"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["format"] == "spe-activation-pairs-v1"
    assert manifest["pair_count"] == 1
    assert manifest["pairs"][0]["positive"]["episode_id"] == "positive"
    assert "positive.txt" in capsys.readouterr().out


def test_activation_pair_lists_must_be_paired(tmp_path, capsys):
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        _episode(store, "positive", "POS-1")
        _episode(store, "negative", "NEG-1")

    assert main([
        "activation", "export-pairs",
        "--workspace", str(workspace),
        "--positive", "positive", "positive",
        "--negative", "negative",
        "--output-dir", str(tmp_path / "pairs"),
    ]) == 2
    assert "same length" in capsys.readouterr().err
