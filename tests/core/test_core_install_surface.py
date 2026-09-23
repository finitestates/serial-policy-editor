"""The core package must remain usable without research extensions."""

from __future__ import annotations

import importlib.abc
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.current_workflow

def test_core_install_surface_without_research_modules(tmp_path):
    """Exercise the reduced install surface in a fresh interpreter import graph."""

    # A subprocess is important here: the normal test process may already have
    # imported research modules for other tests.
    script = r'''
import importlib.abc
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Mirror a clean core-only install instead of importing the compatibility
# package at the repository root, which also exposes optional source trees.
repo_root = Path.cwd()
sys.path[:] = [entry for entry in sys.path if entry not in ("", str(repo_root))]
sys.path.insert(0, str(repo_root / "core" / "src"))
sys.meta_path[:] = [
    finder for finder in sys.meta_path if "EditableFinder" not in repr(finder)
]

import numpy as np

blocked = {
    "trajectory_editor.domain",
    "trajectory_editor.bias_catalog",
    "trajectory_editor.bias_presets",
    "trajectory_editor.lexical_reference",
    "trajectory_editor.token_preference_features",
    "trajectory_editor.token_preference",
    "trajectory_editor.online_learning",
    "trajectory_editor.learning_readout",
    "trajectory_editor.learning_controls",
    "trajectory_editor.learning_observation",
    "trajectory_editor.controller_pipeline",
    "trajectory_editor.episode_policy",
    "trajectory_editor.vector_artifacts",
    "trajectory_editor.vector_impact",
    "trajectory_editor.trajectory_compare",
}


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or any(fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError(f"optional module blocked: {fullname}")
        return None


sys.meta_path.insert(0, Blocker())


class Backend:
    pieces = {0: "<EOG>", 1: "P", 2: " B"}

    def __init__(self):
        self.tokens = []

    def reset(self, token_ids):
        self.tokens = list(token_ids)

    def eval(self, token_ids):
        self.tokens.extend(token_ids)

    def last_logits(self):
        values = np.full(3, -10.0, dtype=np.float32)
        values[2] = 10.0
        values[0] = -9.0
        return values

    def vocabulary_size(self):
        return 3

    def tokenize(self, text, *, add_bos=False, special=False):
        del special
        return [1] if add_bos or text == "P" else [2]

    def render(self, token_ids, *, special=False):
        del special
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def token_text(self, token_id):
        return self.pieces[token_id]

    def is_eog(self, token_id):
        return token_id == 0

    def eog_token_ids(self):
        return (0,)

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {"backend": "smoke", "vocabulary_size": 3}


class DivergingBackend(Backend):
    def last_logits(self):
        values = np.full(3, -10.0, dtype=np.float32)
        values[1] = 10.0
        values[0] = -9.0
        return values


from trajectory_editor import EpisodeEngine, EpisodeRunner, EpisodeStore, SamplerConfig
from trajectory_editor.core.actions import Accept
from trajectory_editor.core.sampler_config import SAMPLING_POLICY_SCHEME
from trajectory_editor.core.sampling import RNG_SCHEME
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_runner import TapeStep
from trajectory_editor.projector import project_episode

exec("from trajectory_editor import *", {})

help_output = io.StringIO()
try:
    with redirect_stdout(help_output):
        main(["-h"], include_vector=False)
except SystemExit as exc:
    assert exc.code == 0
help_text = help_output.getvalue()
assert "--online-learning" not in help_text
assert "--steering-vector" not in help_text
assert "--reference" not in help_text

# The core install can load a portable steering artifact, but it does not
# contain the optional vector production/analysis commands or research modules.
default_help = io.StringIO()
try:
    with redirect_stdout(default_help):
        main(["-h"])
except SystemExit as exc:
    assert exc.code == 0
assert "--steering-vector" in default_help.getvalue()
assert importlib.util.find_spec("trajectory_editor.vector_cli") is None
assert importlib.util.find_spec("trajectory_editor.research") is None
assert importlib.util.find_spec("trajectory_editor.output_head_vectors") is None

workspace = Path(r"__WORKSPACE__")
artifact_path = workspace.with_name("steering.json")
artifact_path.write_text(json.dumps({
    "format": "spe-steering-vector-v1",
    "schema_version": 2,
    "kind": "output-head-steering-vector",
    "model": {},
    "strength": 1.0,
    "method": "core-smoke",
    "vector": [0.25],
    "compatibility": {"model_identity": "metadata-only", "hash_algorithm": None},
}), encoding="utf-8")
from trajectory_editor.activation_vectors import SteeringVectorArtifact
assert not hasattr(SteeringVectorArtifact, "from_prompt_pair")
assert not hasattr(SteeringVectorArtifact, "from_prompt_pairs")
assert not hasattr(SteeringVectorArtifact, "from_hidden_state_prompt_pair")

loaded_vector = SteeringVectorArtifact.from_path(artifact_path)
assert loaded_vector.kind == "output-head-steering-vector"
assert loaded_vector.vector == (0.25,)

# Core preserves an externally supplied vector even when its layer metadata
# cannot establish alignment.  The active backend remains responsible for
# accepting or rejecting the vector when it is installed.
layer_mismatch_path = workspace.with_name("layer-mismatch.json")
layer_mismatch_path.write_text(json.dumps({
    "format": "spe-steering-vector-v1",
    "schema_version": 2,
    "kind": "hidden-state-vector",
    "model": {"hidden_state_width": 3, "hidden_state_layer_count": 4},
    "target": {
        "site": "decoder-block-output-residual",
        "layer_numbering": "one-based",
        "coordinate": "canonical-decoder-block-output-v1",
    },
    "layer_start": 1,
    "layer_end": 4,
    "position": "layers",
    "strength": 1.0,
    "method": "external-smoke",
    "vector": [0.25, 0.5],
    "compatibility": {"model_identity": "metadata-only", "hash_algorithm": None},
}), encoding="utf-8")
layer_mismatch = SteeringVectorArtifact.from_path(layer_mismatch_path)
assert layer_mismatch.dimension == 2
assert layer_mismatch.layer_start == 1
assert layer_mismatch.layer_end == 4

# A historical sampler record may be wider than the current core contract.
# Its unknown research fields are ignored; its replayable fields still load.
legacy = SamplerConfig.from_record({
    "temperature": 0.8,
    "policy_scheme": SAMPLING_POLICY_SCHEME,
    "rng_scheme": RNG_SCHEME,
    "history_scope": "model-visible-prefix-tail-v1",
    "group_controls": [{"research": "ignored"}],
    "token_preference_vector": [1, 2, 3],
    "reference_prior_routes": [],
})
assert legacy.temperature == 0.8

backend = Backend()
source_engine = EpisodeEngine(backend, sampling=SamplerConfig(), initial_token_ids=[1])
with EpisodeStore(workspace) as store:
    source = store.create_episode(
        initial_text="P",
        initial_token_ids=[1],
        sampling=source_engine.sampling,
        stream_fingerprint=source_engine.stream_fingerprint,
        coordinate_offset=source_engine.coordinate_offset,
        max_tokens=None,
        backend=backend.provenance(),
    )
    legacy_record = source_engine.sampling.to_dict()
    legacy_record.update({
        "group_controls": [{"research": "ignored"}],
        "token_preference_vector": [1, 2, 3],
        "reference_prior_routes": [],
    })
    store.connection.execute(
        "UPDATE sampler_segments SET sampling_json = ? WHERE episode_id = ?",
        (json.dumps(legacy_record), source),
    )
    store.connection.commit()
    assert store.sampling_segment(source, 0)["sampling"]["group_controls"]
    outcome = source_engine.apply(Accept())
    store.record_action(source, 0, outcome)
    store.update_episode(source, visible_text=source_engine.text, max_tokens=None, status="open")
    from trajectory_editor.episode_replay_source import replay_procedure
    step = replay_procedure(store, source)[0]
    action, expectation = step["action"], step["expectation"]

    handoff_backend = DivergingBackend()
    handoff_engine = EpisodeEngine(handoff_backend, sampling=SamplerConfig(), initial_token_ids=[1])
    handoff = store.create_episode(
        initial_text="P",
        initial_token_ids=[1],
        sampling=handoff_engine.sampling,
        stream_fingerprint=handoff_engine.stream_fingerprint,
        coordinate_offset=handoff_engine.coordinate_offset,
        max_tokens=None,
        backend=handoff_backend.provenance(),
    )
    handoff_result = EpisodeRunner(handoff_engine, store, handoff, divergence_policy="handoff").run(
        tape=[TapeStep(action, expectation)]
    )
    assert handoff_result.handed_off
    assert handoff_result.replayed_actions == 0
    assert handoff_engine.boundary == 0

    target_backend = Backend()
    target_engine = EpisodeEngine(target_backend, sampling=SamplerConfig(), initial_token_ids=[1])
    target = store.create_episode(
        initial_text="P",
        initial_token_ids=[1],
        sampling=target_engine.sampling,
        stream_fingerprint=target_engine.stream_fingerprint,
        coordinate_offset=target_engine.coordinate_offset,
        max_tokens=None,
        backend=target_backend.provenance(),
    )
    result = EpisodeRunner(target_engine, store, target, divergence_policy="ballistic").run(
        tape=[TapeStep(action, expectation)]
    )
    assert result.replayed_actions == 1
    assert project_episode(store, target).text == "P B"
'''
    script = script.replace("__WORKSPACE__", str(tmp_path / "core.sqlite3"))

    result = __import__("subprocess").run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
