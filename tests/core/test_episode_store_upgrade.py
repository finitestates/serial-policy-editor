"""One-time SQLite workspace format upgrades."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Accept
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore


@pytest.mark.invariant
def test_legacy_workspace_upgrade_drops_sampler_offset_and_preserves_history(tmp_path):
    path = tmp_path / "legacy.db"
    with EpisodeStore(path) as store:
        engine = EpisodeEngine(
            ConformingFakeBackend(), sampling=SamplerConfig(temperature=0.0),
            initial_token_ids=[7],
        )
        episode_id = store.create_episode(
            initial_text=engine.initial_text, initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling, stream_fingerprint=engine.stream_fingerprint,
            max_tokens=engine.max_tokens, backend=engine.backend.provenance(),
        )
        store.record_action(episode_id, 0, engine.apply(Accept()))
        db = store.connection
        db.execute("UPDATE tokens SET raw_model_nll = 1.5, raw_rank = 2, policy_rank = 3")
        db.execute("ALTER TABLE tokens RENAME TO tokens_old")
        db.execute("""CREATE TABLE tokens (
            episode_id TEXT NOT NULL, action_ordinal INTEGER NOT NULL,
            action_token_index INTEGER NOT NULL, boundary INTEGER NOT NULL,
            token_id INTEGER NOT NULL, text TEXT NOT NULL,
            realized_visible INTEGER NOT NULL, is_eog INTEGER NOT NULL,
            sampling_coordinate INTEGER NOT NULL, proposal_token_id INTEGER NOT NULL,
            raw_model_nll REAL NOT NULL, raw_rank INTEGER NOT NULL,
            policy_rank INTEGER NOT NULL, decoder_probability REAL NOT NULL,
            proposal_agreement INTEGER NOT NULL,
            PRIMARY KEY (episode_id, action_ordinal, action_token_index),
            FOREIGN KEY (episode_id, action_ordinal) REFERENCES actions(episode_id, ordinal)
        )""")
        db.execute("INSERT INTO tokens SELECT * FROM tokens_old")
        db.execute("DROP TABLE tokens_old")
        db.execute("ALTER TABLE sampler_segments RENAME TO sampler_segments_current")
        db.execute("""CREATE TABLE sampler_segments (
            episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
            start_boundary INTEGER NOT NULL, sampling_json TEXT NOT NULL,
            stream_fingerprint TEXT NOT NULL, coordinate_offset INTEGER NOT NULL,
            PRIMARY KEY (episode_id, start_boundary)
        )""")
        db.execute("""INSERT INTO sampler_segments(
            episode_id, start_boundary, sampling_json, stream_fingerprint, coordinate_offset
        ) SELECT episode_id, start_boundary, sampling_json, stream_fingerprint, 29
          FROM sampler_segments_current""")
        db.execute("DROP TABLE sampler_segments_current")
        db.execute("UPDATE schema_info SET version = 1")
        db.commit()

    with pytest.raises(EditorError, match="one-time boundary-coordinate upgrade"):
        EpisodeStore(path)

    script = Path(__file__).resolve().parents[2] / "scripts" / "upgrade_boundary_coordinates.py"
    upgraded = subprocess.run(
        [sys.executable, str(script), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    backup = Path(upgraded.stdout.split("Saved backup: ", 1)[1].splitlines()[0])
    assert backup.is_file()

    with EpisodeStore(path) as store:
        assert store.tokens(episode_id)[0]["raw_model_nll"] == 1.5
        assert store.connection.execute("SELECT version FROM schema_info").fetchone()[0] == 3
        assert store.connection.execute("PRAGMA table_info(tokens)").fetchall()[10][3] == 0
        assert store.sampler_segments(episode_id)[0]["stream_fingerprint"] == engine.stream_fingerprint
