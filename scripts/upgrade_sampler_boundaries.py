#!/usr/bin/env python3
"""One-time SQLite upgrade for sampler boundaries.

Usage: python scripts/upgrade_sampler_boundaries.py PATH/TO/episodes.sqlite3

The standalone utility removes the old sampler offset column and renames the
stored draw-position evidence field. EpisodeStore never runs it automatically.
A SQLite backup is saved beside the database before the schema changes.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 4


class UpgradeError(RuntimeError):
    """The database cannot be upgraded by this one-time utility."""


def _backup(source: sqlite3.Connection, database: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = database.with_name(
        f"{database.name}.before-sampler-boundary-upgrade-{stamp}.bak"
    )
    suffix = 1
    while backup.exists():
        backup = database.with_name(
            f"{database.name}.before-sampler-boundary-upgrade-{stamp}-{suffix}.bak"
        )
        suffix += 1
    target = sqlite3.connect(backup)
    try:
        source.backup(target)
    finally:
        target.close()
    return backup


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})")
    }


def upgrade_database(database: Path) -> Path | None:
    """Upgrade sampler storage and return the safety-copy path, if needed."""

    database = database.expanduser().resolve()
    if not database.is_file():
        raise UpgradeError(f"database does not exist: {database}")

    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not {"schema_info", "sampler_segments", "tokens"}.issubset(tables):
            raise UpgradeError("file is not a recognized episode database")

        version_row = connection.execute(
            "SELECT version FROM schema_info LIMIT 1"
        ).fetchone()
        if version_row is None:
            raise UpgradeError("episode database has no schema version")
        version = int(version_row["version"])
        if version not in {2, 3, SCHEMA_VERSION}:
            raise UpgradeError(
                f"schema version {version} is not supported by this one-time upgrader; "
                "open the database with the previous release first"
            )

        sampler_columns = _columns(connection, "sampler_segments")
        token_columns = _columns(connection, "tokens")
        if "sampling_coordinate" in token_columns and "sampling_boundary" in token_columns:
            raise UpgradeError(
                "token table contains both old and new sampling-position fields"
            )
        has_offset = "coordinate_offset" in sampler_columns
        has_old_sampling_position = "sampling_coordinate" in token_columns
        has_new_sampling_position = "sampling_boundary" in token_columns
        if not has_old_sampling_position and not has_new_sampling_position:
            raise UpgradeError("token table has no recognized sampling-position field")
        if version == SCHEMA_VERSION and not has_offset and not has_old_sampling_position:
            return None

        backup = _backup(connection, database)
        connection.execute("BEGIN IMMEDIATE")
        # Recheck after acquiring the write lock in case another process changed
        # the file between the initial inspection and this transaction.
        sampler_columns = _columns(connection, "sampler_segments")
        token_columns = _columns(connection, "tokens")
        if "coordinate_offset" in sampler_columns:
            connection.execute(
                "ALTER TABLE sampler_segments RENAME TO sampler_segments_with_offset"
            )
            connection.execute(
                """CREATE TABLE sampler_segments (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    start_boundary INTEGER NOT NULL,
                    sampling_json TEXT NOT NULL,
                    stream_fingerprint TEXT NOT NULL,
                    PRIMARY KEY (episode_id, start_boundary)
                )"""
            )
            connection.execute(
                """INSERT INTO sampler_segments(
                    episode_id, start_boundary, sampling_json, stream_fingerprint
                ) SELECT episode_id, start_boundary, sampling_json, stream_fingerprint
                  FROM sampler_segments_with_offset"""
            )
            connection.execute("DROP TABLE sampler_segments_with_offset")
        if "sampling_coordinate" in token_columns:
            if "sampling_boundary" in token_columns:
                raise UpgradeError(
                "token table contains both old and new sampling-position fields"
            )
            connection.execute(
                "ALTER TABLE tokens RENAME COLUMN sampling_coordinate TO sampling_boundary"
            )
        elif "sampling_boundary" not in token_columns:
            raise UpgradeError("token table has no recognized sampling-position field")
        connection.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
        connection.commit()

        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise UpgradeError(f"SQLite integrity check failed after upgrade: {integrity}")
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise UpgradeError(
                f"foreign-key check found {len(foreign_key_errors)} issue(s) after upgrade"
            )
        return backup
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite episode workspace to upgrade")
    args = parser.parse_args()
    try:
        backup = upgrade_database(args.database)
    except (UpgradeError, sqlite3.Error) as exc:
        parser.error(str(exc))
    if backup is None:
        print("Episode database is already on the sampler-boundary schema.")
    else:
        print(f"Saved backup: {backup}")
        print("Upgraded sampler evidence and controls to boundary terminology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
