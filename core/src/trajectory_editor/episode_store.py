"""Compact SQLite persistence for policy episodes.

The database is the workspace.  JSON is used only inside typed argument and
metadata columns; the runtime never emits report/event sidecar forests.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.actions import Phrase, Write, Hold, action_from_dict, sampler_after_action
from .core.errors import EditorError
from .core.results import ActionOutcome
from .core.sampler_config import SamplerConfig
from .episode_history import StoredHistoryPrefix, materialize_stored_prefix
from .episode_hash import token_prefix_sha256, validate_boundary, validate_fingerprint

SCHEMA_VERSION = 2


def _core_sampling_record(sampling: SamplerConfig) -> dict[str, Any]:
    """Serialize the canonical sampler record."""

    if not isinstance(sampling, SamplerConfig):
        raise EditorError("episode sampling must implement the core sampler contract")
    return SamplerConfig.from_record(sampling.to_dict()).to_dict()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


class EpisodeStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema()
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
        except Exception:
            self.connection.close()
            raise

    def __enter__(self) -> EpisodeStore:  # noqa: PYI034 - Python 3.10 support
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            with self.connection:
                yield self.connection
        except sqlite3.Error as exc:
            raise EditorError(f"episode database transaction failed: {exc}") from exc

    def _create_schema(self) -> None:
        self._reject_previous_schema()
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    parent_episode_id TEXT REFERENCES episodes(episode_id),
                    fork_boundary INTEGER,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    initial_text TEXT NOT NULL,
                    initial_token_ids_json TEXT NOT NULL,
                    initial_sampling_json TEXT NOT NULL,
                    initial_stream_fingerprint TEXT NOT NULL,
                    visible_text TEXT NOT NULL DEFAULT '',
                    terminal_token_id INTEGER,
                    terminal_reason TEXT,
                    backend_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS actions (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    boundary_before INTEGER NOT NULL,
                    boundary_after INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    resolved_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT NOT NULL,
                    mismatch_json TEXT,
                    PRIMARY KEY (episode_id, ordinal)
                );

                CREATE TABLE IF NOT EXISTS tokens (
                    episode_id TEXT NOT NULL,
                    action_ordinal INTEGER NOT NULL,
                    action_token_index INTEGER NOT NULL,
                    boundary INTEGER NOT NULL,
                    token_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    realized_visible INTEGER NOT NULL,
                    is_eog INTEGER NOT NULL,
                    sampling_boundary INTEGER NOT NULL,
                    proposal_token_id INTEGER NOT NULL,
                    raw_model_nll REAL,
                    raw_rank INTEGER,
                    policy_rank INTEGER,
                    decoder_probability REAL NOT NULL,
                    proposal_agreement INTEGER NOT NULL,
                    PRIMARY KEY (episode_id, action_ordinal, action_token_index),
                    FOREIGN KEY (episode_id, action_ordinal)
                        REFERENCES actions(episode_id, ordinal) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS tokens_by_boundary
                    ON tokens(episode_id, boundary);

                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    boundary INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episode_names (
                    number INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT UNIQUE NOT NULL REFERENCES episodes(episode_id),
                    title TEXT NOT NULL,
                    visited_at TEXT NOT NULL
                );
                """
            )
            token_columns = {row["name"]: row for row in db.execute("PRAGMA table_info(tokens)")}
            raw_nll_column = token_columns.get("raw_model_nll")
            if raw_nll_column is None or raw_nll_column["notnull"]:
                raise EditorError(
                    "episode database has an unsupported token table; create a current workspace"
                )
            row = db.execute("SELECT version FROM schema_info").fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO schema_info(version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            elif int(row["version"]) != SCHEMA_VERSION:
                raise EditorError(
                    "episode workspace schema is unsupported; create a new workspace"
                )

    def _reject_previous_schema(self) -> None:
        """Require a fresh workspace after the sampler and budget format break."""
        tables = {
            str(row["name"])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not tables:
            return

        unsupported = (
            "this workspace uses the previous episode format, which is no longer "
            "supported; create a new workspace (existing episodes were left untouched)"
        )
        if "schema_info" not in tables:
            raise EditorError(unsupported)
        version_row = self.connection.execute(
            "SELECT version FROM schema_info LIMIT 1"
        ).fetchone()
        if version_row is None or int(version_row["version"]) != SCHEMA_VERSION:
            raise EditorError(unsupported)
        if "episodes" not in tables:
            # A partially initialized store must not be completed in place.
            raise EditorError(unsupported)

        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(episodes)")
        }
        old_controls = {"max_tokens", "checkpoint_boundary"} & columns
        old_tables = {"sampler_segments", "budget_segments"} & tables
        required_columns = {"initial_sampling_json", "initial_stream_fingerprint"}
        required_tables = {"actions", "tokens", "interactions", "episode_names"}
        if (
            old_controls
            or old_tables
            or not required_columns.issubset(columns)
            or not required_tables.issubset(tables)
        ):
            raise EditorError(
                unsupported
            )

    def create_episode(
        self,
        *,
        initial_text: str,
        initial_token_ids: Sequence[int],
        sampling: SamplerConfig,
        stream_fingerprint: str,
        backend: Mapping[str, Any],
        episode_id: str | None = None,
        parent_episode_id: str | None = None,
        fork_boundary: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        token_prefix_sha256(list(initial_token_ids))
        validate_fingerprint(stream_fingerprint)
        identifier = episode_id or uuid.uuid4().hex
        if not identifier or any(character.isspace() for character in identifier):
            raise EditorError("episode id must be nonempty and contain no whitespace")
        if parent_episode_id is not None:
            self.get_episode(parent_episode_id)
        with self.transaction() as db:
            try:
                db.execute(
                    """
                    INSERT INTO episodes(
                        episode_id, parent_episode_id, fork_boundary, status,
                        created_at, initial_text, initial_token_ids_json,
                        initial_sampling_json, initial_stream_fingerprint,
                        backend_json, metadata_json
                    ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        parent_episode_id,
                        fork_boundary,
                        _utc_now(),
                        initial_text,
                        _json([int(value) for value in initial_token_ids]),
                        _json(_core_sampling_record(sampling)),
                        stream_fingerprint,
                        _json(dict(backend)),
                        _json(dict(metadata or {})),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise EditorError(f"episode {identifier!r} already exists") from exc
            db.execute("INSERT INTO episode_names(episode_id, title, visited_at) VALUES (?, ?, ?)",
                       (identifier, " ".join(initial_text.split())[:60] or "Untitled", _utc_now()))
        return identifier

    def resolve_id(self, value: str) -> str:
        if value.startswith("#") and value[1:].isdigit():
            row = self.connection.execute("SELECT episode_id FROM episode_names WHERE number = ?", (int(value[1:]),)).fetchone()
            if row is None:
                raise EditorError(f"unknown episode {value}")
            return str(row["episode_id"])
        self.get_episode(value)
        return value

    def label(self, episode_id: str) -> str:
        row = self.connection.execute("SELECT number, title FROM episode_names WHERE episode_id = ?", (episode_id,)).fetchone()
        return f"#{row['number']}  {row['title']}" if row else episode_id

    def rename(self, episode_id: str, title: str) -> None:
        if not title.strip():
            raise EditorError("title must not be empty")
        with self.transaction() as db:
            db.execute("UPDATE episode_names SET title = ? WHERE episode_id = ?", (title.strip(), episode_id))

    def visit(self, episode_id: str) -> None:
        with self.transaction() as db:
            db.execute("UPDATE episode_names SET visited_at = ? WHERE episode_id = ?", (_utc_now(), episode_id))

    def workspace_list(self, *, include_finished: bool = False, current: str | None = None) -> str:
        rows = self.connection.execute("""SELECT e.*, n.number, n.title FROM episodes e
            JOIN episode_names n USING(episode_id) ORDER BY n.visited_at DESC, n.number DESC""").fetchall()
        lines = []
        for row in rows:
            if not include_finished and row["status"] in {"completed", "failed"}:
                continue
            state = ("current" if row["episode_id"] == current else
                     "finished" if row["status"] == "completed" else
                     "failed" if row["status"] == "failed" else "paused")
            lines.append(f"{'●' if row['episode_id'] == current else ' '} #{row['number']}  {row['title']}  ({state})")
            if row["parent_episode_id"]:
                lines.append(f"    From {self.label(row['parent_episode_id'])} at token {row['fork_boundary']}")
            lines.append("    " + " ".join(row["visible_text"].split())[-100:])
        return "\n".join(lines) or "No open episodes."

    def next_action_ordinal(self, episode_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 AS value FROM actions WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        return int(row["value"])

    @staticmethod
    def _require_unsealed(db: sqlite3.Connection, episode_id: str) -> None:
        row = db.execute(
            "SELECT status FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise EditorError(f"unknown episode {episode_id!r}")
        if row["status"] in {"completed", "failed"}:
            raise EditorError(f"episode {episode_id!r} is sealed")

    def update_episode(
        self,
        episode_id: str,
        *,
        visible_text: str,
        status: str = "open",
    ) -> None:
        """Persist the current live edge without sealing the episode."""
        if status not in {"open", "replay-edge"}:
            raise EditorError(f"invalid unsealed episode status {status!r}")
        with self.transaction() as db:
            cursor = db.execute(
                """
                UPDATE episodes
                SET status = ?, visible_text = ?,
                    terminal_token_id = NULL, terminal_reason = NULL, finished_at = NULL
                WHERE episode_id = ? AND status NOT IN ('completed', 'failed')
                """,
                (status, visible_text, episode_id),
            )
            if cursor.rowcount != 1:
                self.get_episode(episode_id)
                raise EditorError(f"episode {episode_id!r} is sealed")

    def rewind_to(
        self,
        episode_id: str,
        boundary: int,
        *,
        visible_text: str,
    ) -> dict[str, Any]:
        """Persist the root-relative history retained through ``boundary``."""

        if type(boundary) is not int or boundary < 0:
            raise EditorError("rewind boundary must be a nonnegative integer")
        episode = self.get_episode(episode_id)
        if episode["status"] in {"completed", "failed"}:
            raise EditorError(f"episode {episode_id!r} is sealed")
        token_rows = self.tokens(episode_id)
        visible_count = sum(bool(row["realized_visible"]) for row in token_rows)
        if boundary > visible_count:
            raise EditorError(
                f"rewind boundary must be between 0 and {visible_count}"
            )
        try:
            retained = materialize_stored_prefix(
                self.actions(episode_id),
                token_rows,
                boundary,
            )
        except ValueError as exc:
            raise EditorError(str(exc)) from exc

        partial = retained.partial
        trimmed_action = (
            {
                "ordinal": partial.ordinal,
                "kind": partial.action.kind,
                "original_boundary_after": partial.outcome.boundary_after,
                "new_boundary_after": boundary,
            }
            if partial is not None
            else None
        )
        with self.transaction() as db:
            self._replace_history_actions(
                db,
                episode_id,
                retained,
                preserve_ordinals=True,
            )
            db.execute(
                "DELETE FROM interactions WHERE episode_id = ? AND boundary >= ?",
                (episode_id, boundary),
            )
            cursor = db.execute(
                """
                UPDATE episodes
                SET status = 'open', visible_text = ?,
                    terminal_token_id = NULL, terminal_reason = NULL,
                    finished_at = NULL
                WHERE episode_id = ?
                """,
                (visible_text, episode_id),
            )
            if cursor.rowcount != 1:
                raise EditorError(f"unknown episode {episode_id!r}")
        return {
            "target_boundary": boundary,
            "trimmed_action": trimmed_action,
        }


    def copy_prefix(
        self,
        source_episode_id: str,
        destination_episode_id: str,
        boundary: int,
        *,
        visible_text: str,
    ) -> None:
        """Persist one source history prefix as a new root-relative episode."""

        self.get_episode(source_episode_id)
        self.get_episode(destination_episode_id)
        if source_episode_id == destination_episode_id:
            raise EditorError("cannot copy an episode prefix onto itself")
        if type(boundary) is not int or boundary < 0:
            raise EditorError("fork boundary must be a nonnegative integer")
        if self.actions(destination_episode_id) or self.tokens(destination_episode_id):
            raise EditorError("destination episode already has recorded history")

        source_actions = self.actions(source_episode_id)
        source_tokens = self.tokens(source_episode_id)
        visible_count = sum(bool(row["realized_visible"]) for row in source_tokens)
        if boundary > visible_count:
            raise EditorError(f"fork boundary must be between 0 and {visible_count}")
        try:
            materialized = materialize_stored_prefix(
                source_actions,
                source_tokens,
                boundary,
            )
        except ValueError as exc:
            raise EditorError(str(exc)) from exc

        # Keep the destination identity and root context, replacing only its
        # initial rows with the copied root-relative prefix.
        with self.transaction() as db:
            self._write_history_actions(
                db,
                destination_episode_id,
                materialized,
                preserve_ordinals=False,
            )
            source = self.get_episode(source_episode_id)
            db.execute(
                """
                UPDATE episodes
                SET visible_text = ?, initial_sampling_json = ?,
                    initial_stream_fingerprint = ?, status = 'open',
                    terminal_token_id = NULL, terminal_reason = NULL,
                    finished_at = NULL
                WHERE episode_id = ?
                """,
                (
                    visible_text,
                    _json(source["initial_sampling"]),
                    source["initial_stream_fingerprint"],
                    destination_episode_id,
                ),
            )

    def _replace_history_actions(
        self,
        db: sqlite3.Connection,
        episode_id: str,
        prefix: StoredHistoryPrefix,
        *,
        preserve_ordinals: bool,
    ) -> None:
        """Replace persisted action/token rows with a projected history prefix."""

        db.execute("DELETE FROM actions WHERE episode_id = ?", (episode_id,))
        self._write_history_actions(
            db,
            episode_id,
            prefix,
            preserve_ordinals=preserve_ordinals,
        )

    @staticmethod
    def _write_history_actions(
        db: sqlite3.Connection,
        episode_id: str,
        prefix: StoredHistoryPrefix,
        *,
        preserve_ordinals: bool,
    ) -> None:
        """Insert projected action evidence using only SQLite primitives."""

        for position, action in enumerate(prefix.actions):
            ordinal = action.ordinal if preserve_ordinals else position
            db.execute(
                """
                INSERT INTO actions(
                    episode_id, ordinal, boundary_before, boundary_after,
                    kind, arguments_json, resolved_text, status,
                    stop_reason, mismatch_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    ordinal,
                    action.boundary_before,
                    action.boundary_after,
                    action.kind,
                    _json(dict(action.arguments)),
                    action.resolved_text,
                    action.status,
                    action.stop_reason,
                    _json(dict(action.mismatch))
                    if action.mismatch is not None
                    else None,
                ),
            )
            for token_index, token in enumerate(action.tokens):
                db.execute(
                    """
                    INSERT INTO tokens(
                        episode_id, action_ordinal, action_token_index,
                        boundary, token_id, text, realized_visible, is_eog,
                        sampling_boundary, proposal_token_id,
                        raw_model_nll, raw_rank, policy_rank,
                        decoder_probability, proposal_agreement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        ordinal,
                        token_index,
                        int(token["boundary"]),
                        int(token["token_id"]),
                        str(token["text"]),
                        int(token["realized_visible"]),
                        int(token["is_eog"]),
                        int(token["sampling_boundary"]),
                        int(token["proposal_token_id"]),
                        float(token["raw_model_nll"]) if token.get("raw_model_nll") is not None else None,
                        int(token["raw_rank"]) if token.get("raw_rank") is not None else None,
                        int(token["policy_rank"]) if token.get("policy_rank") is not None else None,
                        float(token["decoder_probability"]),
                        int(token["proposal_agreement"]),
                    ),
                )


    def record_action(
        self, episode_id: str, ordinal: int, outcome: ActionOutcome,
    ) -> None:
        arguments = outcome.action.to_dict()
        if outcome.diagnostics is not None:
            arguments["diagnostics"] = dict(outcome.diagnostics)
        if outcome.replay_eog_token_id is not None:
            arguments["replay_eog_token_id"] = outcome.replay_eog_token_id
        with self.transaction() as db:
            self._require_unsealed(db, episode_id)
            db.execute(
                """
                INSERT INTO actions(
                    episode_id, ordinal, boundary_before, boundary_after, kind,
                    arguments_json, resolved_text, status, stop_reason, mismatch_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    ordinal,
                    outcome.boundary_before,
                    outcome.boundary_after,
                    outcome.action.kind,
                    _json(arguments),
                    outcome.resolved_text,
                    outcome.status,
                    outcome.stop_reason,
                    _json(outcome.divergence.to_dict())
                    if outcome.divergence is not None
                    else None,
                ),
            )
            for index, evidence in enumerate(outcome.evidence):
                db.execute(
                    """
                    INSERT INTO tokens(
                        episode_id, action_ordinal, action_token_index, boundary,
                        token_id, text, realized_visible, is_eog,
                        sampling_boundary, proposal_token_id, raw_model_nll,
                        raw_rank, policy_rank, decoder_probability,
                        proposal_agreement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        ordinal,
                        index,
                        evidence.boundary,
                        evidence.token_id,
                        evidence.text,
                        int(evidence.realized_visible),
                        int(evidence.is_eog),
                        evidence.sampling_boundary,
                        evidence.proposal_token_id,
                        evidence.raw_model_nll,
                        evidence.raw_rank,
                        evidence.policy_rank,
                        evidence.decoder_probability,
                        int(evidence.proposal_agreement),
                    ),
                )

    def record_interaction(
        self,
        episode_id: str,
        boundary: int,
        kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        with self.transaction() as db:
            self._require_unsealed(db, episode_id)
            db.execute(
                """
                INSERT INTO interactions(
                    episode_id, boundary, kind, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (episode_id, boundary, kind, _json(dict(payload)), _utc_now()),
            )

    def finish_episode(
        self,
        episode_id: str,
        *,
        visible_text: str,
        terminal_token_id: int | None,
        terminal_reason: str,
    ) -> None:
        """Record a genuine terminal event and seal an unsealed episode."""
        if not isinstance(terminal_reason, str) or not terminal_reason:
            raise EditorError("completed episode requires a terminal reason")
        with self.transaction() as db:
            cursor = db.execute(
                """
                UPDATE episodes
                SET status = 'completed', finished_at = ?, visible_text = ?,
                    terminal_token_id = ?, terminal_reason = ?
                WHERE episode_id = ? AND status NOT IN ('completed', 'failed')
                """,
                (
                    _utc_now(),
                    visible_text,
                    terminal_token_id,
                    terminal_reason,
                    episode_id,
                ),
            )
            if cursor.rowcount != 1:
                self.get_episode(episode_id)
                raise EditorError(f"episode {episode_id!r} is sealed")

    def get_episode(self, episode_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise EditorError(f"unknown episode {episode_id!r}")
        result = dict(row)
        result["initial_token_ids"] = _loads(result.pop("initial_token_ids_json"), [])
        result["initial_sampling"] = _loads(result.pop("initial_sampling_json"), {})
        if not isinstance(result["initial_sampling"], dict):
            raise EditorError("saved initial sampler settings must be an object")
        SamplerConfig.from_record(result["initial_sampling"])
        validate_fingerprint(result["initial_stream_fingerprint"])
        result["backend"] = _loads(result.pop("backend_json"), {})
        result["metadata"] = _loads(result.pop("metadata_json"), {})
        return result

    def episode_relation_rows(self) -> list[dict[str, Any]]:
        """Return the flat persistence facts required for lineage projection."""

        rows = self.connection.execute(
            """
            SELECT
                episode.episode_id,
                episode.parent_episode_id,
                episode.fork_boundary,
                episode.status,
                episode.created_at,
                episode.terminal_reason,
                episode.metadata_json,
                COALESCE(
                    (
                        SELECT COUNT(*)
                        FROM tokens
                        WHERE tokens.episode_id = episode.episode_id
                          AND tokens.realized_visible = 1
                    ),
                    0
                ) AS visible_token_count
            FROM episodes AS episode
            ORDER BY episode.created_at, episode.episode_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def actions(self, episode_id: str) -> list[dict[str, Any]]:
        self.get_episode(episode_id)
        rows = self.connection.execute(
            "SELECT * FROM actions WHERE episode_id = ? ORDER BY ordinal",
            (episode_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["arguments"] = _loads(item.pop("arguments_json"), {})
            item["mismatch"] = _loads(item.pop("mismatch_json"), None)
            result.append(item)
        return result

    def action_boundaries(
        self, episode_id: str, *, after_ordinal: int = -1
    ) -> list[dict[str, Any]]:
        """Return the compact action fields used to label visible boundaries."""
        self.get_episode(episode_id)
        rows = self.connection.execute(
            """
            SELECT ordinal, boundary_before, boundary_after, kind
            FROM actions
            WHERE episode_id = ? AND ordinal > ?
            ORDER BY ordinal
            """,
            (episode_id, after_ordinal),
        ).fetchall()
        return [dict(row) for row in rows]

    def tokens(self, episode_id: str) -> list[dict[str, Any]]:
        self.get_episode(episode_id)
        rows = self.connection.execute(
            """
            SELECT * FROM tokens
            WHERE episode_id = ?
            ORDER BY action_ordinal, action_token_index
            """,
            (episode_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def interactions(self, episode_id: str) -> list[dict[str, Any]]:
        self.get_episode(episode_id)
        rows = self.connection.execute(
            """
            SELECT * FROM interactions
            WHERE episode_id = ? ORDER BY interaction_id
            """,
            (episode_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = _loads(item.pop("payload_json"), {})
            result.append(item)
        return result

    def sampling_state_at_boundary(
        self, episode_id: str, boundary: int | None = None
    ) -> dict[str, Any]:
        """Resolve sampler state from the executable action prefix."""
        episode = self.get_episode(episode_id)
        visible_boundary = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM tokens WHERE episode_id = ? AND realized_visible = 1",
                (episode_id,),
            ).fetchone()[0]
        )
        selected = visible_boundary if boundary is None else boundary
        validate_boundary(selected, "boundary")
        if selected > visible_boundary:
            raise EditorError(f"sampler boundary must be between 0 and {visible_boundary}")
        if boundary is None:
            rows = self.connection.execute(
                "SELECT arguments_json FROM actions WHERE episode_id = ? ORDER BY ordinal",
                (episode_id,),
            )
        else:
            rows = self.connection.execute(
                """
                SELECT arguments_json FROM actions
                WHERE episode_id = ?
                  AND (boundary_after < ? OR
                       (boundary_after = ? AND boundary_before < ?))
                ORDER BY ordinal
                """,
                (episode_id, selected, selected, selected),
            )
        sampling = SamplerConfig.from_record(episode["initial_sampling"])
        for row in rows:
            arguments = _loads(row["arguments_json"], {})
            if not isinstance(arguments, Mapping):
                raise EditorError("saved action arguments must be an object")
            action = action_from_dict(arguments)
            sampling = sampler_after_action(sampling, action)
        return {
            "boundary": selected,
            "sampling": sampling.to_dict(),
            "stream_fingerprint": episode["initial_stream_fingerprint"],
        }

    def current_sampling_state(self, episode_id: str) -> dict[str, Any]:
        return self.sampling_state_at_boundary(episode_id)
