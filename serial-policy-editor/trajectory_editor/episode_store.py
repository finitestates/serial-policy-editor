"""Compact SQLite persistence for policy episodes.

The database is the workspace.  JSON is used only inside typed argument and
metadata columns; the runtime never emits report/event sidecar forests.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import EditorError, SamplingConfig
from .episode_actions import Write, Hold, Finish, PolicyAction, action_from_dict
from .episode_engine import ActionOutcome, ReplayExpectation
from .episode_hash import token_prefix_sha256, validate_coordinate, validate_fingerprint

SCHEMA_VERSION = 1


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
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()

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
                    visible_text TEXT NOT NULL DEFAULT '',
                    terminal_token_id INTEGER,
                    terminal_reason TEXT,
                    max_tokens INTEGER NOT NULL,
                    backend_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS budget_segments (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    start_boundary INTEGER NOT NULL,
                    max_tokens INTEGER,
                    checkpoint_boundary INTEGER,
                    PRIMARY KEY (episode_id, start_boundary)
                );

                CREATE TABLE IF NOT EXISTS sampler_segments (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    start_boundary INTEGER NOT NULL,
                    sampling_json TEXT NOT NULL,
                    stream_fingerprint TEXT NOT NULL,
                    coordinate_offset INTEGER NOT NULL,
                    PRIMARY KEY (episode_id, start_boundary)
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
                    sampling_coordinate INTEGER NOT NULL,
                    proposal_token_id INTEGER NOT NULL,
                    raw_model_nll REAL NOT NULL,
                    raw_rank INTEGER NOT NULL,
                    policy_rank INTEGER NOT NULL,
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
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(episodes)")}
            if "checkpoint_boundary" not in columns:
                db.execute("ALTER TABLE episodes ADD COLUMN checkpoint_boundary INTEGER")
                db.execute("UPDATE episodes SET checkpoint_boundary = max_tokens WHERE max_tokens > 0")
            db.execute("""CREATE TABLE IF NOT EXISTS episode_names (
                number INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT UNIQUE NOT NULL REFERENCES episodes(episode_id),
                title TEXT NOT NULL, visited_at TEXT NOT NULL)""")
            for episode in db.execute("SELECT episode_id, initial_text, created_at FROM episodes WHERE episode_id NOT IN (SELECT episode_id FROM episode_names) ORDER BY created_at, rowid").fetchall():
                db.execute("INSERT OR IGNORE INTO episode_names(episode_id, title, visited_at) VALUES (?, ?, ?)",
                           (episode["episode_id"], " ".join(episode["initial_text"].split())[:60] or "Untitled", episode["created_at"]))
            row = db.execute("SELECT version FROM schema_info").fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif int(row["version"]) != SCHEMA_VERSION:
                raise EditorError(
                    f"unsupported episode database schema {row['version']}"
                )

    def create_episode(
        self,
        *,
        initial_text: str,
        initial_token_ids: Sequence[int],
        sampling: SamplingConfig,
        stream_fingerprint: str,
        coordinate_offset: int,
        max_tokens: int | None,
        backend: Mapping[str, Any],
        episode_id: str | None = None,
        parent_episode_id: str | None = None,
        fork_boundary: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_boundary: int | None | str = "initial",
    ) -> str:
        if checkpoint_boundary == "initial":
            checkpoint_boundary = max_tokens
        token_prefix_sha256(list(initial_token_ids))
        validate_fingerprint(stream_fingerprint)
        validate_coordinate(coordinate_offset, "coordinate_offset")
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
                        max_tokens, backend_json, metadata_json
                    ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        parent_episode_id,
                        fork_boundary,
                        _utc_now(),
                        initial_text,
                        _json([int(value) for value in initial_token_ids]),
                        max_tokens or 0,
                        _json(dict(backend)),
                        _json(dict(metadata or {})),
                    ),
                )
                db.execute(
                    "UPDATE episodes SET checkpoint_boundary = ? WHERE episode_id = ?",
                    (checkpoint_boundary, identifier),
                )
                db.execute(
                    """
                    INSERT INTO sampler_segments(
                        episode_id, start_boundary, sampling_json,
                        stream_fingerprint, coordinate_offset
                    ) VALUES (?, 0, ?, ?, ?)
                    """,
                    (
                        identifier,
                        _json(sampling.to_dict()),
                        stream_fingerprint,
                        coordinate_offset,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise EditorError(f"episode {identifier!r} already exists") from exc
            db.execute("INSERT INTO episode_names(episode_id, title, visited_at) VALUES (?, ?, ?)",
                       (identifier, " ".join(initial_text.split())[:60] or "Untitled", _utc_now()))
        self.record_budget(identifier, 0, max_tokens, checkpoint_boundary)
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

    def budget_at(self, episode_id: str, boundary: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT max_tokens, checkpoint_boundary FROM budget_segments "
            "WHERE episode_id = ? AND start_boundary <= ? ORDER BY start_boundary DESC LIMIT 1",
            (episode_id, boundary),
        ).fetchone()
        if row is None:
            return None
        allowance, checkpoint = row["max_tokens"], row["checkpoint_boundary"]
        if allowance is None and checkpoint is None:
            return dict(row)
        if (type(allowance) is not int or allowance <= 0
                or type(checkpoint) is not int or checkpoint < boundary):
            return None
        return dict(row)

    def record_budget(self, episode_id: str, boundary: int, max_tokens: int | None,
                      checkpoint_boundary: int | None) -> None:
        """Record edits and renewals, not ordinary consumption of an allowance."""
        if (max_tokens is None) != (checkpoint_boundary is None):
            raise EditorError("budget allowance and checkpoint must both be set or unlimited")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0
                or type(checkpoint_boundary) is not int or checkpoint_boundary < boundary):
            raise EditorError("invalid budget allowance or checkpoint")
        state = {"max_tokens": max_tokens, "checkpoint_boundary": checkpoint_boundary}
        with self.transaction() as db:
            if self.budget_at(episode_id, boundary) != state:
                db.execute("INSERT OR REPLACE INTO budget_segments VALUES (?, ?, ?, ?)",
                           (episode_id, boundary, max_tokens, checkpoint_boundary))
            db.execute("UPDATE episodes SET max_tokens = ?, checkpoint_boundary = ? WHERE episode_id = ?",
                       (max_tokens or 0, checkpoint_boundary, episode_id))

    def update_episode(
        self,
        episode_id: str,
        *,
        visible_text: str,
        max_tokens: int | None,
        status: str = "open",
    ) -> None:
        """Persist the current live edge without sealing the episode."""
        with self.transaction() as db:
            cursor = db.execute(
                """
                UPDATE episodes
                SET status = ?, visible_text = ?, max_tokens = ?,
                    terminal_token_id = NULL, terminal_reason = NULL, finished_at = NULL
                WHERE episode_id = ?
                """,
                (status, visible_text, max_tokens or 0, episode_id),
            )
            if cursor.rowcount != 1:
                raise EditorError(f"unknown episode {episode_id!r}")

    def rewind_to(
        self,
        episode_id: str,
        boundary: int,
        *,
        visible_text: str,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Truncate one open episode at a visible-token boundary.

        Partial holds become finite holds. Partial writes become exact writes
        of the retained text, keeping the original submission in their metadata.
        """
        if type(boundary) is not int or boundary < 0:
            raise EditorError("rewind boundary must be a nonnegative integer")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise EditorError("max_tokens must be a positive integer")
        episode = self.get_episode(episode_id)
        if episode["status"] in {"completed", "failed"}:
            raise EditorError(f"episode {episode_id!r} is sealed")
        token_rows = self.tokens(episode_id)
        visible_count = sum(bool(row["realized_visible"]) for row in token_rows)
        if boundary > visible_count:
            raise EditorError(
                f"rewind boundary must be between 0 and {visible_count}"
            )

        action_rows = self.actions(episode_id)
        containing: dict[str, Any] | None = None
        first_removed: dict[str, Any] | None = None
        for row in action_rows:
            before = int(row["boundary_before"])
            after = int(row["boundary_after"])
            if before < boundary < after:
                containing = row
                break
            if before >= boundary:
                first_removed = row
                break

        if containing is not None and containing["kind"] not in {"hold", "write"}:
            raise EditorError(
                "seamless rewind cannot enter a multi-token action block; "
                "rewind to its start or end"
            )

        trimmed_action: dict[str, Any] | None = None
        with self.transaction() as db:
            if containing is not None:
                ordinal = int(containing["ordinal"])
                retained_rows = db.execute(
                    """
                    SELECT text FROM tokens
                    WHERE episode_id = ? AND action_ordinal = ?
                      AND boundary < ? AND realized_visible = 1
                    ORDER BY action_token_index
                    """,
                    (episode_id, ordinal, boundary),
                ).fetchall()
                arguments = dict(containing["arguments"])
                resolved_text = "".join(str(row["text"]) for row in retained_rows)
                if containing["kind"] == "write":
                    original_write = arguments.get("original_write", dict(arguments))
                    arguments = {**arguments, "kind": "write", "mode": "exact", "text": resolved_text,
                                 "original_write": original_write}
                else:
                    arguments["limit"] = boundary - int(containing["boundary_before"])
                    arguments["boundary"] = None
                db.execute(
                    "DELETE FROM actions WHERE episode_id = ? AND ordinal > ?",
                    (episode_id, ordinal),
                )
                db.execute(
                    """
                    UPDATE actions
                    SET boundary_after = ?, arguments_json = ?,
                        resolved_text = ?, status = 'completed',
                        stop_reason = ?, mismatch_json = NULL
                    WHERE episode_id = ? AND ordinal = ?
                    """,
                    (
                        boundary,
                        _json(arguments),
                        resolved_text,
                        "completed" if containing["kind"] == "write" else "requested-length",
                        episode_id,
                        ordinal,
                    ),
                )
                trimmed_action = {
                    "ordinal": ordinal,
                    "kind": str(containing["kind"]),
                    "original_boundary_after": int(containing["boundary_after"]),
                    "new_boundary_after": boundary,
                }
            elif first_removed is not None:
                db.execute(
                    "DELETE FROM actions WHERE episode_id = ? AND ordinal >= ?",
                    (episode_id, int(first_removed["ordinal"])),
                )

            db.execute(
                "DELETE FROM tokens WHERE episode_id = ? AND boundary >= ?",
                (episode_id, boundary),
            )
            db.execute(
                "DELETE FROM interactions WHERE episode_id = ? AND boundary >= ?",
                (episode_id, boundary),
            )
            db.execute("DELETE FROM budget_segments WHERE episode_id = ? AND start_boundary > ?",
                       (episode_id, boundary))
            db.execute(
                """
                DELETE FROM sampler_segments
                WHERE episode_id = ? AND start_boundary > ?
                """,
                (episode_id, boundary),
            )
            cursor = db.execute(
                """
                UPDATE episodes
                SET status = 'open', visible_text = ?, max_tokens = ?,
                    terminal_token_id = NULL, terminal_reason = NULL,
                    finished_at = NULL
                WHERE episode_id = ?
                """,
                (visible_text, max_tokens or 0, episode_id),
            )
            if cursor.rowcount != 1:
                raise EditorError(f"unknown episode {episode_id!r}")
        return {
            "target_boundary": boundary,
            "trimmed_action": trimmed_action,
        }

    def record_sampling_segment(
        self,
        episode_id: str,
        *,
        start_boundary: int,
        sampling: SamplingConfig,
        stream_fingerprint: str,
        coordinate_offset: int,
    ) -> None:
        """Record a sampler-policy transition at a live token boundary."""
        self.get_episode(episode_id)
        validate_coordinate(start_boundary, "start_boundary")
        validate_coordinate(coordinate_offset, "coordinate_offset")
        validate_fingerprint(stream_fingerprint)
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO sampler_segments(
                    episode_id, start_boundary, sampling_json,
                    stream_fingerprint, coordinate_offset
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(episode_id, start_boundary) DO UPDATE SET
                    sampling_json = excluded.sampling_json,
                    stream_fingerprint = excluded.stream_fingerprint,
                    coordinate_offset = excluded.coordinate_offset
                """,
                (
                    episode_id,
                    int(start_boundary),
                    _json(sampling.to_dict()),
                    stream_fingerprint,
                    int(coordinate_offset),
                ),
            )

    def record_action(
        self, episode_id: str, ordinal: int, outcome: ActionOutcome,
        *, replay_origin: Mapping[str, Any] | None = None,
    ) -> None:
        arguments = outcome.action.to_dict()
        if replay_origin is not None:
            arguments["replay_origin"] = dict(replay_origin)
        with self.transaction() as db:
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
                        sampling_coordinate, proposal_token_id, raw_model_nll,
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
                        evidence.sampling_coordinate,
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
        terminal_reason: str | None,
        status: str = "completed",
    ) -> None:
        with self.transaction() as db:
            cursor = db.execute(
                """
                UPDATE episodes
                SET status = ?, finished_at = ?, visible_text = ?,
                    terminal_token_id = ?, terminal_reason = ?
                WHERE episode_id = ?
                """,
                (
                    status,
                    _utc_now(),
                    visible_text,
                    terminal_token_id,
                    terminal_reason,
                    episode_id,
                ),
            )
            if cursor.rowcount != 1:
                raise EditorError(f"unknown episode {episode_id!r}")

    def get_episode(self, episode_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise EditorError(f"unknown episode {episode_id!r}")
        result = dict(row)
        result["max_tokens"] = result["max_tokens"] or None
        result["initial_token_ids"] = _loads(result.pop("initial_token_ids_json"), [])
        result["backend"] = _loads(result.pop("backend_json"), {})
        result["metadata"] = _loads(result.pop("metadata_json"), {})
        return result

    def list_episodes(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT episode_id, parent_episode_id, fork_boundary, status,
                   created_at, finished_at, terminal_reason,
                   length(visible_text) AS visible_characters
            FROM episodes ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def lineage(self, episode_id: str) -> dict[str, Any]:
        """Return ordinary fork ancestry plus separately typed replay links.

        ``parent_episode_id`` predates the distinction between a fork and a
        Serial Policy Replay episode.  The persisted ``metadata.mode`` and
        ``metadata.spr_source`` values let this view preserve that distinction
        without adding a schema column or treating replay runs as branches of
        the ordinary family tree.
        """
        self.get_episode(episode_id)
        rows = self.connection.execute(
            """
            SELECT
                episode.*,
                length(episode.visible_text) AS visible_characters,
                COALESCE(
                    (
                        SELECT COUNT(*)
                        FROM tokens
                        WHERE tokens.episode_id = episode.episode_id
                          AND tokens.realized_visible = 1
                    ),
                    0
                ) AS visible_tokens
            FROM episodes AS episode
            ORDER BY episode.created_at, episode.episode_id
            """
        ).fetchall()

        nodes: dict[str, dict[str, Any]] = {}
        for row in rows:
            metadata = _loads(row["metadata_json"], {})
            if not isinstance(metadata, dict):
                metadata = {}
            mode = metadata.get("mode")
            if not isinstance(mode, str) or not mode:
                mode = "interactive"
            replay_source = metadata.get("spr_source")
            if replay_source is not None and not isinstance(replay_source, str):
                replay_source = str(replay_source)
            is_replay = mode == "serial-policy-replay" or replay_source is not None
            if is_replay and replay_source is None:
                replay_source = row["parent_episode_id"]
            nodes[str(row["episode_id"])] = {
                "episode_id": str(row["episode_id"]),
                "parent_episode_id": row["parent_episode_id"],
                "fork_boundary": row["fork_boundary"],
                "status": str(row["status"]),
                "created_at": row["created_at"],
                "finished_at": row["finished_at"],
                "terminal_reason": row["terminal_reason"],
                "visible_characters": int(row["visible_characters"] or 0),
                "visible_tokens": int(row["visible_tokens"] or 0),
                "mode": mode,
                "is_replay": is_replay,
                "replay_source_episode_id": replay_source,
            }

        structural = {
            identifier: node
            for identifier, node in nodes.items()
            if not bool(node["is_replay"])
        }

        def structural_root(identifier: str) -> str:
            seen: set[str] = set()
            current = identifier
            while current in structural:
                if current in seen:
                    # Creation order and foreign keys prevent this in normal
                    # operation; keeping the current node makes old/corrupt
                    # workspaces inspectable instead of hanging here.
                    return current
                seen.add(current)
                parent = structural[current]["parent_episode_id"]
                if not isinstance(parent, str) or parent not in structural:
                    return current
                current = parent
            return identifier

        def structural_seed(identifier: str) -> str | None:
            """Find the ordinary family context for an episode or replay."""
            seen: set[str] = set()
            current = identifier
            while current in nodes and current not in seen:
                seen.add(current)
                node = nodes[current]
                if not bool(node["is_replay"]):
                    return current
                for candidate in (
                    node["parent_episode_id"],
                    node["replay_source_episode_id"],
                ):
                    if isinstance(candidate, str) and candidate in nodes:
                        current = candidate
                        break
                else:
                    return None
            return None

        seed = structural_seed(episode_id)
        family_root_id = structural_root(seed) if seed is not None else None
        family_ids = {
            identifier
            for identifier in structural
            if family_root_id is not None
            and structural_root(identifier) == family_root_id
        }

        children_by_parent: dict[str, list[str]] = {}
        for identifier in family_ids:
            parent = structural[identifier]["parent_episode_id"]
            if isinstance(parent, str) and parent in family_ids:
                children_by_parent.setdefault(parent, []).append(identifier)
        for children in children_by_parent.values():
            children.sort(
                key=lambda child: (
                    str(structural[child]["created_at"]),
                    child,
                )
            )

        def tree_node(identifier: str, seen: set[str]) -> dict[str, Any]:
            node = dict(structural[identifier])
            if identifier in seen:
                node["children"] = []
                return node
            next_seen = {*seen, identifier}
            node["children"] = [
                tree_node(child, next_seen)
                for child in children_by_parent.get(identifier, [])
            ]
            return node

        tree = (
            tree_node(family_root_id, set())
            if family_root_id is not None
            else None
        )

        related_replay_ids: set[str] = {
            identifier
            for identifier, node in nodes.items()
            if bool(node["is_replay"])
            and (
                identifier == episode_id
                or node["parent_episode_id"] in family_ids
                or node["replay_source_episode_id"] in family_ids
            )
        }
        for identifier in family_ids:
            parent = structural[identifier]["parent_episode_id"]
            if isinstance(parent, str) and parent in nodes and nodes[parent]["is_replay"]:
                related_replay_ids.add(parent)

        replays = [
            dict(nodes[identifier])
            for identifier in sorted(
                related_replay_ids,
                key=lambda value: (
                    str(nodes[value]["created_at"]),
                    value,
                ),
            )
        ]
        replay_derived_forks = [
            dict(node)
            for node in nodes.values()
            if not bool(node["is_replay"])
            and isinstance(node["parent_episode_id"], str)
            and node["parent_episode_id"] in related_replay_ids
        ]
        replay_derived_forks.sort(
            key=lambda node: (str(node["created_at"]), str(node["episode_id"]))
        )

        return {
            "selected_episode_id": episode_id,
            "family_root_id": family_root_id,
            "selected": dict(nodes[episode_id]),
            "tree": tree,
            "replays": replays,
            "replay_derived_forks": replay_derived_forks,
        }

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

    def replay_tape(
        self, episode_id: str
    ) -> list[tuple[PolicyAction, ReplayExpectation]]:
        return [
            (action, expectation)
            for action, expectation, _ in self.replay_tape_with_sampling(episode_id)
        ]

    def replay_tape_with_sampling(
        self, episode_id: str
    ) -> list[tuple[PolicyAction, ReplayExpectation, SamplingConfig]]:
        """Return recorded actions with the sampler active at each action boundary."""
        return [
            (step["action"], step["expectation"], step["sampling"])
            for step in self.replay_procedure(episode_id)
        ]

    def replay_procedure(self, episode_id: str) -> list[dict[str, Any]]:
        """Derive surviving moves with source labels and evidence for display."""
        action_rows = self.actions(episode_id)
        token_rows = self.tokens(episode_id)
        if not action_rows:
            return []
        segments = self.connection.execute(
            "SELECT * FROM sampler_segments "
            "WHERE episode_id = ? ORDER BY start_boundary",
            (episode_id,),
        ).fetchall()
        for segment in segments:
            validate_coordinate(segment["start_boundary"], "start_boundary")
            validate_coordinate(segment["coordinate_offset"], "coordinate_offset")
            validate_fingerprint(segment["stream_fingerprint"])
        starts = [segment["start_boundary"] for segment in segments]
        samplers: dict[int, SamplingConfig] = {}
        grouped: dict[int, list[dict[str, Any]]] = {}
        for token in token_rows:
            grouped.setdefault(int(token["action_ordinal"]), []).append(token)
        tape: list[dict[str, Any]] = []
        for row in action_rows:
            records = grouped.get(int(row["ordinal"]), [])
            segment_index = bisect_right(starts, int(row["boundary_before"])) - 1
            if segment_index < 0:
                raise EditorError(f"episode {episode_id!r} has no sampler segment")
            if segment_index not in samplers:
                samplers[segment_index] = SamplingConfig.from_record(
                    _loads(segments[segment_index]["sampling_json"], {})
                )
            sampling = samplers[segment_index]
            if row["status"] == "handed-off":
                # This row records an action that was deliberately not applied.
                # A partially realized autonomous span is retained as the
                # finite Hold that actually happened; later rows came from the
                # live teacher after handoff.
                partial = tuple(
                    int(record["token_id"])
                    for record in records
                    if bool(record["realized_visible"])
                )
                if partial:
                    tape.append(
                        {
                            "action": Hold(len(partial)),
                            "expectation": ReplayExpectation(partial, None, "requested-length"),
                            "sampling": sampling,
                            "boundary": int(row["boundary_before"]),
                            "tokens": records,
                        }
                    )
                continue
            visible = tuple(
                int(record["token_id"])
                for record in records
                if bool(record["realized_visible"])
            )
            terminal = next(
                (
                    int(record["token_id"])
                    for record in records
                    if bool(record["is_eog"])
                ),
                None,
            )
            tape.append(
                {
                    "action": action_from_dict(row["arguments"]),
                    "expectation": ReplayExpectation(visible, terminal, str(row["stop_reason"])),
                    "sampling": sampling,
                    "boundary": int(row["boundary_before"]),
                    "tokens": records,
                }
            )
        return tape

    def replay_until(self, episode_id: str, until: int | None = None) -> list[dict[str, Any]]:
        """Select source boundaries; destination writes retain normal token indexing."""
        if until is None:
            return self.replay_procedure(episode_id)
        length = sum(bool(row["realized_visible"]) for row in self.tokens(episode_id))
        if type(until) is not int or not 0 <= until <= length:
            raise EditorError(f"Replay boundary must be 0..{length}.")
        result = []
        for original in self.replay_procedure(episode_id):
            if original["boundary"] >= until:
                break
            step = dict(original)
            count = until - step["boundary"]
            visible = [row for row in step["tokens"] if row["realized_visible"]]
            if len(visible) > count or (len(visible) == count and isinstance(step["action"], (Hold, Finish))):
                retained = visible[:count]
                if isinstance(step["action"], Write):
                    step["action"] = Write("".join(row["text"] for row in retained), "exact")
                    reason = "completed"
                else:
                    step["action"] = Hold(count)
                    reason = "requested-length"
                step["tokens"] = retained
                step["expectation"] = ReplayExpectation(
                    tuple(row["token_id"] for row in retained), None, reason)
            result.append(step)
        return result

    def final_sampling(self, episode_id: str) -> SamplingConfig:
        """Return the latest source setting, including changes after its last move."""
        row = self.connection.execute(
            "SELECT sampling_json FROM sampler_segments WHERE episode_id = ? "
            "ORDER BY start_boundary DESC LIMIT 1", (episode_id,),
        ).fetchone()
        if row is None:
            raise EditorError(f"episode {episode_id!r} has no sampler segment")
        return SamplingConfig.from_record(_loads(row["sampling_json"], {}))

    def sampling_segment(self, episode_id: str, boundary: int = 0) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT * FROM sampler_segments
            WHERE episode_id = ? AND start_boundary <= ?
            ORDER BY start_boundary DESC LIMIT 1
            """,
            (episode_id, boundary),
        ).fetchone()
        if row is None:
            raise EditorError(f"episode {episode_id!r} has no sampler segment")
        result = dict(row)
        result["sampling"] = _loads(result.pop("sampling_json"), {})
        SamplingConfig.from_record(result["sampling"])
        validate_coordinate(result["start_boundary"], "start_boundary")
        validate_coordinate(result["coordinate_offset"], "coordinate_offset")
        validate_fingerprint(result["stream_fingerprint"])
        return result
