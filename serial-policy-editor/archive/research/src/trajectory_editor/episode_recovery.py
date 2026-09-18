"""Reviewable recovery of sampler metadata; never repair the source in place."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from .domain import EditorError, SamplingConfig
from .episode_hash import token_prefix_sha256, validate_coordinate, validate_fingerprint
from .episode_store import EpisodeStore, _json, _utc_now


@dataclass(frozen=True)
class RecoveryPlan:
    source_id: str
    segments: tuple[dict[str, Any], ...]
    changes: tuple[str, ...]


def inspect_sampler_record(store: EpisodeStore, episode_id: str) -> RecoveryPlan:
    source = store.get_episode(episode_id)
    # Invalid token history cannot be repaired by inventing sampler settings.
    fresh_fingerprint = token_prefix_sha256(source["initial_token_ids"])
    rows = [dict(row) for row in store.connection.execute(
        "SELECT * FROM sampler_segments WHERE episode_id = ? ORDER BY start_boundary",
        (episode_id,),
    )]
    changes: list[str] = []
    defaults = SamplingConfig().to_dict()
    if not rows or rows[0]["start_boundary"] != 0:
        rows.insert(0, dict(start_boundary=0, sampling_json="{}",
                            stream_fingerprint=None, coordinate_offset=None))
        changes.append("Missing initial sampler segment: would establish one at boundary 0.")
    repaired = []
    for row in rows:
        boundary = validate_coordinate(row["start_boundary"], "saved start_boundary")
        label = f"Boundary {boundary}"
        try:
            values = json.loads(row["sampling_json"])
        except (TypeError, ValueError):
            values = None
        if not isinstance(values, dict):
            changes.append(f"{label}: malformed sampler object would be replaced with defaults.")
            values = {}
        values = dict(values)
        for field, default in defaults.items():
            if field not in values:
                changes.append(f"{label}: missing {field} would become {default!r}.")
                values[field] = default
            elif field in {"rng_scheme", "policy_scheme", "history_scope"}:
                if values[field] != default:
                    raise EditorError(f"{label}: unsupported {field} {values[field]!r}; use a compatible build. Source unchanged.")
            else:
                try:
                    SamplingConfig.from_mapping({field: values[field]})
                except EditorError:
                    changes.append(f"{label}: invalid {field} {values[field]!r} would become {default!r}.")
                    values[field] = default
        SamplingConfig.from_record(values)
        fingerprint = row["stream_fingerprint"]
        offset = row["coordinate_offset"]
        try:
            validate_fingerprint(fingerprint)
        except EditorError:
            fingerprint = fresh_fingerprint
            changes.append(f"{label}: invalid/missing stream identity would become {fingerprint} (hash of this episode's initial tokens).")
        try:
            validate_coordinate(offset, "coordinate_offset")
        except EditorError:
            offset = 0
            changes.append(f"{label}: invalid/missing coordinate offset would become 0.")
        repaired.append(dict(start_boundary=boundary, sampling=values,
                             stream_fingerprint=fingerprint, coordinate_offset=offset))
    return RecoveryPlan(episode_id, tuple(repaired), tuple(changes))


def recover_sampler_record(store: EpisodeStore, episode_id: str, io) -> str:
    plan = inspect_sampler_record(store, episode_id)
    if not plan.changes:
        return episode_id
    io.write(f"Saved sampler state in {store.label(episode_id)} needs recovery:")
    for change in plan.changes:
        io.write("  " + change)
    io.write(
        "Proceeding would create a separate recovered copy and leave the source unchanged. "
        "Substituted settings or stream coordinates may change subsequent draws and replay results. "
        "Earlier token evidence would remain historical, not recomputed under the repaired settings."
    )
    while True:
        answer = io.read("Do you wish to proceed? [Y/n]> ")
        if answer is None or answer.strip().lower() in {"n", "no"}:
            raise EditorError("sampler recovery cancelled; source unchanged")
        if answer.strip().lower() in {"", "y", "yes"}:
            break
        io.write("Enter Y to recover a copy, or N to leave the source unchanged.")
    identifier = uuid.uuid4().hex
    source = store.get_episode(episode_id)
    metadata = dict(source["metadata"])
    metadata["sampler_recovery"] = {"source": episode_id, "changes": list(plan.changes)}
    with store.transaction() as db:
        # Copy the ledger as evidence, retaining its original coordinates. Only
        # the explicitly reviewed sampler metadata is repaired in the copy.
        for table in ("episodes", "actions", "tokens", "interactions", "budget_segments"):
            rows = db.execute(f"SELECT * FROM {table} WHERE episode_id = ?", (episode_id,)).fetchall()
            for original in rows:
                record = dict(original)
                record["episode_id"] = identifier
                if table == "episodes":
                    record["metadata_json"] = _json(metadata)
                    record["created_at"] = _utc_now()
                elif table == "interactions":
                    record.pop("interaction_id")
                columns = list(record)
                db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                           tuple(record[column] for column in columns))
        for segment in plan.segments:
            db.execute("""INSERT INTO sampler_segments
                (episode_id, start_boundary, sampling_json, stream_fingerprint, coordinate_offset)
                VALUES (?, ?, ?, ?, ?)""", (identifier, segment["start_boundary"],
                _json(segment["sampling"]), segment["stream_fingerprint"], segment["coordinate_offset"]))
        title = store.label(episode_id).split("  ", 1)[-1] + " · recovered"
        db.execute("INSERT INTO episode_names(episode_id, title, visited_at) VALUES (?, ?, ?)",
                   (identifier, title, _utc_now()))
    io.write(f"Recovered copy: {store.label(identifier)}")
    return identifier
