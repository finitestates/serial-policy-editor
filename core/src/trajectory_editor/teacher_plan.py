"""Portable JSONL tapes for executable policy replay.

The tape is deliberately only the action procedure and optional observations.
An envelope describes the circumstances in which it was recorded, but is
advisory: callers may always run the tape under different conditions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .core.actions import action_from_dict
from .core.errors import EditorError
from .core.results import ReplayExpectation
from .episode_replay_source import replay_procedure
from .episode_runner import ReplayPlan, TapeStep
from .surviving_procedure import ProcedureRecord, project_surviving_procedure


TAPE_FORMAT = "serial-policy-tape"
TAPE_VERSION = 1


@dataclass(frozen=True)
class TeacherTape:
    """An executable plan plus its optional, non-binding description."""

    plan: ReplayPlan
    envelope: dict[str, Any]


def _validate_envelope(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EditorError(f"{label}: expected an object")
    result = dict(value)
    format_name = result.get("format", result.get("type"))
    if format_name != TAPE_FORMAT:
        raise EditorError(f"{label}: unsupported tape format {format_name!r}")
    if result.get("version") != TAPE_VERSION:
        raise EditorError(
            f"{label}: unsupported tape version {result.get('version')!r}"
        )
    prompt = result.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise EditorError(f"{label}: prompt must be a string")
    environment = result.get("environment")
    if environment is not None and not isinstance(environment, Mapping):
        raise EditorError(f"{label}: environment must be an object")
    result["format"] = TAPE_FORMAT
    result["version"] = TAPE_VERSION
    result.pop("type", None)
    return result


def load_teacher_plan(
    records: Iterable[object], *, require_observations: bool = False,
) -> ReplayPlan:
    """Convert external step records into an executable replay plan."""
    steps: list[TapeStep] = []
    for ordinal, record in enumerate(records):
        label = f"teacher plan step {ordinal}"
        if not isinstance(record, Mapping):
            raise EditorError(f"{label}: expected an object")
        supplied_step = record.get("step")
        if type(supplied_step) is not int or supplied_step != ordinal:
            raise EditorError(f"{label}: expected step={ordinal}, got {supplied_step!r}")
        action_record = record.get("action")
        if not isinstance(action_record, Mapping):
            raise EditorError(f"{label}: missing action")
        try:
            action = action_from_dict(action_record)
        except EditorError as exc:
            raise EditorError(f"{label}: invalid action: {exc}") from exc
        observation_record = record.get("observation")
        if observation_record is None:
            if require_observations:
                raise EditorError(f"{label}: missing observation")
            expectation = None
        elif not isinstance(observation_record, Mapping):
            raise EditorError(f"{label}: observation must be an object")
        else:
            try:
                expectation = ReplayExpectation.from_mapping(observation_record)
            except EditorError as exc:
                raise EditorError(f"{label}: invalid observation: {exc}") from exc
        steps.append(TapeStep(action, expectation))
    return ReplayPlan(steps=tuple(steps), follow_source_sampling=False)


def load_teacher_tape_jsonl(
    path: Path,
    *,
    envelope_path: Path | None = None,
    require_observations: bool = False,
) -> TeacherTape:
    """Load a tape with either an embedded header, a JSON sidecar, or both."""
    sidecar: dict[str, Any] = {}
    if envelope_path is not None:
        try:
            sidecar = _validate_envelope(
                json.loads(envelope_path.read_text(encoding="utf-8")),
                label=f"teacher tape envelope {envelope_path}",
            )
        except OSError as exc:
            raise EditorError(f"could not read teacher tape envelope {envelope_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EditorError(f"{envelope_path}: invalid JSON: {exc.msg}") from exc
    records: list[object] = []
    embedded: dict[str, Any] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EditorError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not records and not embedded and isinstance(record, Mapping) and record.get("type") == TAPE_FORMAT:
                    embedded = _validate_envelope(record, label=f"{path}:{line_number}")
                else:
                    records.append(record)
    except OSError as exc:
        raise EditorError(f"could not read teacher plan {path}: {exc}") from exc
    envelope = {**sidecar, **embedded}
    return TeacherTape(load_teacher_plan(records, require_observations=require_observations), envelope)


def _write_teacher_tape(
    path: Path,
    envelope: dict[str, Any],
    steps: Iterable[Any],
    *,
    envelope_path: Path | None,
) -> dict[str, Any]:
    """Write either a stored or live procedure through one serializer."""

    try:
        with path.open("w", encoding="utf-8") as handle:
            if envelope_path is None:
                header = {
                    "type": TAPE_FORMAT,
                    **{
                        key: value
                        for key, value in envelope.items()
                        if key != "format"
                    },
                }
                handle.write(
                    json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n"
                )
            for ordinal, step in enumerate(steps):
                if isinstance(step, Mapping):
                    action = step["action"]
                    expectation = step["expectation"]
                else:
                    action = step.action
                    expectation = step.expectation
                record: dict[str, Any] = {
                    "step": ordinal,
                    "action": action.to_dict(),
                }
                if expectation is not None:
                    record["observation"] = {
                        "token_ids": list(expectation.token_ids),
                        "terminal_token_id": expectation.terminal_token_id,
                        "stop_reason": expectation.stop_reason,
                    }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        if envelope_path is not None:
            envelope_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    except OSError as exc:
        raise EditorError(f"could not write teacher tape {path}: {exc}") from exc
    return envelope


def _project_live_tape(session: Any) -> tuple[TapeStep, ...]:
    """Project live attempts into the same procedure as durable export.

    A live branch keeps both the attempted tape and its runtime outcomes so
    that rewinds and handoffs remain inspectable.  A portable tape should only
    contain the surviving procedure, however.  Keep the original tape
    expectation for ordinary records, while leaving ``visible_text`` empty so
    partial handoffs use SQLite's finite-Hold convention.
    """
    tape = tuple(session.history_tape)
    outcomes = tuple(session.history_outcomes)
    if len(tape) != len(outcomes):
        raise EditorError("live teacher tape and outcomes must align")
    records = tuple(
        ProcedureRecord(
            action=step.action,
            expectation=step.expectation,
            status=outcome.status,
            visible_token_ids=tuple(outcome.visible_token_ids),
            visible_text="",
            boundary_before=outcome.boundary_before,
        )
        for step, outcome in zip(tape, outcomes)
    )
    return project_surviving_procedure(
        records,
        normalize_for_replay=False,
    ).tape


def export_teacher_tape(
    store: Any,
    episode_id: str,
    path: Path,
    *,
    envelope_path: Path | None = None,
) -> dict[str, Any]:
    """Export a stored episode as JSONL plus an optional JSON envelope sidecar."""
    episode = store.get_episode(episode_id)
    initial = store.sampling_segment(episode_id, 0)["sampling"]
    envelope = {
        "format": TAPE_FORMAT,
        "version": TAPE_VERSION,
        "prompt": episode["initial_text"],
        "environment": {"backend": episode["backend"], "sampler": initial},
    }
    return _write_teacher_tape(
        path,
        envelope,
        replay_procedure(store, episode_id),
        envelope_path=envelope_path,
    )


def export_live_teacher_tape(
    session: Any,
    path: Path,
    *,
    envelope_path: Path | None = None,
) -> dict[str, Any]:
    """Export the currently selected non-durable branch as a portable tape."""
    environment = dict(getattr(session, "environment", {}))
    environment.setdefault("sampler", session.sampler.to_dict())
    envelope = {
        "format": TAPE_FORMAT,
        "version": TAPE_VERSION,
        "prompt": session.prompt,
        "environment": environment,
    }
    return _write_teacher_tape(
        path,
        envelope,
        _project_live_tape(session),
        envelope_path=envelope_path,
    )


__all__ = ["TAPE_FORMAT", "TAPE_VERSION", "TeacherTape", "export_live_teacher_tape", "export_teacher_tape", "load_teacher_plan", "load_teacher_tape_jsonl"]
