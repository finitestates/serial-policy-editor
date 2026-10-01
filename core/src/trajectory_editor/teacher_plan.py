"""Portable YAML and JSONL tapes for executable policy replay.

The tape is deliberately only the action procedure and optional observations.
The envelope carries the starting prompt, optional sampler defaults, and
recording circumstances. CLI sampler options can override those defaults.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .core.actions import action_from_dict
from .core.errors import EditorError
from .core.results import ReplayExpectation
from .core.sampler_config import SamplerConfig
from .episode_replay_source import replay_procedure
from .run_loop import ReplayPlan, TapeStep
from .surviving_procedure import ProcedureRecord, project_surviving_procedure

TAPE_FORMAT = "serial-policy-tape"
TAPE_VERSION = 1


@dataclass(frozen=True)
class TeacherTape:
    """An executable plan plus its prompt and optional initial configuration."""

    plan: ReplayPlan
    envelope: dict[str, Any]
    initial_sampling: SamplerConfig | None = None


class _DuplicateKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _DuplicateKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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


def _initial_sampling(envelope: Mapping[str, Any], *, label: str) -> SamplerConfig | None:
    environment = envelope.get("environment")
    if not isinstance(environment, Mapping) or "sampler" not in environment:
        return None
    sampler = environment["sampler"]
    if sampler is None:
        return None
    if not isinstance(sampler, Mapping):
        raise EditorError(f"{label}: environment.sampler must be an object")
    values = SamplerConfig().to_dict()
    values.update(sampler)
    try:
        return SamplerConfig.from_record(values)
    except EditorError as exc:
        raise EditorError(f"{label}: invalid environment.sampler: {exc}") from exc


def _load_yaml(text: str, *, path: Path) -> object:
    try:
        return yaml.load(text, Loader=_DuplicateKeyLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f"{path}:{mark.line + 1}:{mark.column + 1}" if mark else str(path)
        detail = getattr(exc, "problem", None) or str(exc)
        raise EditorError(f"{location}: invalid teacher-plan YAML: {detail}") from exc


def _validate_yaml_step_fields(records: list[object], *, path: Path) -> None:
    step_fields = {"step", "action", "observation"}
    observation_fields = {"token_ids", "terminal_token_id", "stop_reason"}
    action_fields = {
        "accept": {"kind"},
        "select": {"kind", "rank", "selected_rank"},
        "select-raw-rank": {"kind", "rank", "selected_rank"},
        "insert": {"kind", "text", "supplied_text", "mode", "insert_mode"},
        "write": {"kind", "text", "supplied_text", "mode", "insert_mode"},
        "check-phrase": {"kind", "text", "supplied_text", "mode", "max_tokens", "max_shift"},
        "force-phrase": {"kind", "text", "supplied_text", "mode", "max_tokens", "max_shift"},
        "hold": {"kind", "limit", "requested_visible_tokens", "boundary"},
        "teacher-eog": {"kind"},
        "end-generation": {"kind"},
        "reroll": {"kind", "seed"},
        "set-sampler": {"kind", "sampling"},
    }
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        unknown = set(record) - step_fields
        if unknown:
            names = ", ".join(sorted(map(repr, unknown)))
            raise EditorError(f"{path}: teacher plan step {ordinal}: unknown fields: {names}")
        action = record.get("action")
        if isinstance(action, Mapping):
            kind = action.get("kind")
            allowed = action_fields.get(kind) if isinstance(kind, str) else None
            if allowed is not None:
                unknown = set(action) - allowed
                if unknown:
                    names = ", ".join(sorted(map(repr, unknown)))
                    raise EditorError(
                        f"{path}: teacher plan step {ordinal}: unknown action fields: {names}"
                    )
        observation = record.get("observation")
        if isinstance(observation, Mapping):
            unknown = set(observation) - observation_fields
            if unknown:
                names = ", ".join(sorted(map(repr, unknown)))
                raise EditorError(
                    f"{path}: teacher plan step {ordinal}: unknown observation fields: {names}"
                )


def load_teacher_tape_yaml(
    path: Path,
    *,
    envelope_path: Path | None = None,
    require_observations: bool = False,
) -> TeacherTape:
    """Load one human-readable YAML document containing a teacher tape."""
    selected = Path(path)
    try:
        payload = _load_yaml(selected.read_text(encoding="utf-8"), path=selected)
    except OSError as exc:
        raise EditorError(f"could not read teacher plan {selected}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EditorError(f"{selected}: expected a YAML mapping")
    allowed_fields = {"format", "type", "version", "prompt", "environment", "steps"}
    unknown = set(payload) - allowed_fields
    if unknown:
        names = ", ".join(sorted(map(repr, unknown)))
        raise EditorError(f"{selected}: unknown top-level fields: {names}")
    records = payload.get("steps")
    if not isinstance(records, list):
        raise EditorError(f"{selected}: steps must be a list")
    _validate_yaml_step_fields(records, path=selected)

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
    embedded = _validate_envelope(
        {key: value for key, value in payload.items() if key != "steps"},
        label=str(selected),
    )
    envelope = {**sidecar, **embedded}
    try:
        plan = load_teacher_plan(records, require_observations=require_observations)
    except EditorError as exc:
        raise EditorError(f"{selected}: {exc}") from exc
    return TeacherTape(plan, envelope, _initial_sampling(envelope, label=str(selected)))


def load_teacher_tape(
    path: Path,
    *,
    envelope_path: Path | None = None,
    require_observations: bool = False,
) -> TeacherTape:
    """Load a YAML plan by suffix or retain the existing JSONL tape reader."""
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        return load_teacher_tape_yaml(
            path,
            envelope_path=envelope_path,
            require_observations=require_observations,
        )
    return load_teacher_tape_jsonl(
        path,
        envelope_path=envelope_path,
        require_observations=require_observations,
    )


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
    return ReplayPlan(steps=tuple(steps))


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
    plan = load_teacher_plan(records, require_observations=require_observations)
    return TeacherTape(plan, envelope, _initial_sampling(envelope, label=str(path)))


def teacher_tape_yaml_text(envelope: Mapping[str, Any], steps: Iterable[Any]) -> str:
    """Serialize typed teacher steps as one editable YAML document."""
    normalized = _validate_envelope(envelope, label="teacher tape")
    document: dict[str, Any] = {
        "format": TAPE_FORMAT,
        "version": TAPE_VERSION,
    }
    if "prompt" in normalized:
        document["prompt"] = normalized["prompt"]
    if "environment" in normalized:
        document["environment"] = dict(normalized["environment"])

    records: list[dict[str, Any]] = []
    for ordinal, step in enumerate(steps):
        if isinstance(step, Mapping):
            action = step["action"]
            expectation = step.get("expectation")
        else:
            action = step.action
            expectation = step.expectation
        action_record = action if isinstance(action, Mapping) else action.to_dict()
        record: dict[str, Any] = {
            "step": ordinal,
            "action": dict(action_record),
        }
        if expectation is not None:
            if isinstance(expectation, Mapping):
                observation = dict(expectation)
            else:
                observation = {
                    "token_ids": list(expectation.token_ids),
                    "terminal_token_id": expectation.terminal_token_id,
                    "stop_reason": expectation.stop_reason,
                }
            record["observation"] = observation
        records.append(record)
    document["steps"] = records
    return yaml.safe_dump(
        document,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )


def _write_teacher_tape(
    path: Path,
    envelope: dict[str, Any],
    steps: Iterable[Any],
    *,
    envelope_path: Path | None,
) -> dict[str, Any]:
    """Write a stored or live procedure as YAML or JSONL by file suffix."""

    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            path.write_text(teacher_tape_yaml_text(envelope, steps), encoding="utf-8")
            if envelope_path is not None:
                envelope_path.write_text(
                    json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
            return envelope
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
    """Export a stored episode as YAML or JSONL plus an optional JSON sidecar."""
    episode = store.get_episode(episode_id)
    initial = episode["initial_sampling"]
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
    """Export the selected non-durable branch as a portable YAML or JSONL tape."""
    environment = dict(getattr(session, "environment", {}))
    branch_state = session.branch_state
    if callable(branch_state):
        branch_state = branch_state()
    environment["sampler"] = branch_state.initial_sampling.to_dict()
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


__all__ = [
    "TAPE_FORMAT",
    "TAPE_VERSION",
    "TeacherTape",
    "export_live_teacher_tape",
    "export_teacher_tape",
    "load_teacher_plan",
    "load_teacher_tape",
    "load_teacher_tape_jsonl",
    "load_teacher_tape_yaml",
    "teacher_tape_yaml_text",
]
