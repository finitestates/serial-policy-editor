"""A storage-neutral, root-relative history of episode action attempts.

``EpisodeHistory`` is the semantic seam between execution and persistence.
Adapters can turn rows or live outcomes into :class:`RecordedAttempt` values,
then use this module for boundary validation and prefix/rewind projection.
The history deliberately has no tokenizer or backend: visible text comes from
the recorded :class:`~trajectory_editor.core.results.TokenEvidence`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .core.actions import Hold, Phrase, PolicyAction, Write, action_from_dict
from .core.results import ActionOutcome, ReplayExpectation, TokenEvidence


def _visible_evidence(outcome: ActionOutcome) -> tuple[TokenEvidence, ...]:
    return tuple(item for item in outcome.evidence if item.realized_visible)


@dataclass(frozen=True, slots=True)
class RecordedAttempt:
    """One root-relative action attempt and the evidence it produced.

    ``action`` is the requested policy action.  It normally equals
    ``outcome.action``; keeping it as a first-class field makes the adapter
    contract explicit and lets validation reject mixed records.  An explicit
    ``expectation`` is used when the attempt came from an earlier replay; for
    ordinary live attempts, the outcome's expectation is used when exporting
    to a surviving procedure.
    """

    ordinal: int
    action: PolicyAction
    outcome: ActionOutcome
    expectation: ReplayExpectation | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("attempt ordinal must be a nonnegative integer")
        if not isinstance(self.outcome, ActionOutcome):
            raise TypeError("attempt outcome must be an ActionOutcome")
        if self.action != self.outcome.action:
            raise ValueError("attempt action must match outcome action")
        if self.expectation is not None and not isinstance(
            self.expectation, ReplayExpectation
        ):
            raise TypeError("attempt expectation must be a ReplayExpectation or None")

    @property
    def requested_action(self) -> PolicyAction:
        """Alias spelling for adapters that name the action explicitly."""

        return self.action

    @classmethod
    def from_outcome(
        cls,
        ordinal: int,
        outcome: ActionOutcome,
        *,
        expectation: ReplayExpectation | None = None,
    ) -> "RecordedAttempt":
        return cls(ordinal, outcome.action, outcome, expectation)


@dataclass(frozen=True, slots=True)
class HistoryTruncation:
    """The result of retaining a history prefix.

    ``discarded`` contains complete attempts that begin at or after the
    requested boundary.  If the boundary cuts an attempt, that original
    attempt is represented by ``partial`` and its retained form is in
    ``retained``; it is not duplicated in ``discarded``.
    """

    requested_boundary: int
    retained: "EpisodeHistory"
    discarded: tuple[RecordedAttempt, ...]
    partial: RecordedAttempt | None = None

    @property
    def retained_history(self) -> "EpisodeHistory":
        return self.retained

    @property
    def discarded_attempts(self) -> tuple[RecordedAttempt, ...]:
        return self.discarded


@dataclass(frozen=True, slots=True)
class StoredHistoryAction:
    """One persistence-neutral action record in a retained history prefix."""

    ordinal: int
    boundary_before: int
    boundary_after: int
    kind: str
    arguments: Mapping[str, Any]
    resolved_text: str
    status: str
    stop_reason: str
    mismatch: Mapping[str, Any] | None
    tokens: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class StoredHistoryPrefix:
    """A root-relative durable prefix projected from action and token records."""

    actions: tuple[StoredHistoryAction, ...]
    sampler_segments: tuple[Mapping[str, Any], ...]
    budget_segments: tuple[Mapping[str, Any], ...]
    source_boundary: int
    partial: RecordedAttempt | None = None


@dataclass(frozen=True, slots=True)
class EpisodeHistory:
    """Ordered action/evidence attempts on one root-relative visible stream."""

    attempts: tuple[RecordedAttempt, ...] = ()

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        object.__setattr__(self, "attempts", attempts)

        previous_ordinal: int | None = None
        previous_boundary = 0
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, RecordedAttempt):
                raise TypeError("history attempts must be RecordedAttempt values")
            if previous_ordinal is not None and attempt.ordinal <= previous_ordinal:
                raise ValueError("attempt ordinals must be strictly increasing")

            outcome = attempt.outcome
            before = outcome.boundary_before
            after = outcome.boundary_after
            if type(before) is not int or type(after) is not int:
                raise ValueError("attempt boundaries must be integers")
            if before < 0 or after < before:
                raise ValueError("attempt boundaries must be nonnegative and ordered")
            if index == 0 and before != 0:
                raise ValueError("history must begin at root-visible boundary zero")
            if before != previous_boundary:
                raise ValueError("attempt boundaries must be contiguous")
            if after - before != len(outcome.visible_token_ids):
                raise ValueError(
                    "boundary width must equal the number of visible token ids"
                )

            evidence = tuple(outcome.evidence)
            for item in evidence:
                if not isinstance(item, TokenEvidence):
                    raise TypeError("outcome evidence must contain TokenEvidence values")

            for item in evidence:
                if type(item.boundary) is not int or item.boundary < before:
                    raise ValueError("evidence boundaries must be root-relative")
                if item.realized_visible:
                    if item.boundary >= after:
                        raise ValueError(
                            "visible evidence must lie inside its visible boundary span"
                        )
                elif item.boundary > after:
                    raise ValueError(
                        "non-visible evidence must not pass the outcome boundary"
                    )

            visible = tuple(item for item in evidence if item.realized_visible)
            if tuple(item.token_id for item in visible) != tuple(
                outcome.visible_token_ids
            ):
                raise ValueError(
                    "visible evidence token ids must match the outcome visible ids"
                )
            if tuple(item.boundary for item in visible) != tuple(
                range(before, after)
            ):
                raise ValueError(
                    "visible evidence boundaries must be contiguous and ordered"
                )

            previous_ordinal = attempt.ordinal
            previous_boundary = after

    @classmethod
    def from_attempts(cls, attempts: Iterable[RecordedAttempt]) -> "EpisodeHistory":
        return cls(tuple(attempts))

    def __len__(self) -> int:
        return len(self.attempts)

    def __iter__(self) -> Iterator[RecordedAttempt]:
        return iter(self.attempts)

    @property
    def current_boundary(self) -> int:
        """The current root-relative visible boundary."""

        return self.attempts[-1].outcome.boundary_after if self.attempts else 0

    @property
    def visible_boundary(self) -> int:
        return self.current_boundary

    @property
    def visible_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for attempt in self.attempts
            for token_id in attempt.outcome.visible_token_ids
        )

    @property
    def visible_token_evidence(self) -> tuple[TokenEvidence, ...]:
        return tuple(
            item
            for attempt in self.attempts
            for item in _visible_evidence(attempt.outcome)
        )

    @property
    def visible_evidence(self) -> tuple[TokenEvidence, ...]:
        return self.visible_token_evidence

    @property
    def visible_text(self) -> str:
        return "".join(item.text for item in self.visible_token_evidence)

    def _partial_attempt(
        self, attempt: RecordedAttempt, boundary: int
    ) -> RecordedAttempt | None:
        outcome = attempt.outcome
        count = boundary - outcome.boundary_before
        retained_evidence = _visible_evidence(outcome)[:count]
        visible_ids = tuple(item.token_id for item in retained_evidence)
        if not visible_ids:
            return None

        retained_text = "".join(item.text for item in retained_evidence)
        if isinstance(attempt.action, (Write, Phrase)):
            action: PolicyAction = Write(retained_text, mode="exact")
            stop_reason = "completed"
        else:
            action = Hold(len(visible_ids))
            stop_reason = "requested-length"

        retained_outcome = replace(
            outcome,
            action=action,
            boundary_after=boundary,
            resolved_text=retained_text,
            resolved_token_ids=visible_ids,
            visible_token_ids=visible_ids,
            terminal_token_id=None,
            stop_reason=stop_reason,
            evidence=retained_evidence,
            status="completed",
            divergence=None,
            replay_eog_token_id=None,
            diagnostics=None,
        )
        expectation = ReplayExpectation(visible_ids, None, stop_reason)
        return RecordedAttempt(
            ordinal=attempt.ordinal,
            action=action,
            outcome=retained_outcome,
            expectation=expectation,
        )

    def truncate(self, boundary: int) -> HistoryTruncation:
        """Retain the history through ``boundary`` without rebasing it.

        Raw zero-width handed-off attempts before the requested boundary are
        retained as attempted actions.  The surviving-procedure projector is
        responsible for omitting them from replay.  A cut strictly inside an
        attempt creates one transformed retained attempt; all later complete
        attempts are returned as the discarded suffix.
        """

        if type(boundary) is not int or boundary < 0:
            raise ValueError("retained boundary must be a nonnegative integer")
        if boundary > self.current_boundary:
            raise ValueError(
                f"retained boundary must be between 0 and {self.current_boundary}"
            )

        retained: list[RecordedAttempt] = []
        discarded: list[RecordedAttempt] = []
        partial: RecordedAttempt | None = None

        for attempt in self.attempts:
            outcome = attempt.outcome
            before = outcome.boundary_before
            after = outcome.boundary_after

            if after < boundary or (after == boundary and before < boundary):
                retained.append(attempt)
                continue

            if before < boundary < after:
                partial = attempt
                transformed = self._partial_attempt(attempt, boundary)
                if transformed is not None:
                    retained.append(transformed)
                else:
                    discarded.append(attempt)
                continue

            discarded.append(attempt)

        return HistoryTruncation(
            requested_boundary=boundary,
            retained=EpisodeHistory(tuple(retained)),
            discarded=tuple(discarded),
            partial=partial,
        )

    def retain_through(self, boundary: int) -> "EpisodeHistory":
        """Return only the root-relative history retained through ``boundary``."""

        return self.truncate(boundary).retained


def visible_text_prefix(
    tokens: Sequence[Mapping[str, Any]],
    boundary: int,
) -> str:
    """Render visible record text strictly before a root-relative boundary."""

    if type(boundary) is not int or boundary < 0:
        raise ValueError("retained boundary must be a nonnegative integer")
    return "".join(
        str(token["text"])
        for token in tokens
        if bool(token["realized_visible"]) and int(token["boundary"]) < boundary
    )


def materialize_stored_prefix(
    actions: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
    sampler_segments: Sequence[Mapping[str, Any]],
    budget_segments: Sequence[Mapping[str, Any]],
    boundary: int,
) -> StoredHistoryPrefix:
    """Project durable records into one retained root-relative history prefix.

    The records are adapter-shaped mappings rather than SQLite rows.  This
    function delegates action truncation to :class:`EpisodeHistory` and only
    preserves the extra serializable details that execution outcomes do not
    model, such as token diagnostics and replay metadata.
    """

    if type(boundary) is not int or boundary < 0:
        raise ValueError("retained boundary must be a nonnegative integer")
    visible_count = sum(bool(token.get("realized_visible")) for token in tokens)
    if boundary > visible_count:
        raise ValueError(
            f"retained boundary must be between 0 and {visible_count}"
        )

    grouped_tokens: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for token in tokens:
        try:
            ordinal = int(token["action_ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("stored token has no valid action ordinal") from exc
        grouped_tokens[ordinal] = (*grouped_tokens.get(ordinal, ()), token)

    source_actions: dict[int, Mapping[str, Any]] = {}
    attempts: list[RecordedAttempt] = []
    try:
        ordered_actions = sorted(actions, key=lambda item: int(item["ordinal"]))
        for source in ordered_actions:
            ordinal = int(source["ordinal"])
            source_actions[ordinal] = source
            arguments = source.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("stored action arguments must be an object")
            action = action_from_dict(dict(arguments))
            evidence = tuple(
                _evidence_from_record(record)
                for record in grouped_tokens.get(ordinal, ())
            )
            visible = tuple(
                item.token_id for item in evidence if item.realized_visible
            )
            terminal = next(
                (item.token_id for item in evidence if item.is_eog),
                None,
            )
            outcome = ActionOutcome(
                action=action,
                boundary_before=int(source["boundary_before"]),
                boundary_after=int(source["boundary_after"]),
                resolved_text=str(source["resolved_text"]),
                resolved_token_ids=tuple(item.token_id for item in evidence),
                visible_token_ids=visible,
                terminal_token_id=terminal,
                stop_reason=str(source["stop_reason"]),
                evidence=evidence,
                status=str(source["status"]),
            )
            attempts.append(RecordedAttempt(ordinal, action, outcome))
        history = EpisodeHistory(tuple(attempts))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"stored history is invalid: {exc}") from exc

    truncation = history.truncate(boundary)
    partial_ordinal = (
        truncation.partial.ordinal if truncation.partial is not None else None
    )
    materialized_actions = tuple(
        _stored_history_action(
            source_actions[attempt.ordinal],
            grouped_tokens.get(attempt.ordinal, ()),
            attempt,
            boundary=boundary,
            partial=attempt.ordinal == partial_ordinal,
        )
        for attempt in truncation.retained
    )
    return StoredHistoryPrefix(
        actions=materialized_actions,
        sampler_segments=tuple(
            dict(segment)
            for segment in sampler_segments
            if int(segment["start_boundary"]) <= boundary
        ),
        budget_segments=tuple(
            dict(segment)
            for segment in budget_segments
            if int(segment["start_boundary"]) <= boundary
        ),
        source_boundary=boundary,
        partial=truncation.partial,
    )


def _evidence_from_record(record: Mapping[str, Any]) -> TokenEvidence:
    """Adapt a serializable evidence record for storage-neutral validation."""

    boundary = int(record["boundary"])
    # Lightweight adapters that only need boundary/text projection may omit a
    # token id; use the distinct root coordinate as a harmless placeholder.
    token_id = int(record.get("token_id", boundary))
    return TokenEvidence(
        boundary=boundary,
        sampling_coordinate=int(record.get("sampling_coordinate", boundary)),
        token_id=token_id,
        text=str(record["text"]),
        proposal_token_id=int(record.get("proposal_token_id", token_id)),
        raw_model_nll=(float(record["raw_model_nll"]) if record.get("raw_model_nll") is not None else None),
        raw_rank=(int(record["raw_rank"]) if record.get("raw_rank") is not None else None),
        policy_rank=(int(record["policy_rank"]) if record.get("policy_rank") is not None else None),
        decoder_probability=float(record.get("decoder_probability", 0.0)),
        proposal_agreement=bool(record.get("proposal_agreement", False)),
        is_eog=bool(record.get("is_eog", False)),
        realized_visible=bool(record.get("realized_visible", False)),
    )


def _stored_history_action(
    source: Mapping[str, Any],
    source_tokens: Sequence[Mapping[str, Any]],
    attempt: RecordedAttempt,
    *,
    boundary: int,
    partial: bool,
) -> StoredHistoryAction:
    """Keep source records intact except for the one action cut by a prefix."""

    if not partial:
        arguments = source.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("stored action arguments must be an object")
        mismatch = source.get("mismatch")
        return StoredHistoryAction(
            ordinal=attempt.ordinal,
            boundary_before=attempt.outcome.boundary_before,
            boundary_after=attempt.outcome.boundary_after,
            kind=str(source["kind"]),
            arguments=dict(arguments),
            resolved_text=str(source["resolved_text"]),
            status=str(source["status"]),
            stop_reason=str(source["stop_reason"]),
            mismatch=dict(mismatch) if isinstance(mismatch, Mapping) else None,
            tokens=tuple(source_tokens),
        )

    arguments = source.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("stored action arguments must be an object")
    original_arguments = dict(arguments)
    original_kind = str(source["kind"])
    outcome = attempt.outcome
    if isinstance(attempt.action, Write):
        original_key = "original_write" if original_kind == "write" else "original_action"
        arguments = {
            **original_arguments,
            "kind": "write",
            "mode": "exact",
            "text": outcome.resolved_text,
            original_key: original_arguments.get(original_key, original_arguments),
        }
    else:
        arguments = {
            **original_arguments,
            "limit": len(outcome.visible_token_ids),
            "boundary": None,
        }
    return StoredHistoryAction(
        ordinal=attempt.ordinal,
        boundary_before=outcome.boundary_before,
        boundary_after=outcome.boundary_after,
        kind=outcome.action.kind,
        arguments=arguments,
        resolved_text=outcome.resolved_text,
        status=outcome.status,
        stop_reason=outcome.stop_reason,
        mismatch=None,
        tokens=tuple(
            token
            for token in source_tokens
            if int(token["boundary"]) < boundary
        ),
    )


__all__ = [
    "EpisodeHistory",
    "HistoryTruncation",
    "RecordedAttempt",
    "StoredHistoryAction",
    "StoredHistoryPrefix",
    "materialize_stored_prefix",
    "visible_text_prefix",
]
