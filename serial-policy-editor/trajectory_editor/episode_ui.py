"""Terminal policy adapter for the unified episode engine.

The adapter owns presentation and navigation.  In particular, full-vocabulary
``/`` search remains a non-mutating lens over the current observation; it is
never compiled into a replay tape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .domain import Candidate, ChoiceSet, EditorError
from .episode_actions import (
    EndGeneration,
    Finish,
    Hold,
    PolicyAction,
    SelectRawRank,
    Write,
)
from .episode_engine import EpisodeEngine, Observation
from .episode_policy import (
    EdgeRequested,
    ForkRequested,
    SeamlessEdgeRequested,
    SeamlessRewindRequested,
)
from .episode_store import EpisodeStore
from .episode_hash import token_prefix_sha256
from .sampling import raw_rank
from .tui import (
    HELP_TEXT,
    IO,
    BoundaryReview,
    ChoiceFeedback,
    CommandKind,
    ForkAddressKind,
    SEAMLESS_REACTIVATE,
    TerminalIO,
    display_candidates,
    display_choice,
    parse_command,
)


@dataclass
class SearchLens:
    query: str
    token_id: int
    target_rank: int
    lower_rank: int
    upper_rank: int


def _choice_from_observation(
    engine: EpisodeEngine,
    observation: Observation,
    candidates: tuple[Candidate, ...],
    *,
    context_characters: int,
    serial: int,
) -> ChoiceSet:
    return ChoiceSet(
        choice_set_id=f"episode-choice-{serial:06d}",
        prompt_id="episode",
        aligned_step=observation.boundary,
        sampling_coordinate=observation.sampling_coordinate,
        context_token_sha256=token_prefix_sha256(list(observation.prefix_token_ids)),
        context_text_tail=observation.context_text[-context_characters:],
        proposal_token_id=observation.proposal_token_id,
        proposal_text=observation.proposal_text,
        proposal_raw_probability=observation.proposal_raw_probability,
        proposal_decoder_probability=observation.proposal_decoder_probability,
        proposal_is_eog=engine.backend.is_eog(observation.proposal_token_id),
        candidates=candidates,
        vocabulary_size=len(observation.logits),
        proposal_raw_rank=observation.proposal_raw_rank,
        proposal_policy_rank=observation.proposal_policy_rank,
        proposal_policy_probability=None,
        proposal_policy_logit_adjustment=None,
    )


class InteractivePolicy:
    def __init__(
        self,
        *,
        io: IO | None = None,
        menu_size: int = 12,
        search_radius: int = 3,
        default_hold_tokens: int = 24,
        context_characters: int = 0,
        manual_acceptance: bool = False,
        show_policy_rank: bool = False,
        store: EpisodeStore | None = None,
        episode_id: str | None = None,
        seamless: bool = False,
    ) -> None:
        if min(menu_size, search_radius, default_hold_tokens) < 1:
            raise EditorError("menu, search, and hold sizes must be positive")
        self.io = io or TerminalIO()
        self.menu_size = menu_size
        self.search_radius = search_radius
        self.default_hold_tokens = default_hold_tokens
        self.context_characters = context_characters
        self.manual_acceptance = bool(manual_acceptance)
        self.show_policy_rank = bool(show_policy_rank)
        self.store = store
        self.episode_id = episode_id
        self.seamless = bool(seamless)
        self.choice_serial = 0

    def _interaction(
        self, boundary: int, kind: str, payload: Mapping[str, Any]
    ) -> None:
        if self.store is not None and self.episode_id is not None:
            self.store.record_interaction(self.episode_id, boundary, kind, payload)

    @staticmethod
    def _resolve_write(engine: EpisodeEngine, text: str, mode: object) -> str:
        value = getattr(mode, "value", mode)
        _, resolved = engine._write_tokens(Write(text, str(value)))
        return resolved

    def _search(
        self,
        engine: EpisodeEngine,
        observation: Observation,
        query: str,
        invoked_as: str,
    ) -> tuple[SearchLens | None, ChoiceFeedback]:
        try:
            token_ids = engine.backend.tokenize(query, add_bos=False, special=False)
        except Exception as exc:
            raise EditorError(f"the tokenizer rejected {query!r}: {exc}") from exc
        if not token_ids:
            raise EditorError(
                f"no single-token form exists for {query!r}; the tokenizer produced no tokens"
            )
        if len(token_ids) != 1:
            pieces = tuple(
                (
                    int(token_id),
                    engine.backend.render([int(token_id)], special=False),
                )
                for token_id in token_ids
            )
            if any(
                token_id < 0 or token_id >= engine.backend.vocabulary_size()
                for token_id, _ in pieces
            ):
                raise EditorError(
                    "the tokenizer returned a token outside the vocabulary"
                )
            suggestions = tuple(
                "/" + json.dumps(text, ensure_ascii=False) for _, text in pieces
            )
            self._interaction(
                observation.boundary,
                "vocabulary-search-multiple-tokens",
                {
                    "query": query,
                    "invoked_as": invoked_as,
                    "tokens": [
                        {"token_id": token_id, "text": text}
                        for token_id, text in pieces
                    ],
                },
            )
            feedback = ChoiceFeedback(
                category="search",
                title=f"SEARCH · {query!r} returned {len(pieces)} tokens",
                lines=tuple(
                    f"{suggestion}  id {token_id}  text {text!r}"
                    for suggestion, (token_id, text) in zip(suggestions, pieces)
                ),
                completion_commands=suggestions,
            )
            if not getattr(self.io, "supports_live_choices", False):
                self.io.write(feedback.title)
                for line in feedback.lines:
                    self.io.write(line)
            return None, feedback
        token_id = int(token_ids[0])
        if not 0 <= token_id < len(observation.logits):
            raise EditorError("the tokenizer returned a token outside the vocabulary")
        rendered = engine.backend.render([token_id], special=False)
        if rendered != query:
            suggestion = "/" + json.dumps(rendered, ensure_ascii=False)
            raise EditorError(
                f"no exact single-token form exists for {query!r}; token id "
                f"{token_id} renders as {rendered!r}. Search it with {suggestion}"
            )
        rank = raw_rank(observation.logits, token_id)
        lens = SearchLens(
            query=query,
            token_id=token_id,
            target_rank=rank,
            lower_rank=max(1, rank - self.search_radius),
            upper_rank=min(len(observation.logits), rank + self.search_radius),
        )
        self._record_search_view(
            engine, observation, lens, invoked_as=invoked_as, invocation="search"
        )
        return lens, self._search_feedback(lens)

    @staticmethod
    def _search_feedback(lens: SearchLens) -> ChoiceFeedback:
        return ChoiceFeedback(
            category="search",
            title=f"SEARCH · {lens.query!r} matched rank {lens.target_rank}",
            lines=(
                (
                    f"token {lens.token_id} · neighborhood ranks "
                    f"{lens.lower_rank}–{lens.upper_rank}"
                ),
            ),
            initial_tab_command=str(lens.target_rank),
        )

    def _lens_candidates(
        self,
        engine: EpisodeEngine,
        observation: Observation,
        lens: SearchLens,
    ) -> tuple[Candidate, ...]:
        return engine.candidates(
            observation,
            start_rank=lens.lower_rank,
            count=lens.upper_rank - lens.lower_rank + 1,
        )

    def _record_search_view(
        self,
        engine: EpisodeEngine,
        observation: Observation,
        lens: SearchLens,
        *,
        invoked_as: str | None,
        invocation: str,
    ) -> None:
        candidates = self._lens_candidates(engine, observation, lens)
        self._interaction(
            observation.boundary,
            "vocabulary-search-view",
            {
                "query": lens.query,
                "invoked_as": invoked_as,
                "invocation": invocation,
                "target_token_id": lens.token_id,
                "target_rank": lens.target_rank,
                "lower_rank": lens.lower_rank,
                "upper_rank": lens.upper_rank,
                "vocabulary_size": len(observation.logits),
                "candidates": [candidate.to_dict() for candidate in candidates],
            },
        )
        if not getattr(self.io, "supports_live_choices", False):
            self.io.write(
                f"\nExact token search {lens.query!r}: id={lens.token_id}, "
                f"absolute raw rank={lens.target_rank}"
            )
            self.io.write(
                f"Neighborhood ranks {lens.lower_rank}–{lens.upper_rank} of "
                f"{len(observation.logits)} (absolute raw-model ordering)."
            )
            display_candidates(
                self.io, candidates, heading=True, target_token_id=lens.token_id
            )

    def _review(
        self, engine: EpisodeEngine, boundary: int, active: int
    ) -> BoundaryReview:
        prefix = [*engine.initial_token_ids, *engine.visible_token_ids[:boundary]]
        next_token = (
            {
                "token_id": engine.visible_token_ids[boundary],
                "text": engine.backend.token_text(engine.visible_token_ids[boundary]),
                "origin": "episode",
            }
            if boundary < len(engine.visible_token_ids)
            else None
        )
        return BoundaryReview(
            aligned_step=boundary,
            active_aligned_step=active,
            context_text_tail=engine.backend.render(prefix, special=True)[
                -self.context_characters :
            ],
            context_token_sha256=token_prefix_sha256(prefix),
            next_token=next_token,
            position=(
                self._seamless_position(boundary)
                if self.seamless
                else {"kind": "token-boundary"}
            ),
        )

    def _seamless_targets(
        self, engine: EpisodeEngine, active_boundary: int
    ) -> tuple[int, ...]:
        """Return every visible token boundary, including the interior of writes."""
        return tuple(range(active_boundary + 1))

    def _seamless_edge_boundary(self, active_boundary: int) -> int | None:
        # Checkpoints are pauses, not barriers in the editable history.
        return None

    def _seamless_position(self, boundary: int) -> dict[str, Any]:
        if self.store is None or self.episode_id is None:
            return {"kind": "token-boundary"}
        if self._seamless_edge_boundary(boundary + 1) == boundary:
            return {"kind": "edge"}
        rows = self.store.actions(self.episode_id)
        for row in rows:
            if int(row["boundary_before"]) == boundary:
                return {
                    "kind": "action-boundary",
                    "action_kind": str(row["kind"]),
                    "side": "before",
                }
        for row in rows:
            before = int(row["boundary_before"])
            after = int(row["boundary_after"])
            if before < boundary < after:
                kind = str(row["kind"])
                if kind == "hold":
                    return {
                        "kind": "inside-span",
                        "span_type": "hold",
                        "offset_visible_tokens": boundary - before,
                        "total_visible_tokens": after - before,
                    }
                return {
                    "kind": "action-boundary",
                    "action_kind": kind,
                    "side": "inside",
                }
        for row in rows:
            if int(row["boundary_after"]) == boundary:
                return {
                    "kind": "action-boundary",
                    "action_kind": str(row["kind"]),
                    "side": "after",
                }
        return {"kind": "token-boundary"}

    def choose(self, engine: EpisodeEngine, observation: Observation) -> PolicyAction:
        self.choice_serial += 1
        candidates = engine.candidates(
            observation, count=min(self.menu_size, len(observation.logits))
        )
        choice = _choice_from_observation(
            engine,
            observation,
            candidates,
            context_characters=self.context_characters,
            serial=self.choice_serial,
        )
        exposed = {candidate.rank: candidate for candidate in candidates}
        preview_candidates = dict(exposed)

        def resolve_candidate(rank: int) -> Candidate:
            if rank not in preview_candidates:
                preview_candidates[rank] = engine.candidates(
                    observation, start_rank=rank, count=1
                )[0]
            return preview_candidates[rank]

        search: SearchLens | None = None
        search_lens_active = False
        feedback: ChoiceFeedback | None = None
        policy_sort = False
        policy_columns = bool(
            self.show_policy_rank and engine.sampling.history_penalties_active
        )
        review_boundary: int | None = None
        seamless_targets = (
            self._seamless_targets(engine, observation.boundary)
            if self.seamless
            else ()
        )
        proposal_prefill_available = not self.manual_acceptance
        live = bool(
            getattr(self.io, "supports_live_choices", False)
            and callable(getattr(self.io, "read_choice", None))
        )
        if not live:
            display_choice(self.io, choice, remaining_tokens=engine.remaining)
        while True:
            displayed = (
                self._lens_candidates(engine, observation, search)
                if search_lens_active and search is not None
                else choice.candidates
            )
            if live:
                raw = self.io.read_choice(  # type: ignore[attr-defined]
                    choice,
                    remaining_tokens=engine.remaining,
                    candidates=tuple(exposed[rank] for rank in sorted(exposed)),
                    display_candidates=displayed,
                    resolve_candidate=resolve_candidate,
                    resolve_insertion=lambda text, mode: self._resolve_write(
                        engine, text, mode
                    ),
                    target_token_id=search.token_id if search is not None else None,
                    feedback=feedback,
                    initial_command=(
                        str(observation.proposal_raw_rank)
                        if proposal_prefill_available and review_boundary is None
                        else None
                    ),
                    review=(
                        self._review(engine, review_boundary, observation.boundary)
                        if review_boundary is not None
                        else None
                    ),
                    seamless=self.seamless,
                    reactivate_on_review_enter=(
                        self.seamless and review_boundary is not None
                    ),
                    search_lens_active=search_lens_active,
                    policy_active=engine.sampling.history_penalties_active,
                    show_policy_rank=policy_columns,
                    sort_by_policy=policy_sort,
                )
            else:
                raw = self.io.read("\nTeacher action> ")
            if raw is None:
                raise EditorError("teacher input closed; use finish explicitly")
            if (
                self.seamless
                and review_boundary is not None
                and raw == SEAMLESS_REACTIVATE
            ):
                if self._seamless_edge_boundary(observation.boundary + 1) == review_boundary:
                    raise SeamlessEdgeRequested(review_boundary)
                raise SeamlessRewindRequested(review_boundary)
            if raw == "" and review_boundary is None:
                raw = str(observation.proposal_raw_rank)
            proposal_prefill_available = False
            try:
                command = parse_command(
                    raw,
                    menu_size=len(choice.candidates),
                    default_hold_tokens=self.default_hold_tokens,
                    vocabulary_size=len(observation.logits),
                    default_search_radius=self.search_radius,
                )
            except EditorError as exc:
                feedback = ChoiceFeedback("error", "INVALID COMMAND", (str(exc),))
                if not live:
                    self.io.write(f"[invalid command] {exc}")
                continue
            if review_boundary is not None:
                if command.kind == CommandKind.REVIEW_BACK:
                    if self.seamless:
                        previous = [
                            target
                            for target in seamless_targets
                            if target < review_boundary
                        ]
                        if previous:
                            review_boundary = previous[-1]
                    else:
                        review_boundary = max(0, review_boundary - 1)
                    continue
                if command.kind == CommandKind.REVIEW_FORWARD:
                    if self.seamless:
                        following = [
                            target
                            for target in seamless_targets
                            if target > review_boundary
                            and target < observation.boundary
                        ]
                        review_boundary = following[0] if following else None
                    else:
                        review_boundary = min(observation.boundary, review_boundary + 1)
                    continue
                if command.kind == CommandKind.FORK:
                    self._interaction(
                        observation.boundary,
                        "fork-requested",
                        {"boundary": review_boundary},
                    )
                    raise ForkRequested(review_boundary)
                review_boundary = None
                if not live:
                    display_choice(self.io, choice, remaining_tokens=engine.remaining)
                continue
            if command.kind == CommandKind.HELP:
                self.io.write(HELP_TEXT, end="")
                continue
            if command.kind == CommandKind.EDIT:
                assert command.action is not None
                edit = command.action
                if edit.kind.value == "accept":
                    return SelectRawRank(observation.proposal_raw_rank)
                if edit.kind.value == "select":
                    return SelectRawRank(int(edit.selected_rank))
                assert edit.supplied_text is not None and edit.insert_mode is not None
                return Write(edit.supplied_text, edit.insert_mode.value)
            if command.kind == CommandKind.HOLD:
                return Hold(
                    int(command.hold_tokens or self.default_hold_tokens),
                    command.hold_boundary,
                )
            if command.kind == CommandKind.FINISH:
                raise EdgeRequested()
            if command.kind == CommandKind.TEACHER_EOG:
                if not command.force:
                    key = self.io.read_key(
                        "[e/Enter] confirm EOG  [Backspace/Esc] cancel > "
                    )
                    if key not in {"e", "E", "\n", "\r", ""}:
                        feedback = ChoiceFeedback(
                            "status", "EOG CANCELLED", ("Generation remains live.",)
                        )
                        continue
                return EndGeneration()
            if command.kind == CommandKind.MENU_EXPAND:
                if search_lens_active:
                    search_lens_active = False
                additional = int(command.additional_rows or 0)
                target = min(
                    len(observation.logits), len(choice.candidates) + additional
                )
                candidates = engine.candidates(observation, count=target)
                choice = _choice_from_observation(
                    engine,
                    observation,
                    candidates,
                    context_characters=self.context_characters,
                    serial=self.choice_serial,
                )
                exposed.update((candidate.rank, candidate) for candidate in candidates)
                self._interaction(
                    observation.boundary,
                    "menu-expanded",
                    {"visible_rows": len(candidates)},
                )
                if not live:
                    display_choice(self.io, choice, remaining_tokens=engine.remaining)
                continue
            if command.kind == CommandKind.MAIN_MENU:
                search_lens_active = False
                feedback = None
                continue
            if command.kind == CommandKind.TOKEN_SEARCH:
                assert command.search_query is not None
                try:
                    found, feedback = self._search(
                        engine,
                        observation,
                        command.search_query,
                        command.invoked_as or raw,
                    )
                except EditorError as exc:
                    feedback = ChoiceFeedback("error", "SEARCH FAILED", (str(exc),))
                    if not live:
                        self.io.write(f"[search failed] {exc}")
                    continue
                if found is not None:
                    search = found
                    search_lens_active = True
                    exposed.update(
                        (candidate.rank, candidate)
                        for candidate in self._lens_candidates(
                            engine, observation, search
                        )
                    )
                continue
            if command.kind == CommandKind.TOKEN_SEARCH_VIEW:
                if command.search_rank is not None:
                    rank = command.search_rank
                    candidate = resolve_candidate(rank)
                    search = SearchLens(
                        query=candidate.text, token_id=candidate.token_id,
                        target_rank=rank,
                        lower_rank=max(1, rank - self.search_radius),
                        upper_rank=min(len(observation.logits), rank + self.search_radius),
                    )
                if search is None:
                    feedback = ChoiceFeedback(
                        "error", "NO ACTIVE SEARCH", ("Use /TERM or ms N first.",)
                    )
                    continue
                if command.search_direction == "+":
                    search.upper_rank = min(
                        len(observation.logits),
                        search.upper_rank + int(command.search_rows or 0),
                    )
                elif command.search_direction == "-":
                    search.lower_rank = max(
                        1, search.lower_rank - int(command.search_rows or 0)
                    )
                search_lens_active = True
                rows = self._lens_candidates(engine, observation, search)
                exposed.update((candidate.rank, candidate) for candidate in rows)
                self._record_search_view(
                    engine,
                    observation,
                    search,
                    invoked_as=None,
                    invocation=("rank" if command.search_rank is not None else
                                "expanded" if command.search_direction else "redraw"),
                )
                feedback = self._search_feedback(search)
                continue
            if command.kind == CommandKind.CONTEXT:
                extent = command.context_characters
                text = observation.context_text
                shown = text if extent == "all" else text[-int(extent or 2000) :]
                self.io.page(shown)
                self._interaction(
                    observation.boundary,
                    "context-viewed",
                    {"characters": len(shown)},
                )
                continue
            if command.kind in {CommandKind.NOTE_BEFORE, CommandKind.NOTE_AFTER}:
                note = command.note
                if note is None:
                    note = self.io.read("Note> ")
                if note:
                    self._interaction(
                        observation.boundary,
                        "note-before"
                        if command.kind == CommandKind.NOTE_BEFORE
                        else "note-after",
                        {"text": note},
                    )
                continue
            if command.kind == CommandKind.POLICY_VIEW:
                policy_sort = not policy_sort
                continue
            if command.kind == CommandKind.POLICY_COLUMN:
                policy_columns = not policy_columns
                continue
            if command.kind == CommandKind.REVIEW_BACK:
                if self.seamless:
                    previous = [
                        target
                        for target in seamless_targets
                        if target < observation.boundary
                    ]
                    if previous:
                        review_boundary = previous[-1]
                else:
                    review_boundary = max(0, observation.boundary - 1)
                continue
            if command.kind == CommandKind.REVIEW_FORWARD:
                if not self.seamless:
                    review_boundary = observation.boundary
                continue
            if command.kind == CommandKind.FORK:
                address = command.fork_address
                assert address is not None
                if address.kind == ForkAddressKind.CURRENT:
                    target = observation.boundary
                elif address.kind == ForkAddressKind.ABSOLUTE:
                    target = int(address.value or 0)
                elif address.kind == ForkAddressKind.RELATIVE_BACKWARD:
                    target = observation.boundary - int(address.value or 0)
                if not 0 <= target <= observation.boundary:
                    feedback = ChoiceFeedback(
                        "error",
                        "FORK UNAVAILABLE",
                        (f"Boundary {target} is outside this episode.",),
                    )
                    continue
                self._interaction(
                    observation.boundary, "fork-requested", {"boundary": target}
                )
                raise ForkRequested(target)
