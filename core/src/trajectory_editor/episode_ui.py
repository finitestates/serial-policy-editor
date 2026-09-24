"""Terminal policy adapter for the unified episode engine.

The adapter owns command interpretation and navigation. Full-vocabulary
``/`` search remains a non-mutating lens over the current observation; it is
never compiled into a replay tape.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .candidate_columns import CandidateColumns, CandidateViewPlan, next_column_focus
from .chord import ChordRequested
from .core.candidates import Candidate
from .core.errors import EditorError
from .core.ui import ChoiceSet
from .core.actions import (
    EndGeneration,
    Hold,
    Phrase,
    PHRASE_DEFAULT_MAX_SHIFT,
    PHRASE_DEFAULT_MAX_TOKENS,
    PolicyAction,
    SelectRawRank,
    Write,
)
from .episode_engine import EpisodeEngine, Observation
from .episode_runner import (
    EdgeRequested,
    ForkRequested,
    SeamlessRewindRequested,
)
from .episode_store import EpisodeStore
from .episode_hash import token_prefix_sha256
from .core.sampling import raw_rank
from .teacher_commands import HELP_TEXT, CommandKind, CommandState, ForkAddressKind, interpret_command
from .terminal_contracts import (
    BoundaryReview, ChoiceFeedback, ChoiceViewState, PromptRequest, SEAMLESS_REACTIVATE,
    TerminalProtocol,
)
from .tui import TerminalIO


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
    view: CandidateViewPlan | None = None,
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
        proposal_raw_probability=(
            observation.proposal_raw_probability if view and view.needs("raw_probability") else None
        ),
        proposal_decoder_probability=observation.proposal_decoder_probability,
        proposal_is_eog=engine.backend.is_eog(observation.proposal_token_id),
        candidates=candidates,
        vocabulary_size=len(observation.logits),
        proposal_raw_rank=observation.proposal_raw_rank,
        proposal_policy_rank=(
            observation.proposal_policy_rank if view and view.needs("policy_rank") else None
        ),
        proposal_policy_probability=None,
        raw_k1_logit=(
            float(observation.statistics.maximum) if view and view.needs("top_raw_logit") else None
        ),
    )


@dataclass
class PolicyViewPreferences:
    """Session presentation preferences; never part of sampler/replay state."""

    show: bool | None = None
    sort_by_policy: bool = False
    logit_view: str = "none"
    show_model_probabilities: bool = False
    # Single middle-column overlay focus; None = identity (or fall back to l/%).
    column_focus: str | None = None
    overlays: frozenset[str] = frozenset()


class InteractivePolicy:
    def __init__(
        self,
        *,
        io: TerminalProtocol | None = None,
        menu_size: int = 12,
        search_radius: int = 3,
        default_hold_tokens: int = 100,
        phrase_max_tokens: int = PHRASE_DEFAULT_MAX_TOKENS,
        phrase_max_shift: float = PHRASE_DEFAULT_MAX_SHIFT,
        context_characters: int = 0,
        manual_acceptance: bool = False,
        show_policy_rank: bool | None = None,
        logit_view: str = "none",
        show_model_probabilities: bool = False,
        view_preferences: PolicyViewPreferences | None = None,
        store: EpisodeStore | None = None,
        episode_id: str | None = None,
        seamless: bool = False,
    ) -> None:
        if min(menu_size, search_radius, default_hold_tokens, phrase_max_tokens) < 1:
            raise EditorError("menu, search, and hold sizes must be positive")
        if (
            type(phrase_max_shift) not in (int, float)
            or not math.isfinite(float(phrase_max_shift))
            or phrase_max_shift < 0.0
        ):
            raise EditorError("phrase max shift must be finite and nonnegative")
        self.io = io or TerminalIO()
        self.menu_size = menu_size
        self.search_radius = search_radius
        self.default_hold_tokens = default_hold_tokens
        self.phrase_max_tokens = phrase_max_tokens
        self.phrase_max_shift = float(phrase_max_shift)
        self.context_characters = context_characters
        self.manual_acceptance = bool(manual_acceptance)
        self.view_preferences = (
            view_preferences if view_preferences is not None
            else PolicyViewPreferences(
                show=show_policy_rank,
                logit_view=logit_view,
                show_model_probabilities=show_model_probabilities,
            )
        )
        self.store = store
        self.episode_id = episode_id
        self.seamless = bool(seamless)
        self.choice_serial = 0

    def _show_policy_diagnostics(self, engine: EpisodeEngine) -> bool:
        del engine
        return bool(self.view_preferences.show)

    def _view_plan(self, engine: EpisodeEngine) -> CandidateViewPlan:
        """Resolve the visible columns once for lookup and rendering."""
        size = self.io.terminal_size()
        width = size[0] if size is not None else None
        plan = CandidateColumns(
            policy=self._show_policy_diagnostics(engine),
            logit_view=self.view_preferences.logit_view,
            show_model_probabilities=self.view_preferences.show_model_probabilities,
            column_focus=self.view_preferences.column_focus,
            overlays=self.view_preferences.overlays,
            width=width,
        ).plan
        return plan.policy_ordered() if self.view_preferences.sort_by_policy else plan

    def _interaction(
        self, boundary: int, kind: str, payload: Mapping[str, Any]
    ) -> None:
        if self.store is not None and self.episode_id is not None:
            self.store.record_interaction(self.episode_id, boundary, kind, payload)

    def action_rejected(self, action: PolicyAction, reason: str) -> None:
        """Receive a live action rejection without leaving the current edge."""
        if isinstance(action, Phrase):
            self.io.write(f"[{action.kind} rejected] {reason}")

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
                    f"token {lens.token_id} · absolute raw rank={lens.target_rank} · neighborhood ranks "
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
            view=self._view_plan(engine),
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

    def _seamless_position(self, boundary: int) -> dict[str, Any]:
        if self.store is None or self.episode_id is None:
            return {"kind": "token-boundary"}
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
        view = self._view_plan(engine)
        candidates = engine.candidates(
            observation,
            count=min(self.menu_size, len(observation.logits)),
            view=view,
        )
        choice = _choice_from_observation(
            engine,
            observation,
            candidates,
            context_characters=self.context_characters,
            serial=self.choice_serial,
            view=view,
        )
        exposed = {candidate.rank: candidate for candidate in candidates}
        preview_candidates = dict(exposed)
        choice_view = view

        def resolve_candidate(rank: int) -> Candidate:
            if rank not in preview_candidates:
                preview_candidates[rank] = engine.candidates(
                    observation,
                    start_rank=rank,
                    count=1,
                    view=self._view_plan(engine),
                )[0]
            return preview_candidates[rank]

        search: SearchLens | None = None
        search_lens_active = False
        feedback: ChoiceFeedback | None = None
        policy_sort = self.view_preferences.sort_by_policy
        review_boundary: int | None = None
        seamless_targets = (
            self._seamless_targets(engine, observation.boundary)
            if self.seamless
            else ()
        )
        proposal_prefill_available = not self.manual_acceptance
        while True:
            policy_columns = self._show_policy_diagnostics(engine)
            view = self._view_plan(engine)
            if view != choice_view:
                refreshed = engine.candidates(
                    observation,
                    count=len(choice.candidates),
                    view=view,
                )
                choice = replace(
                    choice,
                    candidates=refreshed,
                    proposal_raw_probability=(
                        observation.proposal_raw_probability
                        if view.needs("raw_probability")
                        else None
                    ),
                    proposal_policy_rank=(
                        observation.proposal_policy_rank if view.needs("policy_rank") else None
                    ),
                    raw_k1_logit=(
                        float(observation.statistics.maximum)
                        if view.needs("top_raw_logit") else None
                    ),
                )
                exposed = {
                    rank: engine.candidates(observation, start_rank=rank, count=1, view=view)[0]
                    for rank in exposed
                }
                exposed.update((candidate.rank, candidate) for candidate in refreshed)
                preview_candidates = dict(exposed)
                choice_view = view
            displayed = (
                self._lens_candidates(engine, observation, search)
                if search_lens_active and search is not None
                else (
                    engine.policy_candidates(
                        observation,
                        count=len(choice.candidates),
                        view=view,
                    )
                    if policy_sort
                    else choice.candidates
                )
            )
            if policy_sort and not search_lens_active:
                exposed.update((candidate.rank, candidate) for candidate in displayed)
            raw = self.io.read_choice(ChoiceViewState(
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
                policy_active=engine.sampling.policy_active,
                show_policy_rank=policy_columns,
                sort_by_policy=policy_sort and not search_lens_active,
                logit_view=self.view_preferences.logit_view,
                show_model_probabilities=self.view_preferences.show_model_probabilities,
                column_focus=self.view_preferences.column_focus,
                overlays=self.view_preferences.overlays,
                default_hold_tokens=self.default_hold_tokens,
                default_search_radius=self.search_radius,
                warm_selection=(
                    (lambda rank, token_id, generation, cancelled: engine.speculate_accept(
                        observation,
                        raw_rank=rank,
                        token_id=token_id,
                        generation=generation,
                        cancelled=cancelled,
                    ))
                    if review_boundary is None else None
                ),
                cancel_warm_selection=(
                    engine.discard_speculative_accept
                    if review_boundary is None else None
                ),
            ))
            if raw is None:
                raise EdgeRequested()
            if (
                self.seamless
                and review_boundary is not None
                and raw == SEAMLESS_REACTIVATE
            ):
                raise SeamlessRewindRequested(review_boundary)
            proposal_prefill_available = False
            interpretation = interpret_command(
                raw,
                menu_size=len(choice.candidates),
                default_hold_tokens=self.default_hold_tokens,
                vocabulary_size=len(observation.logits),
                default_search_radius=self.search_radius,
                implicit_accept=review_boundary is None,
            )
            if interpretation.state != CommandState.READY:
                feedback = ChoiceFeedback(
                    "error", "INVALID COMMAND", (interpretation.message,)
                )
                continue
            command = interpretation.command
            assert command is not None
            if command.kind == CommandKind.CHORD:
                if review_boundary is not None:
                    feedback = ChoiceFeedback(
                        "error", "INVALID COMMAND",
                        ("return to the current menu before starting a chord",),
                    )
                    continue
                assert command.chord_ranks is not None
                raise ChordRequested(command.chord_ranks)
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
                continue
            if command.kind == CommandKind.BIAS:
                from .bias_commands import apply_bias_command
                if command.bias_status:
                    lines = [f"{g.name}: shared amount {g.bias:+g}; {len(g.members)} terms; "
                             f"{'enabled' if g.enabled else 'disabled'}"
                             for g in engine.sampling.bias_groups]
                    self.io.page("\n".join(lines) or "No active bias groups.")
                    continue
                try:
                    updated, updates = apply_bias_command(
                        command, engine.backend, engine.sampling, observation,
                        resolve_candidate,
                    )
                except EditorError as exc:
                    feedback = ChoiceFeedback("error", "INVALID BIAS", (str(exc),))
                    continue
                if self.store is not None and self.episode_id is not None:
                    with self.store.transaction():
                        self.store.record_sampling_segment(
                            self.episode_id, start_boundary=engine.boundary, sampling=updated,
                            stream_fingerprint=engine.stream_fingerprint,
                            coordinate_offset=engine.coordinate_offset,
                        )
                        for kind, payload, _label, _value in updates:
                            self._interaction(engine.boundary, kind, payload)
                engine.sampling = updated
                observation = engine.observe()
                ranks = tuple(exposed)
                exposed = {
                    rank: engine.candidates(
                        observation,
                        start_rank=rank,
                        count=1,
                        view=self._view_plan(engine),
                    )[0]
                    for rank in ranks
                }
                preview_candidates = dict(exposed)
                candidates = tuple(resolve_candidate(c.rank) for c in choice.candidates)
                choice = _choice_from_observation(engine, observation, candidates,
                    context_characters=self.context_characters, serial=self.choice_serial,
                    view=self._view_plan(engine))
                choice_view = self._view_plan(engine)
                lines = tuple(f"{label}: {value:+g}" for _kind, _payload, label, value in updates)
                feedback = ChoiceFeedback("status", "STEERING UPDATED", lines)
                continue
            if command.kind == CommandKind.HELP:
                self.io.page(HELP_TEXT)
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
            if command.kind == CommandKind.PHRASE:
                assert command.phrase_text is not None and command.phrase_mode is not None
                return Phrase(
                    command.phrase_text,
                    command.phrase_mode,
                    force=command.phrase_force,
                    max_tokens=self.phrase_max_tokens,
                    max_shift=self.phrase_max_shift,
                )
            if command.kind == CommandKind.HOLD:
                return Hold(
                    int(command.hold_tokens or self.default_hold_tokens),
                    command.hold_boundary,
                )
            if command.kind == CommandKind.FINISH:
                raise EdgeRequested()
            if command.kind == CommandKind.TEACHER_EOG:
                if not command.force:
                    key = self.io.prompt(PromptRequest(
                        "[e/Enter] confirm EOG  [Backspace/Esc] cancel > ",
                        single_key=True,
                    ))
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
                view = self._view_plan(engine)
                candidates = engine.candidates(
                    observation,
                    count=target,
                    view=view,
                )
                choice = _choice_from_observation(
                    engine,
                    observation,
                    candidates,
                    context_characters=self.context_characters,
                    serial=self.choice_serial,
                    view=view,
                )
                choice_view = view
                exposed.update((candidate.rank, candidate) for candidate in candidates)
                self._interaction(
                    observation.boundary,
                    "menu-expanded",
                    {"visible_rows": len(candidates)},
                )
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
                    note = self.io.prompt(PromptRequest("Note> "))
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
                self.view_preferences.sort_by_policy = policy_sort
                continue
            if command.kind == CommandKind.POLICY_COLUMN:
                self.view_preferences.show = not policy_columns
                continue
            if command.kind == CommandKind.LOGIT_VIEW:
                if command.invoked_as == "L":
                    next_view = (
                        "none"
                        if self.view_preferences.logit_view == "both"
                        else "both"
                    )
                else:
                    views = ("none", "raw", "gap")
                    current = self.view_preferences.logit_view
                    try:
                        next_view = views[(views.index(current) + 1) % len(views)]
                    except ValueError:
                        next_view = views[0]
                self.view_preferences.logit_view = next_view
                feedback = ChoiceFeedback(
                    "status",
                    "LOGIT VIEW",
                    (f"columns: {next_view}",),
                )
                continue
            if command.kind == CommandKind.PROBABILITY_VIEW:
                self.view_preferences.show_model_probabilities = (
                    not self.view_preferences.show_model_probabilities
                )
                state = (
                    "on"
                    if self.view_preferences.show_model_probabilities
                    else "off"
                )
                feedback = ChoiceFeedback(
                    "status",
                    "PROBABILITY VIEW",
                    (f"model % overlays: {state}",),
                )
                continue
            if command.kind == CommandKind.COLUMN_FOCUS:
                if command.invoked_as == "C":
                    self.view_preferences.column_focus = None
                    self.view_preferences.logit_view = "none"
                    self.view_preferences.show_model_probabilities = False
                    self.view_preferences.overlays = frozenset()
                    feedback = ChoiceFeedback(
                        "status",
                        "COLUMN FOCUS",
                        ("cleared → identity (rank | token-id | text)",),
                    )
                else:
                    self.view_preferences.column_focus = next_column_focus(
                        self.view_preferences.column_focus
                    )
                    focus = self.view_preferences.column_focus
                    feedback = ChoiceFeedback(
                        "status",
                        "COLUMN FOCUS",
                        (f"middle column: {focus}",),
                    )
                continue
            if command.kind == CommandKind.OVERLAY_TOGGLE:
                assert command.overlay is not None
                enabled = set(self.view_preferences.overlays)
                if command.overlay in enabled:
                    enabled.remove(command.overlay)
                else:
                    enabled.add(command.overlay)
                self.view_preferences.overlays = frozenset(enabled)
                feedback = ChoiceFeedback("status", "OVERLAYS", (", ".join(sorted(enabled)) or "identity",))
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
