"""Live, non-authoritative terminal rendering for one teacher decision."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable

from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.styles import Style

from .teacher_commands import (
    CommandKind, CommandState, ForkAddressKind, TeacherCommand, interpret_command,
)
from .candidate_columns import CandidateColumns
from .core.candidates import Candidate
from .core.errors import EditorError
from .core.ui import ChoiceSet, ContextText, InsertMode
from .terminal_contracts import (
    BoundaryReview,
    ChoiceFeedback,
    ChoiceViewState,
    SEAMLESS_REACTIVATE,
)
from .tui_views import ViewLifecycle


InsertionResolver = Callable[[str, InsertMode], str]


class PreviewPending(Exception):
    """A preview has been requested from the episode thread."""

    def __init__(self, appended_text: str | None = None):
        super().__init__()
        self.appended_text = appended_text


@dataclass(frozen=True)
class ActionPreview:
    kind: str
    label: str
    detail: str
    appended_text: str | None = None
    candidate_rank: int | None = None
    token_id: int | None = None
    raw_probability: float | None = None
    decoder_probability: float | None = None
    policy_rank: int | None = None
    policy_probability: float | None = None
    is_eog: bool = False
    valid: bool = True
    state: str = "ready"
    command: TeacherCommand | None = None


def _candidate_preview(candidate: Candidate, *, label: str) -> ActionPreview:
    return ActionPreview(
        kind="candidate",
        label=label,
        detail="",
        appended_text=candidate.text,
        candidate_rank=candidate.rank,
        token_id=candidate.token_id,
        raw_probability=candidate.raw_probability,
        decoder_probability=candidate.decoder_probability,
        policy_rank=candidate.policy_rank,
        policy_probability=candidate.policy_probability,
        is_eog=candidate.is_eog,
    )


def action_preview(
    choice: ChoiceSet,
    raw: str,
    candidates: tuple[Candidate, ...],
    resolve_insertion: InsertionResolver,
    *,
    remaining_tokens: int | None = None,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    default_hold_tokens: int = 100,
    default_search_radius: int = 3,
) -> ActionPreview:
    """Render the shared interpretation, enriching it with owner-thread previews."""
    interpretation = interpret_command(
        raw,
        menu_size=len(choice.candidates),
        default_hold_tokens=default_hold_tokens,
        vocabulary_size=choice.vocabulary_size or len(candidates),
        default_search_radius=default_search_radius,
    )
    if interpretation.state != CommandState.READY:
        return ActionPreview(
            kind=interpretation.state.value,
            label=("invalid chord" if [part.lower() for part in raw.strip().split()[:1]] == ["chord"]
                   and interpretation.state == CommandState.INVALID
                   else interpretation.state.value + " command"),
            detail=interpretation.message,
            valid=False,
            state=interpretation.state.value,
        )
    command = interpretation.command
    assert command is not None
    by_rank = {candidate.rank: candidate for candidate in candidates}
    sampled_candidate = next(
        (candidate for candidate in candidates
         if candidate.token_id == choice.proposal_token_id),
        None,
    )

    if command.kind == CommandKind.CHORD:
        ranks = command.chord_ranks
        assert ranks is not None
        return ActionPreview(
            kind="effect", label="chord preview",
            detail=(
                f"Press Enter to preview {len(ranks)} paths from raw ranks "
                + ", ".join(str(rank) for rank in ranks)
                + ". No episode action is recorded yet."
            ),
            command=command,
        )

    if command.kind == CommandKind.EDIT:
        action = command.action
        assert action is not None
        if action.kind.value in {"accept", "select"}:
            rank = (choice.proposal_raw_rank if action.kind.value == "accept"
                    else int(action.selected_rank))
            if rank == choice.proposal_raw_rank:
                if sampled_candidate is not None:
                    preview = _candidate_preview(sampled_candidate, label="sampled proposal")
                    return replace(preview, command=command)
                return ActionPreview(
                    kind="candidate", label="sampled proposal", detail="",
                    appended_text=choice.proposal_text,
                    candidate_rank=choice.proposal_raw_rank,
                    token_id=choice.proposal_token_id,
                    raw_probability=choice.proposal_raw_probability,
                    decoder_probability=choice.proposal_decoder_probability,
                    policy_rank=choice.proposal_policy_rank,
                    policy_probability=choice.proposal_policy_probability,
                    is_eog=choice.proposal_is_eog,
                    command=command,
                )
            candidate = by_rank.get(rank)
            if candidate is None and resolve_candidate is not None:
                try:
                    candidate = resolve_candidate(rank)
                except PreviewPending:
                    return ActionPreview(
                        kind="pending", label="selected raw rank",
                        detail=f"Resolving raw rank {rank}…", state="pending",
                        command=command,
                    )
                except EditorError as exc:
                    return ActionPreview(
                        kind="effect", label="selected raw rank",
                        detail=f"Candidate preview unavailable: {exc}",
                        valid=False, state="invalid", command=command,
                    )
            if candidate is not None:
                preview = _candidate_preview(candidate, label="selected candidate")
                return replace(preview, command=command)
            return ActionPreview(
                kind="effect", label="selected raw rank",
                detail=(f"Press Enter to select raw rank {rank}. "
                        "This token is not shown in the current menu."),
                command=command,
            )
        assert action.supplied_text is not None and action.insert_mode is not None
        supplied = action.supplied_text
        label = ("continuation insertion" if action.insert_mode == InsertMode.CONTINUATION
                 else "exact insertion")
        try:
            rendered = resolve_insertion(supplied, action.insert_mode)
        except PreviewPending as pending:
            # Keep the insertion status stable until the latest preview resolves.
            # Retain the last resolved text for the live context display.
            return ActionPreview(
                kind="insertion",
                label=label,
                detail="Tokenization and budget are validated on Enter.",
                appended_text=pending.appended_text,
                command=command,
            )
        except Exception as exc:
            return ActionPreview(
                kind="effect", label=label,
                detail=f"Insertion preview unavailable: {type(exc).__name__}: {exc}",
                valid=False, state="invalid", command=command,
            )
        return ActionPreview(
            kind="insertion", label=label,
            detail="Tokenization and budget are validated on Enter.",
            appended_text=rendered, command=command,
        )

    if command.kind == CommandKind.FORK:
        address = command.fork_address
        assert address is not None
        current = choice.aligned_step
        if address.kind == ForkAddressKind.CURRENT:
            target = current
        elif address.kind == ForkAddressKind.ABSOLUTE:
            target = int(address.value or 0)
        else:
            target = current - int(address.value or 0)
        detail = (
            f"The parent will seal at step {current}; a child will open fresh from step {target}."
            if 0 <= target <= current
            else f"Step {target} is outside the recorded range 0..{current}; Enter validates the boundary."
        )
        return ActionPreview(kind="effect", label="fork recorded boundary",
                             detail=detail, command=command)

    effects = {
        CommandKind.BIAS: ("token bias", "Update this bias rule; stay at this step."),
        CommandKind.PHRASE: (
            "phrase action",
            (
                "Enter validates and applies the phrase as one action; checked phrases "
                "roll back if a token exceeds the shift bound."
            ),
        ),
        CommandKind.HOLD: ("hold", "Hold will release control only after Enter."),
        CommandKind.FINISH: ("finish", "Open the live edge menu on Enter; no tokens are generated."),
        CommandKind.TEACHER_EOG: ("teacher EOG", "Teacher EOG selection begins on Enter."),
        CommandKind.MAIN_MENU: ("main menu", "The main candidate table returns on Enter."),
        CommandKind.MENU_EXPAND: ("more rows", "The main candidate table returns and expands on Enter."),
        CommandKind.TOKEN_SEARCH: ("token search", "Exact-token search executes on Enter; no text is committed."),
        CommandKind.TOKEN_SEARCH_VIEW: ("search view", "The token-search neighborhood updates on Enter."),
        CommandKind.CONTEXT: ("context", "The requested context view opens on Enter."),
        CommandKind.POLICY_VIEW: ("policy view", "The table toggles between raw-model and policy ordering on Enter."),
        CommandKind.POLICY_COLUMN: ("policy columns", "Policy diagnostics toggle on Enter without reordering."),
        CommandKind.LOGIT_VIEW: ("logit view", "Logit view changes on Enter."),
        CommandKind.PROBABILITY_VIEW: ("probability view", "Model probability overlays toggle on Enter."),
        CommandKind.COLUMN_FOCUS: ("column focus", "Middle-column focus changes on Enter."),
        CommandKind.OVERLAY_TOGGLE: ("overlay", f"Toggle the {command.overlay} overlay on Enter."),
        CommandKind.REVIEW_BACK: ("review back", "Review the previous durable token boundary on Enter."),
        CommandKind.REVIEW_FORWARD: ("review forward", "Review the next durable token boundary on Enter."),
        CommandKind.NOTE_BEFORE: ("note before", "The note-before action begins on Enter."),
        CommandKind.NOTE_AFTER: ("note after", "The note-after action begins on Enter."),
        CommandKind.HELP: ("help", "Full command help opens on Enter."),
    }
    label, detail = effects[command.kind]
    if command.kind == CommandKind.TEACHER_EOG and command.force:
        detail = "A teacher EOG is committed on Enter."
    if command.kind == CommandKind.HOLD:
        boundary = (f" through the next {command.hold_boundary} boundary"
                    if command.hold_boundary else "")
        detail = f"Release control for up to {command.hold_tokens} tokens{boundary} on Enter."
    if command.kind == CommandKind.PHRASE:
        label = f"{command.invoked_as} phrase"
    return ActionPreview(kind="effect", label=label, detail=detail, command=command)


def _safe_rendered_text(value: str) -> str:
    output: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\n":
            output.append(character)
        elif character == "\t":
            output.append("\\t")
        elif codepoint < 32 or codepoint == 127:
            output.append(f"\\x{codepoint:02x}")
        else:
            output.append(character)
    return "".join(output)


def _probability(value: float | None) -> str:
    if value is None or value <= 0.0:
        return "--"
    return f"{value:.2%}"


def _clipped_repr(value: str, width: int) -> str:
    rendered = repr(value)
    if len(rendered) <= width:
        return rendered
    if width <= 1:
        return "…"
    return rendered[: width - 1] + "…"


def _visible_candidates(
    candidates: tuple[Candidate, ...],
    selected_rank: int | None,
    maximum_rows: int,
    *,
    sort_by_policy: bool = False,
) -> tuple[tuple[Candidate, ...], int, int]:
    ordered = _ordered_candidates(candidates, sort_by_policy=sort_by_policy)
    if len(ordered) <= maximum_rows:
        return ordered, 0, 0
    selected_index = next(
        (
            index
            for index, candidate in enumerate(ordered)
            if candidate.rank == selected_rank
        ),
        0,
    )
    start = max(0, selected_index - maximum_rows // 2)
    start = min(start, len(ordered) - maximum_rows)
    end = start + maximum_rows
    return ordered[start:end], start, len(ordered) - end


def _ordered_candidates(
    candidates: tuple[Candidate, ...],
    *,
    sort_by_policy: bool = False,
) -> tuple[Candidate, ...]:
    """Return the exact visual row order used by rendering and Tab navigation."""
    return tuple(
        sorted(
            candidates,
            key=(
                (lambda candidate: (
                    candidate.policy_rank
                    if candidate.policy_rank is not None
                    else candidate.rank,
                    candidate.rank,
                ))
                if sort_by_policy
                else (lambda candidate: candidate.rank)
            ),
        )
    )


def _candidate_command_cycle(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    *,
    sort_by_policy: bool = False,
) -> tuple[str, ...]:
    """Return rank commands downward from the proposal, wrapping in view order."""
    ordered = _ordered_candidates(candidates, sort_by_policy=sort_by_policy)
    proposal_command = str(choice.proposal_raw_rank)
    proposal_index = next(
        (
            index
            for index, candidate in enumerate(ordered)
            if candidate.token_id == choice.proposal_token_id
        ),
        None,
    )
    if proposal_index is None:
        return (proposal_command, *(str(candidate.rank) for candidate in ordered))
    after = ordered[proposal_index + 1 :]
    before = ordered[:proposal_index]
    return (
        proposal_command,
        *(str(candidate.rank) for candidate in after),
        *(str(candidate.rank) for candidate in before),
    )


def _navigation_command_cycle(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    feedback: ChoiceFeedback | None,
    *,
    sort_by_policy: bool = False,
    search_lens_active: bool = False,
) -> tuple[str, ...]:
    suggestions = feedback.completion_commands if feedback is not None else ()
    unique_suggestions = tuple(dict.fromkeys(suggestions))
    if search_lens_active:
        ordered = _ordered_candidates(candidates, sort_by_policy=sort_by_policy)
        return unique_suggestions + tuple(str(candidate.rank) for candidate in ordered)
    candidates_cycle = _candidate_command_cycle(
        choice,
        candidates,
        sort_by_policy=sort_by_policy,
    )
    return candidates_cycle[:1] + unique_suggestions + candidates_cycle[1:]


def _terminal_size() -> tuple[int, int]:
    try:
        app = get_app()
        size = app.output.get_size()
        rows = int(size.rows)
        return int(size.columns), max(1, rows)
    except Exception:
        fallback = shutil.get_terminal_size(fallback=(100, 30))
        return fallback.columns, fallback.lines


def _append_wrapped_text(rows: list[StyleAndTextTuples], column: int,
                         text: str, style: str, width: int) -> int:
    for char in text:
        if char == "\n":
            rows.append([])
            column = 0
            continue
        rendered = " " * (8 - column % 8) if char == "\t" else char
        for glyph in rendered:
            cells = max(0, get_cwidth(glyph))
            if column + cells > width:
                rows.append([])
                column = 0
            if rows[-1] and rows[-1][-1][0] == style:
                previous_style, previous_text = rows[-1][-1]
                rows[-1][-1] = (previous_style, previous_text + glyph)
            else:
                rows[-1].append((style, glyph))
            column += cells
    return column


@dataclass(frozen=True)
class _ContextRowsView:
    base_rows: Sequence[StyleAndTextTuples]
    tail_rows: list[StyleAndTextTuples] | None = None

    @property
    def base_count(self) -> int:
        return len(self.base_rows) - (1 if self.tail_rows is not None else 0)

    def __len__(self) -> int:
        if self.tail_rows is None:
            return len(self.base_rows)
        return self.base_count + len(self.tail_rows)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        resolved = index + len(self) if index < 0 else index
        if resolved < 0 or resolved >= len(self):
            raise IndexError("context row index out of range")
        if self.tail_rows is not None and resolved >= self.base_count:
            return self.tail_rows[resolved - self.base_count]
        return self.base_rows[resolved]


@dataclass(frozen=True, slots=True)
class _ContextLayoutUndo:
    snapshot: ContextText | None
    row_count: int
    last_row: tuple[tuple[str, str], ...]
    column: int


class _IncrementalContextLayout:
    """Reversible wrapped rows, extended by only newly decoded text chunks."""

    def __init__(self, snapshot: ContextText, width: int) -> None:
        self.snapshot: ContextText | None = None
        self.width = max(1, width)
        self.rows: list[StyleAndTextTuples] = [[]]
        self.column = 0
        self._undo: list[_ContextLayoutUndo] = []
        self.sync(snapshot, self.width)

    def _append_node(self, node: ContextText) -> None:
        self._undo.append(
            _ContextLayoutUndo(
                snapshot=self.snapshot,
                row_count=len(self.rows),
                last_row=tuple(self.rows[-1]),
                column=self.column,
            )
        )
        if node.chunk:
            self.column = _append_wrapped_text(
                self.rows, self.column, _safe_rendered_text(node.chunk), "", self.width
            )
        self.snapshot = node

    def _rebuild(self, snapshot: ContextText, width: int) -> None:
        self.width = width
        self.rows = [[]]
        self.column = 0
        self.snapshot = None
        self._undo.clear()
        nodes = snapshot.nodes_since(None)
        assert nodes is not None
        for node in nodes:
            self._append_node(node)

    def _step_back(self) -> None:
        undo = self._undo.pop()
        del self.rows[undo.row_count:]
        self.rows[-1] = list(undo.last_row)
        self.column = undo.column
        self.snapshot = undo.snapshot

    def sync(self, snapshot: ContextText, width: int) -> None:
        width = max(1, width)
        if width != self.width:
            self._rebuild(snapshot, width)
            return
        if snapshot is self.snapshot:
            return
        if self.snapshot is not None:
            appended = snapshot.nodes_since(self.snapshot)
            if appended is not None:
                for node in appended:
                    self._append_node(node)
                return
            removed = self.snapshot.nodes_since(snapshot)
            if removed is not None:
                for _node in reversed(removed):
                    self._step_back()
                return
        self._rebuild(snapshot, width)

    def rows_with_proposal(self, proposal: str) -> _ContextRowsView:
        if not proposal:
            return _ContextRowsView(self.rows)
        tail = [list(self.rows[-1])]
        _append_wrapped_text(
            tail,
            self.column,
            _safe_rendered_text(proposal),
            "class:proposal",
            self.width,
        )
        return _ContextRowsView(self.rows, tail)


@dataclass(frozen=True, slots=True)
class _ReviewAnchor:
    boundary: int
    end_character: int | None = None


@dataclass(frozen=True)
class _ReviewContextPage:
    rows: tuple[StyleAndTextTuples, ...]
    top_anchor: _ReviewAnchor
    more_above: bool
    more_below: bool


def _wrap_review_window(
    chunks: Iterable[str], start_character: int, width: int
) -> tuple[list[StyleAndTextTuples], list[int]]:
    """Wrap a bounded chunk slice while retaining absolute source offsets."""
    rows: list[StyleAndTextTuples] = [[]]
    row_starts = [start_character]
    column = 0
    width = max(1, width)
    source_offset = start_character
    for chunk in chunks:
        for character in chunk:
            if character == "\n":
                rows.append([])
                source_offset += 1
                row_starts.append(source_offset)
                column = 0
                continue
            codepoint = ord(character)
            if character == "\t":
                rendered = "\\t"
            elif codepoint < 32 or codepoint == 127:
                rendered = f"\\x{codepoint:02x}"
            else:
                rendered = character
            for glyph in rendered:
                cells = max(0, get_cwidth(glyph))
                if column + cells > width:
                    rows.append([])
                    row_starts.append(source_offset)
                    column = 0
                if rows[-1] and rows[-1][-1][0] == "":
                    previous_style, previous_text = rows[-1][-1]
                    rows[-1][-1] = (previous_style, previous_text + glyph)
                else:
                    rows[-1].append(("", glyph))
                column += cells
            source_offset += 1
    return rows, row_starts


class _ReviewContextViewport:
    """A token-boundary anchored review page that wraps only its visible slice."""

    def __init__(self, review: BoundaryReview) -> None:
        self.review = review
        self.anchor = _ReviewAnchor(self._review_end_boundary(review))
        self._forward: list[_ReviewAnchor] = []
        self._page_cache: tuple[tuple[int, int, int, int], _ReviewContextPage] | None = None

    def update(self, review: BoundaryReview) -> None:
        same_review = (
            review.aligned_step == self.review.aligned_step
            and self._review_end_boundary(review) == self._review_end_boundary(self.review)
            and review.context_snapshots is self.review.context_snapshots
            and review.context_text_tail is self.review.context_text_tail
        )
        self.review = review
        if not same_review:
            self.anchor = _ReviewAnchor(self._review_end_boundary(review))
            self._forward.clear()
        self._page_cache = None

    @staticmethod
    def _review_end_boundary(review: BoundaryReview) -> int:
        return (
            review.aligned_step
            if review.context_boundary is None
            else review.context_boundary
        )

    def _context_at(self, boundary: int) -> ContextText | None:
        snapshots = self.review.context_snapshots
        if 0 <= boundary < len(snapshots):
            context = snapshots[boundary]
            if context is not None:
                return context
        if boundary == self._review_end_boundary(self.review):
            context = self.review.context_text_tail
            if isinstance(context, ContextText):
                return context
            return ContextText.root(context)
        return None

    def _anchor_for_character(
        self, character: int, lower_boundary: int, upper_boundary: int
    ) -> _ReviewAnchor:
        previous: tuple[int, int] | None = None
        for boundary in range(lower_boundary, upper_boundary + 1):
            context = self._context_at(boundary)
            if context is None:
                continue
            count = context.character_count
            if count == character:
                return _ReviewAnchor(boundary)
            if count > character:
                if previous is None:
                    return _ReviewAnchor(boundary, character)
                return _ReviewAnchor(boundary, character)
            previous = (boundary, count)
        if previous is not None:
            boundary, count = previous
            if count == character:
                return _ReviewAnchor(boundary)
            return _ReviewAnchor(upper_boundary, character)
        return _ReviewAnchor(upper_boundary, character)

    def page(self, width: int, budget: int) -> _ReviewContextPage:
        width = max(1, width)
        budget = max(1, budget)
        key = (
            self.anchor.boundary,
            -1 if self.anchor.end_character is None else self.anchor.end_character,
            width,
            budget,
        )
        if self._page_cache is not None and self._page_cache[0] == key:
            return self._page_cache[1]
        end_context = self._context_at(self.anchor.boundary)
        if end_context is None:
            result = _ReviewContextPage((), self.anchor, False, False)
            self._page_cache = (key, result)
            return result

        end_character = self.anchor.end_character
        if end_character is None:
            end_character = end_context.character_count
        end_character = max(0, min(end_character, end_context.character_count))
        floor = max(0, min(self.review.context_character_start, end_context.character_count))
        start_boundary = self.anchor.boundary
        start_character = end_character
        target_characters = width * budget
        selected_newlines = 0

        while (
            start_character - floor < target_characters
            and selected_newlines < budget
        ):
            needed = target_characters - (end_character - start_character)
            if start_boundary > 0:
                previous = self._context_at(start_boundary - 1)
                current = self._context_at(start_boundary)
                if (
                    previous is None
                    or current is None
                    or current.nodes_since(previous) is None
                ):
                    next_start = max(floor, start_character - needed)
                    selected_newlines += sum(
                        chunk.count("\n")
                        for chunk in current.iter_slices(next_start, start_character)
                    ) if current is not None else 0
                    start_character = next_start
                    break
                token_start = previous.character_count
                if start_character - token_start > needed:
                    next_start = start_character - needed
                    selected_newlines += sum(
                        chunk.count("\n")
                        for chunk in current.iter_slices(next_start, start_character)
                    )
                    start_character = next_start
                    break
                next_start = max(floor, token_start)
                selected_newlines += sum(
                    chunk.count("\n")
                    for chunk in current.iter_slices(next_start, start_character)
                )
                start_boundary -= 1
                start_character = next_start
                continue
            next_start = max(floor, start_character - needed)
            selected_newlines += sum(
                chunk.count("\n")
                for chunk in end_context.iter_slices(next_start, start_character)
            )
            start_character = next_start
            break

        start_character = max(floor, min(start_character, end_character))
        wrapped, row_starts = _wrap_review_window(
            end_context.iter_slices(start_character, end_character),
            start_character,
            width,
        )
        dropped_rows = max(0, len(wrapped) - budget)
        visible_rows = wrapped[dropped_rows:]
        top_character = row_starts[dropped_rows]
        top_anchor = self._anchor_for_character(
            top_character, start_boundary, self.anchor.boundary
        )
        review_context = self._context_at(self._review_end_boundary(self.review))
        review_end = (
            review_context.character_count if review_context is not None else end_character
        )
        more_above = top_character > floor or (
            start_boundary > 0
            and self._context_at(start_boundary - 1) is not None
        )
        more_below = (
            self.anchor.boundary < self._review_end_boundary(self.review)
            or end_character < review_end
        )
        result = _ReviewContextPage(
            tuple(visible_rows), top_anchor, more_above, more_below
        )
        self._page_cache = (key, result)
        return result

    def page_up(self, width: int, budget: int) -> None:
        page = self.page(width, budget)
        if page.top_anchor == self.anchor:
            if self.anchor.end_character is not None and self.anchor.end_character > 0:
                next_anchor = _ReviewAnchor(
                    self.anchor.boundary,
                    max(0, self.anchor.end_character - max(1, width * budget)),
                )
            elif (
                self.anchor.boundary > 0
                and self._context_at(self.anchor.boundary - 1) is not None
            ):
                next_anchor = _ReviewAnchor(self.anchor.boundary - 1)
            else:
                return
        else:
            next_anchor = page.top_anchor
        self._forward.append(self.anchor)
        self.anchor = next_anchor
        self._page_cache = None

    def page_down(self) -> None:
        if self._forward:
            self.anchor = self._forward.pop()
            self._page_cache = None


def _review_scroll_hint(page: _ReviewContextPage) -> str:
    hints = []
    if page.more_above:
        hints.append("More above")
    if page.more_below:
        hints.append("More below")
    if not hints:
        hints.append("All context shown")
    return " · ".join((*hints, "PgUp/PgDn scroll"))


@lru_cache(maxsize=1)
def _wrapped_context(context: str, width: int):
    """Keep only the latest context layout, with immutable cached rows."""
    rows: list[StyleAndTextTuples] = [[]]
    column = _append_wrapped_text(rows, 0, context, "", width)
    return tuple(tuple(row) for row in rows), column


@lru_cache(maxsize=1)
def _safe_context_text(context: str | ContextText) -> str:
    if isinstance(context, ContextText):
        context = context.materialize()
    return _safe_rendered_text(context)


def _context_rows(context: str, proposal: str, width: int) -> list[StyleAndTextTuples]:
    """Reuse history wrapping while preserving the changing proposal highlight."""
    cached_rows, column = _wrapped_context(context, width)
    rows = [list(row) for row in cached_rows]
    _append_wrapped_text(rows, column, proposal, "class:proposal", width)
    return rows


def _feedback_line_limit(height: int) -> int:
    return max(1, height // 5 - 1)


def _choice_context_budget(height: int, candidate_count: int,
                           feedback: ChoiceFeedback | None, view_rows: int = 0) -> int:
    feedback_rows = 0
    if feedback is not None:
        shown = min(len(feedback.lines), _feedback_line_limit(height))
        feedback_rows = 2 + shown + int(shown < len(feedback.lines))
    # Reserve the input/footer, headings, and candidate overflow indicators first.
    available = max(2, height - 16 - feedback_rows - view_rows)
    table = min(candidate_count, max(4, available // 3))
    return max(1, available - table)


def _context_scroll_hint(start: int, end: int, total: int) -> str:
    hints = []
    if start > 0:
        hints.append("More above")
    if end < total:
        hints.append("More below")
    if not hints:
        hints.append("All context shown")
    return " · ".join((*hints, "PgUp/PgDn scroll"))


def _context_view(context: str | ContextText, proposal: str, width: int, height: int,
                  offset: int = 0, *, budget: int | None = None,
                  context_layout: _IncrementalContextLayout | None = None
                  ) -> tuple[StyleAndTextTuples, int]:
    rows = (
        context_layout.rows_with_proposal(proposal)
        if context_layout is not None
        else _context_rows(_safe_context_text(context), proposal, max(1, width - 1))
    )
    budget = max(1, height // 3) if budget is None else max(1, budget)
    offset = min(max(0, offset), max(0, len(rows) - budget))
    end = len(rows) - offset
    start = max(0, end - budget)
    fragments: StyleAndTextTuples = []
    for row in rows[start:end]:
        fragments.extend(row)
        fragments.append(("", "\n"))
    fragments.append(("class:muted", _context_scroll_hint(start, end, len(rows)) + "\n"))
    return fragments, end - start + 1


def _is_writing(command: str) -> bool:
    return command[:2].lower() in {"t ", "x "}


def _writing_sizes(height: int) -> tuple[int, int]:
    """Reserve a stable editor and context region, independent of draft length."""
    editor = max(3, height // 3)
    context = max(1, height - editor - 14)
    return editor, context


def _one_line(text: str, width: int) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    result = ""
    for char in text:
        if get_cwidth(result + char) > max(1, width - 2):
            return result + "…"
        result += char
    return result


def _preview_status(preview: ActionPreview) -> tuple[str, str]:
    """Static text and style cues; color is never the only state signal."""
    if preview.state == "invalid":
        return "class:invalid", "INVALID · "
    if preview.state == "incomplete":
        return "class:hint", "INCOMPLETE · "
    if preview.state == "pending":
        return "class:pending", "PENDING · "
    return "class:effect", "READY · "


def _render_writing(choice: ChoiceSet, candidates: tuple[Candidate, ...],
                    preview: ActionPreview, width: int, height: int,
                    offset: int, sort_by_policy: bool,
                    show_policy_rank: bool = False,
                    logit_view: str = "none",
                    show_model_probabilities: bool = False,
                    column_focus: str | None = None,
                    overlays: frozenset[str] = frozenset(),
                    context_layout: _IncrementalContextLayout | None = None
                    ) -> StyleAndTextTuples:
    _, budget = _writing_sizes(height)
    rows = (
        context_layout.rows_with_proposal(preview.appended_text or "")
        if context_layout is not None
        else _context_rows(_safe_context_text(choice.context_text_tail),
                           _safe_rendered_text(preview.appended_text or ""), width - 1)
    )
    end = len(rows) - min(max(0, offset), max(0, len(rows) - budget))
    start = max(0, end - budget)
    fragments: StyleAndTextTuples = [
        ("class:status-strong", f"Step {choice.aligned_step} · Writing\n"),
        ("class:section", "DECISION BOUNDARY\n"),
    ]
    for row in rows[start:end]:
        fragments.extend(row)
        fragments.append(("", "\n"))
    fragments.append(("", "\n" * (budget - (end - start))))
    fragments.append(("class:muted", _context_scroll_hint(start, end, len(rows)) + "\n"))
    if preview.kind in {"insertion", "pending"}:
        text = preview.appended_text or ""
        effect = f"{preview.label} · {text.count(chr(10)) + 1} lines · {len(text)} characters"
    else:
        effect = f"{preview.label} · {preview.detail}"
    style, cue = _preview_status(preview)
    fragments.append((style, _one_line(cue + effect, width) + "\n"))
    fragments.append(("class:rule", "─" * (width - 1) + "\n"))
    columns = CandidateColumns(
        policy=show_policy_rank,
        logit_view=logit_view,
        show_model_probabilities=show_model_probabilities,
        column_focus=column_focus,
        overlays=overlays,
        raw_k1_logit=choice.raw_k1_logit,
    )
    heading = (
        "Candidates · rank / token ID / text"
        if columns.columns == (("token-id", 8),)
        else "Candidates · rank / token ID / overlays / text"
    )
    fragments.append(("class:table-header", _one_line(heading, width) + "\n"))
    shown = _ordered_candidates(candidates, sort_by_policy=sort_by_policy)[:3]
    for candidate in shown:
        fragments.append(("class:table-row", _one_line(
            f"{candidate.rank:>5}"
            + columns.values(candidate)
            + f"  {candidate.text!r}"
            + (f" [bias {candidate.bias:+g}]" if candidate.bias else ""), width) + "\n"))
    fragments.append(("", "\n" * (3 - len(shown))))
    fragments.append(("class:muted", f"{max(0, len(candidates) - 3)} more candidate rows · Ctrl+E restores full table\n"))
    return fragments


def _render_choice(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    command_text: str,
    remaining_tokens: int | None,
    resolve_insertion: InsertionResolver,
    target_token_id: int | None,
    feedback: ChoiceFeedback | None = None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    display_candidates: tuple[Candidate, ...] | None = None,
    search_lens_active: bool = False,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    context_offset: int = 0,
    expanded_editor: bool = False,
    terminal_size: tuple[int, int] | None = None,
    default_hold_tokens: int = 100,
    default_search_radius: int = 3,
    context_layout: _IncrementalContextLayout | None = None,
) -> StyleAndTextTuples:
    width, height = terminal_size or _terminal_size()
    width = max(width, 36)
    preview = action_preview(
        choice,
        command_text,
        candidates,
        resolve_insertion,
        remaining_tokens=remaining_tokens,
        resolve_candidate=resolve_candidate,
        default_hold_tokens=default_hold_tokens,
        default_search_radius=default_search_radius,
    )
    if expanded_editor and _is_writing(command_text):
        return _render_writing(choice, tuple(candidates if display_candidates is None else display_candidates),
                               preview, width, height, context_offset, sort_by_policy,
                               show_policy_rank, logit_view,
                               show_model_probabilities, column_focus, overlays,
                               context_layout=context_layout)
    context = (choice.context_text_tail if context_layout is not None
               else _safe_context_text(choice.context_text_tail))
    proposal = (
        _safe_rendered_text(preview.appended_text)
        if preview.appended_text is not None
        else None
    )
    feedback_line_limit = _feedback_line_limit(height)
    displayed_feedback_lines = (
        ()
        if feedback is None
        else feedback.lines[:feedback_line_limit]
    )
    hidden_feedback_lines = (
        0
        if feedback is None
        else len(feedback.lines) - len(displayed_feedback_lines)
    )
    feedback_rows = (
        0
        if feedback is None
        else 2 + len(displayed_feedback_lines) + bool(hidden_feedback_lines)
    )
    focus_rank = preview.candidate_rank
    if not command_text and search_lens_active and target_token_id is not None:
        focus_rank = next(
            (
                candidate.rank
                for candidate in candidates
                if candidate.token_id == target_token_id
            ),
            focus_rank,
        )
    lens_candidates = tuple(
        candidates if display_candidates is None else display_candidates
    )
    table_candidates = lens_candidates
    external_focus = next(
        (
            candidate
            for candidate in candidates
            if candidate.rank == focus_rank
            and not any(row.rank == focus_rank for row in table_candidates)
        ),
        None,
    )
    current_view_is_lens = search_lens_active
    view_status_rows = int(external_focus is not None) + int(current_view_is_lens)
    context_budget = _choice_context_budget(
        height, len(table_candidates), feedback, view_status_rows,
    )
    context_fragments, approximate_context_lines = _context_view(
        context, proposal or "", width, height, context_offset, budget=context_budget,
        context_layout=context_layout,
    )
    maximum_rows = max(
        1,
        height
        - approximate_context_lines
        - 15
        - feedback_rows
        - view_status_rows,
    )
    shown, hidden_before, hidden_after = _visible_candidates(
        table_candidates,
        focus_rank,
        maximum_rows,
        sort_by_policy=sort_by_policy,
    )
    rule = "─" * max(20, width - 1)
    fragments: StyleAndTextTuples = []

    fragments.extend(
        [
            ("class:status-strong", f"Step {choice.aligned_step}"),
            ("class:muted", " · teacher track"),
            ("", " " * 3),
            ("class:muted", (f"{remaining_tokens} tokens remaining\n" if remaining_tokens is not None else "No token budget\n")),
            ("class:rule", rule + "\n"),
            ("class:section", "DECISION BOUNDARY\n\n"),

        ]
    )
    fragments.extend(context_fragments)
    fragments.append(("", "\n"))

    if preview.kind == "candidate":
        rank = (
            f" · rank {preview.candidate_rank}"
            if preview.candidate_rank is not None
            else ""
        )
        terminal = " · END" if preview.is_eog else ""
        fragments.extend(
            [
                ("class:proposal-label", "READY · " + preview.label),
                ("class:muted", rank),
                ("class:muted", f" · exact {repr(preview.appended_text or '')}"),
                ("class:muted", f" · token {preview.token_id}"),
                ("class:muted", (
                    f" · raw {_probability(preview.raw_probability)}"
                    if preview.raw_probability is not None else ""
                )),
                (
                    "class:muted",
                    (
                        " · decoder "
                        f"{_probability(preview.decoder_probability)}"
                        if preview.decoder_probability is not None else ""
                    ) + terminal,
                ),
                (
                    "class:muted",
                    (
                        f" · policy-rank {preview.policy_rank}"
                        if policy_active and preview.policy_rank is not None
                        else ""
                    )
                    + "\n",
                ),
            ]
        )
    elif preview.kind in {"insertion", "pending"}:
        style, cue = _preview_status(preview)
        fragments.extend(
            [
                (style if cue else "class:proposal-label", cue + preview.label),
                ("class:muted", f" · {preview.detail}\n"),
            ]
        )
    else:
        style, cue = _preview_status(preview)
        fragments.extend(
            [
                (style, cue + preview.label),
                ("class:muted", f" · {preview.detail}\n"),
            ]
        )

    fragments.extend([("class:rule", rule + "\n")])
    columns = CandidateColumns(
        policy=show_policy_rank,
        logit_view=logit_view,
        show_model_probabilities=show_model_probabilities,
        column_focus=column_focus,
        overlays=overlays,
        raw_k1_logit=choice.raw_k1_logit,
    )
    fragments.append((
        "class:table-header", f"    rank{columns.heading}  text\n",
    ))
    if hidden_before:
        fragments.append(
            ("class:muted", f"    … {hidden_before} earlier disclosed row(s) …\n")
        )
    for candidate in shown:
        marker = "▶" if candidate.rank == preview.candidate_rank else " "
        target_suffix = " [MATCH]" if candidate.token_id == target_token_id else ""
        prefix = f"{marker} {candidate.rank:>5}{columns.values(candidate)}  "
        if candidate.bias:
            target_suffix += f" [bias {candidate.bias:+g}]"
        text_width = max(1, width - len(prefix) - 1)
        if candidate.rank == preview.candidate_rank:
            row_style = "class:selected-row"
        elif candidate.token_id == target_token_id:
            row_style = "class:match-row"
        else:
            row_style = "class:table-row"
        fragments.append(
            (
                row_style,
                prefix
                + _clipped_repr(candidate.text, max(1, text_width - len(target_suffix)))
                + target_suffix
                + "\n",
            )
        )
    if hidden_after:
        fragments.append(
            ("class:muted", f"    … {hidden_after} later disclosed row(s) …\n")
        )
    if external_focus is not None:
        fragments.append(
            (
                "class:feedback-info",
                (
                    "PREVIEW OUTSIDE LENS"
                    if search_lens_active
                    else "PREVIEW OUTSIDE TABLE"
                )
                + f" · raw rank {external_focus.rank}\n",
            )
        )
    if current_view_is_lens:
        fragments.append(
            (
                "class:feedback-search",
                "SEARCH LENS · Tab cycles this neighborhood · m/esc main table\n",
            )
        )
    fragments.extend(
        [
            ("class:rule", rule + "\n"),
        ]
    )
    if feedback is not None:
        category = (
            feedback.category
            if feedback.category in {"error", "info", "search"}
            else "info"
        )
        fragments.append(
            (
                f"class:feedback-{category}",
                _safe_rendered_text(feedback.title) + "\n",
            )
        )
        for line in displayed_feedback_lines:
            fragments.append(
                ("class:feedback-detail", "  " + _safe_rendered_text(line) + "\n")
            )
        if hidden_feedback_lines:
            fragments.append(
                (
                    "class:feedback-detail",
                    f"  … {hidden_feedback_lines} more; Tab cycles all suggestions.\n",
                )
            )
        fragments.append(("", "\n"))
    fragments.extend(
        [
            ("class:help-key", "accept"),
            ("class:muted", "  "),
            ("class:help-key", "rank"),
            ("class:muted", " choose  "),
            ("class:help-key", "tab/⇧tab"),
            ("class:muted", " browse  "),
            ("class:help-key", "t TEXT"),
            ("class:muted", " insert  "),
            ("class:help-key", "h [N]"),
            ("class:muted", " hold  "),
            ("class:help-key", "?"),
            ("class:muted", " help\n"),
        ]
    )
    return fragments


def _render_review(
    review: BoundaryReview,
    *,
    seamless: bool = False,
    terminal_size: tuple[int, int] | None = None,
    context_viewport: _ReviewContextViewport | None = None,
) -> StyleAndTextTuples:
    """Render one journal-backed historical boundary, never a live preview."""
    width, height = terminal_size or _terminal_size()
    width = max(width, 36)
    rule = "─" * max(20, width - 1)
    viewport = context_viewport or _ReviewContextViewport(review)
    page = viewport.page(max(1, width - 1), max(1, height - 15))
    position = dict(review.position)
    fragments: StyleAndTextTuples = [
        ("class:status-strong", f"Review boundary {review.aligned_step}"),
        ("class:muted", f" · active boundary {review.active_aligned_step}\n"),
        ("class:rule", rule + "\n"),
        ("class:section", "HISTORICAL BOUNDARY REVIEW\n\n"),
    ]
    for row in page.rows:
        fragments.extend(row)
        fragments.append(("", "\n"))
    fragments.append(("class:muted", _review_scroll_hint(page) + "\n"))
    fragments.append(("", "\n"))
    if position.get("kind") == "inside-span":
        label = str(position.get("span_type") or "span").replace("-", " ").upper()
        fragments.extend(
            [
                ("class:proposal-label", f"Inside {label}"),
                (
                    "class:muted",
                    f" · {position.get('offset_visible_tokens')}/"
                    f"{position.get('total_visible_tokens')} visible tokens realized\n",
                ),
            ]
        )
    elif position.get("kind") == "action-boundary":
        label = str(position.get("action_kind") or "action").replace("-", " ").upper()
        side = {
            "before": "start",
            "inside": "inside",
            "after": "end",
        }.get(position.get("side"), "boundary")
        fragments.extend(
            [
                ("class:proposal-label", f"{label} {side}"),
                (
                    "class:muted",
                    " · Enter deletes the continuation from this token boundary\n",
                ),
            ]
        )
    else:
        fragments.append(("class:proposal-label", "recorded token boundary\n"))
    if review.next_token is not None:
        token = dict(review.next_token)
        fragments.extend(
            [
                ("class:muted", "next recorded token · "),
                ("", repr(token.get("text"))),
                (
                    "class:muted",
                    f" · token {token.get('token_id')} · {token.get('origin')}\n",
                ),
            ]
        )
    fragments.extend(
        [
            ("class:rule", rule + "\n"),
            ("class:help-key", "["),
            ("class:muted", " previous  "),
            ("class:help-key", "]"),
            ("class:muted", " next  "),
            ("class:help-key", "f"),
            ("class:muted", " fork here  "),
            ("class:help-key", "esc"),
            ("class:muted", " live\n"),
            (
                "class:prompt-label",
                (
                    "History · Enter deletes the continuation and resumes here\n"
                    if seamless else "Review action · Enter submits bare f\n"
                ),
            ),
        ]
    )
    return fragments


LIVE_STYLES = {
    "amber-cyan": Style.from_dict(
        {
            "status-strong": "bold",
            "muted": "ansibrightblack",
            "rule": "ansibrightblack",
            "section": "bold",
            "proposal": "ansiyellow bold reverse",
            "proposal-label": "ansiyellow bold",
            "effect": "ansicyan bold",
            "invalid": "ansiyellow bold",
            "pending": "ansicyan",
            "table-header": "ansibrightblack",
            "table-row": "",
            "selected-row": "ansicyan bold reverse",
            "match-row": "ansimagenta bold",
            "help-key": "bold",
            "prompt-label": "ansicyan bold",
            "prompt": "ansicyan bold",
            "input": "",
            "hint": "ansibrightblack",
            "feedback-error": "ansired bold",
            "feedback-info": "ansicyan bold",
            "feedback-search": "ansimagenta bold",
            "feedback-detail": "",
        }
    ),
    "monochrome": Style.from_dict(
        {
            "status-strong": "bold",
            "muted": "",
            "rule": "",
            "section": "bold underline",
            "proposal": "bold reverse",
            "proposal-label": "bold underline",
            "effect": "bold",
            "invalid": "bold underline",
            "pending": "underline",
            "table-header": "underline",
            "table-row": "",
            "selected-row": "bold reverse",
            "match-row": "bold underline",
            "help-key": "bold",
            "prompt-label": "bold",
            "prompt": "bold reverse",
            "input": "",
            "hint": "",
            "feedback-error": "bold underline",
            "feedback-info": "bold",
            "feedback-search": "bold underline",
            "feedback-detail": "",
        }
    ),
    "high-contrast": Style.from_dict(
        {
            "status-strong": "bold underline",
            "muted": "",
            "rule": "bold",
            "section": "bold underline",
            "proposal": "ansibrightyellow bold reverse",
            "proposal-label": "ansibrightyellow bold underline",
            "effect": "ansibrightcyan bold",
            "invalid": "ansibrightyellow bold underline",
            "pending": "ansibrightcyan underline",
            "table-header": "bold underline",
            "table-row": "",
            "selected-row": "ansibrightcyan bold reverse",
            "match-row": "ansibrightmagenta bold reverse",
            "help-key": "bold reverse",
            "prompt-label": "ansibrightcyan bold",
            "prompt": "ansibrightcyan bold reverse",
            "input": "",
            "hint": "bold",
            "feedback-error": "ansibrightred bold reverse",
            "feedback-info": "ansibrightcyan bold",
            "feedback-search": "ansibrightmagenta bold underline",
            "feedback-detail": "bold",
        }
    ),
}


def _live_style(theme: str) -> Style:
    try:
        return LIVE_STYLES[theme]
    except KeyError as exc:
        raise ValueError(f"unknown live UI theme: {theme!r}") from exc


class LiveChoiceView(ViewLifecycle):
    """Reusable layout, bindings and buffer for live choices and history review."""

    def __init__(self, state: ChoiceViewState, *, submit, enabled=lambda: True,
                 terminal_size=None):
        super().__init__(submit=submit)
        self.terminal_size = terminal_size or _terminal_size
        self.command_buffer = Buffer(multiline=True, read_only=Condition(lambda: not enabled()))
        self.bindings = KeyBindings()
        self.context_offset = 0
        self.expanded_editor = False
        self._context_key = None
        self._context_layout: _IncrementalContextLayout | None = None
        self._review_viewport: _ReviewContextViewport | None = None
        self.update(state)

        def _replace_buffer(text: str, *, owned: bool) -> None:
            self.completion_owned = owned
            self.command_buffer.document = Document(text, cursor_position=len(text))

        def _in_authored_text() -> bool:
            return _is_writing(self.command_buffer.text)

        def _editor_expanded() -> bool:
            return self.state.review is None and self.expanded_editor and _in_authored_text()

        def _reset_editor_on_command_change(buffer: Buffer) -> None:
            if not _is_writing(buffer.text):
                self.expanded_editor = False

        self.command_buffer.on_text_changed += _reset_editor_on_command_change

        @self.bindings.add("c-e", filter=Condition(lambda: self.state.review is None and _in_authored_text()))
        def _toggle_editor(event: object) -> None:
            self.expanded_editor = not self.expanded_editor

        review_empty = Condition(lambda: self.state.review is not None and not self.command_buffer.text)
        review_has_input = Condition(
            lambda: self.state.review is not None and bool(self.command_buffer.text)
        )
        active_empty = Condition(
            lambda: self.state.review is None and (not self.command_buffer.text or self.completion_owned)
        )

        def _navigate(direction: int, event: object) -> None:
            # The initial proposal rank is replace-on-first-typing only.  Once Tab
            # is used, the rank in the buffer is a navigation result, so ordinary
            # editing must not treat the next typed character as a replacement.
            # Keep the buffer non-owned for every navigation result, including the
            # first one selected from an otherwise blank prompt.
            if not self.navigation_commands:
                return
            current = self.command_buffer.text
            if not current:
                suggestions = (
                    self.state.feedback.completion_commands if self.state.feedback is not None else ()
                )
                if suggestions:
                    _replace_buffer(
                        suggestions[0] if direction > 0 else suggestions[-1],
                        owned=False,
                    )
                    return
                if (
                    direction > 0
                    and self.state.search_lens_active
                ):
                    match_command = (
                        self.state.feedback.initial_tab_command
                        if self.state.feedback is not None
                        else None
                    )
                    if match_command is None and self.state.target_token_id is not None:
                        match_command = next(
                            (
                                str(candidate.rank)
                                for candidate in self.active_table_candidates
                                if candidate.token_id == self.state.target_token_id
                            ),
                            None,
                        )
                    if match_command is not None:
                        _replace_buffer(match_command, owned=False)
                        return
            if not current:
                if self.state.search_lens_active and self.state.target_token_id is not None:
                    match_command = next(
                        (
                            str(candidate.rank)
                            for candidate in self.active_table_candidates
                            if candidate.token_id == self.state.target_token_id
                        ),
                        self.navigation_commands[0],
                    )
                    index = self.navigation_commands.index(match_command)
                else:
                    _replace_buffer(
                        self.navigation_commands[0] if direction > 0 else self.navigation_commands[-1],
                        owned=False,
                    )
                    return
            else:
                try:
                    index = self.navigation_commands.index(current)
                except ValueError:
                    if self.state.search_lens_active:
                        match_command = next(
                            (
                                str(candidate.rank)
                                for candidate in self.active_table_candidates
                                if candidate.token_id == self.state.target_token_id
                            ),
                            self.navigation_commands[0],
                        )
                        _replace_buffer(match_command, owned=False)
                        return
                    proposal_rank = next(
                        (
                            candidate.rank
                            for candidate in self.active_table_candidates
                            if candidate.token_id == self.state.choice.proposal_token_id
                        ),
                        None,
                    )
                    if current.isdigit() and int(current) == proposal_rank:
                        index = 0
                    else:
                        return
            _replace_buffer(
                self.navigation_commands[(index + direction) % len(self.navigation_commands)],
                owned=False,
            )

        @self.bindings.add("c-g")
        def _explore_rank(event: object) -> None:
            raw = self.command_buffer.text.strip()
            if self.state.review is None and raw.isdigit():
                rank = int(raw)
                if rank >= 1 and (self.state.choice.vocabulary_size is None or rank <= self.state.choice.vocabulary_size):
                    self._finish(event, result=f"ms {rank}")  # type: ignore[attr-defined]

        @self.bindings.add("escape", "enter", filter=Condition(lambda: self.state.review is None and _in_authored_text()))
        def _insert_newline(event: object) -> None:
            self.command_buffer.insert_text("\n")

        def _scroll_budget(height: int, preview: ActionPreview) -> int:
            if _editor_expanded():
                return _writing_sizes(height)[1]
            if self.state.review is not None:
                return max(1, height - 15)
            outside_table = (preview.candidate_rank is not None and
                any(row.rank == preview.candidate_rank for row in self.state.candidates) and
                not any(row.rank == preview.candidate_rank for row in self.active_table_candidates))
            return _choice_context_budget(height, len(self.active_table_candidates), self.state.feedback,
                                          int(self.state.search_lens_active) + int(outside_table))

        @self.bindings.add("pageup")
        def _context_up(event: object) -> None:
            width, height = self.terminal_size()
            if self.state.review is not None:
                if self._review_viewport is not None:
                    self._review_viewport.page_up(
                        max(1, max(36, width) - 1), max(1, height - 15)
                    )
                return
            preview = action_preview(self.state.choice, self.command_buffer.text, self.state.candidates, self.state.resolve_insertion,
                                     remaining_tokens=self.state.remaining_tokens, resolve_candidate=self.state.resolve_candidate,
                                     default_hold_tokens=self.state.default_hold_tokens,
                                     default_search_radius=self.state.default_search_radius)
            budget = _scroll_budget(height, preview)
            self._sync_context_layout(self.state, max(1, max(36, width) - 1))
            if self._context_layout is not None:
                rows = self._context_layout.rows_with_proposal(preview.appended_text or "")
            else:
                rows = _context_rows(_safe_context_text(self.state.choice.context_text_tail),
                                     _safe_rendered_text(preview.appended_text or ""), max(1, max(36, width) - 1))
            self.context_offset = min(max(0, len(rows) - budget), self.context_offset + max(1, budget - 1))

        @self.bindings.add("pagedown")
        def _context_down(event: object) -> None:
            _, height = self.terminal_size()
            if self.state.review is not None:
                if self._review_viewport is not None:
                    self._review_viewport.page_down()
                return
            preview = action_preview(self.state.choice, self.command_buffer.text, self.state.candidates, self.state.resolve_insertion,
                                     remaining_tokens=self.state.remaining_tokens, resolve_candidate=self.state.resolve_candidate,
                                     default_hold_tokens=self.state.default_hold_tokens,
                                     default_search_radius=self.state.default_search_radius)
            budget = _scroll_budget(height, preview)
            self.context_offset = max(0, self.context_offset - max(1, budget - 1))

        @self.bindings.add("enter")
        def _submit(event: object) -> None:
            result = self.command_buffer.text
            if self.state.review is not None:
                if self.state.seamless and self.state.reactivate_on_review_enter and not result.strip():
                    result = SEAMLESS_REACTIVATE
                elif result.strip().lower() not in {"f", "fork"}:
                    result = "\x1b"
            self._finish(event, result=result)  # type: ignore[attr-defined]

        @self.bindings.add("[", filter=active_empty | review_empty)
        def _review_back(event: object) -> None:
            self._finish(event, result="[")  # type: ignore[attr-defined]

        @self.bindings.add("]", filter=active_empty | review_empty)
        def _review_forward(event: object) -> None:
            self._finish(event, result="]")  # type: ignore[attr-defined]

        @self.bindings.add("escape", filter=Condition(lambda: self.state.review is not None))
        def _leave_review(event: object) -> None:
            self._finish(event, result="\x1b")  # type: ignore[attr-defined]

        @self.bindings.add(
            "escape",
            filter=Condition(lambda: self.state.review is None and self.state.search_lens_active),
        )
        def _leave_search_lens(event: object) -> None:
            self._finish(event, result="\x1b")  # type: ignore[attr-defined]

        @self.bindings.add("f", filter=review_empty)
        def _review_fork(event: object) -> None:
            del event
            _replace_buffer("f", owned=False)

        @self.bindings.add(Keys.Any, filter=review_empty)
        def _consume_and_leave_review(event: object) -> None:
            self._finish(event, result="\x1b")  # type: ignore[attr-defined]

        @self.bindings.add(Keys.Any, filter=review_has_input)
        def _consume_extra_review_input(event: object) -> None:
            self._finish(event, result="\x1b")  # type: ignore[attr-defined]

        @self.bindings.add("backspace", filter=review_has_input)
        def _leave_review_on_backspace(event: object) -> None:
            self._finish(event, result="\x1b")  # type: ignore[attr-defined]

        @self.bindings.add("c-c")
        def _interrupt(event: object) -> None:
            self._finish(event, exception=KeyboardInterrupt())  # type: ignore[attr-defined]

        @self.bindings.add("c-d", filter=Condition(lambda: not self.command_buffer.text))
        def _closed(event: object) -> None:
            self._finish(event, result=None)  # type: ignore[attr-defined]

        @self.bindings.add("tab")
        def _next_completion(event: object) -> None:
            if self.state.review is not None:
                self._finish(event, result="\x1b")  # type: ignore[attr-defined]
                return
            if _in_authored_text():
                self.command_buffer.insert_text("\t")
                return
            _navigate(1, event)

        @self.bindings.add("s-tab")
        def _previous_completion(event: object) -> None:
            if self.state.review is not None:
                self._finish(event, result="\x1b")  # type: ignore[attr-defined]
                return
            if _in_authored_text():
                self.command_buffer.insert_text("\t")
                return
            _navigate(-1, event)

        completion_filter = Condition(lambda: self.completion_owned)

        @self.bindings.add("backspace", filter=completion_filter)
        def _clear_completion(event: object) -> None:
            del event
            _replace_buffer("", owned=False)

        @self.bindings.add(Keys.BracketedPaste, filter=completion_filter)
        def _replace_completion_with_paste(event: object) -> None:
            _replace_buffer(event.data, owned=False)  # type: ignore[attr-defined]

        @self.bindings.add(Keys.Any, filter=completion_filter)
        def _replace_completion_with_typing(event: object) -> None:
            _replace_buffer(event.data, owned=False)  # type: ignore[attr-defined]

        choice_control = FormattedTextControl(self._render)
        input_control = BufferControl(buffer=self.command_buffer, focusable=True)
        prompt_row = VSplit(
            [
                Window(
                    FormattedTextControl([("class:prompt", "› ")]),
                    width=Dimension.exact(2),
                    height=1,
                ),
                Window(input_control, height=lambda: Dimension.exact(
                           _writing_sizes(self.terminal_size()[1])[0] if _editor_expanded() else 1),
                       wrap_lines=True, style="class:input"),
            ]
        )
        root = HSplit(
            [
                Window(
                    choice_control,
                    wrap_lines=True,
                    always_hide_cursor=True,
                ),
                prompt_row,
                Window(
                    FormattedTextControl(
                        lambda: [
                            (
                                "class:hint",
                                (
                                    "Historical review is read-only; bare f forks this boundary."
                                    if self.state.review is not None
                                    else ("Ctrl+E " + ("collapse" if _editor_expanded() else "expand") +
                                          " · Alt+Enter newline · Tab indent · Enter commits")
                                    if _in_authored_text()
                                    else (
                                        (f"Next live edge in {self.state.remaining_tokens} {self.remaining_label} · " if self.state.remaining_tokens is not None else "q opens the live edge · ") +
                                        "Enter commits · Alt+Enter newline in t/x · PgUp/PgDn context · Ctrl+G explores rank."
                                    )
                                ),
                            )
                        ]
                    ),
                    wrap_lines=True,
                    dont_extend_height=True,
                    always_hide_cursor=True,
                ),
            ]
        )
        self.layout = Layout(root, focused_element=input_control)

    def update(self, state: ChoiceViewState) -> None:
        context_key = (
            state.choice.context_token_sha256,
            state.review.aligned_step if state.review else None,
        )
        if context_key != self._context_key:
            self.context_offset = 0
        self._context_key = context_key
        self.state = state
        if state.review is None:
            self._review_viewport = None
            self._sync_context_layout(
                state, max(1, max(36, self.terminal_size()[0]) - 1)
            )
        elif self._review_viewport is None:
            self._review_viewport = _ReviewContextViewport(state.review)
        else:
            self._review_viewport.update(state.review)
        self.expanded_editor = False
        self.completion_owned = bool(state.initial_command and state.review is None)
        self.active_table_candidates = tuple(
            state.candidates if state.display_candidates is None else state.display_candidates
        )
        self.navigation_commands = _navigation_command_cycle(
            state.choice, self.active_table_candidates, state.feedback,
            sort_by_policy=state.sort_by_policy, search_lens_active=state.search_lens_active,
        )
        self.remaining_label = "token" if state.remaining_tokens == 1 else "tokens"
        text = state.initial_command if self.completion_owned else ""
        self.command_buffer.reset(document=Document(text, cursor_position=len(text)))

    def _sync_context_layout(self, state: ChoiceViewState, width: int) -> None:
        if state.review is not None:
            return
        context = state.choice.context_text_tail
        if not isinstance(context, ContextText):
            self._context_layout = None
            return
        if self._context_layout is None:
            self._context_layout = _IncrementalContextLayout(context, width)
        else:
            self._context_layout.sync(context, width)

    def _render(self):
        state = self.state
        terminal_size = self.terminal_size()
        if state.review is not None:
            return _render_review(
                state.review,
                seamless=state.seamless,
                terminal_size=terminal_size,
                context_viewport=self._review_viewport,
            )
        self._sync_context_layout(
            state, max(1, max(36, terminal_size[0]) - 1)
        )
        return _render_choice(
            state.choice, state.candidates, self.command_buffer.text,
            state.remaining_tokens, state.resolve_insertion, state.target_token_id,
            state.feedback, state.policy_active, state.show_policy_rank,
            state.sort_by_policy, state.logit_view, state.show_model_probabilities,
            state.column_focus,
            state.overlays,
            state.display_candidates,
            state.search_lens_active,
            state.resolve_candidate, self.context_offset,
            self.expanded_editor and _is_writing(self.command_buffer.text),
            terminal_size=terminal_size,
            default_hold_tokens=state.default_hold_tokens,
            default_search_radius=state.default_search_radius,
            context_layout=self._context_layout,
        )
