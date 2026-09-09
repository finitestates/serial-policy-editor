"""Live, non-authoritative terminal rendering for one teacher decision."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable

from prompt_toolkit.application import Application, get_app
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

from .domain import Candidate, ChoiceSet, EditorError, InsertMode
from .tui import (
    BoundaryReview,
    ChoiceFeedback,
    ForkAddressKind,
    normalize_command_syntax,
    parse_fork_address,
    SEAMLESS_REACTIVATE,
)
from .ui_themes import DEFAULT_LIVE_THEME


InsertionResolver = Callable[[str, InsertMode], str]


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
) -> ActionPreview:
    """Project an unsubmitted input buffer without changing editor state."""
    by_rank = {candidate.rank: candidate for candidate in candidates}
    sampled_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.token_id == choice.proposal_token_id
        ),
        None,
    )
    stripped = raw.strip()
    lower = stripped.lower()

    if not stripped or lower == "accept":
        if sampled_candidate is not None:
            return _candidate_preview(sampled_candidate, label="sampled proposal")
        return ActionPreview(
            kind="candidate",
            label="sampled proposal",
            detail="",
            appended_text=choice.proposal_text,
            candidate_rank=choice.proposal_raw_rank,
            token_id=choice.proposal_token_id,
            raw_probability=choice.proposal_raw_probability,
            decoder_probability=choice.proposal_decoder_probability,
            policy_rank=choice.proposal_policy_rank,
            policy_probability=choice.proposal_policy_probability,
            is_eog=choice.proposal_is_eog,
        )

    if stripped.isdigit():
        requested_rank = int(stripped)
        if requested_rank < 1 or (
            choice.vocabulary_size is not None and requested_rank > choice.vocabulary_size
        ):
            return ActionPreview(
                kind="effect", label="invalid rank",
                detail=f"Choose a rank from 1 through {choice.vocabulary_size}.",
                valid=False,
            )
        if requested_rank == choice.proposal_raw_rank:
            if sampled_candidate is not None:
                return _candidate_preview(sampled_candidate, label="sampled proposal")
            return ActionPreview(
                kind="candidate",
                label="sampled proposal",
                detail="",
                appended_text=choice.proposal_text,
                candidate_rank=choice.proposal_raw_rank,
                token_id=choice.proposal_token_id,
                raw_probability=choice.proposal_raw_probability,
                decoder_probability=choice.proposal_decoder_probability,
                policy_rank=choice.proposal_policy_rank,
                policy_probability=choice.proposal_policy_probability,
                is_eog=choice.proposal_is_eog,
            )
        candidate = by_rank.get(requested_rank)
        if candidate is None and resolve_candidate is not None:
            candidate = resolve_candidate(requested_rank)
        if candidate is not None:
            return _candidate_preview(candidate, label="selected candidate")
        return ActionPreview(
            kind="effect",
            label="selected raw rank",
            detail=(
                f"Press Enter to select raw rank {requested_rank}. "
                "This token is not shown in the current menu."
            ),
            valid=True,
        )

    if len(raw) >= 2 and raw[:2].lower() in {"t ", "x "}:
        supplied = raw[2:]
        if not supplied:
            label = (
                "continuation insertion"
                if raw[:1].lower() == "t"
                else "exact insertion"
            )
            return ActionPreview(
                kind="effect",
                label=label,
                detail="Type text after the insertion command.",
                valid=True,
            )
        mode = (
            InsertMode.CONTINUATION
            if raw[:1].lower() == "t"
            else InsertMode.EXACT
        )
        try:
            rendered = resolve_insertion(supplied, mode)
        except Exception as exc:
            return ActionPreview(
                kind="invalid",
                label="insertion cannot be previewed",
                detail=f"{type(exc).__name__}: {exc}",
                valid=False,
            )
        return ActionPreview(
            kind="insertion",
            label=(
                "continuation insertion"
                if mode == InsertMode.CONTINUATION
                else "exact insertion"
            ),
            detail="Tokenization and budget are validated on Enter.",
            appended_text=rendered,
        )

    if lower in {"t", "x"}:
        label = "continuation insertion" if lower == "t" else "exact insertion"
        return ActionPreview(
            kind="effect",
            label=label,
            detail="Add a space and the text to insert.",
            valid=True,
        )

    try:
        fork_address = parse_fork_address(raw)
    except EditorError as exc:
        return ActionPreview(
            kind="invalid",
            label="invalid fork address",
            detail=str(exc),
            valid=False,
        )
    if fork_address is not None:
        current = choice.aligned_step
        if fork_address.kind == ForkAddressKind.CURRENT:
            target = current
        elif fork_address.kind == ForkAddressKind.ABSOLUTE:
            target = int(fork_address.value or 0)
        else:
            target = current - int(fork_address.value or 0)
        if not 0 <= target <= current:
            return ActionPreview(
                kind="invalid",
                label="fork boundary unavailable",
                detail=f"Step {target} is outside this run's recorded range 0..{current}.",
                valid=False,
            )
        return ActionPreview(
            kind="effect",
            label="fork recorded boundary",
            detail=(
                f"The parent will seal at step {current}; a child will open "
                f"fresh from step {target}."
            ),
        )

    normalized = normalize_command_syntax(raw)
    normalized_lower = normalized.lower()
    effects = {
        "h": "Hold will release control only after Enter.",
        "hold": "Hold will release control only after Enter.",
        "q": "Open the live edge menu on Enter; no tokens are generated.",
        "quit": "Open the live edge menu on Enter; no tokens are generated.",
        "finish": "Open the live edge menu on Enter; no tokens are generated.",
        "e": "Teacher EOG selection begins on Enter.",
        "eog": "Teacher EOG selection begins on Enter.",
        "e!": "A recognized teacher EOG is committed on Enter.",
        "eog!": "A recognized teacher EOG is committed on Enter.",
        "m": "The main candidate table returns without disclosing rows on Enter.",
        "more": "The main candidate table returns without disclosing rows on Enter.",
        "v": "The table toggles between raw-model and policy ordering on Enter.",
        "policy-view": "The table toggles between raw-model and policy ordering on Enter.",
        "policy-sort": "The table toggles between raw-model and policy ordering on Enter.",
        "policy-column": "The policy-rank column toggles on Enter without reordering.",
        "policy-rank-column": "The policy-rank column toggles on Enter without reordering.",
        "ms": "The active token-search neighborhood redraws on Enter.",
        "c": "The requested context view opens on Enter.",
        "context": "The requested context view opens on Enter.",
        "n": "The note-before action begins on Enter.",
        "p": "The note-after action begins on Enter.",
        "?": "Full command help opens on Enter.",
        "help": "Full command help opens on Enter.",
    }
    head = normalized_lower.split(maxsplit=1)[0] if normalized_lower else ""
    if raw == "/":
        return ActionPreview(
            kind="effect",
            label="token search",
            detail="Type the exact token text after /.",
            valid=True,
        )
    if raw.startswith("/"):
        detail = "Exact-token search executes on Enter; no text is committed."
    elif normalized_lower.startswith(("m ", "more ")):
        detail = "The main candidate table returns and expands on Enter."
    elif normalized == "V":
        detail = "The policy-rank column toggles on Enter without reordering."
    elif head in effects:
        detail = effects[head]
    else:
        return ActionPreview(
            kind="invalid",
            label="unrecognized command",
            detail="Enter will submit it to the ordinary parser, which may reject it.",
            valid=False,
        )
    return ActionPreview(
        kind="effect",
        label="command effect",
        detail=detail,
        valid=True,
    )


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
        try:
            rows -= int(app.renderer.rows_above_layout)
        except Exception:
            # Before prompt_toolkit receives its cursor-position response, the
            # inline layout height can be unknown. The conservative row budget
            # below still protects the fixed input shell in that first frame.
            pass
        return int(size.columns), max(1, rows)
    except Exception:
        fallback = shutil.get_terminal_size(fallback=(100, 30))
        return fallback.columns, fallback.lines


def _context_rows(context: str, proposal: str, width: int) -> list[StyleAndTextTuples]:
    """Wrap styled context into terminal rows, preserving the proposal highlight."""
    rows: list[StyleAndTextTuples] = [[]]
    column = 0
    for style, text in (("", context), ("class:proposal", proposal)):
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
    return rows


def _context_view(context: str, proposal: str, width: int, height: int,
                  offset: int = 0) -> tuple[StyleAndTextTuples, int]:
    rows = _context_rows(context, proposal, max(1, width - 1))
    budget = max(3, min(12, height // 3))
    offset = min(max(0, offset), max(0, len(rows) - budget))
    end = len(rows) - offset
    start = max(0, end - budget)
    fragments: StyleAndTextTuples = []
    for row in rows[start:end]:
        fragments.extend(row)
        fragments.append(("", "\n"))
    fragments.append(("class:muted",
                      f"Context rows {start + 1}–{end}/{len(rows)} · PgUp/PgDn scroll\n"))
    return fragments, end - start + 1


def _is_writing(command: str) -> bool:
    return command[:2].lower() in {"t ", "x "}


def _writing_sizes(height: int) -> tuple[int, int]:
    """Reserve a stable editor and context region, independent of draft length."""
    editor = max(3, min(16, height // 3))
    context = max(1, min(12, height - editor - 13))
    return editor, context


def _one_line(text: str, width: int) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    result = ""
    for char in text:
        if get_cwidth(result + char) > max(1, width - 2):
            return result + "…"
        result += char
    return result


def _render_writing(choice: ChoiceSet, candidates: tuple[Candidate, ...],
                    preview: ActionPreview, width: int, height: int,
                    offset: int, sort_by_policy: bool) -> StyleAndTextTuples:
    _, budget = _writing_sizes(height)
    rows = _context_rows(_safe_rendered_text(choice.context_text_tail),
                         _safe_rendered_text(preview.appended_text or ""), width - 1)
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
    fragments.append(("class:muted", f"Context rows {start + 1}–{end}/{len(rows)} · PgUp/PgDn\n"))
    if preview.kind == "insertion":
        text = preview.appended_text or ""
        effect = f"{preview.label} · {text.count(chr(10)) + 1} lines · {len(text)} characters"
    else:
        effect = f"{preview.label} · {preview.detail}"
    fragments.append(("class:effect" if preview.valid else "class:invalid", _one_line(effect, width) + "\n"))
    fragments.append(("class:rule", "─" * (width - 1) + "\n"))
    fragments.append(("class:table-header", "Candidates · rank / raw probability / text\n"))
    shown = _ordered_candidates(candidates, sort_by_policy=sort_by_policy)[:3]
    for candidate in shown:
        fragments.append(("class:table-row", _one_line(
            f"{candidate.rank:>5}  {_probability(candidate.raw_probability)}  {candidate.text!r}", width) + "\n"))
    fragments.append(("", "\n" * (3 - len(shown))))
    fragments.append(("class:muted", f"{max(0, len(candidates) - 3)} more candidate rows · clear t/x prefix to restore table\n"))
    fragments.append(("class:prompt-label", "Write text · Enter commits\n"))
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
    display_candidates: tuple[Candidate, ...] | None = None,
    search_lens_active: bool = False,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    context_offset: int = 0,
) -> StyleAndTextTuples:
    width, height = _terminal_size()
    width = max(width, 36)
    preview = action_preview(
        choice,
        command_text,
        candidates,
        resolve_insertion,
        remaining_tokens=remaining_tokens,
        resolve_candidate=resolve_candidate,
    )
    if _is_writing(command_text):
        return _render_writing(choice, tuple(candidates if display_candidates is None else display_candidates),
                               preview, width, height, context_offset, sort_by_policy)
    context = _safe_rendered_text(choice.context_text_tail)
    proposal = (
        _safe_rendered_text(preview.appended_text)
        if preview.appended_text is not None
        else None
    )
    context_fragments, approximate_context_lines = _context_view(
        context, proposal or "", width, height, context_offset,
    )
    feedback_line_limit = max(2, height // 4)
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
    maximum_rows = max(
        4,
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
                ("class:proposal-label", preview.label),
                ("class:muted", rank),
                ("class:muted", f" · exact {repr(preview.appended_text or '')}"),
                ("class:muted", f" · token {preview.token_id}"),
                ("class:muted", f" · raw {_probability(preview.raw_probability)}"),
                (
                    "class:muted",
                    " · decoder "
                    f"{_probability(preview.decoder_probability)}{terminal}",
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
    elif preview.kind == "insertion":
        fragments.extend(
            [
                ("class:proposal-label", preview.label),
                ("class:muted", f" · rendered {repr(preview.appended_text or '')}"),
                ("class:muted", f" · {preview.detail}\n"),
            ]
        )
    else:
        style = "class:invalid" if not preview.valid else "class:effect"
        fragments.extend(
            [
                (style, preview.label),
                ("class:muted", f" · {preview.detail}\n"),
            ]
        )

    fragments.extend([("class:rule", rule + "\n")])
    show_token_id = width >= (88 if show_policy_rank else 78)
    policy_heading = "  pol-rank" if show_policy_rank else ""
    if show_token_id:
        fragments.append(
            (
                "class:table-header",
                f"    rank{policy_heading}     raw-p  decode-p  token-id  text\n",
            )
        )
    else:
        fragments.append(
            (
                "class:table-header",
                f"    rank{policy_heading}     raw-p  decode-p  text\n",
            )
        )
    if hidden_before:
        fragments.append(
            ("class:muted", f"    … {hidden_before} earlier disclosed row(s) …\n")
        )
    for candidate in shown:
        marker = "▶" if candidate.rank == preview.candidate_rank else " "
        target_suffix = " [MATCH]" if candidate.token_id == target_token_id else ""
        decoder = _probability(candidate.decoder_probability)
        raw_probability = _probability(candidate.raw_probability)
        policy_column = (
            f"  {candidate.policy_rank:>8}"
            if show_policy_rank and candidate.policy_rank is not None
            else ""
        )
        if show_token_id:
            prefix = (
                f"{marker} {candidate.rank:>5}{policy_column}  {raw_probability:>8}  "
                f"{decoder:>8}  {candidate.token_id:>8}  "
            )
        else:
            prefix = (
                f"{marker} {candidate.rank:>5}{policy_column}  {raw_probability:>8}  "
                f"{decoder:>8}  "
            )
        text_width = max(8, width - len(prefix) - 1)
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
            ("class:muted", " proposal  "),
            ("class:help-key", "rank"),
            ("class:muted", " choose  "),
            ("class:help-key", "tab/⇧tab"),
            ("class:muted", " browse  "),
            ("class:help-key", "t TEXT"),
            ("class:muted", " insert  "),
            ("class:help-key", "h [N]"),
            ("class:muted", " hold  "),
            ("class:help-key", "?"),
            ("class:muted", " all commands\n"),
            ("class:prompt-label", "Teacher action · Enter commits\n"),
        ]
    )
    return fragments


def _render_review(
    review: BoundaryReview,
    *,
    seamless: bool = False,
    context_offset: int = 0,
) -> StyleAndTextTuples:
    """Render one journal-backed historical boundary, never a live preview."""
    width, height = _terminal_size()
    width = max(width, 36)
    rule = "─" * max(20, width - 1)
    context = _safe_rendered_text(review.context_text_tail)
    position = dict(review.position)
    fragments: StyleAndTextTuples = [
        ("class:status-strong", f"Review boundary {review.aligned_step}"),
        ("class:muted", f" · active boundary {review.active_aligned_step}\n"),
        ("class:rule", rule + "\n"),
        ("class:section", "HISTORICAL BOUNDARY REVIEW\n\n"),
    ]
    context_fragments, _ = _context_view(context, "", width, height, context_offset)
    fragments.extend(context_fragments)
    fragments.append(("", "\n"))
    if position.get("kind") == "edge":
        fragments.extend(
            [
                ("class:proposal-label", "LIVE EDGE"),
                ("class:muted", " · Enter opens the live-edge menu\n"),
            ]
        )
    elif position.get("kind") == "inside-span":
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
        side = "start" if position.get("side") == "before" else "end"
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
                    (
                        "Seamless review · Enter opens the edge menu\n"
                        if position.get("kind") == "edge"
                        else "History · Enter deletes the continuation and resumes here\n"
                    )
                    if seamless
                    else "Review action · Enter submits bare f\n"
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
            "invalid": "ansired bold",
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
            "invalid": "ansibrightred bold reverse",
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


def read_live_choice(
    choice: ChoiceSet,
    *,
    remaining_tokens: int | None,
    candidates: tuple[Candidate, ...],
    display_candidates: tuple[Candidate, ...] | None = None,
    resolve_insertion: InsertionResolver,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    target_token_id: int | None = None,
    input_device: object | None = None,
    output_device: object | None = None,
    theme: str = DEFAULT_LIVE_THEME,
    feedback: ChoiceFeedback | None = None,
    initial_command: str | None = None,
    review: BoundaryReview | None = None,
    seamless: bool = False,
    reactivate_on_review_enter: bool = False,
    search_lens_active: bool = False,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
) -> str | None:
    """Read one command with a live preview and a stable raw-text editor."""
    command_buffer = Buffer(multiline=True)
    if initial_command and review is None:
        command_buffer.document = Document(
            initial_command,
            cursor_position=len(initial_command),
        )
    bindings = KeyBindings()
    context_offset = 0
    completion_owned = bool(initial_command)
    active_table_candidates = tuple(
        candidates if display_candidates is None else display_candidates
    )
    navigation_commands = _navigation_command_cycle(
        choice,
        active_table_candidates,
        feedback,
        sort_by_policy=sort_by_policy,
        search_lens_active=search_lens_active,
    )

    def _replace_buffer(text: str, *, owned: bool) -> None:
        nonlocal completion_owned
        completion_owned = owned
        command_buffer.document = Document(text, cursor_position=len(text))

    def _in_authored_text() -> bool:
        return _is_writing(command_buffer.text)

    review_empty = Condition(lambda: review is not None and not command_buffer.text)
    review_has_input = Condition(
        lambda: review is not None and bool(command_buffer.text)
    )
    active_empty = Condition(
        lambda: review is None and (not command_buffer.text or completion_owned)
    )

    def _navigate(direction: int, event: object) -> None:
        # The initial proposal rank is replace-on-first-typing only.  Once Tab
        # is used, the rank in the buffer is a navigation result, so ordinary
        # editing must not treat the next typed character as a replacement.
        # Keep the buffer non-owned for every navigation result, including the
        # first one selected from an otherwise blank prompt.
        if not navigation_commands:
            return
        current = command_buffer.text
        if not current:
            suggestions = (
                feedback.completion_commands if feedback is not None else ()
            )
            if suggestions:
                _replace_buffer(
                    suggestions[0] if direction > 0 else suggestions[-1],
                    owned=False,
                )
                return
            if (
                direction > 0
                and search_lens_active
            ):
                match_command = (
                    feedback.initial_tab_command
                    if feedback is not None
                    else None
                )
                if match_command is None and target_token_id is not None:
                    match_command = next(
                        (
                            str(candidate.rank)
                            for candidate in active_table_candidates
                            if candidate.token_id == target_token_id
                        ),
                        None,
                    )
                if match_command is not None:
                    _replace_buffer(match_command, owned=False)
                    return
        if not current:
            if search_lens_active and target_token_id is not None:
                match_command = next(
                    (
                        str(candidate.rank)
                        for candidate in active_table_candidates
                        if candidate.token_id == target_token_id
                    ),
                    navigation_commands[0],
                )
                index = navigation_commands.index(match_command)
            else:
                _replace_buffer(
                    navigation_commands[0] if direction > 0 else navigation_commands[-1],
                    owned=False,
                )
                return
        else:
            try:
                index = navigation_commands.index(current)
            except ValueError:
                if search_lens_active:
                    match_command = next(
                        (
                            str(candidate.rank)
                            for candidate in active_table_candidates
                            if candidate.token_id == target_token_id
                        ),
                        navigation_commands[0],
                    )
                    _replace_buffer(match_command, owned=False)
                    return
                proposal_rank = next(
                    (
                        candidate.rank
                        for candidate in active_table_candidates
                        if candidate.token_id == choice.proposal_token_id
                    ),
                    None,
                )
                if current.isdigit() and int(current) == proposal_rank:
                    index = 0
                else:
                    return
        _replace_buffer(
            navigation_commands[(index + direction) % len(navigation_commands)],
            owned=False,
        )

    @bindings.add("c-g")
    def _explore_rank(event: object) -> None:
        raw = command_buffer.text.strip()
        if review is None and raw.isdigit():
            rank = int(raw)
            if rank >= 1 and (choice.vocabulary_size is None or rank <= choice.vocabulary_size):
                event.app.exit(result=f"ms {rank}")  # type: ignore[attr-defined]

    @bindings.add("escape", "enter", filter=Condition(lambda: review is None and _in_authored_text()))
    def _insert_newline(event: object) -> None:
        command_buffer.insert_text("\n")

    @bindings.add("pageup")
    def _context_up(event: object) -> None:
        nonlocal context_offset
        width, height = _terminal_size()
        preview = action_preview(choice, command_buffer.text, candidates, resolve_insertion,
                                 remaining_tokens=remaining_tokens, resolve_candidate=resolve_candidate)
        rows = _context_rows(_safe_rendered_text(review.context_text_tail if review else choice.context_text_tail),
                             "" if review else _safe_rendered_text(preview.appended_text or ""), max(1, max(36, width) - 1))
        budget = _writing_sizes(height)[1] if _in_authored_text() else max(3, min(12, height // 3))
        context_offset = min(max(0, len(rows) - budget), context_offset + max(1, budget - 1))

    @bindings.add("pagedown")
    def _context_down(event: object) -> None:
        nonlocal context_offset
        _, height = _terminal_size()
        budget = _writing_sizes(height)[1] if _in_authored_text() else max(3, min(12, height // 3))
        context_offset = max(0, context_offset - max(1, budget - 1))

    @bindings.add("enter")
    def _submit(event: object) -> None:
        result = command_buffer.text
        if review is not None:
            if seamless and reactivate_on_review_enter and not result.strip():
                result = SEAMLESS_REACTIVATE
            elif result.strip().lower() not in {"f", "fork"}:
                result = "\x1b"
        event.app.exit(result=result)  # type: ignore[attr-defined]

    @bindings.add("[", filter=active_empty | review_empty)
    def _review_back(event: object) -> None:
        event.app.exit(result="[")  # type: ignore[attr-defined]

    @bindings.add("]", filter=active_empty | review_empty)
    def _review_forward(event: object) -> None:
        event.app.exit(result="]")  # type: ignore[attr-defined]

    @bindings.add("escape", filter=Condition(lambda: review is not None))
    def _leave_review(event: object) -> None:
        event.app.exit(result="\x1b")  # type: ignore[attr-defined]

    @bindings.add(
        "escape",
        filter=Condition(lambda: review is None and search_lens_active),
    )
    def _leave_search_lens(event: object) -> None:
        event.app.exit(result="\x1b")  # type: ignore[attr-defined]

    @bindings.add("f", filter=review_empty)
    def _review_fork(event: object) -> None:
        del event
        _replace_buffer("f", owned=False)

    @bindings.add(Keys.Any, filter=review_empty)
    def _consume_and_leave_review(event: object) -> None:
        event.app.exit(result="\x1b")  # type: ignore[attr-defined]

    @bindings.add(Keys.Any, filter=review_has_input)
    def _consume_extra_review_input(event: object) -> None:
        event.app.exit(result="\x1b")  # type: ignore[attr-defined]

    @bindings.add("backspace", filter=review_has_input)
    def _leave_review_on_backspace(event: object) -> None:
        event.app.exit(result="\x1b")  # type: ignore[attr-defined]

    @bindings.add("c-c")
    def _interrupt(event: object) -> None:
        event.app.exit(exception=KeyboardInterrupt())  # type: ignore[attr-defined]

    @bindings.add("c-d", filter=Condition(lambda: not command_buffer.text))
    def _closed(event: object) -> None:
        event.app.exit(result=None)  # type: ignore[attr-defined]

    @bindings.add("tab")
    def _next_completion(event: object) -> None:
        if review is not None:
            event.app.exit(result="\x1b")  # type: ignore[attr-defined]
            return
        if _in_authored_text():
            command_buffer.insert_text("\t")
            return
        _navigate(1, event)

    @bindings.add("s-tab")
    def _previous_completion(event: object) -> None:
        if review is not None:
            event.app.exit(result="\x1b")  # type: ignore[attr-defined]
            return
        if _in_authored_text():
            command_buffer.insert_text("\t")
            return
        _navigate(-1, event)

    completion_filter = Condition(lambda: completion_owned)

    @bindings.add("backspace", filter=completion_filter)
    def _clear_completion(event: object) -> None:
        del event
        _replace_buffer("", owned=False)

    @bindings.add(Keys.BracketedPaste, filter=completion_filter)
    def _replace_completion_with_paste(event: object) -> None:
        _replace_buffer(event.data, owned=False)  # type: ignore[attr-defined]

    @bindings.add(Keys.Any, filter=completion_filter)
    def _replace_completion_with_typing(event: object) -> None:
        _replace_buffer(event.data, owned=False)  # type: ignore[attr-defined]

    choice_control = FormattedTextControl(
        text=(
            (
                lambda: _render_review(
                    review,
                    seamless=seamless,
                    context_offset=context_offset,
                )
            )
            if review is not None
            else lambda: _render_choice(
                choice,
                candidates,
                command_buffer.text,
                remaining_tokens,
                resolve_insertion,
                target_token_id,
                feedback,
                policy_active,
                show_policy_rank,
                sort_by_policy,
                display_candidates,
                search_lens_active,
                resolve_candidate,
                context_offset,
            )
        )
    )
    input_control = BufferControl(buffer=command_buffer, focusable=True)
    prompt_row = VSplit(
        [
            Window(
                FormattedTextControl([("class:prompt", "› ")]),
                width=Dimension.exact(2),
                height=1,
            ),
            Window(input_control, height=lambda: Dimension.exact(
                       _writing_sizes(_terminal_size()[1])[0] if review is None and _in_authored_text() else 1),
                   wrap_lines=True, style="class:input"),
        ]
    )
    remaining_label = "token" if remaining_tokens == 1 else "tokens"
    root = HSplit(
        [
            Window(
                choice_control,
                wrap_lines=True,
                dont_extend_height=True,
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
                                if review is not None
                                else "Alt+Enter newline · Tab indent · Enter commits · PgUp/PgDn context"
                                if _in_authored_text()
                                else (
                                    (f"Next live edge in {remaining_tokens} {remaining_label} · " if remaining_tokens is not None else "q opens the live edge · ") +
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
    layout = Layout(root, focused_element=input_control)
    application: Application[str | None] = Application(
        layout=layout,
        key_bindings=bindings,
        style=_live_style(theme),
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
        input=input_device,  # type: ignore[arg-type]
        output=output_device,  # type: ignore[arg-type]
    )
    return application.run()
