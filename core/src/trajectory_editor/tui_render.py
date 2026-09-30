"""Pure command-preview and candidate-rendering helpers for terminal views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from rich.text import Text

from .candidate_columns import CandidateColumns
from .core.candidates import Candidate
from .core.errors import EditorError
from .core.ui import ChoiceSet, ContextText, InsertMode
from .teacher_commands import (
    CommandKind,
    CommandState,
    ForkAddressKind,
    TeacherCommand,
    format_beam_rank_ranges,
    interpret_command,
)
from .terminal_contracts import BoundaryReview, ChoiceFeedback
from .ui_themes import semantic_style

InsertionResolver = Callable[[str, InsertMode], str]
Fragments = list[tuple[str, str]]


def _render_fragments(
    fragments: Fragments,
    *,
    theme: str = "amber-cyan",
    environment=None,
) -> Text:
    rendered = Text()
    for semantic, value in fragments:
        name = semantic.removeprefix("class:")
        rendered.append(value, style=semantic_style(name, theme, environment=environment))
    return rendered


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
        first_word = raw.strip().split(maxsplit=1)[0].lower() if raw.strip() else ""
        is_invalid = interpretation.state == CommandState.INVALID
        if is_invalid and first_word == "chord":
            label = "invalid chord"
        elif is_invalid and first_word in {"beam", "gbeam"}:
            label = "invalid beam"
        else:
            label = "command"
        return ActionPreview(
            kind=interpretation.state.value,
            label=label,
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

    if command.kind == CommandKind.BEAM:
        width = command.beam_width
        assert width is not None
        mode = (
            "stochastic Gumbel-Top-k"
            if command.beam_stochastic else
            "cumulative log-p"
        )
        root_skip = (
            f" Skip first-step model ranks {format_beam_rank_ranges(command.beam_skip_rank_ranges)}; "
            "they remain available later."
            if command.beam_skip_rank_ranges else ""
        )
        root_add = (
            " Force and protect first-step model ranks "
            + ", ".join(str(rank) for rank in command.beam_add_model_ranks)
            + "; beam width stays fixed."
            if command.beam_add_model_ranks else ""
        )
        return ActionPreview(
            kind="effect",
            label=("stochastic beam preview" if command.beam_stochastic else "beam search preview"),
            detail=(
                f"Open a width-{width} {mode} beam.{root_skip}{root_add} "
                "Select a branch to commit."
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
                detail="Tokenization and action validity are checked on Enter.",
                appended_text=pending.appended_text,
                command=command,
            )
        except Exception as exc:  # noqa: BLE001 - preview failures are shown inline while Enter remains authoritative.
            return ActionPreview(
                kind="effect", label=label,
                detail=f"Insertion preview unavailable: {type(exc).__name__}: {exc}",
                valid=False, state="invalid", command=command,
            )
        return ActionPreview(
            kind="insertion", label=label,
            detail="Tokenization and action validity are checked on Enter.",
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
        CommandKind.BIAS: ("token bias", "Update a group or token bias; stay at this step."),
        CommandKind.BEAM: (
            "beam search",
            "Open the temporary branch leaderboard; root-rank skips apply only to step one.",
        ),
        CommandKind.SAMPLER: (
            "sampler settings",
            (
                "Enter opens sampler settings input; an accepted change is recorded "
                "in the action tape."
                if command.sampler_text is None
                else "Enter applies and records these sampler settings at this boundary."
            ),
        ),
        CommandKind.REROLL: (
            "reroll",
            (
                f"Enter rerolls the draw and records seed {command.reroll_seed}."
                if command.reroll_seed is not None
                else "Enter chooses a fresh draw seed and records it in the action tape."
            ),
        ),
        CommandKind.DRAW: (
            "targeted draw",
            (
                f"Enter searches for a seed that draws the token at raw rank "
                f"{command.draw_raw_rank}; "
                "the resulting seed is recorded as a reroll."
            ),
        ),
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
        CommandKind.POLICY_VIEW: ("candidate order", "The table advances to the next model, policy, or Gumbel order on Enter."),
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


def _preview_status(preview: ActionPreview) -> tuple[str, str]:
    """Static text and style cues; color is never the only state signal."""
    if preview.state == "invalid":
        return "class:invalid", "INVALID · "
    if preview.state == "incomplete":
        return "class:hint", "INCOMPLETE · "
    if preview.state == "pending":
        return "class:pending", "PENDING · "
    return "class:effect", "READY · "


def _is_writing(command: str) -> bool:
    return command[:2].lower() in {"t ", "x "}


def _ordered_candidates(
    candidates: tuple[Candidate, ...],
    *,
    sort_by_policy: bool = False,
    sort_by_gumbel: bool = False,
) -> tuple[Candidate, ...]:
    """Return the exact visual row order used by rendering and Tab navigation."""
    if sort_by_gumbel:
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.gumbel_rank is None,
                    candidate.gumbel_rank
                    if candidate.gumbel_rank is not None
                    else candidate.rank,
                    candidate.rank,
                ),
            )
        )
    if sort_by_policy:
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.policy_rank
                    if candidate.policy_rank is not None
                    else candidate.rank,
                    candidate.rank,
                ),
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.rank))


def _candidate_command_cycle(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    *,
    sort_by_policy: bool = False,
    sort_by_gumbel: bool = False,
) -> tuple[str, ...]:
    """Return rank commands downward from the proposal, wrapping in view order."""
    ordered = _ordered_candidates(
        candidates,
        sort_by_policy=sort_by_policy,
        sort_by_gumbel=sort_by_gumbel,
    )
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
    sort_by_gumbel: bool = False,
    search_lens_active: bool = False,
) -> tuple[str, ...]:
    suggestions = feedback.completion_commands if feedback is not None else ()
    unique_suggestions = tuple(dict.fromkeys(suggestions))
    if search_lens_active:
        ordered = _ordered_candidates(
            candidates,
            sort_by_policy=sort_by_policy,
            sort_by_gumbel=sort_by_gumbel,
        )
        return unique_suggestions + tuple(str(candidate.rank) for candidate in ordered)
    candidates_cycle = _candidate_command_cycle(
        choice,
        candidates,
        sort_by_policy=sort_by_policy,
        sort_by_gumbel=sort_by_gumbel,
    )
    return candidates_cycle[:1] + unique_suggestions + candidates_cycle[1:]



def _safe_context_text(context: str | ContextText) -> str:
    if isinstance(context, ContextText):
        context = context.materialize()
    return _safe_rendered_text(context)



def _preview_fragment_parts(
    preview: ActionPreview, *, policy_active: bool
) -> Fragments:
    fragments: Fragments = []
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
                ("class:muted", f" · exact {preview.appended_text or ''!r}"),
                ("class:muted", f" · token {preview.token_id}"),
                (
                    "class:muted",
                    f" · raw {_probability(preview.raw_probability)}"
                    if preview.raw_probability is not None else "",
                ),
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
                    f" · policy-rank {preview.policy_rank}"
                    if policy_active and preview.policy_rank is not None else "",
                ),
                ("", "\n"),
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
    return fragments




def _choice_render_fragment_parts(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    command_text: str,
    resolve_insertion: InsertionResolver,
    target_token_id: int | None,
    feedback: ChoiceFeedback | None = None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    sort_by_gumbel: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    display_candidates: tuple[Candidate, ...] | None = None,
    search_lens_active: bool = False,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    default_hold_tokens: int = 100,
    default_search_radius: int = 3,
    preview: ActionPreview | None = None,
) -> tuple[Fragments, int]:
    preview = preview or action_preview(
        choice,
        command_text,
        candidates,
        resolve_insertion,
        resolve_candidate=resolve_candidate,
        default_hold_tokens=default_hold_tokens,
        default_search_radius=default_search_radius,
    )
    fragments: Fragments = [
        ("class:status-strong", f"Step {choice.aligned_step} · teacher track\n"),
    ]
    fragments.extend(_preview_fragment_parts(preview, policy_active=policy_active))

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
    table_candidates = tuple(
        candidates if display_candidates is None else display_candidates
    )
    external_focus = next(
        (
            candidate
            for candidate in candidates
            if candidate.rank == focus_rank
            and not any(row.rank == focus_rank for row in table_candidates)
        ),
        None,
    )

    columns = CandidateColumns(
        policy=show_policy_rank,
        logit_view=logit_view,
        show_model_probabilities=show_model_probabilities,
        column_focus=column_focus,
        overlays=overlays,
        raw_k1_logit=choice.raw_k1_logit,
    )
    fragments.extend(
        [
            ("class:section", "Candidates\n"),
            ("class:table-header", f"    rank{columns.heading}  text\n"),
        ]
    )
    ordered = _ordered_candidates(
        table_candidates,
        sort_by_policy=sort_by_policy,
        sort_by_gumbel=sort_by_gumbel,
    )
    cursor_line = 2
    for candidate in ordered:
        marker = "▶" if candidate.rank == focus_rank else " "
        suffix = " [MATCH]" if candidate.token_id == target_token_id else ""
        if candidate.bias:
            suffix += f" [bias {candidate.bias:+g}]"
        row_style = (
            "class:selected-row" if candidate.rank == focus_rank
            else "class:match-row" if candidate.token_id == target_token_id
            else "class:table-row"
        )
        if candidate.rank == focus_rank:
            cursor_line = sum(text.count("\n") for _style, text in fragments)
        fragments.append(
            (
                row_style,
                (
                    f"{marker} {candidate.rank:>5}{columns.values(candidate)}  "
                    f"{candidate.text!r}{suffix}\n"
                ),
            )
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
    if search_lens_active:
        fragments.append(
            (
                "class:feedback-search",
                "SEARCH LENS · Tab cycles this neighborhood · m/esc main table\n",
            )
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
        for line in feedback.lines:
            fragments.append(
                ("class:feedback-detail", "  " + _safe_rendered_text(line) + "\n")
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
    return fragments, cursor_line



def _render_review_fragment_parts(
    review: BoundaryReview,
    *,
    seamless: bool = False,
) -> Fragments:
    position = dict(review.position)
    fragments: Fragments = [
        (
            "class:status-strong",
            (
                f"Review boundary {review.aligned_step} · "
                f"active boundary {review.active_aligned_step}\n"
            ),
        ),
        ("class:section", "HISTORICAL BOUNDARY REVIEW\n"),
    ]
    if position.get("kind") == "inside-span":
        label = str(position.get("span_type") or "span").replace("-", " ").upper()
        fragments.extend(
            [
                ("class:proposal-label", f"Inside {label}"),
                (
                    "class:muted",
                    (
                        f" · {position.get('offset_visible_tokens')}/"
                        f"{position.get('total_visible_tokens')} visible tokens realized\n"
                    ),
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


def _preview_fragments(
    preview: ActionPreview,
    *,
    policy_active: bool,
    theme: str = "amber-cyan",
    environment=None,
) -> Text:
    return _render_fragments(
        _preview_fragment_parts(preview, policy_active=policy_active),
        theme=theme,
        environment=environment,
    )


def _choice_render_data(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    command_text: str,
    resolve_insertion: InsertionResolver,
    target_token_id: int | None,
    feedback: ChoiceFeedback | None = None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    sort_by_gumbel: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    display_candidates: tuple[Candidate, ...] | None = None,
    search_lens_active: bool = False,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    default_hold_tokens: int = 100,
    default_search_radius: int = 3,
    preview: ActionPreview | None = None,
    theme: str = "amber-cyan",
    environment=None,
) -> tuple[Text, int]:
    fragments, cursor_line = _choice_render_fragment_parts(
        choice,
        candidates,
        command_text,
        resolve_insertion,
        target_token_id,
        feedback,
        policy_active,
        show_policy_rank,
        sort_by_policy,
        sort_by_gumbel,
        logit_view,
        show_model_probabilities,
        column_focus,
        overlays,
        display_candidates,
        search_lens_active,
        resolve_candidate,
        default_hold_tokens,
        default_search_radius,
        preview,
    )
    return _render_fragments(fragments, theme=theme, environment=environment), cursor_line


def _render_choice(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    command_text: str,
    resolve_insertion: InsertionResolver,
    target_token_id: int | None,
    feedback: ChoiceFeedback | None = None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    sort_by_gumbel: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    display_candidates: tuple[Candidate, ...] | None = None,
    search_lens_active: bool = False,
    resolve_candidate: Callable[[int], Candidate] | None = None,
    default_hold_tokens: int = 100,
    default_search_radius: int = 3,
    theme: str = "amber-cyan",
    environment=None,
) -> Text:
    rendered, _cursor_line = _choice_render_data(
        choice,
        candidates,
        command_text,
        resolve_insertion,
        target_token_id,
        feedback,
        policy_active,
        show_policy_rank,
        sort_by_policy,
        sort_by_gumbel,
        logit_view,
        show_model_probabilities,
        column_focus,
        overlays,
        display_candidates,
        search_lens_active,
        resolve_candidate,
        default_hold_tokens,
        default_search_radius,
        theme=theme,
        environment=environment,
    )
    return rendered


def _render_review(
    review: BoundaryReview,
    *,
    seamless: bool = False,
    theme: str = "amber-cyan",
    environment=None,
) -> Text:
    return _render_fragments(
        _render_review_fragment_parts(review, seamless=seamless),
        theme=theme,
        environment=environment,
    )


@dataclass(frozen=True)
class CandidateTablePlan:
    columns: CandidateColumns
    candidates: tuple[Candidate, ...]
    focus_rank: int | None
    external_focus: Candidate | None


def candidate_table_plan(
    choice: ChoiceSet,
    candidates: tuple[Candidate, ...],
    command_text: str,
    *,
    target_token_id: int | None,
    policy_active: bool = False,
    show_policy_rank: bool = False,
    sort_by_policy: bool = False,
    sort_by_gumbel: bool = False,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
    display_candidates: tuple[Candidate, ...] | None = None,
    search_lens_active: bool = False,
    preview: ActionPreview | None = None,
) -> CandidateTablePlan:
    """Prepare visible row order, focus marker, and candidate columns."""
    preview = preview or action_preview(choice, command_text, candidates, lambda _t, _m: "")
    focus_rank = preview.candidate_rank
    if not command_text and search_lens_active and target_token_id is not None:
        focus_rank = next(
            (candidate.rank for candidate in candidates if candidate.token_id == target_token_id),
            focus_rank,
        )
    visible = tuple(candidates if display_candidates is None else display_candidates)
    external_focus = next(
        (
            candidate
            for candidate in candidates
            if candidate.rank == focus_rank
            and not any(row.rank == focus_rank for row in visible)
        ),
        None,
    )
    columns = CandidateColumns(
        policy=show_policy_rank,
        logit_view=logit_view,
        show_model_probabilities=show_model_probabilities,
        column_focus=column_focus,
        overlays=overlays,
        raw_k1_logit=choice.raw_k1_logit,
    )
    return CandidateTablePlan(
        columns=columns,
        candidates=_ordered_candidates(
            visible,
            sort_by_policy=sort_by_policy,
            sort_by_gumbel=sort_by_gumbel,
        ),
        focus_rank=focus_rank,
        external_focus=external_focus,
    )


def candidate_table_row(
    candidate: Candidate,
    columns: CandidateColumns,
    *,
    focus_rank: int | None,
    target_token_id: int | None,
    theme: str,
    environment=None,
) -> tuple[Text, ...]:
    """Return a DataTable row as Rich cells with the old selection markers."""
    selected = candidate.rank == focus_rank
    matched = candidate.token_id == target_token_id
    semantic = "selected-row" if selected else "match-row" if matched else "table-row"
    style = semantic_style(semantic, theme, environment=environment)
    marker = "▶" if selected else " "
    suffix = " [MATCH]" if matched else ""
    if candidate.bias:
        suffix += f" [bias {candidate.bias:+g}]"
    numeric_values = columns.values(candidate).split()
    cells = [Text(marker, style=style), Text(f"{candidate.rank:>5}", style=style)]
    cells.extend(Text(value, style=style) for value in numeric_values)
    cells.append(Text(f"{candidate.text!r}{suffix}", style=style))
    return tuple(cells)
