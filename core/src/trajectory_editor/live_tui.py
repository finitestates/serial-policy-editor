"""Live, non-authoritative terminal rendering for one teacher decision."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, VSplit, VerticalAlign, Window
from prompt_toolkit.layout.controls import (
    BufferControl, FormattedTextControl, GetLinePrefixCallable, UIContent, UIControl,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
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
        return ActionPreview(
            kind="effect",
            label=("stochastic beam preview" if command.beam_stochastic else "beam search preview"),
            detail=f"Open a width-{width} {mode} beam; select a branch to commit.",
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
        except Exception as exc:
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
            "Open the temporary deterministic or stochastic branch leaderboard on Enter.",
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
            f"Enter searches for a seed that draws the token at raw rank "
            f"{command.draw_raw_rank}; "
            "the resulting seed is recorded as a reroll.",
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


class _ContextLineStore:
    """Keep logical context lines incrementally, independent of screen width."""

    def __init__(self) -> None:
        self.snapshot: ContextText | None = None
        self.source: str | ContextText | None = None
        self.rows: list[list[tuple[str, str]]] = [[]]
        self._undo: list[
            tuple[ContextText | None, int, tuple[tuple[str, str], ...]]
        ] = []

    def _append_node(self, node: ContextText) -> None:
        self._undo.append(
            (self.snapshot, len(self.rows), tuple(self.rows[-1]))
        )
        parts = _safe_rendered_text(node.chunk).split("\n")
        for index, part in enumerate(parts):
            if part:
                self.rows[-1].append(("", part))
            if index < len(parts) - 1:
                self.rows.append([])
        self.snapshot = node

    def _rebuild(self, snapshot: ContextText) -> None:
        self.snapshot = None
        self.rows = [[]]
        self._undo.clear()
        nodes = snapshot.nodes_since(None)
        assert nodes is not None
        for node in nodes:
            self._append_node(node)

    def _step_back(self) -> None:
        snapshot, row_count, last_row = self._undo.pop()
        del self.rows[row_count:]
        self.rows[-1] = list(last_row)
        self.snapshot = snapshot

    def sync(self, source: str | ContextText) -> None:
        if isinstance(source, str):
            if source == self.source and self.snapshot is not None:
                return
            snapshot = ContextText.root(source)
        else:
            snapshot = source
            if snapshot is self.snapshot:
                return

        if self.snapshot is not None:
            appended = snapshot.nodes_since(self.snapshot)
            if appended is not None:
                for node in appended:
                    self._append_node(node)
                self.source = source
                return
            removed = self.snapshot.nodes_since(snapshot)
            if removed is not None:
                for _node in reversed(removed):
                    self._step_back()
                self.source = source
                return

        self._rebuild(snapshot)
        self.source = source


class _ContextControl(UIControl):
    """Supply unwrapped logical context lines to prompt-toolkit's Window."""

    def __init__(self, get_view: Callable[[], tuple[str, str | ContextText, str, bool]]):
        self.get_view = get_view
        self.store = _ContextLineStore()

    def create_content(self, width: int, height: int) -> UIContent:
        del width, height
        heading, context, proposal, follow_tail = self.get_view()
        self.store.sync(context)
        base_rows = self.store.rows
        base_count = len(base_rows)
        proposal_parts = (
            _safe_rendered_text(proposal).split("\n") if proposal else []
        )
        logical_count = base_count + max(0, len(proposal_parts) - 1)

        def get_line(line_number: int) -> StyleAndTextTuples:
            if line_number == 0:
                return [("class:section", heading)]
            index = line_number - 1
            if index < base_count - 1 or not proposal_parts:
                return list(base_rows[index])
            if index == base_count - 1:
                line = list(base_rows[-1])
                if proposal_parts[0]:
                    line.append(("class:proposal", proposal_parts[0]))
                return line
            proposal_index = index - base_count + 1
            text = proposal_parts[proposal_index]
            return [("class:proposal", text)] if text else []

        cursor = None
        if follow_tail:
            cursor_row = logical_count
            while cursor_row > 0 and not any(
                text for _style, text in get_line(cursor_row)
            ):
                cursor_row -= 1
            cursor = Point(0, cursor_row)
        return UIContent(
            get_line=get_line,
            line_count=logical_count + 1,
            cursor_position=cursor,
        )

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: GetLinePrefixCallable | None,
    ) -> int:
        content = self.create_content(width, max_available_height)
        if not wrap_lines:
            return content.line_count
        height = 0
        for line_number in range(content.line_count):
            height += content.get_height_for_line(
                line_number, width, get_line_prefix
            )
            if height >= max_available_height:
                return max_available_height
        return height


def _safe_context_text(context: str | ContextText) -> str:
    if isinstance(context, ContextText):
        context = context.materialize()
    return _safe_rendered_text(context)


def _preview_fragments(
    preview: ActionPreview, *, policy_active: bool
) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = []
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
) -> tuple[StyleAndTextTuples, Point]:
    preview = preview or action_preview(
        choice,
        command_text,
        candidates,
        resolve_insertion,
        resolve_candidate=resolve_candidate,
        default_hold_tokens=default_hold_tokens,
        default_search_radius=default_search_radius,
    )
    fragments: StyleAndTextTuples = [
        ("class:status-strong", f"Step {choice.aligned_step} · teacher track\n"),
    ]
    fragments.extend(_preview_fragments(preview, policy_active=policy_active))

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
                f"{marker} {candidate.rank:>5}{columns.values(candidate)}  "
                f"{repr(candidate.text)}{suffix}\n",
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
    return fragments, Point(0, cursor_line)


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
) -> StyleAndTextTuples:
    fragments, _cursor = _choice_render_data(
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
    )
    return fragments


def _render_review(
    review: BoundaryReview,
    *,
    seamless: bool = False,
) -> StyleAndTextTuples:
    position = dict(review.position)
    fragments: StyleAndTextTuples = [
        (
            "class:status-strong",
            f"Review boundary {review.aligned_step} · "
            f"active boundary {review.active_aligned_step}\n",
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
            "beam-selected": "ansicyan bold",
            "beam-score": "ansibrightblack",
            "beam-continuation": "",
            "beam-pane": "",
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
            "beam-selected": "bold underline",
            "beam-score": "",
            "beam-continuation": "",
            "beam-pane": "",
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
            "beam-selected": "ansicyan bold",
            "beam-score": "ansibrightblack",
            "beam-continuation": "",
            "beam-pane": "",
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

    def __init__(self, state: ChoiceViewState, *, submit, enabled=lambda: True):
        super().__init__(submit=submit)
        self.state = state
        self.command_buffer = Buffer(multiline=True, read_only=Condition(lambda: not enabled()))
        self.bindings = KeyBindings()
        self.expanded_editor = False
        self._context_follow_tail = True
        self._preview_cache_key = None
        self._preview_cache: ActionPreview | None = None
        self._choice_cursor = Point(0, 0)
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
            self._context_follow_tail = True
            self._preview_cache_key = None

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

        def _scroll_context(direction: int, event: object) -> None:
            info = self.context_window.render_info
            if info is None:
                return
            self._context_follow_tail = False
            last_scroll = max(0, info.content_height - info.window_height)
            page = max(1, info.window_height - 1)
            self.context_window.vertical_scroll = min(
                last_scroll,
                max(0, self.context_window.vertical_scroll + direction * page),
            )
            event.app.invalidate()  # type: ignore[attr-defined]

        @self.bindings.add("pageup")
        def _context_up(event: object) -> None:
            _scroll_context(-1, event)

        @self.bindings.add("pagedown")
        def _context_down(event: object) -> None:
            _scroll_context(1, event)

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

        self._context_control = _ContextControl(self._context_view)
        self.context_window = Window(
            self._context_control,
            wrap_lines=True,
            dont_extend_height=True,
            always_hide_cursor=True,
        )
        choice_control = FormattedTextControl(
            self._render,
            get_cursor_position=self._choice_cursor_position,
        )
        choice_window = Window(
            choice_control,
            wrap_lines=True,
            dont_extend_height=True,
            always_hide_cursor=True,
        )
        input_control = BufferControl(buffer=self.command_buffer, focusable=True)
        prompt_row = VSplit(
            [
                Window(
                    FormattedTextControl([("class:prompt", "› ")]),
                    width=Dimension.exact(2),
                    height=1,
                ),
                Window(
                    input_control,
                    height=lambda: (
                        Dimension(min=3, preferred=8)
                        if _editor_expanded()
                        else Dimension.exact(1)
                    ),
                    wrap_lines=True,
                    style="class:input",
                ),
            ]
        )
        root = HSplit(
            [
                self.context_window,
                choice_window,
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
                                        "q opens the live edge · " +
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
            ],
            align=VerticalAlign.TOP,
        )
        self.layout = Layout(root, focused_element=input_control)

    def update(self, state: ChoiceViewState) -> None:
        self.state = state
        self._preview_cache_key = None
        self._preview_cache = None
        self._context_follow_tail = True
        if hasattr(self, "context_window"):
            self.context_window.vertical_scroll = 0
        self.expanded_editor = False
        self.completion_owned = bool(state.initial_command and state.review is None)
        self.active_table_candidates = tuple(
            state.candidates if state.display_candidates is None else state.display_candidates
        )
        self.navigation_commands = _navigation_command_cycle(
            state.choice, self.active_table_candidates, state.feedback,
            sort_by_policy=state.sort_by_policy,
            sort_by_gumbel=state.sort_by_gumbel,
            search_lens_active=state.search_lens_active,
        )
        text = state.initial_command if self.completion_owned else ""
        self.command_buffer.reset(document=Document(text, cursor_position=len(text)))

    def _current_preview(self) -> ActionPreview:
        command_text = self.command_buffer.text
        key = (id(self.state), command_text)
        # Insertion previews finish asynchronously; recheck them after the
        # owner-thread completion invalidates the application.
        if (
            self._preview_cache_key != key
            or self._preview_cache is None
            or _is_writing(command_text)
            or self._preview_cache.state == "pending"
        ):
            self._preview_cache = action_preview(
                self.state.choice,
                command_text,
                self.state.candidates,
                self.state.resolve_insertion,
                resolve_candidate=self.state.resolve_candidate,
                default_hold_tokens=self.state.default_hold_tokens,
                default_search_radius=self.state.default_search_radius,
            )
            self._preview_cache_key = key
        return self._preview_cache

    def _context_view(self) -> tuple[str, str | ContextText, str, bool]:
        if self.state.review is not None:
            return (
                "HISTORICAL CONTEXT",
                self.state.review.context_text_tail,
                "",
                self._context_follow_tail,
            )
        return (
            "DECISION BOUNDARY",
            self.state.choice.context_text_tail,
            self._current_preview().appended_text or "",
            self._context_follow_tail,
        )

    def _choice_cursor_position(self) -> Point:
        if self.state.review is not None:
            fragments = _render_review(
                self.state.review,
                seamless=self.state.seamless,
            )
            text = "".join(value for _style, value in fragments)
            row = text.count("\n") - int(text.endswith("\n"))
            return Point(0, max(0, row))
        return self._choice_cursor

    def _render(self):
        state = self.state
        if state.review is not None:
            return _render_review(
                state.review,
                seamless=state.seamless,
            )
        fragments, self._choice_cursor = _choice_render_data(
            state.choice,
            state.candidates,
            self.command_buffer.text,
            state.resolve_insertion,
            state.target_token_id,
            state.feedback,
            state.policy_active,
            state.show_policy_rank,
            state.sort_by_policy,
            state.sort_by_gumbel,
            state.logit_view,
            state.show_model_probabilities,
            state.column_focus,
            state.overlays,
            state.display_candidates,
            state.search_lens_active,
            state.resolve_candidate,
            state.default_hold_tokens,
            state.default_search_radius,
            preview=self._current_preview(),
        )
        return fragments
