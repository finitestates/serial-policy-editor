"""Request views: state, key handling, and rendering for each request kind.

A view never draws outside :meth:`render`, and :meth:`render` reads only the
view's current state, so each frame shows exactly one state.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from rich.style import Style
from rich.text import Text

from ..core.errors import EditorError
from ..edge_help import edge_help
from ..teacher_commands import HELP_TEXT
from ..terminal_contracts import (
    SEAMLESS_REACTIVATE,
    BeamInput,
    BeamViewState,
    ChoiceFeedback,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
)
from ..tui_render import (
    ActionPreview,
    PreviewPending,
    _candidate_preview,
    _is_writing,
    _navigation_command_cycle,
    _preview_fragments,
    _render_review,
    _safe_context_text,
    action_preview,
    candidate_table_plan,
    candidate_table_row,
)
from .editor import TextEditor, draw_editor, editor_height
from .keys import Key, Mouse
from .palette import (
    BEAM_PALETTE_ENTRIES,
    PaletteEntry,
    edge_insert_command,
    palette_entries,
)
from .widgets import (
    Column,
    RenderContext,
    Scroll,
    Slot,
    TableRow,
    TableView,
    allocate,
    clip,
    draw_lines,
    draw_table,
    runs,
)

if TYPE_CHECKING:
    from .app import Lifecycle, TerminalApp


def help_document(state: Any) -> Text:
    document = Text(HELP_TEXT)
    document.append("\nTerminal\n", style="bold underline")
    document.append("  F1                       This help\n")
    document.append("  Ctrl+K                   Search commands\n")
    document.append("  Ctrl+L                   Open captured output\n")
    document.append("  Ctrl+C                   Interrupt\n")
    if isinstance(state, EdgeViewState):
        document.append(f"\n{state.mode.title()} edge commands\n", style="bold underline")
        for item in edge_help(state.mode):
            document.append(f"  {item.command:<24} {item.description}\n")
    elif isinstance(state, BeamViewState):
        document.append("\nBeam commands\n", style="bold underline")
        for item in BEAM_PALETTE_ENTRIES:
            document.append(f"  {item.title:<24} {item.help}\n")
    return document


def pack_hint(text: str, width: int, height: int) -> list[str]:
    """Greedily pack ' · '-separated hint items into at most ``height`` lines."""
    items = [item.strip() for part in text.split("\n") for item in part.split(" · ") if item.strip()]
    lines: list[str] = []
    for item in items:
        if lines and len(lines[-1]) + 3 + len(item) <= width:
            lines[-1] += " · " + item
        else:
            lines.append(item)
    if len(lines) > height:
        kept = lines[:height]
        overflow = " · ".join(lines[height - 1:])
        kept[-1] = overflow
        return kept
    return lines


class RequestView:
    """One request's UI state. The app owns submission and input gating."""

    palette_enabled = True

    def __init__(self, app: TerminalApp, lifecycle: Lifecycle) -> None:
        self.app = app
        self.lifecycle = lifecycle
        self.generation = lifecycle.generation
        self.accepting = False
        self.editor = TextEditor()
        self._owner_preview_values: dict[tuple[Any, ...], Any] = {}
        self._owner_preview_errors: dict[tuple[Any, ...], BaseException] = {}
        self._owner_preview_pending: set[tuple[Any, ...]] = set()

    @property
    def state(self) -> Any:
        return self.lifecycle.state

    # -- app hooks -----------------------------------------------------------

    def submit(self, value: Any) -> None:
        self.app.submit(self, value)

    def palette_entries(self) -> tuple[PaletteEntry, ...]:
        return palette_entries(self.state)

    def help_document(self) -> Text:
        return help_document(self.state)

    def insert_command(self, command: str) -> None:
        self.editor.set(command)
        self.on_edit()

    def on_edit(self) -> None:
        """Called after the editor text changed through user input."""

    def owner_preview_ready(self, key: tuple[Any, ...], result: Any,
                            error: BaseException | None) -> None:
        self._owner_preview_pending.discard(key)
        if error is None:
            self._owner_preview_values[key] = result
            self._owner_preview_errors.pop(key, None)
        else:
            self._owner_preview_errors[key] = error

    def on_key(self, key: Key) -> None:
        raise NotImplementedError

    def on_paste(self, text: str) -> None:
        self.editor.insert(text)
        self.on_edit()

    def on_mouse(self, event: Mouse) -> bool:
        return False

    def render(self, ctx: RenderContext) -> None:
        raise NotImplementedError

    # -- shared drawing ------------------------------------------------------

    def edit_key(self, key: Key, *, multiline: bool = False) -> bool:
        before = (self.editor.text, self.editor.replace_on_type)
        if not self.editor.handle_key(key.name, key.char, multiline=multiline):
            return False
        if (self.editor.text, self.editor.replace_on_type) != before:
            self.on_edit()
        return True

    def input_style(self, ctx: RenderContext) -> Style:
        if self.accepting:
            return ctx.styles("prompt-input-focus")
        return ctx.styles("prompt-input-idle")

    def draw_command_row(self, ctx: RenderContext, label: str, y: int, height: int,
                         *, read_only: bool = False, placeholder: str = "") -> None:
        canvas = ctx.canvas
        width = canvas.width
        label_text = f"{label} "
        label_style = ctx.styles("prompt-label") if self.accepting else ctx.styles("muted")
        x = canvas.put(0, y, label_text, label_style)
        for row in range(1, height):
            canvas.put(0, y + row, " " * len(label_text))
        field_width = max(1, width - x)
        field_style = ctx.styles("prompt-input-idle") if read_only else self.input_style(ctx)
        draw_editor(
            canvas, self.editor, x, y, field_width, height,
            style=field_style,
            selected_style=ctx.styles("prompt-input-selected"),
            placeholder="" if not self.accepting else placeholder,
            placeholder_style=self.input_style(ctx) + Style(dim=True),
            show_cursor=self.accepting and not read_only,
        )

    def draw_hint(self, ctx: RenderContext, text: str, y: int, height: int) -> None:
        """Centered key hints, re-flowed on ' · ' boundaries to fit the width."""
        if height <= 0:
            return
        width = ctx.canvas.width
        lines = pack_hint(text, width, height)
        muted = ctx.styles("hint")
        for offset in range(height):
            line = lines[offset] if offset < len(lines) else ""
            clipped = clip(((line, muted),), width) if len(line) > width else ((line, muted),)
            size = min(width, len(line))
            left = max(0, (width - size) // 2)
            ctx.canvas.put(0, y + offset, " " * left)
            ctx.canvas.put_line(left, y + offset, clipped, width=width - left)

    def wrap_rows(self, ctx: RenderContext, text: Text, width: int, *, cache_key: Any = None):
        return ctx.layout.lines(text, max(1, width), cache_key=cache_key)


# ---------------------------------------------------------------------------
# Choice and historical review


class ChoiceView(RequestView):
    """A teacher decision, or a read-only historical boundary review."""

    def __init__(self, app: TerminalApp, lifecycle: Lifecycle) -> None:
        super().__init__(app, lifecycle)
        state: ChoiceViewState = lifecycle.state
        self.review = state.review is not None
        self._completion_owned = bool(state.initial_command and not self.review)
        if self._completion_owned:
            self.editor.set(state.initial_command, replace_on_type=True)
        self._expanded = False
        self.context_scroll = Scroll(follow=True, sticky=True)
        self.table_view = TableView()
        self._preview_key: tuple[int, str] | None = None
        self._preview: ActionPreview | None = None
        self._local_feedback: ChoiceFeedback | None = None
        self._preview_notice: str | None = None
        self._navigation = self._navigation_commands()
        self._last_focus: int | None = None
        self._refresh_preview()

    @property
    def command_text(self) -> str:
        return self.editor.text

    # -- semantics ported from the teacher decision screen ------------------

    def _navigation_commands(self) -> tuple[str, ...]:
        state = self.state
        visible = tuple(state.candidates if state.display_candidates is None else state.display_candidates)
        return _navigation_command_cycle(
            state.choice, visible, state.feedback,
            sort_by_policy=state.sort_by_policy,
            sort_by_gumbel=state.sort_by_gumbel,
            search_lens_active=state.search_lens_active,
        )

    def _candidate_table_plan(self):
        state = self.state
        return candidate_table_plan(
            state.choice, state.candidates, self.command_text,
            target_token_id=state.target_token_id,
            policy_active=state.policy_active,
            show_policy_rank=state.show_policy_rank,
            sort_by_policy=state.sort_by_policy,
            sort_by_gumbel=state.sort_by_gumbel,
            logit_view=state.logit_view,
            show_model_probabilities=state.show_model_probabilities,
            column_focus=state.column_focus,
            overlays=state.overlays,
            display_candidates=state.display_candidates,
            search_lens_active=state.search_lens_active,
            preview=self._preview,
        )

    def _set_command(self, value: str) -> None:
        if value != self.command_text:
            self._preview_key = None
        self.editor.set(value)
        self._completion_owned = False
        self._local_feedback = None
        self._refresh_preview()

    def on_edit(self) -> None:
        self._completion_owned = self.editor.replace_on_type
        self._local_feedback = None
        self._preview_notice = None
        self._expanded = self._expanded and _is_writing(self.command_text)
        self.context_scroll.follow = True
        self._refresh_preview()

    def _preview_request_key(self, command_text: str, kind: str, value: Any) -> tuple[Any, ...]:
        return (self.generation, kind, command_text, value)

    def _owner_resolver(self, command_text: str):
        def resolve_insertion(text: str, mode) -> str:
            key = self._preview_request_key(command_text, "insertion", (text, mode))
            if key in self._owner_preview_errors:
                raise self._owner_preview_errors[key]
            if key in self._owner_preview_values:
                return self._owner_preview_values[key]
            if key not in self._owner_preview_pending:
                self._owner_preview_pending.add(key)
                state = self.state
                self.app.request_owner_preview(
                    self.generation, key, lambda: state.resolve_insertion(text, mode),
                )
            previous = self._preview.appended_text if self._preview is not None else None
            raise PreviewPending(previous)

        def resolve_candidate(rank: int):
            state = self.state
            if state.resolve_candidate is None:
                raise EditorError(f"raw rank {rank} is not available in the current choice")
            key = self._preview_request_key(command_text, "candidate", rank)
            if key in self._owner_preview_errors:
                raise self._owner_preview_errors[key]
            if key in self._owner_preview_values:
                return self._owner_preview_values[key]
            if key not in self._owner_preview_pending:
                self._owner_preview_pending.add(key)
                self.app.request_owner_preview(
                    self.generation, key, lambda: state.resolve_candidate(rank),
                )
            raise PreviewPending()

        return resolve_insertion, resolve_candidate

    def _refresh_preview(self) -> None:
        if self.review:
            return
        key = (self.generation, self.command_text)
        if self._preview_key == key and self._preview is not None:
            return
        state = self.state
        resolve_insertion, resolve_candidate = self._owner_resolver(self.command_text)
        self._preview = action_preview(
            state.choice, self.command_text, state.candidates, resolve_insertion,
            resolve_candidate=resolve_candidate,
            default_hold_tokens=state.default_hold_tokens,
            default_search_radius=state.default_search_radius,
        )
        self._preview_key = key

    def owner_preview_ready(self, key: tuple[Any, ...], result: Any,
                            error: BaseException | None) -> None:
        super().owner_preview_ready(key, result, error)
        if key[0] != self.generation or key[2] != self.command_text:
            return
        preview = self._preview
        if preview is None:
            return
        if error is not None:
            if key[1] == "candidate" and isinstance(error, EditorError):
                self.app.write_diagnostic("candidate preview failed", error)
                self._preview = ActionPreview(
                    kind="effect", label="selected raw rank",
                    detail="Candidate preview unavailable; see Ctrl+L captured output.",
                    valid=False, state="invalid", command=preview.command,
                )
            else:
                self.app.write_diagnostic("choice preview failed", error)
                self._preview_notice = "Preview unavailable; see Ctrl+L captured output."
        elif key[1] == "insertion":
            self._preview = ActionPreview(
                kind="insertion", label=preview.label, detail=preview.detail,
                appended_text=result, command=preview.command,
            )
        elif key[1] == "candidate":
            self._preview = replace(
                _candidate_preview(result, label="selected candidate"),
                command=preview.command,
            )

    def warm_completed(self, target: tuple[int, int], _value: bool,
                       error: BaseException | None) -> None:
        if target != self.state.search_warm_target:
            return
        # Warm-up is speculative, so failures belong in diagnostics only.
        if error is not None:
            self.app.write_diagnostic("search warm-up failed; cold search continues", error)

    def _navigate(self, direction: int) -> None:
        state = self.state
        if self.review:
            self.submit("\x1b")
            return
        if not self._navigation:
            return
        current = self.command_text
        if not current:
            suggestions = state.feedback.completion_commands if state.feedback else ()
            if suggestions:
                self._set_command(suggestions[0] if direction > 0 else suggestions[-1])
                return
            if direction > 0 and state.search_lens_active:
                match_command = state.feedback.initial_tab_command if state.feedback else None
                if match_command is None and state.target_token_id is not None:
                    match_command = next(
                        (str(candidate.rank) for candidate in state.display_candidates or ()
                         if candidate.token_id == state.target_token_id),
                        None,
                    )
                if match_command is not None:
                    self._set_command(match_command)
                    return
        if not current:
            if state.search_lens_active and state.target_token_id is not None:
                match_command = next(
                    (str(candidate.rank) for candidate in state.display_candidates or ()
                     if candidate.token_id == state.target_token_id),
                    self._navigation[0],
                )
                index = self._navigation.index(match_command) if match_command in self._navigation else 0
            else:
                self._set_command(self._navigation[0] if direction > 0 else self._navigation[-1])
                return
        else:
            try:
                index = self._navigation.index(current)
            except ValueError:
                if state.search_lens_active:
                    match_command = next(
                        (str(candidate.rank) for candidate in state.display_candidates or ()
                         if candidate.token_id == state.target_token_id),
                        self._navigation[0],
                    )
                    self._set_command(match_command)
                    return
                proposal_rank = next(
                    (candidate.rank for candidate in state.display_candidates or ()
                     if candidate.token_id == state.choice.proposal_token_id),
                    None,
                )
                if current.isdigit() and int(current) == proposal_rank:
                    index = 0
                else:
                    return
        self._set_command(self._navigation[(index + direction) % len(self._navigation)])

    def _submit_command(self) -> None:
        state = self.state
        raw = self.command_text
        if self.review:
            if not raw.strip() and state.seamless and state.reactivate_on_review_enter:
                self.submit(SEAMLESS_REACTIVATE)
            elif raw.strip().lower() in {"f", "fork"}:
                self.submit(raw)
            else:
                self.submit("\x1b")
            return
        self._refresh_preview()
        preview = self._preview
        if preview is not None and preview.state in {"invalid", "incomplete"}:
            self._local_feedback = ChoiceFeedback(
                "error" if preview.state == "invalid" else "info",
                preview.label.upper(), (preview.detail,),
            )
            return
        if preview is not None and preview.state == "pending":
            self._local_feedback = ChoiceFeedback("info", "PREVIEW PENDING", (preview.detail,))
            return
        self.submit(raw)

    def insert_command(self, command: str) -> None:
        self._set_command(command)

    # -- input ---------------------------------------------------------------

    def on_key(self, key: Key) -> None:
        name = key.name
        state = self.state
        if self.review:
            self._review_key(key)
            return
        if name == "enter":
            self._submit_command()
        elif name in {"tab", "shift+tab"}:
            if _is_writing(self.command_text):
                self.editor.insert("\t")
                self.on_edit()
            else:
                self._navigate(1 if name == "tab" else -1)
        elif name == "ctrl+g":
            raw = self.command_text.strip()
            if raw.isdigit():
                rank = int(raw)
                size = state.choice.vocabulary_size
                if rank >= 1 and (size is None or rank <= size):
                    self.submit(f"ms {rank}")
        elif name == "ctrl+e":
            if _is_writing(self.command_text):
                self._expanded = not self._expanded
        elif name == "alt+enter":
            if _is_writing(self.command_text):
                self.editor.insert("\n")
                self.on_edit()
        elif name in {"pageup", "pagedown"}:
            self.context_scroll.page(-1 if name == "pageup" else 1)
        elif name == "escape":
            if state.search_lens_active:
                self.submit("\x1b")
        elif name == "f2":
            self.app.open_details(self._details_document())
        elif name in {"[", "]"}:
            if self._completion_owned or not self.command_text:
                self.submit(name)
            else:
                self.edit_key(key)
        elif name == "ctrl+d":
            if not self.command_text:
                self.submit(None)
        elif name == "up" or name == "down":
            self._move_focus(-1 if name == "up" else 1)
        else:
            self.edit_key(key, multiline=self._expanded)

    def _move_focus(self, direction: int) -> None:
        """Arrow keys walk the visible candidate rows like Tab does."""
        if _is_writing(self.command_text) and self._expanded:
            if direction < 0:
                self.editor.up()
            else:
                self.editor.down()
            return
        self._navigate(direction)

    def _review_key(self, key: Key) -> None:
        name = key.name
        if name == "enter":
            self._submit_command()
        elif name == "escape":
            self.submit("\x1b")
        elif name in {"[", "]"}:
            self.submit(name)
        elif name == "f":
            if not self.command_text:
                self._set_command("f")
        elif name in {"tab", "shift+tab"}:
            self.submit("\x1b")
        elif name in {"pageup", "pagedown"}:
            self.context_scroll.page(-1 if name == "pageup" else 1)
        elif name == "f2":
            self.app.open_details(self._details_document())
        elif name == "ctrl+d":
            if not self.command_text:
                self.submit(None)
        else:
            self.submit("\x1b")

    def on_paste(self, text: str) -> None:
        if self.review:
            return
        super().on_paste(text)

    def on_mouse(self, event: Mouse) -> bool:
        return False

    def _select_rank(self, rank: int) -> None:
        if self.review or not self.accepting:
            return
        if self._preview is not None and self._preview.candidate_rank == rank:
            return
        self._set_command(str(rank))

    # -- documents -----------------------------------------------------------

    def _context_text(self, ctx: RenderContext) -> tuple[Text, Any]:
        state = self.state
        if self.review:
            tail = _safe_context_text(state.review.context_text_tail)
            heading = "HISTORICAL CONTEXT\n"
        else:
            tail = _safe_context_text(state.choice.context_text_tail)
            heading = "DECISION BOUNDARY\n"
        text = Text()
        text.append(heading, style=ctx.styles("section"))
        text.append(tail)
        return text, ("context", heading, tail)

    def _feedback_text(self, ctx: RenderContext) -> Text:
        feedback = self._local_feedback or self.state.feedback
        rendered = Text()
        if feedback is not None:
            category = feedback.category if feedback.category in {"error", "info", "search"} else "info"
            rendered.append(_safe_context_text(feedback.title) + "\n", style=ctx.styles(f"feedback-{category}"))
            for line in feedback.lines:
                rendered.append("  " + _safe_context_text(line) + "\n", style=ctx.styles("feedback-detail"))
        if self._preview_notice:
            rendered.append(self._preview_notice + "\n", style=ctx.styles("feedback-error"))
        return rendered

    def _preview_text(self) -> Text:
        if self._preview is None:
            return Text()
        return _preview_fragments(
            self._preview, policy_active=self.state.policy_active,
            theme=self.app.styles.theme, environment=self.app.styles.environment,
        )

    def _details_document(self) -> Text:
        styles = self.app.styles
        document = Text()
        document.append("Current choice details\n", style="bold underline")
        sections: list[tuple[str, Text]] = []
        if self.review:
            sections.append((
                "Historical review",
                _render_review(self.state.review, theme=styles.theme, environment=styles.environment),
            ))
        else:
            sections.append(("Current proposal", self._preview_text()))
        context = Text(_safe_context_text(
            self.state.review.context_text_tail if self.review else self.state.choice.context_text_tail
        ))
        sections.append(("Decision context", context))
        feedback = Text()
        current = self._local_feedback or self.state.feedback
        if current is not None:
            feedback.append(_safe_context_text(current.title) + "\n")
            for line in current.lines:
                feedback.append("  " + _safe_context_text(line) + "\n")
        if self._preview_notice:
            feedback.append(self._preview_notice + "\n")
        sections.append(("Feedback", feedback))
        for title, content in sections:
            document.append(f"\n{title}\n", style="bold")
            document.append_text(content)
            if not document.plain.endswith("\n"):
                document.append("\n")
        return document

    def _hint(self, height: int, width: int) -> str:
        tiny = height < 9
        short = height < 18
        if self.review:
            enter = "Enter rewind" if self.state.seamless else "Enter live"
            if tiny:
                return "[ / ] move · f fork · Esc live"
            if short or width < 70:
                return f"[ / ] move · f fork · Esc live\n{enter} · F1 help · F2 details"
            return f"[ / ] move · f fork · Esc live · {enter} · F2 details · F1 help"
        if tiny:
            return "Enter commit · F2 details · F1 help"
        if self._expanded:
            return "Alt+Enter newline · Enter commit · Tab inserts · F2 details · F1 help"
        if width < 80:
            return "Tab browse · Enter commit · F2 details\nF1 help · Ctrl+K commands · PgUp/PgDn context"
        return "Tab/↑↓ browse · Enter commit · PgUp/Dn context · Ctrl+K commands · F2 details · F1 help"

    # -- rendering -------------------------------------------------------------

    def render(self, ctx: RenderContext) -> None:
        if self.review:
            self._render_review(ctx)
        else:
            self._render_live(ctx)

    def _render_live(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        width, height = canvas.width, canvas.height
        styles = ctx.styles
        state = self.state
        short = height < 18

        heading = Text(f"Step {state.choice.aligned_step} · teacher track", style=styles("status-strong"))
        context_text, context_key = self._context_text(ctx)
        context_lines = ctx.layout.lines(context_text, max(1, width - 1), cache_key=context_key)
        preview_text = self._preview_text()
        preview_lines = ctx.layout.lines(preview_text, width) if preview_text.plain else []
        feedback_text = self._feedback_text(ctx)
        feedback_lines = ctx.layout.lines(feedback_text, width) if feedback_text.plain else []

        plan = self._candidate_table_plan()
        columns = [
            Column("marker", "", width=1),
            Column("rank", "rank", width=5, align_right=True),
            *(Column(label, label, width=column_width) for label, column_width in plan.columns.columns),
            Column("text", "text", flex=True),
        ]
        rows = []
        focus_index = None
        for index, candidate in enumerate(plan.candidates):
            cells = candidate_table_row(
                candidate, plan.columns, focus_rank=plan.focus_rank,
                target_token_id=state.target_token_id,
                theme=styles.theme, environment=styles.environment,
            )
            focused = candidate.rank == plan.focus_rank
            rows.append(TableRow(candidate.rank, cells, style=styles("selected-row") if focused else None))
            if focused:
                focus_index = index
        if focus_index != self._last_focus:
            self.table_view.manual = False
            self._last_focus = focus_index

        hint = self._hint(height, width)
        hint_rows = 1 if short else min(2, len(pack_hint(hint, width, 99)))
        editor_width = max(1, width - len("Command > "))
        editor_rows = editor_height(self.editor, editor_width)
        editor_cap = (8 if height >= 30 else 4) if self._expanded else (4 if height >= 18 else 2)
        slots = [
            Slot("command", 1, min(editor_rows if not self._expanded else editor_cap, editor_cap), priority=0),
            Slot("table", 2, len(rows) + 1, priority=1, flex=True, share=1.0),
            Slot("preview", 1, 1 if short else min(3, max(1, len(preview_lines))), priority=1),
            Slot("hint", 1, hint_rows, priority=2),
            Slot("feedback", 1 if feedback_lines else 0,
                 1 if short else min(len(feedback_lines), max(2, height // 6)), priority=2),
            Slot("heading", 1, 1, priority=3),
            Slot("context", 3 if height >= 20 else 1, None, priority=4, flex=True),
            Slot("label", 0 if short else 1, 0 if short else 1, priority=5),
        ]
        heights = allocate(height, slots)
        y = 0
        if heights["heading"]:
            canvas.put_line(0, y, runs(ctx.layout, heading), width=width)
            y += 1
        if heights["context"]:
            draw_lines(ctx, context_lines, 0, y, width, heights["context"], self.context_scroll)
            y += heights["context"]
        if heights["preview"]:
            if heights["preview"] == 1:
                draw_lines(ctx, [clip(runs(ctx.layout, preview_text), width)], 0, y, width, 1)
            else:
                draw_lines(ctx, preview_lines[:heights["preview"]], 0, y, width, heights["preview"], scrollbar=False)
            y += heights["preview"]
        if heights["feedback"]:
            if heights["feedback"] == 1:
                draw_lines(ctx, [clip(runs(ctx.layout, feedback_text), width)], 0, y, width, 1)
            else:
                draw_lines(ctx, feedback_lines[:heights["feedback"]], 0, y, width, heights["feedback"], scrollbar=False)
            y += heights["feedback"]
        if heights["label"]:
            canvas.put_line(0, y, (("Candidates", styles("section")),), width=width)
            y += 1
        if heights["table"]:
            draw_table(
                ctx, columns, rows, 0, y, width, heights["table"],
                focus=focus_index, view=self.table_view,
                on_click=lambda rank: self._select_rank(int(rank)),
            )
            y += heights["table"]
        if heights["command"]:
            self.draw_command_row(ctx, "Command >", y, heights["command"])
            y += heights["command"]
        if heights["hint"]:
            self.draw_hint(ctx, hint, y, heights["hint"])

    def _render_review(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        width, height = canvas.width, canvas.height
        styles = ctx.styles
        state = self.state
        header_text = _render_review(state.review, theme=styles.theme, environment=styles.environment)
        header_lines = ctx.layout.lines(header_text, width)
        context_text, context_key = self._context_text(ctx)
        context_lines = ctx.layout.lines(context_text, max(1, width - 1), cache_key=context_key)
        feedback_text = self._feedback_text(ctx)
        feedback_lines = ctx.layout.lines(feedback_text, width) if feedback_text.plain else []
        hint = self._hint(height, width)
        slots = [
            Slot("command", 1, 1, priority=0),
            Slot("header", 1, len(header_lines), priority=1),
            Slot("hint", 1, min(2, len(pack_hint(hint, width, 99))), priority=2),
            Slot("feedback", 1 if feedback_lines else 0, min(len(feedback_lines), 4), priority=3),
            Slot("rule", 1, 1, priority=5),
            Slot("context", 1, None, priority=4, flex=True),
        ]
        heights = allocate(height, slots)
        y = 0
        if heights["header"]:
            draw_lines(ctx, header_lines[:heights["header"]], 0, y, width, heights["header"], scrollbar=False)
            y += heights["header"]
        if heights["rule"]:
            canvas.put(0, y, "─" * width, styles("rule"))
            y += 1
        if heights["context"]:
            draw_lines(ctx, context_lines, 0, y, width, heights["context"], self.context_scroll)
            y += heights["context"]
        if heights["feedback"]:
            draw_lines(ctx, feedback_lines[:heights["feedback"]], 0, y, width, heights["feedback"], scrollbar=False)
            y += heights["feedback"]
        if heights["command"]:
            self.draw_command_row(ctx, "Review >", y, 1, read_only=True)
            y += 1
        if heights["hint"]:
            self.draw_hint(ctx, hint, y, heights["hint"])


# ---------------------------------------------------------------------------
# Live edge


class EdgeView(RequestView):
    def __init__(self, app: TerminalApp, lifecycle: Lifecycle) -> None:
        super().__init__(app, lifecycle)
        self.table_view = TableView()
        self._row: int | None = None

    def _items(self):
        return edge_help(self.state.mode)

    def _choose(self, index: int) -> None:
        items = self._items()
        if not items:
            return
        self._row = max(0, min(len(items) - 1, index))
        self.table_view.manual = False
        self.editor.set(edge_insert_command(items[self._row].command))

    def on_key(self, key: Key) -> None:
        name = key.name
        if name == "enter":
            self.submit(self.editor.text)
        elif name == "ctrl+d":
            self.submit(None)
        elif name in {"up", "down"}:
            if self._row is None:
                self._choose(0 if name == "down" else len(self._items()) - 1)
            else:
                self._choose(self._row + (1 if name == "down" else -1))
        elif name in {"pageup", "pagedown"}:
            self.table_view.manual = True
            self.table_view.top = max(0, self.table_view.top + (-5 if name == "pageup" else 5))
        else:
            self.edit_key(key)

    def on_paste(self, text: str) -> None:
        super().on_paste(text.replace("\n", " "))

    def render(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        width, height = canvas.width, canvas.height
        styles = ctx.styles
        state = self.state
        title = "LIVE SESSION" if state.mode == "session" else "LIVE EDGE"
        entity = "Branch" if state.mode == "session" else "Episode"
        header = Text(
            f"{title}\n{entity} {state.episode_id} · boundary {state.boundary}\n"
            f"Sampler · {state.sampler_summary}",
            style=styles("status-strong"),
        )
        header_lines = ctx.layout.lines(header, width)
        items = self._items()
        columns = [Column("command", "command"), Column("description", "description", flex=True, wrap=True)]
        rows = [
            TableRow(index, (item.command, item.description),
                     style=styles("selected-row") if index == self._row else None,
                     cache_key=("edge", item.command, item.description))
            for index, item in enumerate(items)
        ]
        hint = (
            "↑↓ or click a template · type to edit · Enter submits\n"
            "Blank continues · Ctrl+C interrupt · Ctrl+D quit\n"
            "Ctrl+K commands · Ctrl+L output · F1 help"
        )
        slots = [
            Slot("command", 1, 1, priority=0),
            Slot("header", 1, len(header_lines), priority=1),
            Slot("hint", 1, 3, priority=2),
            Slot("table", 2, None, priority=3, flex=True),
        ]
        heights = allocate(height, slots)
        y = 0
        draw_lines(ctx, header_lines[:heights["header"]], 0, y, width, heights["header"], scrollbar=False)
        y += heights["header"]
        if heights["table"]:
            draw_table(ctx, columns, rows, 0, y, width, heights["table"],
                       focus=self._row, view=self.table_view,
                       on_click=lambda index: self._choose(int(index)) if self.accepting else None)
            y += heights["table"]
        self.draw_command_row(ctx, "Command >", y, 1)
        y += 1
        self.draw_hint(ctx, hint, y, heights["hint"])


# ---------------------------------------------------------------------------
# Beam


class BeamView(RequestView):
    SIDE_BY_SIDE_MIN_WIDTH = 120

    def __init__(self, app: TerminalApp, lifecycle: Lifecycle) -> None:
        super().__init__(app, lifecycle)
        state: BeamViewState = lifecycle.state
        labels = [row.label for row in state.rows]
        self.selected_label = state.selected_label if state.selected_label in labels else (labels[0] if labels else None)
        self.table_view = TableView()
        self.detail_scroll = Scroll()

    @property
    def labels(self) -> list[str]:
        return [row.label for row in self.state.rows]

    def _select(self, label: str) -> None:
        if label in self.labels and label != self.selected_label:
            self.selected_label = label
            self.detail_scroll.reset()
            self.table_view.manual = False

    def _move(self, direction: int) -> None:
        labels = self.labels
        if not labels:
            return
        try:
            current = labels.index(self.selected_label)
        except ValueError:
            current = 0 if direction >= 0 else len(labels) - 1
        self._select(labels[max(0, min(len(labels) - 1, current + direction))])

    def _submit(self, command: str) -> None:
        self.submit(BeamInput(command, self.selected_label))

    def on_key(self, key: Key) -> None:
        name = key.name
        state = self.state
        empty = not self.editor.text
        blank = not self.editor.text.strip()
        if name == "enter":
            command = self.editor.text.strip()
            if not command:
                command = (
                    "resume" if state.at_edge
                    else f"select {self.selected_label}" if self.selected_label is not None
                    else ""
                )
            self._submit(command)
        elif name in {"up", "down"}:
            self._move(-1 if name == "up" else 1)
        elif name in {"pageup", "pagedown"}:
            self.detail_scroll.page(-1 if name == "pageup" else 1)
        elif name == "backspace" and empty and not state.at_edge:
            self._submit(f"kill {self.selected_label}" if self.selected_label is not None else "k")
        elif name == "p" and empty and not state.at_edge and not state.stochastic:
            self._submit("protect")
        elif name == "f" and empty and not state.at_edge:
            self._submit("families")
        elif name == "right" and blank:
            self._submit("resume" if state.at_edge else "advance 1")
        elif name == "left" and blank:
            self._submit("resume" if state.at_edge else "rewind")
        elif name in {"escape", "ctrl+d"}:
            self._submit("return")
        else:
            self.edit_key(key)

    def on_paste(self, text: str) -> None:
        super().on_paste(text.replace("\n", " "))

    def _details(self, ctx: RenderContext) -> Text:
        state = self.state
        row = next((item for item in state.rows if item.label == self.selected_label), None)
        label = self.selected_label or "—"
        protected = " · PROTECTED" if row is not None and row.protected else ""
        rendered = Text()
        rendered.append(f"SELECTED: {label}{protected}\n", style="bold underline")
        if row is None:
            rendered.append("(no retained branch)")
            return rendered
        rendered.append(f"STATE: {row.state} · SCORE: {row.score.replace('-', '−')}\n")
        if state.stochastic:
            model_logp = "—" if row.model_log_probability is None else f"{row.model_log_probability:.3f}"
            rendered.append(f"G {row.score.replace('-', '−')} · log-p {model_logp}\n")
        model_rank = "—" if row.model_rank is None else str(row.model_rank)
        step_logp = (
            "—" if row.step_log_probability is None
            else f"{row.step_log_probability:.3f}".replace("-", "−")
        )
        rendered.append(f"Model rank: {model_rank} · Step log-p: {step_logp}\n")
        if row.model_log_probability is not None and not state.stochastic:
            rendered.append(f"Model log-p: {row.model_log_probability:.3f}".replace("-", "−") + "\n")
        if row.family_metadata:
            rendered.append(row.family_metadata + "\n", style=ctx.styles("beam-family"))
        rendered.append(row.continuation + "\n\n")
        rendered.append("Recent steps:\n", style="bold")
        if row.recent_steps:
            for step in row.recent_steps:
                rendered.append(f"  {step}\n")
        else:
            rendered.append("  No generated steps yet.\n")
        return rendered

    def _hint(self, width: int, height: int) -> str:
        state = self.state
        if height < 9:
            return "Enter resume · Esc return · F1" if state.at_edge else "↑↓ move · Enter · PgUp/Dn · F1"
        if state.at_edge:
            return "Enter/→ resume · Esc/Ctrl+D return\nCtrl+K commands · Ctrl+L output · F1 help"
        protect = "p protect · " if not state.stochastic else ""
        if width >= self.SIDE_BY_SIDE_MIN_WIDTH:
            return (
                "↑↓ select · ←/→ step · Enter commit · PgUp/Dn details · Backspace kill · "
                f"{protect}f family\nEsc/Ctrl+D return · Ctrl+K commands · Ctrl+L output · F1 help"
            )
        return (
            "↑↓ select · ←/→ step · Enter commit · PgUp/Dn details\n"
            f"Backspace kill · {protect}f family · Esc/Ctrl+D return\n"
            "Ctrl+K commands · Ctrl+L output · F1 help"
        )

    def render(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        width, height = canvas.width, canvas.height
        styles = ctx.styles
        state = self.state
        title = state.title
        if state.at_edge:
            title = f"BEAM OPTIONS   ·   {title.removeprefix('BEAM   ')}"
        heading = runs(ctx.layout, Text(title, style=styles("status-strong")))
        context = Text()
        context.append("Shared context: ", style=styles("section"))
        context.append(_safe_context_text(state.shared_context))
        context_lines = ctx.layout.lines(context, width, cache_key=("beam-context", context.plain))
        hint = self._hint(width, height)
        notice = state.notice or ""
        slots = [
            Slot("command", 1, 1, priority=0),
            Slot("body", 3, None, priority=1, flex=True),
            Slot("heading", 1, 1, priority=2),
            Slot("hint", 1, len(pack_hint(hint, width, 99)), priority=3),
            Slot("notice", 1 if notice else 0, 1 if notice else 0, priority=4),
            Slot("context", 1, min(len(context_lines), max(1, height // 5)), priority=5),
        ]
        heights = allocate(height, slots)
        y = 0
        if heights["heading"]:
            canvas.put_line(0, y, heading, width=width)
            y += 1
        if heights["context"]:
            lines = context_lines[:heights["context"]]
            if len(context_lines) > heights["context"] and lines:
                lines[-1] = clip((*lines[-1], ("…" * 2, None)), width)
            draw_lines(ctx, lines, 0, y, width, heights["context"], scrollbar=False)
            y += heights["context"]
        if heights["body"]:
            self._render_body(ctx, y, heights["body"])
            y += heights["body"]
        if heights["notice"]:
            canvas.put_line(0, y, clip(runs(ctx.layout, Text(notice)), width), width=width)
            y += 1
        self.draw_command_row(ctx, "Beam >", y, 1)
        y += 1
        self.draw_hint(ctx, hint, y, heights["hint"])

    def _table(self, ctx: RenderContext) -> tuple[list[Column], list[TableRow], int | None]:
        state = self.state
        styles = ctx.styles
        columns = [
            Column("marker", "", width=2),
            Column("rank", "#", width=3, align_right=True),
            Column("label", "label"),
            Column("state", "state"),
            Column("score", "score"),
            Column("continuation", "continuation", flex=True, wrap=True),
        ]
        rows: list[TableRow] = []
        focus = None
        for rank, row in enumerate(state.rows, 1):
            selected = row.label == self.selected_label
            marker = ">" if selected else " "
            if row.protected:
                marker = "◆" if marker == " " else ">◆"
            score = row.score.replace("-", "−")
            if state.stochastic:
                model_logp = "—" if row.model_log_probability is None else f"{row.model_log_probability:.3f}"
                score = f"G {score} · log-p {model_logp}"
            continuation = row.continuation.replace("\n", " ↵ ")
            if selected:
                focus = rank - 1
            rows.append(TableRow(
                row.label,
                (marker, str(rank), row.label, row.state, score, continuation),
                style=styles("beam-selected") if selected else None,
                cache_key=("beam", row.label, continuation),
            ))
        return columns, rows, focus

    def _render_body(self, ctx: RenderContext, y: int, height: int) -> None:
        canvas = ctx.canvas
        width = canvas.width
        columns, rows, focus = self._table(ctx)
        details = self._details(ctx)
        on_click = (lambda label: self._select(str(label)) if self.accepting else None)
        if width >= self.SIDE_BY_SIDE_MIN_WIDTH:
            detail_width = max(36, width // 3)
            table_width = width - detail_width - 1
            draw_table(ctx, columns, rows, 0, y, table_width, height, focus=focus,
                       view=self.table_view, on_click=on_click, max_row_lines=4)
            for row in range(height):
                canvas.put(table_width, y + row, "│", ctx.styles("section"))
            detail_lines = ctx.layout.lines(details, detail_width - 2)
            draw_lines(ctx, detail_lines, table_width + 2, y, detail_width - 2, height, self.detail_scroll)
            canvas.fill(table_width + 1, y, 1, height)
            return
        table_needed = 1 + sum(
            min(3, len(ctx.layout.lines(row.cells[-1], max(4, width // 2)) or [()]))
            for row in rows
        )
        detail_height = 0
        if height >= 6:
            detail_height = max(height // 3, height - table_needed - 1)
            detail_height = max(0, min(detail_height, height - 4))
        table_height = height - detail_height - (1 if detail_height else 0)
        draw_table(ctx, columns, rows, 0, y, width, table_height, focus=focus,
                   view=self.table_view, on_click=on_click, max_row_lines=3)
        if detail_height:
            canvas.put(0, y + table_height, "─" * width, ctx.styles("section"))
            detail_lines = ctx.layout.lines(details, width)
            draw_lines(ctx, detail_lines, 0, y + table_height + 1, width, detail_height, self.detail_scroll)


# ---------------------------------------------------------------------------
# Prompt, key, multiline, page, chord


class PromptView(RequestView):
    def __init__(self, app: TerminalApp, lifecycle: Lifecycle) -> None:
        super().__init__(app, lifecycle)
        request: PromptRequest = lifecycle.state
        self.palette_enabled = not request.single_key
        self.body_scroll = Scroll()
        self._escape_pending = False
        self._status: tuple[str, str] | None = None  # (semantic, text)

    def on_key(self, key: Key) -> None:
        name = key.name
        request: PromptRequest = self.state
        if name in {"pageup", "pagedown"}:
            self.body_scroll.page(-1 if name == "pageup" else 1)
            return
        if request.page:
            if name in {"enter", "escape", "q"}:
                self.submit("")
            elif name == "ctrl+d":
                self.submit(None)
            elif name in {"up", "down"}:
                self.body_scroll.by(-1 if name == "up" else 1)
            elif name in {"home", "end"}:
                self.body_scroll.by(-10**9 if name == "home" else 10**9)
            elif name == " ":
                self.body_scroll.page(1)
            return
        if request.single_key:
            if name == "enter":
                self.submit("\n")
            elif name == "escape":
                self.submit("\x1b")
            elif name == "backspace":
                self.submit("\x7f")
            elif name == "ctrl+d":
                self.submit(None)
            elif key.char is not None:
                self.submit(key.char)
            elif name not in {"ctrl+k"}:
                self.submit(name)
            return
        if request.multiline:
            if name == "enter":
                if self._escape_pending:
                    self._escape_pending = False
                    if not self.editor.text.strip():
                        self._status = ("feedback-error", "Write at least one character.")
                        return
                    self.submit(self.editor.text)
                else:
                    self.editor.insert("\n")
                    self.on_edit()
            elif name == "escape":
                self._escape_pending = True
                self._status = ("feedback-info", "Escape pressed · press Enter to submit")
            elif name == "ctrl+d":
                self.submit(None)
            elif name == "tab":
                column = self.editor.cursor - (self.editor.text.rfind("\n", 0, self.editor.cursor) + 1)
                self.editor.insert(" " * (4 - column % 4))
                self.on_edit()
            else:
                self.edit_key(key, multiline=True)
            return
        if name == "enter":
            self.submit(self.editor.text)
        elif name in {"escape", "ctrl+d"}:
            self.submit(None)
        else:
            self.edit_key(key)

    def on_edit(self) -> None:
        self._escape_pending = False
        self._status = None

    def on_paste(self, text: str) -> None:
        request: PromptRequest = self.state
        if request.page or request.single_key:
            return
        if not request.multiline:
            text = text.replace("\n", " ")
        super().on_paste(text)

    def _hint(self) -> str:
        request: PromptRequest = self.state
        if request.page:
            return "PgUp/PgDn ↑↓ scroll · Enter/Esc/q return · Ctrl+L output · F1 help"
        if request.single_key:
            return "Press a key · Backspace returns DEL · Esc returns ESC\nCtrl+D cancels · Ctrl+L output · F1 help"
        if request.multiline:
            return "Enter adds a line · Esc then Enter submits\nCtrl+D cancels · Ctrl+L output · F1 help"
        return "Enter submits · Esc/Ctrl+D cancels · PgUp/PgDn scroll · Ctrl+L output · F1 help"

    def render(self, ctx: RenderContext) -> None:
        request: PromptRequest = self.state
        if request.page:
            self._render_page(ctx)
        else:
            self._render_prompt(ctx)

    def _render_page(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        width, height = canvas.width, canvas.height
        body = Text(self.state.body)
        lines = ctx.layout.lines(body, max(1, width - 1), cache_key=("page", self.state.body))
        hint = self._hint()
        heights = allocate(height, [
            Slot("return", 1, 1, priority=0),
            Slot("body", 1, None, priority=1, flex=True),
            Slot("hint", 1, 1, priority=2),
        ])
        draw_lines(ctx, lines, 0, 0, width, heights["body"], self.body_scroll)
        y = heights["body"]
        label_style = ctx.styles("prompt-label") if self.accepting else ctx.styles("muted")
        text = "Enter, Esc or q returns"
        if self.body_scroll.total > self.body_scroll.height:
            last = min(self.body_scroll.total, self.body_scroll.top + self.body_scroll.height)
            text += f"   ·   lines {self.body_scroll.top + 1}–{last} of {self.body_scroll.total}"
        canvas.put(0, y, " " + text, label_style)
        y += 1
        self.draw_hint(ctx, hint, y, heights["hint"])

    def _render_prompt(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        width, height = canvas.width, canvas.height
        request: PromptRequest = self.state
        styles = ctx.styles
        group_width = min(width, 96)
        left = (width - group_width) // 2
        inner = max(1, group_width - 2)
        body_lines = (
            ctx.layout.lines(Text(request.body), max(1, inner - 1), cache_key=("prompt-body", request.body))
            if request.body else []
        )
        label_lines = ctx.layout.lines(Text(request.prompt, style=styles("prompt-label")), inner) if request.prompt else []
        instructions = (
            ctx.layout.lines(Text("Write the new prompt. Enter adds a line; Esc then Enter submits.",
                                  style=styles("muted")), inner)
            if request.multiline else []
        )
        status_lines = (
            ctx.layout.lines(Text(self._status[1], style=styles(self._status[0])), inner)
            if self._status else []
        )
        if request.multiline:
            input_desired = max(3, min(max(8, editor_height(self.editor, inner)), max(3, height * 2 // 5)))
        elif request.single_key:
            input_desired = 1
        else:
            input_desired = min(3, editor_height(self.editor, inner))
        hint = self._hint()
        hint_rows = len(hint.split("\n"))
        slots = [
            Slot("input", 1, input_desired, priority=0),
            Slot("label", 1 if label_lines else 0, len(label_lines), priority=1),
            Slot("hint", 1, hint_rows, priority=2),
            Slot("status", 1 if status_lines else 0, len(status_lines), priority=2),
            Slot("body", 3 if body_lines else 0, None if body_lines else 0, priority=3, flex=True),
            Slot("instructions", 1 if instructions else 0, len(instructions), priority=4),
            Slot("margin", 0, 2 if height >= 18 else 0, priority=6),
        ]
        heights = allocate(height, slots)
        if body_lines:
            heights_body = min(heights["body"], len(body_lines) + 2)
            spare = heights["body"] - heights_body
            heights["body"] = heights_body
        else:
            spare = 0
        y = heights["margin"]
        if heights["body"]:
            from .widgets import draw_box
            bx, by, bw, bh = draw_box(ctx, left, y, group_width, heights["body"], style=styles("section"))
            draw_lines(ctx, body_lines, bx, by, bw, bh, self.body_scroll)
            y += heights["body"]
        if heights["instructions"]:
            draw_lines(ctx, instructions, left + 1, y, inner, heights["instructions"], scrollbar=False)
            y += heights["instructions"]
        if heights["label"]:
            draw_lines(ctx, label_lines, left + 1, y, inner, heights["label"], scrollbar=False)
            y += heights["label"]
        input_style = self.input_style(ctx)
        if request.single_key:
            text = "Press a key" if self.accepting else "…"
            canvas.put(left + 1, y, text.ljust(inner), input_style, limit=left + 1 + inner)
            if self.accepting:
                canvas.cursor = (left + 1, y)
        else:
            draw_editor(
                canvas, self.editor, left + 1, y, inner, heights["input"],
                style=input_style,
                placeholder=("Write at least one character" if request.multiline else "Response"),
                placeholder_style=input_style + Style(dim=True),
                show_cursor=self.accepting,
            )
        y += heights["input"]
        if heights["status"]:
            draw_lines(ctx, status_lines, left + 1, y, inner, heights["status"], scrollbar=False)
            y += heights["status"]
        y += spare
        hint_y = height - heights["hint"]
        self.draw_hint(ctx, hint, hint_y, heights["hint"])
