"""Small synchronous curses renderer for the terminal request protocol."""

from __future__ import annotations

import curses
import locale
import textwrap
from collections.abc import Callable

from .core.ui import ContextText
from .edge_help import edge_help
from .teacher_commands import HELP_TEXT, CommandState, interpret_command
from .terminal_contracts import (
    SEAMLESS_REACTIVATE,
    BeamInput,
    BeamViewState,
    ChoiceFeedback,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
)
from .tui_render import _is_writing, _navigation_command_cycle, candidate_table_plan
from .ui_themes import DEFAULT_LIVE_THEME, resolve_live_theme


_OUTPUT_LIMIT = 16_000

_CHOICE_PALETTE = (
    ("accept proposal", "accept", "Commit the sampled proposal."),
    ("candidate rank", "1", "Commit a candidate by its raw model rank."),
    ("sampler settings", "s ", "Change sampler settings."),
    ("reroll seed", "reroll ", "Change the replayable draw seed."),
    ("token bias", "b ", "Inspect or change token bias."),
    ("adjust raw rank bias", "1+", "Adjust a token by raw model rank."),
    ("set raw rank bias", "1=0", "Set or clear a direct token adjustment."),
    ("bias groups", "groups", "List token bias groups."),
    ("insert continuation", "t ", "Insert continuation text."),
    ("insert exact text", "x ", "Insert exact text."),
    ("checked continuation", "check ", "Commit a checked continuation phrase."),
    ("checked exact text", "checkx ", "Commit a checked exact phrase."),
    ("force continuation", "force ", "Force a continuation phrase."),
    ("force exact text", "forcex ", "Force an exact phrase."),
    ("hold", "h", "Release control for a requested token count."),
    ("reveal more candidates", "m ", "Return to the main table and reveal more rows."),
    ("return to candidates", "m", "Return to the main candidate table."),
    ("search token", "/", "Find a token and show its rank neighborhood."),
    ("search raw rank", "ms ", "Show the neighborhood of a raw model rank."),
    ("find a draw seed", "draw ", "Find a seed that draws a chosen raw rank."),
    ("chord preview", "chord ", "Preview several temporary continuations."),
    ("beam search", "beam", "Open deterministic beam search."),
    ("stochastic beam", "gbeam", "Open stochastic beam search."),
    ("page context", "context ", "Show more of the current context."),
    ("cycle candidate order", "v", "Cycle model, policy, and Gumbel ordering."),
    ("toggle policy diagnostics", "V", "Toggle policy diagnostics."),
    ("cycle logit view", "l", "Cycle the displayed logit values."),
    ("toggle logits and gaps", "L", "Toggle model logits and top-rank gap."),
    ("toggle probability columns", "%", "Toggle model and decoder probabilities."),
    ("cycle numeric column", "c", "Cycle the focused candidate column."),
    ("toggle candidate overlay", "overlay ", "Toggle a named candidate overlay."),
    ("clear candidate overlays", "C", "Clear overlays and shortcuts."),
    ("previous boundary", "[", "Review the previous token boundary."),
    ("next boundary", "]", "Review the next token boundary."),
    ("fork here", "f", "Fork at the current boundary."),
    ("fork boundary", "f ", "Fork at an absolute token boundary."),
    ("note before", "n ", "Add a note before this decision."),
    ("note after", "p ", "Add a note after the last update."),
    ("preview end token", "e", "Preview a teacher-selected end token."),
    ("commit end token", "e!", "Commit a teacher-selected end token immediately."),
    ("finish", "q", "Open the live edge menu."),
    ("help", "", "Show the full command list."),
)

_BEAM_PALETTE = (
    ("resume", "resume", "Continue from the beam edge."),
    ("select branch", "select ", "Commit the selected beam branch."),
    ("advance", "advance 1", "Advance one beam step."),
    ("rewind", "rewind", "Return one beam step."),
    ("kill branch", "kill ", "Remove a beam branch."),
    ("protect branch", "protect", "Protect a deterministic lineage."),
    ("toggle families", "families", "Toggle branch family details."),
    ("return to teacher", "return", "Return to the teacher decision."),
    ("help", "", "Show the full command list."),
)


class CursesTerminalSession:
    """Render one request at a time on the caller's thread."""

    def __init__(
        self,
        *,
        theme: str = DEFAULT_LIVE_THEME,
        environment: dict[str, str] | None = None,
        restore_output: Callable[[], None] | None = None,
        capture_output: Callable[[], None] | None = None,
    ) -> None:
        self.theme = resolve_live_theme(theme, environment=environment)
        self.environment = dict(environment or {})
        self._restore_output = restore_output or (lambda: None)
        self._capture_output = capture_output or (lambda: None)
        self._screen = None
        self._styles: dict[str, int] = {}
        self._captured_output = ""

    def open(self) -> None:
        try:
            locale.setlocale(locale.LC_CTYPE, "")
        except locale.Error:
            pass
        self._screen = curses.initscr()
        try:
            curses.noecho()
            curses.cbreak()
            self._screen.keypad(True)
            self._init_styles()
            try:
                curses.curs_set(1)
            except curses.error:
                pass
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._screen is None:
            return
        try:
            self._screen.keypad(False)
            curses.nocbreak()
            curses.echo()
        except curses.error:
            pass
        try:
            curses.endwin()
        finally:
            self._screen = None

    def _init_styles(self) -> None:
        self._styles = {"normal": curses.A_NORMAL}
        try:
            if not curses.has_colors():
                self._init_mono_styles()
                return
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            if self.theme == "high-contrast":
                colors = {
                    "primary": curses.COLOR_YELLOW,
                    "secondary": curses.COLOR_CYAN,
                    "accent": curses.COLOR_MAGENTA,
                    "error": curses.COLOR_RED,
                    "muted": curses.COLOR_WHITE,
                }
            elif self.theme == "monochrome":
                self._init_mono_styles()
                return
            else:
                colors = {
                    "primary": curses.COLOR_YELLOW,
                    "secondary": curses.COLOR_CYAN,
                    "accent": curses.COLOR_MAGENTA,
                    "error": curses.COLOR_RED,
                    "muted": curses.COLOR_WHITE,
                }
            for pair, role in enumerate(colors, 1):
                curses.init_pair(pair, colors[role], background)
                self._styles[role] = curses.color_pair(pair)
        except curses.error:
            self._init_mono_styles()
            return
        self._styles.update(
            {
                "header": self._styles.get("primary", 0) | curses.A_BOLD,
                "section": self._styles.get("secondary", 0) | curses.A_BOLD,
                "selected": self._styles.get("secondary", 0) | curses.A_REVERSE | curses.A_BOLD,
                "matched": self._styles.get("accent", 0) | curses.A_BOLD,
                "invalid": self._styles.get("error", 0) | curses.A_BOLD,
                "input": self._styles.get("secondary", 0) | curses.A_BOLD,
                "hint": self._styles.get("muted", 0) | curses.A_DIM,
                "help": self._styles.get("primary", 0) | curses.A_BOLD,
                "feedback": self._styles.get("secondary", 0) | curses.A_BOLD,
            }
        )

    def _init_mono_styles(self) -> None:
        self._styles = {
            "normal": curses.A_NORMAL,
            "header": curses.A_BOLD,
            "section": curses.A_BOLD | curses.A_UNDERLINE,
            "selected": curses.A_REVERSE | curses.A_BOLD,
            "matched": curses.A_UNDERLINE,
            "invalid": curses.A_BOLD | curses.A_UNDERLINE,
            "input": curses.A_REVERSE | curses.A_BOLD,
            "hint": curses.A_DIM,
            "help": curses.A_BOLD,
            "feedback": curses.A_BOLD,
        }

    def _run_ui(self, callback):
        if self._screen is None:
            raise RuntimeError("enter TerminalIO.session() before live requests")
        self._restore_output()
        try:
            return callback()
        finally:
            self._capture_output()

    def _size(self) -> tuple[int, int]:
        assert self._screen is not None
        height, width = self._screen.getmaxyx()
        return max(1, height), max(1, width)

    @staticmethod
    def _safe(text: object) -> str:
        value = str(text)
        output: list[str] = []
        for character in value:
            code = ord(character)
            if character == "\n":
                output.append("\n")
            elif character == "\t":
                output.append("\\t")
            elif code < 32 or code == 127:
                output.append(f"\\x{code:02x}")
            else:
                output.append(character)
        return "".join(output)

    @staticmethod
    def _wrap(text: object, width: int) -> list[str]:
        width = max(1, width)
        source = CursesTerminalSession._safe(text)
        lines: list[str] = []
        for source_line in source.splitlines() or [""]:
            lines.extend(
                textwrap.wrap(
                    source_line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=True,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return lines

    def _put(self, y: int, x: int, text: object, style: str = "normal") -> None:
        assert self._screen is not None
        height, width = self._size()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        value = self._safe(text).replace("\n", " ")
        try:
            self._screen.addnstr(y, x, value, max(0, width - x - 1), self._styles.get(style, 0))
        except curses.error:
            pass

    def _flush(self, cursor: tuple[int, int] | None = None) -> None:
        assert self._screen is not None
        height, width = self._size()
        if cursor is not None:
            y = min(max(0, cursor[0]), height - 1)
            x = min(max(0, cursor[1]), width - 1)
            try:
                self._screen.move(y, x)
            except curses.error:
                pass
        self._screen.noutrefresh()
        curses.doupdate()

    def _begin_frame(self) -> tuple[int, int]:
        assert self._screen is not None
        self._screen.erase()
        try:
            return self._size()
        except curses.error:
            return 24, 80

    @staticmethod
    def _is_key(key: object, name: str) -> bool:
        key_code = getattr(curses, f"KEY_{name}", None)
        if key_code is not None and key == key_code:
            return True
        return name == "ENTER" and key in {"\n", "\r"}

    @staticmethod
    def _is_backspace(key: object) -> bool:
        return key in {"\x08", "\x7f", curses.KEY_BACKSPACE} or CursesTerminalSession._is_key(key, "BACKSPACE")

    @staticmethod
    def _is_delete(key: object) -> bool:
        return CursesTerminalSession._is_key(key, "DC")

    def _key(self):
        assert self._screen is not None
        return self._screen.get_wch()

    def _page(self, title: str, body: str, *, close_on_q: bool = True) -> str:
        assert self._screen is not None
        scroll = 0
        try:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            while True:
                height, width = self._begin_frame()
                self._put(0, 0, title, "header")
                wrapped = self._wrap(body, width - 2)
                visible = max(1, height - 3)
                scroll = max(0, min(scroll, max(0, len(wrapped) - visible)))
                for row, line in enumerate(wrapped[scroll : scroll + visible], 1):
                    self._put(row, 1, line)
                footer = "PgUp/PgDn scroll · Enter/Esc/q returns"
                if title != "Captured output":
                    footer += " · Ctrl+L output"
                self._put(height - 1, 0, footer, "hint")
                self._flush()
                key = self._key()
                if self._is_key(key, "RESIZE"):
                    continue
                if self._is_key(key, "PAGEUP") or self._is_key(key, "UP"):
                    scroll = max(0, scroll - max(1, visible - 1))
                elif self._is_key(key, "PAGEDOWN") or self._is_key(key, "DOWN"):
                    scroll = min(max(0, len(wrapped) - visible), scroll + max(1, visible - 1))
                elif key == "\x0c" and title != "Captured output":
                    self._output_page()
                elif key == "\x0c" and title == "Captured output":
                    return ""
                elif self._is_key(key, "ENTER") or key == "\x1b" or (close_on_q and key in {"q", "Q"}):
                    return ""
        finally:
            try:
                curses.curs_set(1)
            except curses.error:
                pass

    def _help(self, state) -> None:
        if isinstance(state, EdgeViewState):
            body = "\n".join(
                f"{item.command:<24} {item.description}" for item in edge_help(state.mode)
            )
            title = "EDGE commands"
        elif isinstance(state, BeamViewState):
            body = (
                "Beam commands\n\n"
                "Up/Down selects a branch · Enter commits · Left rewinds · Right advances\n"
                "Backspace kills · p protects a deterministic branch · f toggles family details\n"
                "PgUp/PgDn scroll details · Ctrl+K opens commands · Esc/Ctrl+D returns\n\n"
                + HELP_TEXT
            )
            title = "Beam help"
        else:
            body = (
                "Choice keys\n\n"
                "Tab/Shift+Tab cycles candidates · Ctrl+G inspects a typed rank\n"
                "PgUp/PgDn scroll context · Ctrl+E expands · Ctrl+O inserts a line\n"
                "Ctrl+D opens EDGE · Ctrl+K opens commands · Ctrl+L shows captured output\n\n"
                + HELP_TEXT
            )
            title = "Teacher commands"
        self._page(title, body)

    def _palette_entries(self, state) -> tuple[tuple[str, str, str], ...]:
        if isinstance(state, EdgeViewState):
            return tuple(
                (item.command, _edge_insert_command(item.command), item.description)
                for item in edge_help(state.mode)
            ) + (("help", "", "Show the full command list."),)
        if isinstance(state, BeamViewState):
            return _BEAM_PALETTE
        if isinstance(state, ChoiceViewState) and state.review is not None:
            return (
                ("fork reviewed boundary", "f", "Stage a fork at this boundary."),
                ("help", "", "Show the full command list."),
            )
        if isinstance(state, ChoiceViewState):
            return _CHOICE_PALETTE
        return ()

    def _command_palette(self, state) -> str | None:
        """Offer a small searchable command list without leaving curses."""
        assert self._screen is not None
        entries = self._palette_entries(state)
        query = ""
        selected = 0
        scroll = 0
        while True:
            height, width = self._begin_frame()
            self._put(0, 0, "COMMANDS · type to filter", "header")
            self._put(1, 0, f"Filter: {query}", "input")
            matches = tuple(
                entry for entry in entries
                if not query or query.casefold() in " ".join(entry).casefold()
            )
            selected = min(selected, max(0, len(matches) - 1))
            visible = max(0, height - 4)
            if selected < scroll:
                scroll = selected
            elif selected >= scroll + visible and visible:
                scroll = selected - visible + 1
            if matches:
                for index, (title, _insert, description) in enumerate(
                    matches[scroll : scroll + visible]
                ):
                    style = "selected" if scroll + index == selected else "normal"
                    self._put(2 + index, 0, f"{title} · {description}", style)
            else:
                self._put(2, 1, "No matching commands.", "hint")
            self._put(height - 1, 0, "Type filters · Up/Down selects · Enter inserts · Esc returns", "hint")
            self._flush((1, min(width - 1, len("Filter: ") + len(query))))
            key = self._key()
            if self._is_key(key, "RESIZE"):
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"\x1b", "\x04"}:
                return None
            if key == "\x0c":
                self._output_page()
                continue
            if self._is_key(key, "F1"):
                self._help(state)
                continue
            if self._is_key(key, "UP"):
                selected = max(0, selected - 1)
                continue
            if self._is_key(key, "DOWN"):
                selected = min(max(0, len(matches) - 1), selected + 1)
                continue
            if self._is_key(key, "PAGEUP"):
                selected = max(0, selected - max(1, visible - 1))
                continue
            if self._is_key(key, "PAGEDOWN"):
                selected = min(max(0, len(matches) - 1), selected + max(1, visible - 1))
                continue
            if self._is_key(key, "ENTER"):
                if not matches:
                    continue
                title, insertion, _description = matches[selected]
                if title == "help":
                    self._help(state)
                    return None
                return insertion
            if self._is_backspace(key):
                query = query[:-1]
                selected = 0
                scroll = 0
                continue
            if isinstance(key, str) and len(key) == 1 and key.isprintable():
                query += key
                selected = 0
                scroll = 0

    def _output_page(self) -> None:
        self._page("Captured output", self._captured_output, close_on_q=True)

    def add_output(self, text: str) -> None:
        self._captured_output = (self._captured_output + text)[-_OUTPUT_LIMIT:]

    def write(self, text: str, *, end: str = "\n") -> None:
        value = text + end
        self.add_output(value)

    def page(self, text: str) -> None:
        self._run_ui(lambda: self._page("Policy Editor", text))

    def read_choice(self, state: ChoiceViewState) -> str | None:
        return self._run_ui(lambda: self._read_choice(state))

    def _read_choice(self, state: ChoiceViewState) -> str | None:
        command = state.initial_command or ""
        completion_owned = bool(state.initial_command and state.review is None)
        cursor = len(command)
        expanded = False
        context_scroll = 0
        table_scroll = 0
        feedback = state.feedback
        review = state.review
        navigation = _navigation_command_cycle(
            state.choice,
            state.display_candidates or state.candidates,
            feedback,
            sort_by_policy=state.sort_by_policy,
            sort_by_gumbel=state.sort_by_gumbel,
            search_lens_active=state.search_lens_active,
        )
        while True:
            status = None
            syntax = None
            if review is None:
                syntax = interpret_command(
                    command,
                    menu_size=len(state.choice.candidates),
                    default_hold_tokens=state.default_hold_tokens,
                    vocabulary_size=state.choice.vocabulary_size or len(state.candidates),
                    default_search_radius=state.default_search_radius,
                )
                if syntax.state == CommandState.INVALID:
                    status = ("INVALID COMMAND", "invalid")
                elif syntax.state == CommandState.INCOMPLETE:
                    status = ("INCOMPLETE COMMAND", "hint")
                elif command.strip():
                    status = ("READY · Enter applies this command", "feedback")
            rows, cursor_at, selection_index = self._draw_choice(
                state,
                command,
                cursor,
                expanded=expanded,
                context_scroll=context_scroll,
                table_scroll=table_scroll,
                status=status,
                syntax_message=syntax.message if syntax and syntax.message else None,
                feedback=feedback,
            )
            if selection_index is not None and rows > 0:
                if selection_index < table_scroll:
                    table_scroll = selection_index
                    continue
                if selection_index >= table_scroll + rows:
                    table_scroll = max(0, selection_index - rows + 1)
                    continue
            self._flush(cursor_at)
            key = self._key()
            if self._is_key(key, "RESIZE"):
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x04":
                if not command:
                    return None
                if completion_owned:
                    continue
                command = command[:cursor] + command[cursor + 1 :]
                feedback = None
                continue
            if self._is_key(key, "F1"):
                self._help(state)
                continue
            if key == "\x0c":
                self._output_page()
                continue
            if key == "\x0b":
                replacement = self._command_palette(state)
                if replacement is not None:
                    command = replacement
                    cursor = len(command)
                    completion_owned = False
                    feedback = None
                continue
            if review is not None:
                if key == "\x1b":
                    return "\x1b"
                if key == "[":
                    return "["
                if key == "]":
                    return "]"
                if key in {"f", "F"}:
                    command, cursor = "f", 1
                    continue
                if self._is_key(key, "ENTER"):
                    if state.seamless and state.reactivate_on_review_enter:
                        return SEAMLESS_REACTIVATE
                    return command if command.strip().lower() in {"f", "fork"} else "\x1b"
                continue
            if key == "\x1b":
                if state.search_lens_active:
                    return "\x1b"
                continue
            if key == "\x07" and command.strip().isdecimal():
                rank = int(command.strip())
                size = state.choice.vocabulary_size
                if rank >= 1 and (size is None or rank <= size):
                    return f"ms {rank}"
            if key == "\x05" and _is_writing(command):
                expanded = not expanded
                continue
            if not command and state.search_lens_active and navigation:
                match_command = feedback.initial_tab_command if feedback is not None else None
                if match_command is None and state.target_token_id is not None:
                    match_command = next(
                        (
                            str(candidate.rank)
                            for candidate in state.display_candidates or ()
                            if candidate.token_id == state.target_token_id
                        ),
                        None,
                    )
                if match_command in navigation and (key == "\t" or self._is_key(key, "TAB")):
                    command = match_command
                    cursor = len(command)
                    continue
            if key in {"[", "]"} and (not command.strip() or completion_owned):
                return key
            if key == "\t" or self._is_key(key, "TAB") or self._is_key(key, "BTAB"):
                if _is_writing(command):
                    command, cursor = _edit_insert(command, cursor, "\t")
                elif navigation:
                    direction = -1 if self._is_key(key, "BTAB") else 1
                    if command not in navigation:
                        next_index = 0 if direction > 0 else len(navigation) - 1
                    else:
                        next_index = (navigation.index(command) + direction) % len(navigation)
                    command = navigation[next_index]
                    cursor = len(command)
                    completion_owned = False
                feedback = None
                continue
            if self._is_key(key, "PAGEUP"):
                context_scroll += 1
                continue
            if self._is_key(key, "PAGEDOWN"):
                context_scroll = max(0, context_scroll - 1)
                continue
            if key == "\x0f" and _is_writing(command):
                command, cursor = _edit_insert(command, cursor, "\n")
                expanded = True
                feedback = None
                continue
            if self._is_key(key, "ENTER"):
                if syntax is not None and syntax.state != CommandState.READY:
                    feedback = ChoiceFeedback(
                        "error" if syntax.state == CommandState.INVALID else "info",
                        status[0] if status else "COMMAND",  # type: ignore[index]
                        (syntax.message,),
                    )
                    continue
                return command
            previous_command = command
            if completion_owned and (
                self._is_backspace(key)
                or self._is_delete(key)
                or key in {"\x0b", "\x15", "\x17"}
            ):
                continue
            command, cursor, completion_owned = _edit_key(
                command,
                cursor,
                key,
                completion_owned=completion_owned,
            )
            if command != previous_command:
                feedback = None

    def _draw_choice(
        self,
        state: ChoiceViewState,
        command: str,
        cursor: int,
        *,
        expanded: bool,
        context_scroll: int,
        table_scroll: int,
        status: tuple[str, str] | None,
        syntax_message: str | None,
        feedback: ChoiceFeedback | None,
    ) -> tuple[int, tuple[int, int], int | None]:
        height, width = self._begin_frame()
        input_rows = 3 if expanded else 1
        input_y = max(1, height - 2 - input_rows + 1)
        status_y = input_y - 1
        detail_y = status_y - 1
        choice = state.choice
        self._put(0, 0, f"POLICY EDITOR · Step {choice.aligned_step} · teacher track", "header")
        proposal_rank = choice.proposal_raw_rank
        probability = (
            "--" if choice.proposal_raw_probability is None
            else f"{choice.proposal_raw_probability:.2%}"
        )
        self._put(
            1,
            0,
            f"Proposal {proposal_rank if proposal_rank is not None else '?'} · {choice.proposal_text!r} · model {probability}",
            "feedback",
        )
        context = state.review.context_text_tail if state.review is not None else choice.context_text_tail
        context_text = context.materialize() if isinstance(context, ContextText) else str(context)
        context_lines = self._wrap(context_text, max(1, width - 2))
        context_rows = max(1, min(3, max(1, detail_y - 7)))
        self._put(2, 0, "Historical context" if state.review else "Context · PgUp/PgDn scroll", "section")
        max_context_scroll = max(0, len(context_lines) - context_rows)
        context_scroll = min(max(0, context_scroll), max_context_scroll)
        visible_context = context_lines[
            max_context_scroll - context_scroll : max_context_scroll - context_scroll + context_rows
        ]
        for offset, line in enumerate(visible_context, 3):
            self._put(offset, 1, line, "hint")

        row_count = 0
        selection_index = None
        if state.review is not None:
            row = 3 + len(visible_context) + 1
            self._put(row, 0, f"Review boundary {state.review.aligned_step} of {state.review.active_aligned_step}", "section")
            position = dict(state.review.position)
            if position.get("kind") == "inside-span":
                self._put(row + 1, 1, f"Inside {position.get('span_type', 'span')} · {position.get('offset_visible_tokens')}/{position.get('total_visible_tokens')} tokens")
            elif position.get("kind") == "action-boundary":
                self._put(row + 1, 1, f"{position.get('action_kind', 'action')} {position.get('side', '')}")
            if state.review.next_token is not None:
                self._put(row + 2, 1, f"Next token: {state.review.next_token.get('text')!r}")
            self._put(detail_y, 0, "[ / ] review boundaries · f stage fork · Enter returns to live · Esc leaves review", "hint")
        else:
            plan = candidate_table_plan(
                choice,
                state.candidates,
                command,
                target_token_id=state.target_token_id if state.search_lens_active else None,
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
            )
            table_y = 3 + len(visible_context) + 1
            labels = "    rank" + plan.columns.heading + "  text"
            self._put(table_y, 0, "Candidates · Tab cycles rows", "section")
            self._put(table_y + 1, 0, labels[: max(0, width - 1)], "hint")
            table_start = table_y + 2
            row_count = max(0, detail_y - table_start)
            focus_rank = plan.focus_rank
            if focus_rank is not None:
                selection_index = next(
                    (index for index, candidate in enumerate(plan.candidates) if candidate.rank == focus_rank),
                    None,
                )
            table_scroll = max(0, min(table_scroll, max(0, len(plan.candidates) - row_count)))
            for local_index, candidate in enumerate(plan.candidates[table_scroll : table_scroll + row_count]):
                marker = ">" if candidate.rank == focus_rank else " "
                suffix = " [MATCH]" if candidate.token_id == state.target_token_id else ""
                if candidate.bias:
                    suffix += f" [bias {candidate.bias:+g}]"
                values = plan.columns.values(candidate)
                line = f"{marker} {candidate.rank:>5}{values}  {candidate.text!r}{suffix}"
                style = (
                    "selected" if candidate.rank == focus_rank
                    else "matched" if candidate.token_id == state.target_token_id
                    else "normal"
                )
                self._put(table_start + local_index, 0, line, style)
            if not plan.candidates:
                self._put(table_start, 1, "No candidates are available.", "hint")
            if state.search_lens_active:
                self._put(detail_y, 0, "SEARCH LENS · Esc returns to main candidates", "help")

        if feedback is not None:
            feedback_style = "invalid" if feedback.category == "error" else "feedback"
            self._put(status_y, 0, feedback.title, feedback_style)
            if feedback.lines:
                self._put(detail_y, 0, feedback.lines[0], "hint")
        if status is not None:
            self._put(status_y, 0, status[0], status[1])
        if syntax_message:
            self._put(detail_y, 0, syntax_message, "hint")
        footer = (
            "Enter submit · Tab browse · Ctrl+G rank · Ctrl+D EDGE · Ctrl+K menu · F1 help"
            if state.review is None
            else "Enter resumes here · f then Enter forks · [/] review · Esc live · Ctrl+K menu"
        )
        self._put(height - 1, 0, footer, "hint")
        cursor_at = self._draw_editor("Command > ", command, cursor, input_y, input_rows)
        return row_count, cursor_at, selection_index

    def read_edge(self, state: EdgeViewState) -> str | None:
        return self._run_ui(lambda: self._read_edge(state))

    def _read_edge(self, state: EdgeViewState) -> str | None:
        command = ""
        cursor = 0
        selected = -1
        item_scroll = 0
        items = edge_help(state.mode)
        while True:
            height, width = self._begin_frame()
            title = "LIVE SESSION" if state.mode == "session" else "LIVE EDGE"
            entity = "Branch" if state.mode == "session" else "Episode"
            self._put(0, 0, title, "header")
            self._put(1, 0, f"{entity} {state.episode_id} · boundary {state.boundary}", "section")
            self._put(2, 0, f"Sampler · {state.sampler_summary}", "hint")
            table_start = 4
            input_y = max(1, height - 2)
            visible_items = max(0, input_y - table_start)
            item_scroll = max(0, min(item_scroll, max(0, len(items) - visible_items)))
            if selected >= 0 and selected < item_scroll:
                item_scroll = selected
            elif selected >= item_scroll + visible_items and visible_items:
                item_scroll = selected - visible_items + 1
            for index, item in enumerate(items[item_scroll : item_scroll + visible_items]):
                absolute_index = item_scroll + index
                style = "selected" if absolute_index == selected else "normal"
                self._put(table_start + index, 0, f"{item.command:<28} {item.description}", style)
            self._put(input_y - 1, 0, "Blank continues · Ctrl+D quits · Up/Down or Tab picks a template", "hint")
            self._put(height - 1, 0, "Enter submit · Ctrl+K commands · F1 help · Ctrl+L output", "hint")
            cursor_at = self._draw_editor("EDGE > ", command, cursor, input_y, 1)
            self._flush(cursor_at)
            key = self._key()
            if self._is_key(key, "RESIZE"):
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x04":
                return None
            if self._is_key(key, "F1"):
                self._help(state)
                continue
            if key == "\x0c":
                self._output_page()
                continue
            if key == "\x0b":
                replacement = self._command_palette(state)
                if replacement is not None:
                    command = replacement
                    cursor = len(command)
                continue
            if key == "\t" or self._is_key(key, "TAB"):
                selected = (0 if selected < 0 else selected + 1) % len(items) if items else -1
                if selected >= 0:
                    command = _edge_insert_command(items[selected].command)
                    cursor = len(command)
                continue
            if self._is_key(key, "UP") or self._is_key(key, "DOWN"):
                direction = -1 if self._is_key(key, "UP") else 1
                if items:
                    selected = 0 if selected < 0 else max(0, min(len(items) - 1, selected + direction))
                if selected >= 0:
                    command = _edge_insert_command(items[selected].command)
                    cursor = len(command)
                continue
            if self._is_key(key, "ENTER"):
                return command
            command, cursor, _owned = _edit_key(command, cursor, key)

    def read_beam(self, state: BeamViewState) -> BeamInput | None:
        return self._run_ui(lambda: self._read_beam(state))

    def _read_beam(self, state: BeamViewState) -> BeamInput | None:
        labels = [row.label for row in state.rows]
        selected = state.selected_label if state.selected_label in labels else (labels[0] if labels else None)
        command = ""
        cursor = 0
        detail_scroll = 0
        branch_scroll = 0
        while True:
            height, width = self._begin_frame()
            title = "STOCHASTIC BEAM" if state.stochastic else "BEAM SEARCH"
            self._put(0, 0, f"{title} · {state.title}", "header")
            self._put(1, 0, "Gumbel-Top-k without replacement." if state.stochastic else "Cumulative log-p over the policy adjusted vocabulary.", "hint")
            context_lines = self._wrap(state.shared_context, width - 2)
            context_rows = min(2, max(0, height - 9))
            self._put(2, 0, "Shared context", "section")
            for i, line in enumerate(context_lines[-context_rows:] if context_rows else (), 3):
                self._put(i, 1, line, "hint")
            table_y = 3 + context_rows
            self._put(table_y, 0, "Survivors · Up/Down selects", "section")
            input_y = max(1, height - 2)
            available_list_rows = max(1, (input_y - table_y - 4) // 2)
            list_rows = min(len(state.rows), available_list_rows)
            selected_index = labels.index(selected) if selected in labels else 0
            branch_scroll = max(0, min(branch_scroll, max(0, len(labels) - list_rows)))
            if selected_index < branch_scroll:
                branch_scroll = selected_index
            elif selected_index >= branch_scroll + list_rows:
                branch_scroll = selected_index - list_rows + 1
            detail_y = table_y + 2 + list_rows
            self._put(
                table_y,
                0,
                f"Survivors {branch_scroll + 1}-{branch_scroll + list_rows} / {len(state.rows)} · Up/Down selects",
                "section",
            )
            for index, row in enumerate(state.rows[branch_scroll : branch_scroll + list_rows]):
                absolute_index = branch_scroll + index
                marker = ">" if row.label == selected else " "
                protected = " protected" if row.protected else ""
                score_label = "gumbel" if state.stochastic else "beam"
                model_logp = (
                    f" · model-logp {row.model_log_probability:.6f}"
                    if state.stochastic and row.model_log_probability is not None
                    else ""
                )
                line = (
                    f"{marker}{absolute_index + 1:>2} {row.label:<3} {row.state:<4} "
                    f"model-rank {row.model_rank if row.model_rank is not None else '—'} "
                    f"step-logp {row.step_log_probability if row.step_log_probability is not None else '—'} "
                    f"{score_label} {row.score}{model_logp}{protected} · "
                    f"{row.continuation.replace(chr(10), ' ')}"
                )
                self._put(table_y + 1 + index, 0, line, "selected" if row.label == selected else "normal")
            selected_row = next((row for row in state.rows if row.label == selected), None)
            if selected_row is not None:
                detail_lines = ["Continuation:"]
                detail_lines.extend(self._wrap(selected_row.continuation, width - 2))
                if selected_row.family_metadata:
                    detail_lines.extend(["", "Family:", *self._wrap(selected_row.family_metadata, width - 2)])
                if selected_row.protected:
                    detail_lines.append("Protected lineage: yes")
                if selected_row.recent_steps:
                    detail_lines.extend(["", "Recent steps:"])
                    for step in selected_row.recent_steps:
                        detail_lines.extend(self._wrap(step, width - 2))
                visible_detail = max(0, input_y - detail_y - 2)
                detail_scroll = min(
                    max(0, detail_scroll),
                    max(0, len(detail_lines) - visible_detail),
                )
                self._put(detail_y, 0, f"Selected {selected_row.label} · branch details", "section")
                for offset, line in enumerate(
                    detail_lines[detail_scroll : detail_scroll + visible_detail],
                    detail_y + 1,
                ):
                    self._put(offset, 1, line, "hint")
            if state.notice:
                self._put(input_y - 1, 0, state.notice, "feedback")
            if state.at_edge:
                footer = "Enter/Right resume · Left resume · q for beam EDGE · Esc/Ctrl+D return"
            else:
                footer = "Enter commit · Up/Down select · PgUp/Dn detail · Backspace kill · p/f action · Ctrl+K menu"
            self._put(height - 1, 0, footer, "hint")
            cursor_at = self._draw_editor("Beam > ", command, cursor, input_y, 1)
            self._flush(cursor_at)
            key = self._key()
            if self._is_key(key, "RESIZE"):
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"\x04", "\x1b"}:
                return BeamInput("return", selected)
            if self._is_key(key, "F1"):
                self._help(state)
                continue
            if key == "\x0c":
                self._output_page()
                continue
            if key == "\x0b":
                replacement = self._command_palette(state)
                if replacement is not None:
                    command = replacement
                    cursor = len(command)
                continue
            if self._is_key(key, "UP") and labels:
                index = labels.index(selected) if selected in labels else 0
                selected = labels[max(0, index - 1)]
                detail_scroll = 0
                continue
            if self._is_key(key, "DOWN") and labels:
                index = labels.index(selected) if selected in labels else -1
                selected = labels[min(len(labels) - 1, index + 1)]
                detail_scroll = 0
                continue
            if not command.strip() and self._is_key(key, "RIGHT"):
                return BeamInput("resume" if state.at_edge else "advance 1", selected)
            if not command.strip() and self._is_key(key, "LEFT"):
                return BeamInput("resume" if state.at_edge else "rewind", selected)
            if not command and self._is_key(key, "ENTER"):
                return BeamInput(
                    "resume" if state.at_edge else (f"select {selected}" if selected else ""),
                    selected,
                )
            if not command and self._is_backspace(key):
                if not state.at_edge:
                    return BeamInput(f"kill {selected}" if selected else "k", selected)
                continue
            if not command and key in {"p", "P"} and not state.at_edge and not state.stochastic:
                return BeamInput("protect", selected)
            if not command and key in {"f", "F"} and not state.at_edge:
                return BeamInput("families", selected)
            if self._is_key(key, "PAGEUP"):
                detail_scroll = max(0, detail_scroll - max(1, input_y - detail_y - 2))
                continue
            if self._is_key(key, "PAGEDOWN"):
                detail_scroll += max(1, input_y - detail_y - 2)
                continue
            if self._is_key(key, "ENTER"):
                submitted = command.strip() or (
                    "resume" if state.at_edge else f"select {selected}" if selected else ""
                )
                return BeamInput(submitted, selected)
            command, cursor, _owned = _edit_key(command, cursor, key)

    def prompt(self, request: PromptRequest) -> str | None:
        return self._run_ui(lambda: self._prompt(request))

    def _prompt(self, request: PromptRequest) -> str | None:
        if request.page:
            return self._page("Policy Editor", request.body)
        if request.single_key:
            return self._single_key_prompt(request)
        command = ""
        cursor = 0
        escaped = False
        prompt_message = ""
        scroll = 0
        input_rows = 3 if request.multiline else 1
        while True:
            height, width = self._begin_frame()
            self._put(0, 0, "POLICY EDITOR", "header")
            body_lines = self._wrap(request.body, width - 2) if request.body else []
            input_y = max(2, height - 2 - input_rows + 1)
            body_visible = max(0, min(len(body_lines), input_y - 4))
            scroll = min(max(0, scroll), max(0, len(body_lines) - body_visible))
            for i, line in enumerate(body_lines[scroll : scroll + body_visible], 1):
                self._put(i, 1, line)
            self._put(input_y - 1, 0, request.prompt, "section")
            if request.multiline:
                instructions = "Enter adds a line · Esc then Enter submits · Ctrl+D cancels"
                self._put(input_y - 2, 0, instructions, "hint")
            elif request.body:
                self._put(input_y - 2, 0, "Enter submits · Esc/Ctrl+D cancels · PgUp/PgDn scrolls", "hint")
            if prompt_message:
                self._put(input_y - 3, 0, prompt_message, "invalid")
            self._put(height - 1, 0, "F1 help · Ctrl+L captured output", "hint")
            available_input_rows = max(1, height - input_y - 1)
            cursor_at = self._draw_editor(
                "", command, cursor, input_y, min(input_rows, available_input_rows)
            )
            if escaped:
                self._put(input_y - 1, 0, "Escape armed · Enter submits", "feedback")
            self._flush(cursor_at)
            key = self._key()
            if self._is_key(key, "RESIZE"):
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x04":
                return None
            if self._is_key(key, "F1"):
                self._page("Prompt help", "Enter submits · Esc/Ctrl+D cancels\nPgUp/PgDn scrolls the request body")
                continue
            if key == "\x0c":
                self._output_page()
                continue
            if request.body and self._is_key(key, "PAGEUP"):
                scroll = max(0, scroll - max(1, body_visible - 1))
                continue
            if request.body and self._is_key(key, "PAGEDOWN"):
                scroll += max(1, body_visible - 1)
                continue
            if key == "\x1b":
                if request.multiline:
                    escaped = True
                else:
                    return None
                continue
            if self._is_key(key, "ENTER"):
                if request.multiline and not escaped:
                    command, cursor = _edit_insert(command, cursor, "\n")
                    prompt_message = ""
                    continue
                if request.multiline and not command.strip():
                    escaped = False
                    prompt_message = "Write at least one character."
                    continue
                return command
            previous_command = command
            command, cursor, _owned = _edit_key(command, cursor, key, allow_newline=request.multiline)
            if command != previous_command:
                prompt_message = ""
            if escaped:
                escaped = False

    def _single_key_prompt(self, request: PromptRequest) -> str | None:
        assert self._screen is not None
        while True:
            height, width = self._begin_frame()
            self._put(0, 0, "POLICY EDITOR", "header")
            body_lines = self._wrap(request.body, width - 2) if request.body else []
            for index, line in enumerate(body_lines[: max(0, height - 6)], 1):
                self._put(index, 1, line)
            y = max(2, min(height - 2, len(body_lines) + 2))
            self._put(y, 0, request.prompt + "Press a key", "section")
            self._put(height - 1, 0, "Enter = newline · Backspace = DEL · Esc returns · Ctrl+D cancels", "hint")
            self._flush((y, min(width - 1, len(request.prompt) + len("Press a key"))))
            key = self._key()
            if self._is_key(key, "RESIZE"):
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x04":
                return None
            if self._is_key(key, "F1"):
                self._page("Key input help", "Press one key to continue.")
                continue
            if key == "\x0c":
                self._output_page()
                continue
            if self._is_key(key, "PAGEUP") or self._is_key(key, "PAGEDOWN") or key == "\x0b":
                continue
            if self._is_key(key, "ENTER"):
                return "\n"
            if key == "\x1b":
                return "\x1b"
            if self._is_backspace(key):
                return "\x7f"
            if isinstance(key, str):
                return key
            try:
                name = curses.keyname(key)
                return name.decode("ascii", errors="replace") if isinstance(name, bytes) else str(name)
            except (curses.error, TypeError):
                return str(key)

    def _draw_editor(
        self,
        label: str,
        value: str,
        cursor: int,
        y: int,
        rows: int,
    ) -> tuple[int, int]:
        height, width = self._size()
        rows = max(1, min(rows, height - y))
        x = min(width - 1, len(label))
        self._put(y, 0, label, "input")
        parts = value.split("\n")
        cursor = max(0, min(cursor, len(value)))
        before = value[:cursor]
        cursor_line = before.count("\n")
        cursor_col = len(before.rsplit("\n", 1)[-1])
        start_line = max(0, cursor_line - rows + 1)
        available = max(1, width - x - 1)
        cursor_at = (y, x)
        for line_index, line in enumerate(parts[start_line : start_line + rows]):
            row_y = y + line_index
            actual_index = start_line + line_index
            offset = 0
            if actual_index == cursor_line and cursor_col >= available:
                offset = cursor_col - available + 1
            visible = line[offset : offset + available]
            self._put(row_y, x, visible, "input")
            if actual_index == cursor_line:
                cursor_at = (row_y, x + min(available - 1, cursor_col - offset))
        return cursor_at


def _edit_insert(value: str, cursor: int, insertion: str) -> tuple[str, int]:
    return value[:cursor] + insertion + value[cursor:], cursor + len(insertion)


def _edit_key(
    value: str,
    cursor: int,
    key: object,
    *,
    completion_owned: bool = False,
    allow_newline: bool = False,
) -> tuple[str, int, bool]:
    owned = completion_owned
    if owned and isinstance(key, str) and len(key) == 1 and key.isprintable() and key not in {"\t"}:
        value, cursor, owned = "", 0, False
    if CursesTerminalSession._is_key(key, "LEFT"):
        return value, max(0, cursor - 1), owned
    if CursesTerminalSession._is_key(key, "RIGHT"):
        return value, min(len(value), cursor + 1), owned
    if CursesTerminalSession._is_key(key, "UP") or CursesTerminalSession._is_key(key, "DOWN"):
        line_start = value.rfind("\n", 0, cursor) + 1
        column = cursor - line_start
        if CursesTerminalSession._is_key(key, "UP"):
            previous_end = line_start - 1
            if previous_end < 0:
                return value, cursor, owned
            previous_start = value.rfind("\n", 0, previous_end) + 1
            return value, min(previous_end, previous_start + column), owned
        current_end = value.find("\n", cursor)
        if current_end < 0:
            return value, cursor, owned
        next_start = current_end + 1
        next_end = value.find("\n", next_start)
        if next_end < 0:
            next_end = len(value)
        return value, min(next_end, next_start + column), owned
    if CursesTerminalSession._is_key(key, "HOME") or key == "\x01":
        return value, 0, owned
    if CursesTerminalSession._is_key(key, "END") or key == "\x05":
        return value, len(value), owned
    if CursesTerminalSession._is_backspace(key):
        if cursor:
            return value[: cursor - 1] + value[cursor:], cursor - 1, owned
        return value, cursor, owned
    if CursesTerminalSession._is_delete(key):
        if cursor < len(value):
            return value[:cursor] + value[cursor + 1 :], cursor, owned
        return value, cursor, owned
    if key == "\x15":
        return "", 0, False
    if key == "\x0b":
        return value[:cursor], cursor, owned
    if key == "\x17":
        start = cursor
        while start > 0 and value[start - 1].isspace():
            start -= 1
        while start > 0 and not value[start - 1].isspace():
            start -= 1
        return value[:start] + value[cursor:], start, owned
    if key == "\x0a" and allow_newline:
        value, cursor = _edit_insert(value, cursor, "\n")
        return value, cursor, owned
    if key == "\t":
        value, cursor = _edit_insert(value, cursor, "\t")
        return value, cursor, owned
    if isinstance(key, str) and len(key) == 1 and key.isprintable():
        value, cursor = _edit_insert(value, cursor, key)
    return value, cursor, owned


def _edge_insert_command(command: str) -> str:
    if command.startswith("s "):
        return "s "
    if command.startswith("reroll"):
        return "reroll "
    if command.startswith("f "):
        return "f "
    if command.startswith("#"):
        return "#"
    if command.startswith("new "):
        return "new "
    if command.startswith("name "):
        return "name "
    if command.startswith("rewind "):
        return "rewind "
    if command.startswith("export "):
        return "export "
    if command.startswith("save"):
        return "save "
    if command.startswith("spr"):
        return "spr "
    if command.startswith("switch "):
        return "switch "
    if " / " in command:
        return command.split(" / ", 1)[0]
    return command.split()[0]
