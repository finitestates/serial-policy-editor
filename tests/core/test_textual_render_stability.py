"""Pilot checks for first-frame Textual rendering and stable table fitting."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.text import Text
from textual.widgets import DataTable, Static

from trajectory_editor.terminal_contracts import ChoiceFeedback
from trajectory_editor.textual_tui import BeamScreen, ChoiceScreen, PolicyEditorApp

from tests.core.textual_support import beam_state, choice_state, install_request, run_pilot


def _plain(value: object) -> str:
    return value.plain if isinstance(value, Text) else str(value)


@pytest.mark.current_workflow
def test_choice_context_preview_and_feedback_are_present_in_the_first_frame():
    async def scenario():
        app = PolicyEditorApp()
        first_frame: dict[str, str] = {}

        def capture_first_frame() -> None:
            screen = app._active_screen
            if (
                first_frame
                or not isinstance(screen, ChoiceScreen)
                or not screen.is_mounted
                or app.screen is not screen
            ):
                return
            first_frame.update(
                context=_plain(screen.query_one("#context", Static).content),
                preview=_plain(screen.query_one("#choice-preview", Static).content),
                feedback=_plain(screen.query_one("#choice-feedback", Static).content),
            )

        app.post_display_hook = capture_first_frame
        state = replace(
            choice_state(),
            initial_command="1",
            feedback=ChoiceFeedback("info", "Saved feedback", ("context is ready",)),
        )

        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)

        assert first_frame["context"].startswith("DECISION BOUNDARY\ncontext")
        assert "sampled proposal" in first_frame["preview"]
        assert "Saved feedback" in first_frame["feedback"]
        assert "context is ready" in first_frame["feedback"]

    run_pilot(scenario)


@pytest.mark.current_workflow
@pytest.mark.parametrize("size", [(104, 24), (120, 40), (160, 50)])
def test_beam_continuation_column_is_fitted_before_its_first_frame(size):
    async def scenario():
        base = beam_state(row_count=3)
        continuation = (
            "The wind howls outside, and the shutters rattle against the old stone "
            "walls while rain streams down. "
        ) * 4
        state = replace(
            base,
            rows=(replace(base.rows[0], continuation=continuation), *base.rows[1:]),
        )
        app = PolicyEditorApp()
        first_frame: list[tuple[int, int, bool]] = []

        def capture_first_frame() -> None:
            screen = app._active_screen
            if (
                first_frame
                or not isinstance(screen, BeamScreen)
                or not screen.is_mounted
                or app.screen is not screen
            ):
                return
            table = screen.query_one("#beam-table", DataTable)
            first_frame.append(
                (
                    table.virtual_size.width,
                    table.scrollable_content_region.width,
                    table.styles.get_rule("visibility") != "hidden",
                )
            )

        app.post_display_hook = capture_first_frame
        async with app.run_test(size=size) as pilot:
            await install_request(app, pilot, state)
            table = app._active_screen.query_one("#beam-table", DataTable)
            assert first_frame
            first_width, first_viewport, first_visible = first_frame[0]
            assert not first_visible or first_width <= first_viewport
            assert table.styles.get_rule("visibility") != "hidden"
            assert table.virtual_size.width <= table.scrollable_content_region.width

            await pilot.resize_terminal(size[0] + 20, size[1] + 10)
            assert table.virtual_size.width <= table.scrollable_content_region.width

    run_pilot(scenario)


@pytest.mark.current_workflow
def test_identical_choice_feedback_does_not_replace_static_content():
    async def scenario():
        app = PolicyEditorApp()
        state = replace(
            choice_state(),
            feedback=ChoiceFeedback("info", "Saved feedback", ("context is ready",)),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            target = screen.query_one("#choice-feedback", Static)
            original_content = target.content

            screen._render_feedback()

            assert target.content is original_content
            assert target.display
            assert "Saved feedback" in _plain(target.content)

            screen.state = replace(screen.state, feedback=None)
            screen._render_feedback()
            hidden_content = target.content
            assert not target.display
            assert not _plain(hidden_content)

            screen._render_feedback()
            assert target.content is hidden_content
            assert not target.display

    run_pilot(scenario)
