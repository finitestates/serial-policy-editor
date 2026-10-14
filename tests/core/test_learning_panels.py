"""Stable learning layout, read-only inspection, and literal negative controls."""

from dataclasses import replace

import numpy as np
import pytest

from trajectory_editor.learning_diagnostics import (
    LearningSnapshot, LearningStep, compare, distribution_drift, probabilities,
)
from trajectory_editor.term.learning import LearningOverlay, sparkline
from tests.core.term_support import Harness, choice_state


def snapshot(step=0, *, rival=1, status="training"):
    logits = np.array([1.5 + step / 2, 2.1, 0.5])
    if rival == 2:
        logits[2] = 2.6
    point = compare(logits, 0, lambda index: ["house", "room", "door"][index])
    return LearningSnapshot(
        step + 1, 1, step, 0.001, 5, 20,
        LearningStep(point, step, 20, 5.0, True, 0.9, 0.8, step + 1,
                     status, 0.5 if step else None, 0.0 if step else None,
                     0.5 if step else None, rival == 2),
        tuple(-0.6 + i / 2 for i in range(step + 1)),
    )


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
def test_dock_reserves_space_before_training_and_never_moves_context(size):
    with Harness(size=size) as h:
        h.app.update_learning(replace(snapshot(), current=None, history=()))
        h.show(choice_state())
        context_rows = [(i, line) for i, line in enumerate(h.lines)
                        if "The story so far" in line or line.startswith("Step 0") or "─" * size[0] == line]
        assert context_rows
        command_row = h.row_of("Command >")
        baseline = h.lines[:command_row]
        for step in range(1, 5):
            h.app.update_learning(snapshot(step))
            h.app.write_output(f"[online SGD] step {step}/20\n")
            h.frame()
            assert h.lines[:command_row] == baseline
            assert h.row_of("Command >") == command_row
            assert "Online SGD" in h.lines[size[1] - 5]
            assert "Target" in h.lines[size[1] - 4]
            assert "Rival" in h.lines[size[1] - 3]
            assert "Margin" in h.lines[size[1] - 2]
        h.app.update_learning(snapshot(4, status="margin"))
        h.frame()
        assert h.lines[:command_row] == baseline
        assert "margin" in h.lines[size[1] - 5]


def test_rival_switch_and_exact_step_values_are_visible():
    with Harness(choice_state(), size=(100, 30)) as h:
        h.app.update_learning(snapshot(2, rival=2))
        h.frame()
        assert "'house'" in h.lines[26] and "2.500" in h.lines[26]
        assert "'door'" in h.lines[27] and "2.600" in h.lines[27]
        assert "rival changed" in h.lines[27]
        assert "-0.100" in h.lines[28]


def test_f3_refresh_uses_owner_queue_and_preserves_command_draft():
    calls = []
    def refresh():
        calls.append("owner")
        return snapshot(4)
    with Harness(replace(choice_state(), learning_overview=refresh), size=(100, 30)) as h:
        h.app.update_learning(snapshot(1))
        h.type("2 ? 5.0")
        h.press("f3")
        assert isinstance(h.app.overlay, LearningOverlay)
        assert calls == []  # Rendering / key handling must not run model code.
        assert "Refreshing comparisons" in h.text
        assert not h.done
        h.run_previews()
        h.frame()
        assert calls == ["owner"]
        assert "4 SGD steps" in h.text
        h.press("r")
        assert calls == ["owner"]
        h.run_previews()
        h.press("f3")
        assert h.app.overlay is None
        assert h.view.editor.text == "2 ? 5.0"
        assert not h.done


def test_f3_can_open_and_close_while_sgd_is_busy_and_stays_open_across_requests():
    with Harness(choice_state(), size=(100, 30)) as h:
        h.app.update_learning(snapshot(1))
        h.view.accepting = False
        h.press("f3")
        assert "Learning overview" in h.text
        assert "busy" in h.text
        h.press("r", "enter")
        assert not h.done
        h.app.update_learning(snapshot(3))
        h.frame()
        assert "3 SGD steps" in h.text
        h.show(choice_state())
        assert isinstance(h.app.overlay, LearningOverlay)
        h.press("escape")
        assert h.app.overlay is None
        assert h.view.editor.text == ""


def test_refresh_failure_is_visible_without_destroying_the_choice():
    def fail():
        raise ValueError("probe failure sentinel")
    with Harness(replace(choice_state(), learning_overview=fail)) as h:
        h.app.update_learning(snapshot())
        h.press("f3")
        h.run_previews()
        h.frame()
        assert "Cannot refresh: probe failure sentinel" in h.text
        assert not h.done


@pytest.mark.parametrize("size", [(40, 12), (20, 6), (19, 5)])
def test_small_terminal_learning_panel_does_not_hide_command_or_overflow(size):
    with Harness(choice_state(), size=size) as h:
        h.app.update_learning(snapshot(2))
        h.frame()
        assert all(len(row) == size[0] for row in h.canvas.cells)
        if size[0] >= 20:
            assert "Command >" in h.text
        else:
            assert "Enlarge the termina" in h.text


def test_nonlearning_choice_has_no_reserved_dock():
    with Harness(choice_state(), size=(80, 24)) as h:
        before = h.text
        assert "Online SGD" not in before
        assert "Target" not in before
        h.press("f3")
        assert "Enable --online-sgd" in h.text
        h.press("escape")
        assert h.text == before


def test_sparkline_and_drift_have_independent_numeric_oracles():
    assert sparkline([0, 1, 2]) == "▁▄█"
    assert sparkline([2, 1, 0]) == "█▄▁"
    assert sparkline([1, 1, 1]) == "▄▄▄"
    assert sparkline(range(100), 8) == "▁▂▃▄▅▆▇█"
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert distribution_drift(p, p) == 0
    assert distribution_drift(p, q) == pytest.approx(1.0)
    # Common logit offsets change no probabilities / margins.
    np.testing.assert_allclose(probabilities([1000, 999]), probabilities([1, 0]))
    sample = compare(np.array([1.5, 2.1, 0.5]), 0, str)
    assert sample.rival_id == 1 and sample.margin == pytest.approx(-0.6)
    assert sample.rank == 2 and sample.argmax_id == 1
