"""Measure prepared-screen submission-to-next-render latency headlessly."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--columns", type=int, default=120)
    parser.add_argument("--output", type=Path)
    return parser


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * .95))], 3),
    }


def _load_package(package_root: Path):
    source_root = package_root.resolve() / "core" / "src"
    expected_module = source_root / "trajectory_editor" / "textual_tui.py"
    if not expected_module.is_file():
        raise SystemExit(f"no Textual terminal implementation under {source_root}")
    sys.path.insert(0, str(source_root))

    import trajectory_editor.textual_tui as loaded_terminal
    from trajectory_editor.core.candidates import Candidate
    from trajectory_editor.core.ui import ChoiceSet
    from trajectory_editor.terminal_contracts import (
        BeamViewRow,
        BeamViewState,
        BoundaryReview,
        ChoiceFeedback,
        ChoiceViewState,
        EdgeViewState,
        PromptRequest,
    )
    from trajectory_editor.textual_tui import PolicyEditorApp, _RequestLifecycle

    if Path(loaded_terminal.__file__).resolve() != expected_module.resolve():
        raise RuntimeError("benchmark loaded a Textual terminal from the wrong checkout")
    return (
        Candidate,
        ChoiceSet,
        BeamViewRow,
        BeamViewState,
        BoundaryReview,
        ChoiceFeedback,
        ChoiceViewState,
        EdgeViewState,
        PromptRequest,
        PolicyEditorApp,
        _RequestLifecycle,
    )


async def _measure(args: argparse.Namespace, modules) -> dict[str, object]:
    (
        Candidate,
        ChoiceSet,
        BeamViewRow,
        BeamViewState,
        BoundaryReview,
        ChoiceFeedback,
        ChoiceViewState,
        EdgeViewState,
        PromptRequest,
        PolicyEditorApp,
        RequestLifecycle,
    ) = modules
    row_count = max(1, min(args.rows, 100))
    candidates = tuple(
        Candidate(rank, rank, f" candidate {rank}", .01, False, .01)
        for rank in range(1, row_count + 1)
    )
    context = "\n".join(
        f"History line {index}: previously generated text."
        for index in range(200)
    )
    choice = ChoiceSet(
        "bench", "bench", 200, 200, "0" * 64, context,
        1, " candidate 1", .01, .01, False, candidates,
        vocabulary_size=128000, proposal_raw_rank=1,
    )
    base = ChoiceViewState(choice, candidates, lambda text, mode: text)
    chord_body = (
        "Shared context (last 4 lines):\nPreviously generated text.\n\nPaths:\n"
        + "\n".join(
            f"{label}  rank {rank}  LIVE\n"
            + "   A possible continuation for this path.\n" * 8
            for rank, label in enumerate("abc", 1)
        )
    )
    variants = (
        ("menu", base, ("1", "enter")),
        ("expanded menu", replace(base, candidates=candidates), ("1", "enter")),
        (
            "rank neighborhood",
            replace(
                base,
                display_candidates=candidates[max(0, row_count // 2 - 3):max(1, row_count // 2 + 4)],
                search_lens_active=True,
                target_token_id=max(1, row_count // 2),
                feedback=ChoiceFeedback("search", "SEARCH · rank neighborhood"),
            ),
            ("1", "enter"),
        ),
        (
            "history review",
            replace(
                base,
                review=BoundaryReview(
                    200, 199, context, {"kind": "token-boundary"},
                ),
            ),
            ("enter",),
        ),
        ("edge", EdgeViewState("bench", 200, "temperature 1"), ("q", "enter")),
        ("prompt", PromptRequest("Name › "), ("n", "a", "m", "e", "enter")),
        ("page", PromptRequest("", body=context, page=True), ("enter",)),
        (
            "beam",
            BeamViewState(
                "BEAM · width 3 · depth 8", "Shared generated context",
                tuple(
                    BeamViewRow(
                        f"b{rank}", f"continuation {rank}", "-0.45", "LIVE",
                        (f"step {rank}",), model_rank=rank,
                    )
                    for rank in range(1, 4)
                ),
                "b1",
            ),
            ("enter",),
        ),
        (
            "chord",
            PromptRequest("Chord › ", body=chord_body, isolated=True),
            ("a", "enter"),
        ),
        (
            "chord advance",
            PromptRequest("Chord › ", body=chord_body + "   Another token.", isolated=True),
            ("a", "enter"),
        ),
        ("menu after chord", base, ("1", "enter")),
    )

    app = PolicyEditorApp()
    ready_times: list[float] = []
    original_screen_ready = app._screen_ready

    def note_ready(screen):
        ready_times.append(time.perf_counter())
        original_screen_ready(screen)

    app._screen_ready = note_ready
    measurements: dict[str, list[float]] = {}
    previous_submit: float | None = None
    previous_label: str | None = None
    total_requests = args.iterations + 8

    async with app.run_test(size=(args.columns, args.rows)) as pilot:
        for iteration in range(total_requests):
            label, state, keys = variants[iteration % len(variants)]
            ready_count = len(ready_times)
            lifecycle = RequestLifecycle(
                generation=iteration + 1,
                state=state,
                response=Future(),
                owner_queue=None,
            )
            await app.show_request(lifecycle)
            await pilot.pause()
            while len(ready_times) == ready_count:
                await pilot.pause(.01)
            if previous_submit is not None and iteration >= 8:
                assert previous_label is not None
                measurements.setdefault(previous_label, []).append(
                    (ready_times[-1] - previous_submit) * 1000
                )
            await pilot.press(*keys)
            await pilot.pause()
            if not lifecycle.response.done():
                raise RuntimeError(f"Pilot did not submit prepared {label} screen")
            lifecycle.response.result()
            previous_submit = time.perf_counter()
            previous_label = label

    values = [value for group in measurements.values() for value in group]
    return {
        "implementation": "textual-headless",
        "package_root": str(args.package_root.resolve()),
        "terminal": {"rows": args.rows, "columns": args.columns},
        "samples": len(values),
        "transitions": {label: _summary(samples) for label, samples in measurements.items()},
        "all_transitions": _summary(values),
        "renders": len(ready_times),
        "scope": "Prepared Textual screens; excludes inference, database work, and terminal painting.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.rows < 8 or args.columns < 20:
        parser.error("iterations must be positive and the headless console must be at least 20x8")
    modules = _load_package(args.package_root)
    report = asyncio.run(_measure(args, modules))
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
