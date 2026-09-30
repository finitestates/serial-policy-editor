# Development branches

The active release represented by this tree is `v0.8.5`. Interactive and
replayed execution run through the in-memory runtime; SQLite workspaces are
used for explicit restoration and saving. Sampler changes and rerolls are
ordered actions, and episodes no longer have a global token budget.

Develop experimental work in a separate checkout. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

The live TTY interface uses Textual in a dedicated UI thread. Its request
screens are exercised headlessly with Textual Pilot; the synchronous engine
continues to own backend and storage work. See `TERMINAL_API.md` for the thread
bridge, screen map, theme behavior, and prepared-screen transition benchmark.
`tests/core/test_textual_driver_pty.py` also drives `run_session_roster()`
through a real POSIX PTY, checks runtime state across Choice, EDGE, prompt, and
Beam transitions, and exercises page/single-key/chord/multiline request input.
Resize checkpoints replay captured terminal output through pyte.
pyte encoding and terminal-display evidence must be treated as prima facie
evidence of what happens on the terminal screen. This instruction applies to
all agents doing terminal work; see `AGENTS.md` for the repository rule and
concrete evidence pointers.
Ordered markers distinguish individual writer calls from Textual post-display passes;
the Beam journey checks 30 successive production Right-key submissions, a
gated waiting view, retained screen/widget identity, complete captured panes,
and a separate synchronized resize cycle. An isolated archived-baseline run
fails the remount assertion in both terminal sizes. The checked-in comparison
indexes the ordered passes for before/after review.
The PTY capture stores byte offsets and ordered markers, not monotonic time per
event or monitor scanout. Replaying it through pyte establishes the terminal
cell grid and styles at each marked boundary. A timestamped desktop recording
adds compositor-visible frames and bounds transitions by its encoded frame
timestamps; report sampling gaps with those bounds.

## Verify user-visible behavior before hand-back

Use this workflow for terminal/UI changes and for any bug report whose symptom
depends on what a person sees. A final widget state or passing unit suite is not
evidence that intermediate output was correct.

1. **Turn the report into an observable contract.** Record the exact entry
   point, input sequence, expected screen/output, dimensions, and the visible
   state that would count as failure. Trace that route through the production
   caller before changing code. Use a deterministic backend or fixture where
   possible so unrelated model/runtime variance cannot hide the defect.
2. **Capture the failing behavior before fixing it.** Drive the actual
   application path. For Textual, use the production driver in a POSIX PTY;
   send keys/mouse input through the PTY and synchronize on explicit readiness
   or barriers, not arbitrary sleeps. Preserve raw PTY bytes, terminal size and
   resize events, ordered write/display markers, input/request generations,
   widget identity, pane geometry, cursor/focus, and the decoded terminal grid
   with cell styles. Save the capture before assertions so failures retain
   their evidence.
3. **Assert every visible boundary, not just the settled screen.** Hook actual
   compositor display passes and enqueue each marker through the same ordered
   writer path as terminal bytes. At every pass assert the user-visible
   contract: required content is present, pane bounds fit, wrapped text is
   complete, selection/focus/styles remain correct, and preserved regions are
   not erased or replaced by transient blank/partial layouts. Keep expected
   content changes separate from unrelated redraws. Test a slow/gated result,
   a fast result, repeated input, resize, and the relevant modal/return paths.
4. **Prove the test can detect the defect.** Add small negative controls that
   intentionally produce the class of failure under test—for example, paint
   one incomplete table pass, clear a preserved pane, remount a same-kind
   screen, or drop focus styling. The corresponding assertion must fail on
   that control and pass on the corrected production journey. Do not accept
   positive counters, widget identity alone, or a final screenshot as a visual
   oracle.
5. **Reconstruct and inspect the captured output.** Replay the raw ANSI/UTF-8
   PTY stream at each recorded marker through a terminal emulator (the current
   helper uses pyte), verify the replayed grid against the independently
   captured grid, and export selected frames or a sequence as SVG/PNG. Inspect
   the failure and corrected sequence at original cell dimensions. Keep the
   capture JSON and raw byte stream beside the images so another developer can
   rerender the evidence instead of relying on a description.
6. **Measure timing at the layer that owns the symptom.** PTY byte offsets
   establish ordering only. When duration matters, timestamp input, writes,
   and display markers with a monotonic clock at capture time. For
   compositor-visible flashes, also retain a desktop recording and use its
   frame presentation timestamps; report first bad, last bad, nearest good
   frames, and sampling gaps. Do not equate a Textual callback, PTY write, or
   terminal-emulator replay frame with a physical monitor refresh.
7. **Hand back a reproducible result.** Include the exact command and
   environment, the scenario and dimensions, links to raw and rendered
   artifacts, the frame/event range where failure appears, the assertions that
   rejected it, and the same evidence from the fixed build. State precisely
   which layer was observed (widget, PTY stream, terminal replay, desktop
   recording, or physical display). When required evidence is absent, inspect
   the capture format and source, identify the missing measurement, and add
   that instrumentation to the next run; do not stop at “unknown” or ask the
   user to rediscover the issue.

For the observability model behind this workflow—what each test layer can and cannot establish, why ordered invalid frames do not require timestamps, and how synchronized updates affect presentation eligibility—see [`tests/VISIBLE_FRAME_CORRECTNESS.md`](tests/VISIBLE_FRAME_CORRECTNESS.md).

For current Textual journeys, `core/scripts/record_textual_journey.py` records
the production PTY test and exports marked grids; `core/scripts/render_textual_capture.py`
replays selected display markers as styled images. The reusable Beam example,
including old-build negative results and ordered-frame assertions, is in
`tests/core/test_textual_driver_pty.py` and
`tests/core/test_textual_render_stability.py`.

Version metadata in `core/pyproject.toml`, `vector/pyproject.toml`, and
`core/src/trajectory_editor/version.py` must agree. The vector package's core
dependency lower bound should track the core release. Use a development suffix
for subsequent snapshots and reserve final versions and tags for releases.
Model weights, episode databases, exported writing, and old distribution archives
are local assets, not source release contents.

## Release checklist

1. Update `CHANGELOG.md` with the release date and user-visible changes. Check
   `README.md`, package READMEs, and the compatibility statement for stale
   behavior or version-specific workspace requirements.
2. Set the same release version in `core/pyproject.toml`,
   `vector/pyproject.toml`, and `core/src/trajectory_editor/version.py`.
   Update `vector/pyproject.toml`'s `policy-editor-core` lower bound to match.
3. Run the core, vector, and reference-kernel suites, then build both source
   distributions and wheels and check the artifacts with Twine:

   ```bash
   python -m pip install --upgrade pip build twine
   python -m pip install -e "core[test]" -e ./vector
   python -m pytest -q tests/core
   python -m pytest -q tests/vectors
   python -m pytest -q reference-kernel
   python -m build --sdist --wheel --outdir dist/core ./core
   python -m build --sdist --wheel --outdir dist/vector ./vector
   python -m twine check --strict dist/core/* dist/vector/*
   ```

4. Install the built wheels into a fresh virtual environment and smoke-test
   both installed commands:

   ```bash
   python3 -m venv /tmp/policy-editor-release-venv
   . /tmp/policy-editor-release-venv/bin/activate
   python -m pip install dist/core/*.whl dist/vector/*.whl
   policy-editor --help
   policy-editor-vector --help
   deactivate
   ```

5. Review the final diff and package contents, then create the version tag.
   The repository currently has no tag-triggered publishing workflow, so
   publishing built packages is a separate release action.
