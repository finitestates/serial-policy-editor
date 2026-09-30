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

Follow [Terminal rendering validation and handoff](tests/TERMINAL_RENDERING_GUIDE.md)
for terminal/UI changes. It is the canonical workflow, command reference,
artifact map and handoff checklist. Intermediate production-driver transitions
must be inspected and asserted; settled screenshots alone are insufficient.
See [Visible-frame correctness](tests/VISIBLE_FRAME_CORRECTNESS.md) for replay
interpretation. Root `AGENTS.md` governs evidence acceptance.

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
