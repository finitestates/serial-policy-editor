# Development branches

The active release represented by this tree is `v0.8.5`. Interactive and
replayed execution run through the in-memory runtime; SQLite workspaces are
used for explicit restoration and saving. Sampler changes and rerolls are
ordered actions, and episodes no longer have a global token budget.

Develop experimental work in a separate checkout. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

The live TTY interface uses a synchronous curses renderer on the
episode-owning thread. That thread also owns backend, action, and storage work.
See `TERMINAL_API.md` for the request protocol, controls, output capture, and
theme behavior. Live curses rendering belongs in
`core/src/trajectory_editor/curses_tui.py`; the plain fallback belongs in
`plain_tui.py`.

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
