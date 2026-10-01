# Development branches

The active release represented by this tree is `v0.8.5`. Interactive and
replayed execution run through the in-memory runtime; SQLite workspaces are
used for explicit restoration and saving. Sampler changes and rerolls are
ordered actions, and episodes no longer have a global token budget.

Develop experimental work in a separate checkout. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

The live TTY interface (`core/src/trajectory_editor/term/`) is immediate-mode:
one UI thread renders each frame from one state snapshot and writes only the
changed lines in a single synchronized, erase-free transaction. The synchronous
engine thread keeps backend and storage ownership. See `TERMINAL_API.md` for the
design and thread bridge.

## Verify user-visible behavior before hand-back

Follow [Terminal rendering validation and handoff](tests/TERMINAL_RENDERING_GUIDE.md)
for terminal/UI changes. The PTY journeys in
`tests/core/test_live_terminal_pty.py` replay the raw terminal bytes through
pyte and compare them with every intended frame; pyte replay is evidence of what
the terminal shows (see `AGENTS.md`). Look at rendered screens with
`core/scripts/render_live_screens.py`.

Version metadata in `core/pyproject.toml`, `vector/pyproject.toml`, and
`core/src/trajectory_editor/version.py` must agree. The vector package's core
dependency lower bound should track the core release. Use a development suffix
for subsequent snapshots and reserve final versions and tags for releases.
Model weights, episode databases, exported writing, and old distribution archives
are local assets, not source release contents.

## Hypothesis tests

Install the core test extra from the repository root with
`python -m pip install -e "core[test]"`. Hypothesis tests run under pytest with
the rest of `tests/core`; no separate runner is needed.

The initial state machine generates writes, forks, new roots, and session
switches, checking retained histories, lineage, public addresses, and restoration
of the active session's backend against an independent token-history model:

```bash
python -m pytest -q tests/core/test_live_roster_stateful.py --hypothesis-show-statistics
HYPOTHESIS_PROFILE=ci python -m pytest -q tests/core/test_live_roster_stateful.py
```

`tests/conftest.py` selects the `dev` profile by default: 50 examples, up to 30
state-machine steps per example, and no wall-clock deadline. Select repeatable
generation with `HYPOTHESIS_PROFILE=ci`; this profile uses the same budget and
`derandomize=True`. The CI test job selects this profile. The pytest option
`--hypothesis-profile=ci` also selects it. Use `--hypothesis-show-statistics` to
inspect generated example counts, and `--hypothesis-seed=12345` to replay a
chosen random seed.

For reactive behavior, generate actions and completion order while controlling
pending work with futures or explicit gates. Each example must start with fresh
state, and an invariant should check observable behavior after each action.
Use an independent expected-state model so the test can reject plausible bugs,
such as an old completion replacing the latest request. Sleeping for a guessed
duration makes generation and shrinking unreliable.

Hypothesis shrinks failing values and action sequences and reuses saved examples
from the ignored `.hypothesis/` directory on later local runs. Keep the printed
minimal sequence and any supplied reproduction blob when reporting a failure; add a readable
regression test for the discovered behavior. The `ci` profile disables the
example database through `derandomize=True`. See the upstream
[stateful testing guide](https://hypothesis.readthedocs.io/en/latest/stateful.html)
for rules, preconditions, invariants, and exporting a state machine as a pytest
test case. Generated terminal journeys must also follow the terminal rendering
validation guide linked above.

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
