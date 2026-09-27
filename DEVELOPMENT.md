# Development branches

The active release represented by this tree is `v0.8.5`. Interactive and
replayed execution run through the in-memory runtime; SQLite workspaces are
used for explicit restoration and saving. Sampler changes and rerolls are
ordered actions, and episodes no longer have a global token budget.

Develop experimental work in a separate checkout. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

Version metadata in `core/pyproject.toml`, `vector/pyproject.toml`, and
`core/src/trajectory_editor/version.py` must agree. The vector package's core
dependency lower bound should track the core release. Use a development suffix
for subsequent snapshots and reserve final versions and tags for releases.
Model weights, episode databases, exported writing, and old distribution archives
are local assets, not source release contents.
