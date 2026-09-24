# Development branches

`main` is the current release line for `0.7.5`. It includes teacher replay
plans, storage-independent runtime seams, ephemeral runs, shared live/durable
execution, and root-relative episode/replay semantics.

Develop experimental work in a separate checkout. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

Version metadata in `core/pyproject.toml`, `vector/pyproject.toml`, and
`core/src/trajectory_editor/version.py` must agree. The vector package's core
dependency lower bound should track the core release. Use a development suffix
for subsequent snapshots and reserve final versions and tags for releases.
Model weights, episode databases, exported writing, and old distribution archives
are local assets, not source release contents.
