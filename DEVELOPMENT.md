# Development branches

`main` contains the released 0.3.6 code. `dev/0.3.6` collected the reviewed
performance, multiline input, persistent terminal rendering, and 100-token
hold improvements, including `567aaca`.

`codex/history-edit` contains experimental history replacement, fork-edit and
sibling variations, and token-preserving replacement replay. Those experiments
are not part of 0.3.6. The 100-token hold default was promoted independently.

`fix/persistent-fullscreen-tui` is superseded by the persistent terminal rendering
refactor already incorporated into `dev/0.3.6`; it needs no separate release merge.
`codex/preserve-2026-09-09` is an archival snapshot, not a release branch.

Develop experimental work in a separate checkout. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

Version metadata in `serial-policy-editor/pyproject.toml` and
`serial-policy-editor/trajectory_editor/version.py` must agree. Use a development
suffix for subsequent snapshots and reserve final versions and tags for releases.
Model weights, episode databases, exported writing, and old distribution archives
are local assets, not source release contents.
