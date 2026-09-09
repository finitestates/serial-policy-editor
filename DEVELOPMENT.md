# Development branches

`main` contains the reviewed improvements intended for the next release. Its
current version is `0.3.6.dev0`: a development snapshot, not a final 0.3.6 release.
The existing `v0.3.5` tag and release remain the released baseline.

`codex/history-edit` contains experimental history replacement, fork-edit and
sibling variations, and token-preserving replacement replay. Its version carries
`+historyedit` to distinguish it from main. The proposed 100-token hold default
is a separate commit on that branch; main retains 24.

`codex/preserve-2026-09-09` is an archival snapshot of the combined local code
and tests before these changes were separated. It is not a release branch.

Develop experimental work in a separate checkout. Bring reviewed improvements
from main into the experimental branch as needed. Promote individual features
with focused changes and tests; do not merge the entire experimental branch
into main merely to synchronize repositories.

Version metadata in `serial-policy-editor/pyproject.toml` and
`serial-policy-editor/trajectory_editor/version.py` should agree. Advance the
development suffix for subsequent snapshots; reserve `0.3.6` and its release tag
for the completed release. Model weights, episode databases, exported writing,
and old distribution archives are local assets, not source release contents.
