# Next agent handoff

Stage 2's numeric policy surface is complete. Read
[`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md) for its scope,
parity results, real-model evidence, and exact validation.

For the next requested work, begin Stage 3 by mapping the in-memory
`EpisodeEngine` observation/action/outcome/evidence transitions against the
existing `episode-history` crate. The initial boundary is described in
[`NEXT_SLICE.md`](NEXT_SLICE.md); read
[`ROADMAP.md`](ROADMAP.md), `tests/CORE_CONTRACTS.md`, and the relevant engine
contract tests before proposing a code slice. Keep model inference and SQLite
behind their current boundaries, retain Python as the oracle, preserve existing
workspace changes, and leave work uncommitted unless Graham asks.
