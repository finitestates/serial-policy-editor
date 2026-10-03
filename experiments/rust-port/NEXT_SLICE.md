# Stage 2 status and next Rust slice

**Stage 2 status:** complete for the numeric policy surface. Rust now covers
history penalties, output-head activation adjustments, direct/grouped biases,
ephemeral biases, filtering, sparse probabilities, CFG logit blending, and
lazy policy metrics. Evidence is in
[`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md).

`SamplerConfig` validation/serialization, action/replay serialization, model
inference, and CFG backend lifecycle remain Python-owned boundaries. The
current parity adapter passes already-resolved configuration values into the
Rust numeric kernels. No Stage 2 code remains pending in the sampler crate.

## Next stage

Stage 3 is episode execution and replay, as scoped in
[`ROADMAP.md`](ROADMAP.md). Start by mapping `EpisodeEngine` observation,
action, outcome, and evidence transitions against the existing episode-history
crate. Select a bounded in-memory transition slice before editing; leave
SQLite, terminal I/O, and model inference in their current adapters until the
stage 3 contracts require them.

Keep Python as the oracle, preserve workspace changes, use shared fixtures for
cross-language behavior, and leave work uncommitted unless Graham asks.
