# Serial Policy Editor 0.3.7

Released September 12, 2026.

0.3.7 completes the model-aware biasing work. Human-readable terms can be
compiled into deterministic logical token routes, catalog groups can be used by
name, and live sessions can create append-only runtime groups with one shared
bias amount.

## Changes

- Add YAML bias catalog compilation with case, spacing, plural, suffix, route
  depth, route-budget, and deterministic round-robin controls.
- Use one logical bias-rule matcher for direct text, catalog terms, catalog
  groups, and runtime groups.
- Add telescoping `path` semantics for lexical terms that decompose into
  multiple tokens while preserving tail semantics for phrases.
- Add `b name -> {member, ...}` runtime group construction. Definitions,
  membership changes, and bias updates follow sampler-state rewind, fork, and
  replay boundaries.
- Change bias preset output to `spe-bias-rules-v2`, with full group metadata by
  default and `--rules-only` for flattened ordinary logical rules.
- Remove the unused experimental `logit_bias`, `sequence_bias`, and
  `scoped_bias` representations and legacy preset-loading paths.

## Verification

- Full test suite: 702 passed, 15 skipped with host access for loopback and PTY
  checks.
