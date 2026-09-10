# Serial Policy Editor 0.3.6

Released September 10, 2026.

0.3.6 improves terminal responsiveness and writing input, keeps one terminal
application alive throughout an interactive session, and increases the default
hold to 100 tokens.

## Changes

- Four performance improvements: reuse sampler statistics when history penalties
  are inactive, decode observation context only when requested, cache wrapped
  terminal context between redraws, and skip evidence loading for plain projection.
- Persistent full-screen rendering across choices, menus, review, confirmations,
  and episode navigation, with terminal restoration on exit.
- Scrollable context, multiline raw input, and explicit Ctrl+E input expansion.
- Remove the transient busy indicator to prevent layout shifts (`567aaca`).
- Keep insertion previews and mode labels visible while typing `t`/`x`, preventing
  context rows from briefly collapsing between updates. Remove redundant preview
  text and commit hints.
- Increase the default hold from 24 to 100 tokens (`841b569`). Use
  `--hold-default 24` to retain the previous default.

The release includes the reviewed work from `dev/0.3.6`. Experimental history
replacement and fork-edit variations from `codex/history-edit` are excluded.
The older fullscreen experiment is superseded by the incorporated rendering
refactor; no additional merge from `fix/persistent-fullscreen-tui` is needed.

## Verification

- Default regression suite: 658 passed, 12 opt-in model tests skipped.
- Local llama.cpp smoke suite using Llama 3.2 1B Instruct IQ4_XS: 6 passed,
  including persistent terminal navigation.
- Local Transformers smoke suite using GPT-2: 6 passed.
- Wheel and source distribution built successfully; wheel metadata and runtime
  version both verified as 0.3.6. CLI reports `policy-editor 0.3.6`.

Terminal regression checks include persistent application reuse, input handling,
resize behavior, and terminal restoration. Results describe this local test
environment, not exhaustive coverage of all terminal emulators and platforms.
