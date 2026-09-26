# TODO for 1.0.0

Working backlog from a read-only review and our follow-up discussion. The project already has a substantial foundation; 1.0 can focus on clear promises, dependable release steps, and a few targeted polish items. Contribution and dependency maintenance are optional choices.

## Product polish

- [ ] Decide whether speculative execution for token selection earns its complexity. Measure its latency benefit on supported backends and weigh it against the warmup, cancellation, and promotion logic. If the benefit is small or inconsistent, remove that path and keep selection on the ordinary commit flow.
- [ ] Bring non-runtime menus into line with the main runtime menu’s visual conventions. Review screens such as EDGE, search/review, and setup; make selection, context, status, and available commands easy to read in the same way.
- [ ] Add an end-to-end regression for Tab/Enter through `run_plan()`, checking the recorded action and next view. If speculative execution stays, check that the warmed target matches the committed selection; if it goes, check the ordinary selection flow. The current [selection warm test](tests/core/test_selection_warm_terminal.py#L65) stops at applying the action directly to the engine. The [interaction notes](RUNTIME_MENU_ENTER_CYCLE.md#L82) describe the mixed Tab cycle and target invariant.

## Replay authoring and independent correctness

- [ ] Build a minimal `reference-kernel-oracle` that invokes llama.cpp directly and stays independent of `episode_engine` and its sampling path. Limit its actions to accepting the proposal and `hold X`. Compare its token sequences with the editor under matching inputs, and include a small reproducible demonstration of how deterministic sampler coordinates can produce different behavior.
- [ ] Define a plain-text teacher-plan format modeled on `--procedure`, then add a parser with useful line-numbered errors. Check whether the current procedure view includes everything replay needs; decide what grammar and expected-result details the format must add. Keep JSONL plans usable.

## Tokenizer identity and streaming

- [ ] Decide how model-change forks define sampler identity and coordinates. The current fork path re-tokenizes the root prompt with the destination tokenizer, and those root token IDs affect the sampler fingerprint. Decide whether to preserve the source fingerprint and define how token counts map after boundary zero, or require the same tokenizer for forks and use replay to change tokenizers.
- [ ] Audit and test how streamed partial UTF-8 sequences affect token-step accounting and saved text. Token boundaries advance by token IDs, while a live stream may buffer bytes until a character is complete. Cover display, rewind, partial-write replay, procedure export, and cross-tokenizer forks; keep token IDs authoritative where decoded text cannot faithfully represent a retained prefix. The review identified this as a code-path risk to investigate, not a demonstrated failure.

## Decide what 1.0 promises

- [ ] Document supported Python versions, operating systems, model backends, and artifact formats. Say which Python API and CLI behavior users can rely on.
- [ ] Set a policy for saved workspaces and exported files: supported upgrade paths, behavior when a migration is unsupported, and how users should back up their work. The database is at schema version 2 and rejects unsupported versions; importing the old 0.2.x report format is deliberately out of scope. See [episode_store.py](core/src/trajectory_editor/episode_store.py#L25) and [CUT_NOTES.md](CUT_NOTES.md#L59).

## Documentation and release readiness

- [ ] Remove or correct references to the removed `archive/` tree in the [README](README.md#L7), test inventory, an old [changelog entry](CHANGELOG.md#L196), and a [sampling module comment](core/src/trajectory_editor/core/sampling.py#L1).
- [ ] Reconcile the release history: the packages and `main` identify as `0.7.5`, while the changelog begins at `0.7.0`. Add missing release notes or explain which versions were not separate releases. The [README](README.md#L193) also says 55 core test slots, while the contract document adds four terminal slots to the original 55.
- [ ] Choose the Python and operating-system support matrix, then make CI match it. Current test jobs cover Ubuntu with Python 3.10 and the latest Python; real-model checks are opt-in. See [ci.yml](.github/workflows/ci.yml#L17).
- [ ] Write down the release steps: update the changelog and synchronized versions, build both packages, install the built artifacts in a clean environment, and smoke-test their commands. CI already builds wheels and source archives, checks them with Twine, and runs command help checks ([ci.yml](.github/workflows/ci.yml#L53)); there is no tag-triggered publishing workflow.

## Optional if inviting outside contributors

- [ ] Add contribution and security-reporting instructions, and decide whether the CODEOWNERS snippet should become an active file.
- [ ] Consider Python dependency updates in Dependabot. Its current config covers GitHub Actions only ([dependabot.yml](.github/dependabot.yml#L1)).
