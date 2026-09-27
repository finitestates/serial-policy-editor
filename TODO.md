# TODO for 1.0.0

## Product polish

- [ ] Decide whether speculative execution for token selection earns its complexity. Measure its latency benefit on supported backends and weigh it against the warmup, cancellation, and promotion logic. If the benefit is small or inconsistent, remove that path and keep selection on the ordinary commit flow.
- [ ] Bring non-runtime menus into line with the main runtime menu’s visual conventions. Review screens such as EDGE, search/review, and setup; make selection, context, status, and available commands easy to read in the same way.
- [x] Add an end-to-end regression for Tab/Enter through `run_plan()`, checking the recorded action and next view. The [selection test](tests/core/test_selection_warm_terminal.py) checks the ordinary selection flow and confirms that candidate navigation does not trigger speculative warming.
- [ ] Remove stale functions/data fields/tests/aliases; cut-down on processes that make reads/writes/copies for no obvious purpose.
- [ ] Fix the bias system. Right now, it is a lot of residual complexity lacking a clear identity about what it is trying to accomplish.

## Replay authoring and independent correctness

- [ ] Build a minimal `reference-kernel-oracle` that invokes llama.cpp directly and stays independent of `episode_engine` and its sampling path. Limit its actions to accepting the proposal and `hold X`. Compare its token sequences with the editor under matching inputs, and include a small reproducible demonstration of how deterministic sampler draw coordinates can produce different behavior.
- [ ] Define a plain-text teacher-plan format modeled on `--procedure`, then add a parser with useful line-numbered errors. Check whether the current procedure view includes everything replay needs; decide what grammar and expected-result details the format must add. Keep JSONL plans usable.

## Tokenizer identity and streaming

- [x] Change how mixed-model forks are handled. If a model has a different tokenizer than the episode it is forking from, it will have different sampler coordinates. As such, it's really not a fork. A different model with the same tokenizer could be considered a fork assuming the token IDs of the prompt/prefix match. In order to enforce this, we need to have better provenance data about model & tokenizer (currently, the system resolves "same model" by evaluating absolute path, which is stupid).
- [ ] Audit and test how streamed partial UTF-8 sequences affect token-step accounting and saved text. Token boundaries advance by token IDs, while a live stream may buffer bytes until a character is complete. Cover display, rewind, partial-write replay, procedure export, and cross-tokenizer forks; keep token IDs authoritative where decoded text cannot faithfully represent a retained prefix. The review identified this as a code-path risk to investigate, not a demonstrated failure.

## Decide what 1.0 promises

- [ ] Document supported Python versions, operating systems, model backends, and artifact formats. Say which Python API and CLI behavior users can rely on.
- [x] Set a policy for saved workspaces and exported files.

## Documentation and release readiness

- [x] Remove or correct references to the removed `archive/` tree in the [README](README.md), test inventory, old changelog link, and sampling module comment.
- [x] Reconcile the release history and prepare the packages, changelog, and README for `0.8.0`; document the `0.7.5` release line and link the test count to the contract matrix.
- [ ] Choose the Python and operating-system support matrix, then make CI match it. Current test jobs cover Ubuntu with Python 3.10 and the latest Python; real-model checks are opt-in. See [ci.yml](.github/workflows/ci.yml#L17).
- [ ] Write down the release steps: update the changelog and synchronized versions, build both packages, install the built artifacts in a clean environment, and smoke-test their commands. CI already builds wheels and source archives, checks them with Twine, and runs command help checks ([ci.yml](.github/workflows/ci.yml#L53)); there is no tag-triggered publishing workflow.

## Optional if inviting outside contributors

- [ ] Add contribution and security-reporting instructions, and decide whether the CODEOWNERS snippet should become an active file.
- [ ] Consider Python dependency updates in Dependabot. Its current config covers GitHub Actions only ([dependabot.yml](.github/dependabot.yml#L1)).
