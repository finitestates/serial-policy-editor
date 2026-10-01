# TODO for 1.0.0

## Product polish

- [x] Keep speculative token prewarming only for active search matches. Ordinary candidate navigation and selection use the normal commit path; search warming runs off the UI thread, cancels when its target changes, and promotes only an exact match. Covered by the [ordinary-selection regression](tests/core/test_selection_warm_terminal.py) and [search-warm tests](tests/core/test_search_warm_terminal.py).
- [ ] Identify and validate a viable path to improve overall terminal rendering. Menu conventions across EDGE, search/review, and setup have improved, but the rendered output remains unsatisfactory; no concrete rendering change is selected.
- [x] Strengthen terminal-rendering validation with production PTY frame capture, raw-byte replay through pyte at every frame boundary, completeness checks, and negative controls. See the [rendering guide](tests/TERMINAL_RENDERING_GUIDE.md) and [PTY journeys](tests/core/test_live_terminal_pty.py).
- [x] Add an end-to-end regression for Tab/Enter through `run_plan()`, checking the recorded action and next view. The [selection test](tests/core/test_selection_warm_terminal.py) checks the ordinary selection flow and confirms that candidate navigation does not trigger speculative warming.
- [x] Complete a focused cleanup of confirmed stale aliases/functions and obsolete model-identity/persistence fields. Larger BranchState/registry overlap remains a separate architectural question.
- [x] Fix the bias system. Right now, it is a lot of residual complexity lacking a clear identity about what it is trying to accomplish.

## Replay authoring and independent correctness

- [x] Build a minimal independent model oracle with `atomic` proposal menus and `pattern` continuation. Supports llama.cpp and Transformers, text/JSONL output, production sampler parity, and a reproducible draw-coordinate demonstration. See [usage and scope](reference-kernel/README.md) and [oracle checks](reference-kernel/test_reference_kernel_oracle.py).
- [ ] Define a plain-text teacher-plan format modeled on `--procedure`, then add a parser with useful line-numbered errors. Check whether the current procedure view includes everything replay needs; decide what grammar and expected-result details the format must add. Keep JSONL plans usable.

## Tokenizer identity and streaming

- [x] Change how mixed-model forks are handled. If a model has a different tokenizer than the episode it is forking from, it will have different sampler coordinates. As such, it's really not a fork. A different model with the same tokenizer could be considered a fork assuming the token IDs of the prompt/prefix match. In order to enforce this, we need to have better provenance data about model & tokenizer (currently, the system resolves "same model" by evaluating absolute path, which is stupid).
- [ ] Audit and test how streamed partial UTF-8 sequences affect token-step accounting and saved text. Token boundaries advance by token IDs, while a live stream may buffer bytes until a character is complete. Cover display, rewind, partial-write replay, procedure export, and cross-tokenizer forks; keep token IDs authoritative where decoded text cannot faithfully represent a retained prefix. The review identified this as a code-path risk to investigate, not a demonstrated failure.

## Decide what 1.0 promises

- [x] Document the Python and operating-system support matrix, model backends, file formats, and the Python API and CLI compatibility promise ([README](README.md#10-support-and-compatibility)).
- [x] Set a policy for saved workspaces and exported files.

## Documentation and release readiness

- [x] Remove or correct references to the removed `archive/` tree in the [README](README.md), test inventory, old changelog link, and sampling module comment.
- [x] Reconcile the release history and prepare the packages, changelog, and README for `0.8.0`; document the `0.7.5` release line and link the test count to the contract matrix.
- [x] Choose the Python and operating-system support matrix, then make CI match it: Ubuntu with Python 3.10 and the latest Python; real-model checks remain opt-in. See [ci.yml](.github/workflows/ci.yml#L17) and the [support statement](README.md#10-support-and-compatibility).
- [x] Write down the release steps: update the changelog and synchronized versions, build both packages, install the built artifacts in a clean environment, and smoke-test their commands. See the [release checklist](DEVELOPMENT.md#release-checklist); CI also builds both distributions, checks them with Twine, and checks command help ([ci.yml](.github/workflows/ci.yml#L53)).

## Optional if inviting outside contributors

- [ ] Add contribution and security-reporting instructions, and decide whether the CODEOWNERS snippet should become an active file.
- [ ] Consider Python dependency updates in Dependabot. Its current config covers GitHub Actions only ([dependabot.yml](.github/dependabot.yml#L1)).
