# Serial Policy Editor 0.3.5

0.3.5 is a documentation-focused update to 0.3.4. Editor behavior, command-line
options, dependencies, sampling, and the SQLite workspace format are unchanged.
The only Python source change is the reported version number.

## Documentation

- A public-facing root README with a working installation entry point.
- A first-session walkthrough and a comparison of resume, fork, rewind, and replay.
- Troubleshooting for terminal display, model paths, workspaces, and token allowances.
- Backup and plain-text export guidance.
- Replay and budget details grouped with user documentation rather than tests.
- Prior model results clearly identified as 0.3.4 verification history.

## Verification

Local checks completed for 0.3.5:

| Check | Result |
| --- | --- |
| Default pytest suite | 474 passed; 11 opt-in model tests skipped |
| Real llama.cpp smoke suite: Llama 3.2 1B Instruct Q8_0 | 5 passed |
| Real Transformers smoke suite: GPT-2 | 6 passed |

The real-model checks used llama-cpp-python 0.3.35 and Transformers 5.16.1.
They exercise sampling, EDGE configuration, rewind, forks, resume, and replay;
the Transformers suite also compares cached and complete-prefix inference.
These checks do not establish identical output across different hardware or
model/runtime combinations, or validate the full-screen terminal renderer.

See the [user guide](README.md) for installation and commands to repeat these
checks. Existing 0.3.4 workspaces need no migration for this release.
