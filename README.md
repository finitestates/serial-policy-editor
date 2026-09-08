# Serial Policy Editor 0.3.5

Steer a local language model in your terminal: choose individual tokens, insert
text, delegate a span, and rewind or fork the result. Serial Policy Replay (SPR)
lets you apply the recorded editing procedure again and inspect where its
outcome changes.

SPE supports **llama.cpp (GGUF)** and **Hugging Face Transformers (local model
directories)**. A SQLite workspace keeps episodes, editing actions, and token
evidence for resumption, replay, and text or evidence projection.

**0.3.5 is 0.3.4 with improved documentation and updated release metadata.**
There are no new features, command-line switches, or workspace-format changes.

## Get started

The installable Python project is in [`serial-policy-editor/`](serial-policy-editor/).
You need Python 3.10 or newer and your own local model files.

```bash
cd serial-policy-editor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[llama]'
policy-editor --backend llama.cpp --model /path/to/model.gguf \
  --new-prompt 'Once upon a time' --max-tokens 100
```

Press Enter to commit the displayed proposal, `h 10` to delegate ten tokens,
and `q` to open the EDGE menu. At EDGE, `quit` leaves your work resumable;
`end` seals it. The token allowance is a checkpoint, so reaching it does not
finish the episode.

## Documentation

- [Installation, first session, and complete user guide](serial-policy-editor/README.md)
- [Replay semantics](serial-policy-editor/README.md#serial-policy-replay)
- [Troubleshooting and backing up work](serial-policy-editor/README.md#troubleshooting)
- [Testing and real-model smoke checks](serial-policy-editor/README.md#tests)
- [Changelog](serial-policy-editor/CHANGELOG.md)
- [0.3.5 release notes](serial-policy-editor/RELEASE_NOTES_0.3.5.md)
- [Historical architecture overview (0.3.3)](serial-policy-editor/ARCHITECTURE_0.3.3.md)
- [Scope of the reduced editor](serial-policy-editor/CUT_NOTES.md)

The repository contains source and tests. Model weights, local workspaces,
virtual environments, and historical local archives are not release inputs.
