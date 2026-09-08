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
  --new-prompt 'Once upon a time'
```

Or
```bash
policy-editor --backend llama.cpp --model/path/to/model.gguf
```
And the program will ask you for a prompt. Type in anything, then hit the Escape key, followed by the Enter key.

Nearly everything in the program uses the Enter key before it does anything with the exception of using Tab to cycle through options and using `[` and `]` to move backwards and forwards through tokens you have already selected.

To keep things simple, assume that any command described within the program interface itself is followed by hitting Enter.

The default interaction pattern is fairly simple: pressing Enter without doing anything will select the sampler's proposed token. Here are a few other things you can do:
- If you want to select a different token, type the number on the left-hand column
- Tab or Shift-Tab cycle through the token-selection menu
- `t TEXT` or `x TEXT` allow you to enter any text you want (the only difference is that `t` automatically inserts whitespace and `x` will not); if the last token is `an`, `t avocado` will form `an avocado`, whereas `x other` will form `another`
- `/TERM` lets you search the model's full vocabulary for TERM (this is whitespace sensitive, so `/by` and `/ by` search for different tokens -- much of the time you are searching for the next word so you will want to search for a `TERM` with a space between `/` and `TERM`)
- `m N` temporarily expands the token-selection menu by N rows (e.g. `m 10` expands it by 10 rows); this resets on the next token, so feel free to expand the available menu as much as you want at a given token position
- `q` accesses the EDGE menu; from here you can:
    - quit an episode (you can always resume it later)
    - end an episode (this seals it, but that doesn't prevent you from forking it, replaying it, or doing other things with it)
    - change your sampler settings
    - list the episodes that are in your workspace
    - fork from a particular point in this episode (`fm` lets you see a map of available points; I highly recommend using it)
    - resume the current episode
- `[` and `]` let you navigate forwards and backwards to different token positions in a live episode.

Using `[` and `]` for navigation is handy for backtracking especially if you have decided that you want to undo some recent actions: all you have to do is hit `[` as many times as necessary and then hit Enter. However, it's important to note that this type of undo action is irreversible. It is roughly equivalent to using the Backspace key. If you are not sure whether or not you want to permanently delete something, it is better to create a fork instead (input `f` at the point you have navigated to, or use the forking option from the EDGE menu).

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
