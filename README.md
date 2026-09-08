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

## Getting started

The installable Python project is in [`serial-policy-editor/`](serial-policy-editor/).
You need Python 3.10 or newer and your own local model files.

```bash
cd serial-policy-editor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[llama]'
```
For Transformers, use:
```bash
python -m pip install -e '.[transformers]'
```

Once installed, you can open the program a few different ways:
```bash
policy-editor --backend llama.cpp --model /path/to/model.gguf \
  --new-prompt 'Once upon a time'
```
Or
```bash
policy-editor --backend transformers --model /path/to/transformers/directory \
  --new-prompt 'It was a dark and stormy'
```

You can omit `--new-prompt` and you will be asked for one interactively. Type your prompt, then press **Escape**, followed by **Enter**, to submit it. The backend doesn't need to be specified for llama.cpp, but does need to be provided if you are using transformers.

So, assuming you are using llama.cpp, the fastest way to start the program is just:
```bash
policy-editor --model /path/to/model.gguf
```

For more info, just enter:
```bash
policy-editor --help
```
Or
```bash
policy-editor -h
```
To see a full list of command line flags. There are quite a number, but I've tried to minimize the amount of flags you need to use for basic episode management.


## Basic command usage
Once an episode is live, nearly everything uses the **Enter** key before it does anything with a few notable exceptions: 
- **Tab** and **Shift-Tab** cycle through token-selection options or search options
- `[` and `]` move backwards and forwards through tokens you have already selected
- **CTRL-G** executes a search based on token rank number

To keep things simple, assume that any command described below is followed by hitting **Enter**.

Some of the commands follow a pattern of `letter [operator] number` (e.g. `m 10` or `h . 5`). While these are presented with whitespace for readability purposes, in the program itself, whitespace is not necessary (e.g. `m 10` and `m10` parse the same; this is also true for things like `h . 5` and `h.5`).

Other commands (e.g. `t` and `x` especially) require whitespace after the initial letter in order to do anything (`t TEXT` enters `TEXT`; `tTEXT` will just throw an error).

`/` is whitespace sensitive since tokens themselves can include or not include whitespace. So, `/TERM` and `/ TERM` are different searches.

### Core syntax
The default interaction pattern is fairly simple:
- From a new token position (such as the first token that could appear after the prompt you entered when you started the program), pressing **Enter** without doing anything will select the sampler's proposed token
- If you want to select a different token, type the number on the left-hand column (the raw rank number)
- You can also use **Tab** or **Shift-Tab** to cycle through the token-selection menu
- `t TEXT` or `x TEXT` allow you to enter any text you want, including multiple tokens at once (the only difference is that `t` automatically inserts whitespace and `x` will not); if the last token is `an`, `t avocado` will form `an avocado`, whereas `x other` will form `another`
- `m N` temporarily expands the token-selection menu by N rows (e.g. `m 10` expands it by 10 rows); this resets on the next token, so feel free to expand the available menu as much as you want at a given token position
- `h N` delegates the next `N` decisions to the model & sampler (e.g. `h 5` lets the model & sampler automatically pick the next 5 tokens):
    - `h . N` will delegate `N` decisions to the model & sampler up to a sentence boundary (e.g. `.`, `!`, `?`) or `N`, whichever comes first.
    - `h | N` will do the same but up to a newline (e.g. `\n`).
    - When using `h . N` or `h | N`, tokens that contain either a sentence boundary or a newline plus some other text, will not be split-up (e.g. `h . N` will include the full token for things like `."` or `.\n\n`).
- `?` brings up a list of commands with short descriptions of what they do.

After you select a token or enter text, you will automatically be taken to the next position.

### Search the vocabulary
`/TERM` lets you search the model's full vocabulary for `TERM` (this is whitespace sensitive, so `/by` and `/ by` search for different tokens -- much of the time you are searching for the next word so you will want to search for a `TERM` with a space between `/` and `TERM`):
- Once you have searched for something, it will bring up a smaller menu, which contains `TERM` and its nearest neighbors based on raw rank
- To return to the main token-selection menu from the search menu, use `m`
- If you have already searched for something and are on the token-selection menu, `ms` returns you to search menu
- You can expand the search menu by `N` rows up or down using `ms + N` or `ms - N`  (e.g. `ms + 10` adds 10 additional rows below the end of the current search menu).
- If `TERM` is a multi-token expression (sometimes different model tokenizers will break words into several different tokens), you will get a search suggestion instead of being taken to the search menu
- To automatically enter a search suggestion into the action field, immediately after your search, just hit Tab and it should auto-populate the action field with the first search suggestion

_For example: you search for the word "elephantine" using `/ elephantine` but get told that this particular tokenizer has tokens for `elephant` and `ine`; from there, hit Tab, and the action field should show `/ elephant`; then, enter to search for "elephant"._

If you type in any number, it will preview the token at that position even if you haven't searched for it and even if it is not currently on the token-selection menu. 

_For example: you can just type in `205` to see what token raw-rank 205 is. If you want to perform a search for that token, just hit **CTRL-G** and it will automatically take you to a search menu for that token._

### Access the EDGE menu
`q` accesses the EDGE menu; from here you can:
- quit an episode (you can always resume it later)
- end an episode (this seals it, but that doesn't prevent you from forking it, replaying it, or doing other things with it)
- change your sampler settings
- list the episodes that are in your workspace
- fork from a particular point in this episode (`fm` lets you see a map of available points; I highly recommend using it)
- resume the current episode

### Rewinding
`[` and `]` let you navigate forwards and backwards to different token positions in a live episode.

Using `[` and `]` for navigation is handy for backtracking especially if you have decided that you want to undo some recent actions: all you have to do is hit `[` as many times as necessary and then hit Enter. However, it's important to note that this type of undo action is irreversible. It is roughly equivalent to using the Backspace key. If you are not sure whether or not you want to permanently delete something, it is better to create a fork instead (input `f` at the point you have navigated to, or use the forking option from the EDGE menu).

## Replay

You may notice that there is both `--resume` and `--replay`. They sound like they might be doing the same thing, so why have both? Resume continues an existing unfinished episode; replay creates a new episode by executing its recorded editing actions. Replay is like a swiss-army-knife command that can function as a quick way to clone an existing episode, as a stress test for your system, as a counterfactual generator, or--when used within an episode--as a splicing tool. Replay executes each command that generated an episode sequentially using a "tape" of the other episode's actions.

So: let's say you started an episode by selecting token `7`, which corresponded to `night`. If you replay that episode, the program will (by default) import the same settings of the original episode, fire the model up, and select `7` again. Often, this corresponds to the same token, but it may not.

Replay will continue acting based on the available tape. But it can be initiated with two different stopping conditions:
- Handoff (default): if an action is about to select a different token than the one from the source episode at the same position, replay will end and the EDGE menu will open. From there, action proceeds like any other live episode.
- Ballistic: replay just continues until the tape is exhausted even if different tokens are selected. The only thing that can stop ballistic mode from exhausting the full tape is a rather narrow range of conditions. This can generate episodes that are markedly different than the original especially if you start the replay with a different PRNG seed (`policy-editor --replay '#1' --random-seed` or `policy-editor --replay '#1' --seed 77`) or alter the sampler settings (not every setting will cause a divergence; some are more prone to that than others).

For example, to replay episode `#1` in ballistic mode:

```bash
policy-editor --replay '#1' --divergence-policy ballistic
```

Replace `#1` with an episode number from your workspace, and keep the quotes in shell commands so `#` is not treated as a comment. The examples assume you are using the same workspace; add `--workspace /path/to/episodes.sqlite3` if needed.

The crucial thing about replay is that it always terminates at the EDGE menu, regardless of if you use handoff or ballistic. It is using another episode as a source in order to create a new live episode. It's one of the things that makes the program special in my opinion: episodes you create do not simply generate archival transcripts (although they do that also), but can be used to create new live episodes with very little effort.

Another way to use replay is within an episode itself. To do this you enter `spr #1` from the EDGE menu, replacing `#1` with the source episode's number. The prompt of that other episode will be entered as raw text as if using the `x` command, followed by its recorded actions. This appends to the current episode and keeps the destination's model, sampler settings, and random stream; source sampler settings are not imported. Replay when used this way can function as a powerful splicing tool, allowing you to compose episodes out of other episodes.

There are some finer points to all this, which are covered elsewhere, but it is worth drawing attention to this function.

**One final note about replay:** People often over-index how likely a replayed episode is to diverge from the source episode. In my experience, using the same model and comparable runtime settings, replay usually reproduces the source unless I deliberately change something to encourage divergence. Since token selection happens via raw rank or directly tokenized text, those tend to be pretty stable, unless you change one of a handful of things about the source sampler config or if your episode contains raw rank selection from deep within the probability distribution. `policy-editor --replay '#1'` without anything else more often than not produces an episode that looks the same as the source. The underlying math may have shifted slightly, but not enough to matter. Further, whether or not divergence is even undesirable depends on what you are trying to accomplish via replay.

## Why "teacher"?

The user in this program is referred to as "Teacher" as a reference to Teacher-forced answers in machine learning. That being said, this program is not a model-training tool. Anything you can do here, you can probably accomplish more directly and efficiently via other means. In fact, some of the most fun things to do involve taking the less direct path than you could (e.g. searching for a token and entering it from the search menu as opposed to just entering it directly via `t TEXT`). The purpose of this program is educational, exploratory, and creative--it provides you with a very granular view into how the next token arrives and gives you as much or as little control as you want over that process.


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

## License

Copyright (c) 2026 Graham Christopher Andrews.

Serial Policy Editor is released under the [MIT License](LICENSE).
Third-party dependencies and model weights remain subject to their own licenses.
