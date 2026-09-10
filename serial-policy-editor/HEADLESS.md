# SPE 0.3.6 headless preview

This branch adds a local browser editor and a small HTTP API over SPE's existing
episode engine, runner, and SQLite workspace. The service owns one loaded model
and one active episode. Opening or forking an episode switches that active state.

## Run

From the installable project directory, using the Python environment that already
runs SPE:

```sh
python -m trajectory_editor.headless --model /path/to/model.gguf
```

Open http://127.0.0.1:8765. After reinstalling the editable package, the equivalent
entry point is `policy-editor-server`. No additional server or frontend packages
are required. Browser assets are included in the Python package.

The default workspace is `headless-episodes.sqlite3`, separate from the terminal's
usual workspace. Override it with `--workspace /path/to/workspace.sqlite3`.
`--port`, `--n-ctx`, `--n-gpu-layers`, and `--backend transformers` are also supported.
The preview binds only to IPv4 loopback. This is a local development service, not
a public deployment server. Model paths are chosen at launch, never by an HTTP
client.

Do not run another editor against the same workspace while this server owns it.
To continue in the terminal, close the browser session, stop the server, then use:

```sh
policy-editor --workspace /path/to/workspace.sqlite3 --resume EPISODE_ID
```

The browser presents saved episodes, context, token candidates, text insertion,
bounded holds, forks, destructive rewind with confirmation, budget continuation,
and episode sealing. A fork switches to the child and leaves the parent saved.
Whitespace is shown as dots in candidate labels; the underlying token text is
unchanged. Raw probability is shown in the candidate list; the tooltip includes
sampler probability.

## HTTP contract

All data responses use JSON. Send `Content-Type: application/json` for mutations.
Read `/api/session` first to obtain its current `revision`. Every mutation needs
that revision and a unique `request_id` (1–128 characters):

```json
{
  "revision": "PROCESS_NONCE:0",
  "request_id": "create-1",
  "prompt": "It was a dark and stormy"
}
```

Send this object to `POST /api/session` to start an episode. To resume, replace
`prompt` with `episode_id`. Model provenance must match the running backend;
model replacement is left to the terminal in this preview.

| Method | Route | Additional request fields / response |
| --- | --- | --- |
| GET | `/api/session` | Revision, active episode, boundary, sampler, allowance, notices |
| GET | `/api/episodes` | Up to 100 saved episode summaries, newest first |
| GET | `/api/observation?start=1&count=12` | Revision, proposal, raw-ranked candidates; count ≤ 100 |
| POST | `/api/session` | `prompt` or `episode_id` |
| POST | `/api/session/actions` | `action`: existing SPE action object |
| POST | `/api/session/fork` | `boundary`: visible token boundary in active episode |
| POST | `/api/session/rewind` | `boundary`: permanently removes later history |
| POST | `/api/session/settings` | Optional `sampling` object and `max_tokens` |
| POST | `/api/session/close` | Release active episode, leaving its saved record open |
| POST | `/api/session/end` | Seal without generating another token |

For example, an action request is:

```json
{
  "revision": "PROCESS_NONCE:1",
  "request_id": "hold-1",
  "action": {"kind": "hold", "limit": 40, "boundary": "sentence"}
}
```

Supported actions are `accept`, `select-raw-rank`, `write`, `hold`, and
`end-generation`. Holds are capped at 256 tokens per request. The legacy `finish`
action is rejected; it means generation delegation in SPE, not episode sealing.
Use `/end` to seal. Writes use `mode: "continuation"` or `mode: "exact"`.

Mutation responses include the new session state and `result`. Action results
include outcomes, token evidence, stopping reasons, and any instruction rejection
in `handoff_reason`. Such a rejection is a normal engine handoff with no committed
action, returned as HTTP 200; malformed requests return 400. Stale revisions or
conflicting request IDs return 409. Refresh after either before trying a new
command. Revision changes include sampler edits and attempted mutations, not just
new token boundaries.

Repeating the *exact* same successful mutation, including revision and request ID,
returns its original response without executing again. Receipts last for the
server process; a new process uses a fresh revision nonce. Failed requests reserve
their IDs too; repeat failures return 409. At 10,000 receipts, restart the service.
Request bodies are limited to 64 KiB. No cross-origin access is enabled, and Host
and Origin checks reject requests from unrelated sites.

A settings request with only `sampling` preserves the remaining allowance unless
it has already been exhausted, following the engine's continuation semantics.
An explicit positive `max_tokens` begins a fresh allowance; null removes it.
An empty settings request continues/renews the existing allowance.

## Architecture and current limits

- `episode_lifecycle.py` contains the extracted lifecycle helpers. The terminal
  imports the same functions; notices can be directed to an HTTP response.
- `EpisodeRunner.run(max_live_actions=1)` performs one live action using its
  existing persistence and failure handling. The terminal keeps its normal loop.
- `headless.Session` owns state and revision/receipt handling. The HTTP server is
  deliberately single-threaded, so backend and SQLite operations stay on their
  owner thread and requests cannot interleave mutations.
- The browser is plain HTML/CSS/JavaScript served from the same origin as the API.
  It keeps drafts in memory and displays model text through text nodes.

This first slice does not expose replay, vocabulary text search, streaming,
cancellation, simultaneous model sessions, or authentication. Completed/failed
saved episodes are listed but cannot be reopened in the browser; use terminal
replay/fork tools. An episode just ended in the active session can still be forked.
Inference requests block other requests until the action completes. Closing a
browser tab does not cancel generation. The workspace persists across restarts;
active ownership and request receipts do not. SQLite and inference are not a
single atomic transaction, so unexpected execution failures still require the
existing episode recovery workflow.

## Validation

```sh
python -m pytest -q
```

`tests/test_headless.py` exercises a real loopback HTTP server with a deterministic
backend, duplicate and stale requests, model mismatch, sealing, and a full
edit/fork/rewind/reopen journey with evidence and budget assertions. It needs local
socket permission. Existing optional real-model suites retain their normal setup.
