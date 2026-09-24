# Hot-path rules

These rules apply to every change that touches an interactive path. They are
enforced by `tests/hot_path/`, and CI runs the **base branch's** copy of that
directory, so editing the checker or the tests in your PR has no effect.

"Interactive path" means everything reachable from the roots in
`tests/hot_path/rules.py::INTERACTIVE_ROOTS`: keystrokes, frames, Enter,
`[`/`]`, chord open/advance/rewind/select/promote/discard, review navigation,
and warm selection.

## The rules

1. **Never on an interactive path:** `save_state`, `load_state`,
   `snapshot_state`, `restore_state`, `copy.deepcopy`, anything from
   `hashlib`, `threading.Thread`, `threading.Timer`. Referencing them counts,
   not just calling them: aliases, `getattr(x, "name")`, bound-method
   variables, and lambdas are all caught.
2. **No full prefill:** no `backend.reset(...)` / `guidance_backend.reset(...)`
   on an interactive path. Reposition with `branch_to_prefix`, `truncate_to`,
   or slots.
3. **Invalidate, don't refresh.** These functions record that something changed
   and must not read model output, directly or through anything they call:
   `Chord.rewind`, `Chord._activate`, `Chord.discard`,
   `EpisodeEngine.rewind_to`, `EpisodeEngine._commit_token`,
   `EpisodeEngine._ensure_backend_positioned`,
   `EpisodeEngine.adopt_preview_state`, `_ContextRenderCursor.prewarm`.
   "Reading" means `observe`, `last_logits`, `engine.candidates`,
   `engine.policy_candidates`, `backend.render`, `backend.token_text`,
   `backend.new_text_stream`, `_append_stream`, `_rebuild_stream_only`.
4. **Pure bookkeeping does no backend work at all:**
   `EpisodeEngine._invalidate_observation`, `_invalidate_guidance`,
   `discard_speculative_accept`, `terminate`. (Rolling back or committing a
   speculation is allowed; both are free by contract.)
5. **No swallowed backend errors:** no `except Exception:` /
   `except BaseException:` / bare `except:` around a backend call on an
   interactive path. If an optional fast path can fail, check capability
   first; don't try-and-swallow.
6. **Every method on a watched class** (`Chord`, `EpisodeEngine`,
   `_ContextRenderCursor`, `PersistentTerminalSession`, `LlamaCppDecoder`,
   `TransformersBackend`) is either reachable from an interactive root or
   listed `COLD`. A new unreachable method fails the suite. If it's dead
   code, delete it. If it's new, stop and ask the owner to classify it.

## Runtime budgets (`test_runtime_budgets.py`)

Operations run under a tripwire that makes banned calls raise `Tripwire`, a
`BaseException` that `except Exception` cannot catch. Every banned call is
also recorded and checked at teardown, so `except BaseException` can't hide it
either. The fake backend counts every method call and has no cost model to
argue with.

| Operation | Budget |
|---|---|
| repeat `observe()` of an unchanged decision | 0 backend calls |
| chord rewind | 0 prefills, ≤ 1 model call, 0 `last_logits` |
| chord advance round | 0 prefills, 0 refresh evals, positions ≤ Σ(suffix + 1), `last_logits` ≤ live paths |
| any chord sequence | 0 prefills; path tokens equal a plain run |
| speculative warm | exactly 1 position |
| commit after warm (hit) | **0** positions. A declined warm costs 1 here, so skipping speculation fails |
| commit after warm (miss) | exactly 1 position |
| logits read while a speculative token is cached | forbidden |
| review step back | ≤ 8 stream appends, 0 `render`, 0 prefills, 0 new threads |

Speculation and chord results must also equal a run without the optimization.

## What you may and may not edit

- You may **delete DEBT entries** in `rules.py` that the suite reports as
  cleared. That is the only edit allowed there; CI rejects any other
  difference from the base branch.
- You may not add ALLOW or DEBT entries, roots, symbols, or COLD entries, and
  you may not edit anything else under `tests/hot_path/`.
- If a rule blocks correct work, stop. In the PR description, name the rule,
  the function, and why. The owner decides.

## When your task is done

- `test_task_debt_cleared[<your task>]` passes.
- `test_debt_ratchet_delete_cleared_entries` passes, meaning you deleted those
  entries.
- Every test in `test_runtime_budgets.py` for your task passes, with no new
  failures elsewhere in `tests/hot_path` or `tests/core`.
