"""Runtime enforcement for hot-path rules.

`guard.interactive(label)` patches banned functions for the duration of one
interactive operation. A banned call is RECORDED first and then raised as
`Tripwire`, a BaseException subclass, so production `except Exception:`
blocks cannot swallow it. Tests also assert the record is empty at the end,
so even `except BaseException:` cannot hide a violation.

`CountingBackend` is a self-contained fake: it counts every method call by
name, mirrors the real backends' branch semantics exactly, and treats
snapshot/restore as banned. It offers snapshot support on purpose, so code
that *would* use snapshots is caught trying.
"""

from __future__ import annotations

import copy
import hashlib
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from trajectory_editor.core.backend import BackendStateSnapshot

from . import rules

PACKAGE_DIR = Path(sys.modules["trajectory_editor"].__file__).resolve().parent


class Tripwire(BaseException):
    """A banned operation ran on an interactive path."""


def _tolerated_callers(symbol: str) -> set[str]:
    """ALLOW entries and untriaged (U) debt only. Task debt is never tolerated."""
    keys = set(rules.ALLOW) | {k for k, tag in rules.DEBT.items() if tag == "U"}
    return {fn for rule, fn, sym in keys
            if rule == "never-on-interactive-path" and sym == symbol}


def _is_tolerated(caller: str, tolerated: set[str]) -> bool:
    caller = caller.split(".<locals>")[0]
    if caller in tolerated:
        return True
    if sys.version_info < (3, 11):  # no co_qualname: match module + function name
        module, _, name = caller.partition(":")
        return any(
            key.partition(":")[0] == module and key.rsplit(".", 1)[-1].split(":")[-1] == name
            for key in tolerated
        )
    return False


def _package_frames():
    frame = sys._getframe(2)
    while frame is not None:
        path = Path(frame.f_code.co_filename).resolve()
        if PACKAGE_DIR in path.parents:
            module = ".".join(path.relative_to(PACKAGE_DIR).with_suffix("").parts)
            qual = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
            yield f"{module}:{qual}"
        frame = frame.f_back


class Guard:
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.label: str | None = None

    def banned(self, symbol: str) -> None:
        if self.label is None:
            return
        callers = list(_package_frames())
        first = callers[0] if callers else "<outside package>"
        if callers and _is_tolerated(callers[0], _tolerated_callers(symbol)):
            return
        message = f"{symbol} during {self.label!r} from {first} (stack: {' <- '.join(callers[:6])})"
        self.violations.append(message)
        raise Tripwire(message)

    @contextmanager
    def interactive(self, label: str):
        guard = self
        originals = {
            ("copy", "deepcopy"): copy.deepcopy,
            **{("hashlib", n): getattr(hashlib, n)
               for n in ("sha256", "sha1", "md5", "blake2b", "sha512", "new")},
        }

        def wrap(symbol, original):
            def banned(*args, **kwargs):
                guard.banned(symbol)
                return original(*args, **kwargs)
            return banned

        replacements = {
            key: wrap("copy.deepcopy" if key[0] == "copy" else "hashlib", fn)
            for key, fn in originals.items()
        }
        thread_start = threading.Thread.start

        def start(thread, *a, **k):
            guard.banned("threading.Timer" if isinstance(thread, threading.Timer) else "threading.Thread")
            return thread_start(thread, *a, **k)

        rebound = []
        for (mod, name), fn in replacements.items():
            setattr(sys.modules[mod], name, fn)
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith("trajectory_editor") or module is None:
                continue
            for attr, value in list(vars(module).items()):
                for key, original in originals.items():
                    if value is original:
                        rebound.append((module, attr, value))
                        setattr(module, attr, replacements[key])
        threading.Thread.start = start
        previous, self.label = self.label, label
        try:
            yield self
        finally:
            self.label = previous
            threading.Thread.start = thread_start
            for (mod, name), fn in originals.items():
                setattr(sys.modules[mod], name, fn)
            for module, attr, value in rebound:
                setattr(module, attr, value)


class CountingBackend:
    """Fake with exact branch semantics and per-method call counts."""

    VOCAB = 16

    def __init__(self, guard: Guard) -> None:
        self.guard = guard
        self.tokens: list[int] = []
        self.calls: Counter[str] = Counter()
        self.positions = 0
        self.full_prefills = 0
        self._token = object()
        self.speculative_token: int | None = None
        self._pre_speculation_logits = None
        self._restored_logits = None

    def tokenizer_id(self) -> str:
        return "hot-path-counting-tokenizer-v1"

    # --- accounting ---------------------------------------------------------
    def mark(self) -> tuple[Counter, int, int]:
        return Counter(self.calls), self.positions, self.full_prefills

    def since(self, mark) -> dict:
        calls, positions, prefills = mark
        delta = Counter(self.calls)
        delta.subtract(calls)
        return {
            "calls": +delta,
            "positions": self.positions - positions,
            "full_prefills": self.full_prefills - prefills,
            "model_calls": delta["reset"] + delta["eval"] + delta["branch_refresh"],
        }

    # --- speculation (the brief's in-place API) ---------------------------------
    def speculate(self, token_id):
        self.calls["speculate"] += 1
        self._settle()
        self._pre_speculation_logits = self.last_logits(_count=False)
        self.tokens.append(int(token_id))
        self.positions += 1
        self.speculative_token = int(token_id)
        return True

    def commit_speculation(self):
        self.calls["commit_speculation"] += 1
        self.speculative_token = None
        self._pre_speculation_logits = None

    def rollback_speculation(self):
        self.calls["rollback_speculation"] += 1
        if self.speculative_token is None:
            return
        self.tokens.pop()
        self._restored_logits = self._pre_speculation_logits
        self.speculative_token = None
        self._pre_speculation_logits = None

    def _settle(self):
        """Backend-enforced: any mutation while speculating rolls back first."""
        if self.speculative_token is not None:
            self.rollback_speculation()
        self._restored_logits = None

    # --- model work ----------------------------------------------------------
    def reset(self, prefix):
        self.calls["reset"] += 1
        self._settle()
        self.tokens = [int(t) for t in prefix]
        self.positions += len(self.tokens)
        self.full_prefills += 1

    def eval(self, token_ids):
        self.calls["eval"] += 1
        self._settle()
        values = [int(t) for t in token_ids]
        self.tokens.extend(values)
        self.positions += len(values)

    def branch_to_prefix(self, prefix):
        self.calls["branch_to_prefix"] += 1
        self._settle()
        prefix = [int(t) for t in prefix]
        if len(prefix) > len(self.tokens) or self.tokens[: len(prefix)] != prefix:
            self.reset(prefix)
        elif len(prefix) < len(self.tokens):
            self.tokens = prefix
            self.calls["branch_refresh"] += 1   # real backends re-evaluate one token
            self.positions += 1

    def truncate_to(self, n):
        self.calls["truncate_to"] += 1
        self._settle()
        self.tokens = self.tokens[:n]
        return True

    def snapshot_state(self):
        self.calls["snapshot_state"] += 1
        self.guard.banned("snapshot_state")
        return BackendStateSnapshot(self._token, tuple(self.tokens), tuple(self.tokens))

    def restore_state(self, snapshot):
        self.calls["restore_state"] += 1
        self.guard.banned("restore_state")
        self.tokens = list(snapshot.prefix_token_ids)
        return True

    # --- reads -----------------------------------------------------------------
    def last_logits(self, _count=True):
        if _count:
            self.calls["last_logits"] += 1
            if self.speculative_token is not None:
                message = "last_logits() read while a speculative token is in the cache"
                self.guard.violations.append(message)
                raise Tripwire(message)
        if self._restored_logits is not None:
            return self._restored_logits.copy()
        seed = 0
        for t in self.tokens:
            seed = (seed * 1_000_003 + t + 1) % (1 << 32)
        logits = np.random.default_rng(seed).normal(size=self.VOCAB).astype(np.float32)
        logits[0] = -50.0   # never end on its own
        return logits

    def render(self, token_ids, *, special=False):
        self.calls["render"] += 1
        return "".join(self.token_text_raw(t) for t in token_ids)

    def token_text_raw(self, token_id):
        return "<EOG>" if int(token_id) == 0 else ("P" if int(token_id) == 7 else f" t{int(token_id)}")

    def token_text(self, token_id):
        self.calls["token_text"] += 1
        return self.token_text_raw(token_id)

    def new_text_stream(self, *, special=False):
        self.calls["new_text_stream"] += 1
        backend = self

        class Stream:
            # Exact, tiny state: the fake decodes per token, so there is none
            # beyond identity. checkpoint/restore are free, as they should be.
            def checkpoint(self):
                backend.calls["stream.checkpoint"] += 1
                return ()

            def restore(self, state):
                backend.calls["stream.restore"] += 1

            def append(self, token_ids):
                backend.calls["stream.append"] += 1
                return "".join(backend.token_text_raw(t) for t in token_ids)

        return Stream()

    # --- metadata ----------------------------------------------------------------
    def vocabulary_size(self):
        return self.VOCAB

    def tokenize(self, text, *, add_bos=False, special=False):
        return [7] if text else []

    def is_eog(self, token_id):
        return int(token_id) == 0

    def eog_token_ids(self):
        return (0,)

    def provenance(self, *, include_model_sha256=True):
        return {"backend": "counting-fake", "vocabulary_size": self.VOCAB}
