"""The checker must catch evasions. Each case is a tiny fake package."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

from . import rules as real_rules
from .checker import Package


def _rules(**overrides):
    base = {n: getattr(real_rules, n) for n in dir(real_rules) if n.isupper()}
    base.update(
        INTERACTIVE_ROOTS=["mod:Hot.run"], NO_EAGER_READS=["mod:Hot.invalidate"],
        PURE_BOOKKEEPING=["mod:Hot.mark"], ALLOW={}, DEBT={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _violations(tmp_path, source, **overrides):
    (tmp_path / "mod.py").write_text(textwrap.dedent(source))
    return {(v.rule, v.symbol) for v in Package(tmp_path, _rules(**overrides)).check()}


SKELETON = """
import copy, hashlib, threading
from copy import deepcopy as dc
from threading import Timer
class Hot:
    def run(self): {body}
    def invalidate(self): pass
    def mark(self): pass
    def helper(self): {helper}
"""


@pytest.mark.parametrize("body,helper,expected", [
    ("self.backend.snapshot_state()", "pass", ("never-on-interactive-path", "snapshot_state")),
    ("dc(self.x)", "pass", ("never-on-interactive-path", "copy.deepcopy")),
    ("copy.deepcopy(self.x)", "pass", ("never-on-interactive-path", "copy.deepcopy")),
    ("hashlib.sha256()", "pass", ("never-on-interactive-path", "hashlib")),
    ("Timer(1, print).start()", "pass", ("never-on-interactive-path", "threading.Timer")),
    ("f = self.backend.save_state; f()", "pass", ("never-on-interactive-path", "save_state")),
    ("getattr(self.backend, 'load_state')()", "pass", ("never-on-interactive-path", "load_state")),
    ("self.helper()", "self.backend.restore_state(1)", ("never-on-interactive-path", "restore_state")),
    ("(lambda: self.backend.reset([1]))()", "pass", ("no-full-prefill", "reset")),
])
def test_interactive_evasions_are_caught(tmp_path, body, helper, expected):
    found = _violations(tmp_path, SKELETON.format(body=body, helper=helper))
    assert expected in found, found


def test_swallowed_backend_call_is_caught(tmp_path):
    body = "\n        try:\n            self.backend.eval([1])\n        except Exception:\n            pass"
    found = _violations(tmp_path, SKELETON.format(body=body, helper="pass"))
    assert ("no-swallowed-backend-errors", "eval") in found


def test_eager_read_in_invalidator_is_caught(tmp_path):
    source = SKELETON.format(body="pass", helper="self.engine.observe()").replace(
        "def invalidate(self): pass", "def invalidate(self): self.helper()")
    found = _violations(tmp_path, source)
    assert ("invalidate-dont-refresh", "observe") in found


def test_backend_work_in_pure_bookkeeping_is_caught(tmp_path):
    source = SKELETON.format(body="pass", helper="pass").replace(
        "def mark(self): pass", "def mark(self): self.backend.eval([1])")
    found = _violations(tmp_path, source)
    assert ("pure-bookkeeping", "eval") in found


def test_unrelated_reset_is_not_flagged(tmp_path):
    found = _violations(tmp_path, SKELETON.format(body="self.buffer.reset()", helper="pass"))
    assert ("no-full-prefill", "reset") not in found


def test_missing_root_fails_loudly(tmp_path):
    with pytest.raises(AssertionError, match="do not exist"):
        _violations(tmp_path, SKELETON.format(body="pass", helper="pass"),
                    INTERACTIVE_ROOTS=["mod:Hot.renamed"])
