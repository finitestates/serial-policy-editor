"""Static hot-path rules. See rules.py (owner-edited) and checker.py."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import pytest

from . import rules
from .checker import Package
from .tripwire import PACKAGE_DIR

REPO = PACKAGE_DIR.parents[2]
TASK_TAGS = {"A", "B", "C"}


@lru_cache(maxsize=1)
def package() -> Package:
    return Package(PACKAGE_DIR, rules)


@lru_cache(maxsize=1)
def violations():
    return package().check()


def _fmt(items) -> str:
    return "\n".join(str(v) for v in items)


def test_no_untracked_violations():
    tracked = set(rules.ALLOW) | set(rules.DEBT)
    new = [v for v in violations() if v.key() not in tracked]
    assert not new, (
        "Hot-path rule violations. Fix the code; do not edit rules.py.\n" + _fmt(new)
    )


def test_no_stale_allow_entries():
    found = {v.key() for v in violations()}
    stale = [k for k in rules.ALLOW if k not in found]
    assert not stale, f"ALLOW entries that no longer match (owner must remove): {stale}"


def test_debt_ratchet_delete_cleared_entries():
    found = {v.key() for v in violations()}
    cleared = [k for k in rules.DEBT if k not in found]
    assert not cleared, (
        "These DEBT entries are fixed. Delete them from rules.py "
        "(deleting DEBT lines is the only edit you may make there):\n"
        + "\n".join(map(str, cleared))
    )


@pytest.mark.parametrize("task", sorted(TASK_TAGS))
def test_task_debt_cleared(task):
    found = {v.key() for v in violations()}
    remaining = [k for k, tag in rules.DEBT.items() if tag == task and k in found]
    assert not remaining, f"Task {task} is not done; still violating:\n" + "\n".join(map(str, remaining))


def test_allow_and_debt_are_disjoint_and_well_formed():
    assert not set(rules.ALLOW) & set(rules.DEBT)
    assert set(rules.DEBT.values()) <= TASK_TAGS | {"U"}
    for key, entry in rules.ALLOW.items():
        assert len(entry.get("why", "")) >= 40, f"{key}: justification too short"
        path, _, name = entry["test"].partition("::")
        source = (REPO / path).read_text()
        assert f"def {name}(" in source, f"{key}: bounding test {entry['test']} does not exist"


def test_every_watched_method_is_classified():
    pkg = package()
    reachable = pkg.closure(rules.INTERACTIVE_ROOTS)
    unclassified = [
        f"{cls}.{name}"
        for cls in rules.WATCHED_CLASSES
        for name in sorted(pkg.methods_by_class[cls])
        if f"{cls}.{name}" not in reachable and f"{cls}.{name}" not in rules.COLD
    ]
    assert not unclassified, (
        "Methods that are neither reachable from an interactive root nor classified "
        "COLD. If a method is now dead (e.g. snapshot_state after Task B), delete it. "
        "Otherwise stop and ask the owner to classify it:\n" + "\n".join(unclassified)
    )
    missing = [m for m in rules.COLD if m.rsplit(".", 1)[0] in rules.WATCHED_CLASSES
               and m.rsplit(".", 1)[1] not in pkg.methods_by_class[m.rsplit(".", 1)[0]]]
    assert not missing, f"COLD entries for methods that no longer exist: {missing}"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("base_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rules_ratchet_against_base_branch():
    """In CI, SPE_BASE_RULES points at rules.py from the protected base branch.

    A PR may only delete DEBT entries. Every other value must be identical.
    """
    base_path = os.environ.get("SPE_BASE_RULES")
    if not base_path:
        if os.environ.get("CI"):
            pytest.fail("SPE_BASE_RULES must be set in CI")
        pytest.skip("local run: ratchet is enforced in CI")
    base = _load(Path(base_path))
    names = {n for n in dir(base) if n.isupper()} | {n for n in dir(rules) if n.isupper()}
    for name in sorted(names - {"DEBT"}):
        assert getattr(rules, name, None) == getattr(base, name, None), (
            f"rules.{name} differs from the base branch; only the owner may change it"
        )
    assert set(rules.DEBT.items()) <= set(base.DEBT.items()), "DEBT may only shrink"
