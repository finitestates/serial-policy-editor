#!/usr/bin/env python3
"""Compare the built extension with the current Python history reference."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "core" / "src"))

from rust_episode_history import _native  # noqa: E402
from generate_fixtures import python_result  # noqa: E402


def main() -> None:
    fixtures = json.loads(
        (EXPERIMENT_ROOT / "fixtures" / "history-cases.json").read_text(
            encoding="utf-8"
        )
    )
    failures = []
    for case in fixtures["cases"]:
        name = case["name"]
        expected = case["expected"]
        reference = python_result(case["request"])
        if reference != expected:
            failures.append(f"{name}: checked-in fixture differs from Python reference")
            continue
        request = json.dumps(
            case["request"], ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        rust = json.loads(_native.execute_json(request))
        if expected["ok"]:
            if rust != expected:
                failures.append(f"{name}: Rust result differs from Python result")
        else:
            if rust.get("ok") is not False or rust["error"]["kind"] != expected["error"]["kind"]:
                failures.append(f"{name}: Rust error category differs from Python")
            elif name == "action-unknown-kind" and rust["error"]["message"] != expected["error"]["message"]:
                failures.append(f"{name}: unknown-action diagnostic differs from Python")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"{len(fixtures['cases'])} shared cases match Python and Rust")


if __name__ == "__main__":
    main()

