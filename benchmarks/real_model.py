#!/usr/bin/env python3
"""Opt-in real-model correctness and overhead harness; run from the repository root."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from uuid import uuid4

import numpy as np
import yaml

from trajectory_editor.controller_profiles import _DuplicateKeyLoader, parse_controller_profile, profile_arguments
from trajectory_editor.core.cli_config import sampler_from_args
from trajectory_editor.core.errors import EditorError
from trajectory_editor.episode_backend_loader import load_backend
from trajectory_editor.episode_cli import build_parser
from benchmarks.real_model_scenarios import SCENARIOS, continuation


FORMAT = "spe-real-model-report-v1"
HARNESS_VERSION = 2
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "spe-real-model-results"


class ProfileError(ValueError):
    pass


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ProfileError(f"{label} must be a mapping")
    return value


def load_profile(path: Path, model_root: Path | None, model_override: str | None,
                 backend_override: str | None, requested_scenarios: list[str] | None,
                 repeats_override: int | None) -> dict:
    try:
        payload = _mapping(yaml.load(path.read_text(encoding="utf-8"), Loader=_DuplicateKeyLoader), "profile")
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"cannot read profile {path}: {exc}") from exc
    allowed = {"name", "model", "backend", "launch", "rtol", "atol", "scenarios", "repeats", "timeout_s", "warmup"}
    if set(payload) - allowed:
        raise ProfileError(f"unknown profile fields: {sorted(set(payload) - allowed)}")
    raw_model = model_override if model_override is not None else payload.get("model")
    if not isinstance(raw_model, str) or not raw_model:
        raise ProfileError("profile needs a model path")
    model_path = Path(raw_model).expanduser()
    if not model_path.is_absolute():
        if model_root is None:
            raise ProfileError("relative model path requires --model-root")
        model_path = model_root / model_path
    model_path = model_path.resolve()
    backend = backend_override if backend_override is not None else payload.get("backend")
    if backend not in {"llama.cpp", "transformers"}:
        raise ProfileError("backend must be llama.cpp or transformers")
    if not model_path.exists() or (backend == "llama.cpp" and not model_path.is_file()) or (backend == "transformers" and not model_path.is_dir()):
        raise ProfileError(f"selected {backend} model is missing or has the wrong type: {model_path}")
    dependency = "llama_cpp" if backend == "llama.cpp" else "transformers"
    if importlib.util.find_spec(dependency) is None or (backend == "transformers" and importlib.util.find_spec("torch") is None):
        raise ProfileError(f"missing {backend} dependency; install the matching ./core extra")
    launch = _mapping(payload.get("launch", {}), "launch")
    forbidden = {"model", "backend", "profile", "teacher-plan", "workspace", "ephemeral", "new-prompt", "output"}
    if forbidden & set(launch):
        raise ProfileError(f"launch contains harness-owned option: {sorted(forbidden & set(launch))}")
    parser = build_parser(include_vector=False)
    try:
        values, _ = parse_controller_profile({"values": launch}, parser)
        tokens, _ = profile_arguments(parser, values)
        options = parser.parse_args(tokens)
    except (EditorError, SystemExit) as exc:
        raise ProfileError(f"invalid launch settings: {exc}") from exc
    options.model = model_path
    options.backend = backend
    names = requested_scenarios or payload.get("scenarios", ["continuation", "action-jsonl", "observed-jsonl", "rewind-replace", "fork-switch", "save-resume", "long-context", "cli-jsonl"])
    if not isinstance(names, list) or not names or any(name not in SCENARIOS for name in names) or len(names) != len(set(names)):
        raise ProfileError(f"scenarios must be a nonempty list of {sorted(SCENARIOS)}")
    repeats = repeats_override if repeats_override is not None else payload.get("repeats", 1)
    warmup = payload.get("warmup", 1)
    timeout_s = payload.get("timeout_s", 90)
    if any(type(number) is not int or number < minimum for number, minimum in ((repeats, 1), (warmup, 0), (timeout_s, 1))):
        raise ProfileError("repeats and timeout_s must be positive integers; warmup must be nonnegative")
    rtol = payload.get("rtol", 1e-3)
    atol = payload.get("atol", 1e-2)
    if any(type(number) not in (int, float) or not np.isfinite(number) or number < 0 for number in (rtol, atol)):
        raise ProfileError("rtol and atol must be finite nonnegative numbers")
    return {"name": payload.get("name", path.stem), "path": str(path.resolve()),
            "model_path": model_path, "backend": backend, "launch": values,
            "launch_tokens": tokens, "options": options, "scenarios": names,
            "repeats": repeats, "warmup": warmup, "timeout_s": timeout_s,
            "rtol": float(rtol), "atol": float(atol)}


@contextmanager
def deadline(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    def timed_out(_signum, _frame):
        raise TimeoutError(f"scenario exceeded {seconds} seconds")
    old = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _git_identity() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip()
    revision = git("rev-parse", "HEAD")
    state = git("status", "--porcelain", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard")
    checksum = hashlib.sha256((state + diff).encode())
    for name in untracked.splitlines():
        file = Path(name)
        if file.is_file():
            checksum.update(name.encode())
            checksum.update(file.read_bytes())
    return {"revision": revision, "dirty": bool(state),
            "dirty_state_sha256": checksum.hexdigest() if state else None}


def _preflight(backend) -> dict:
    ids = backend.tokenize("A short preflight.", add_bos=True, special=True)
    if not ids:
        raise AssertionError("tokenizer returned an empty preflight prompt")
    backend.reset(ids)
    logits = np.asarray(backend.last_logits())
    if logits.ndim != 1 or len(logits) != backend.vocabulary_size() or not np.all(np.isfinite(logits)):
        raise AssertionError(f"invalid logit shape/finiteness: {logits.shape}")
    return {"prompt_tokens": len(ids), "vocabulary_size": len(logits)}


def _hardware() -> dict:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model, "python": platform.python_version()}


def _runtime_identity(backend, profile: dict) -> dict:
    selected = profile["model_path"]
    identity = {"backend": profile["backend"], "cache_mode": profile["options"].cache}
    if selected.is_dir():
        files = ("config.json", "tokenizer.json", "tokenizer_config.json")
        identity["component_sha256"] = {
            name: hashlib.sha256((selected / name).read_bytes()).hexdigest()
            for name in files if (selected / name).is_file()
        }
        torch = getattr(backend, "_torch", None)
        if torch is not None:
            identity["torch_version"] = torch.__version__
            identity["cuda_available"] = bool(torch.cuda.is_available())
            identity["cuda_devices"] = [torch.cuda.get_device_name(index)
                                         for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    else:
        metadata = getattr(getattr(backend, "_model", None), "metadata", {})
        identity["gguf_file_type"] = metadata.get("general.file_type")
        identity["gguf_quantization_version"] = metadata.get("general.quantization_version")
        identity["gguf_tokenizer_pre"] = metadata.get("tokenizer.ggml.pre")
        info = getattr(getattr(backend, "_llama_cpp", None), "llama_print_system_info", None)
        if callable(info):
            try:
                value = info()
                identity["llama_system_info"] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
            except (RuntimeError, TypeError):
                identity["llama_system_info"] = None
    return identity


def _save(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _summary(samples: list[dict]) -> dict:
    metrics = [sample["result"]["metrics"] for sample in samples if sample["status"] == "passed" and sample["result"].get("metrics")]
    if not metrics:
        return {}
    fields = ("active_wall_s", "backend_eval_wall_s", "outside_backend_eval_wall_s",
              "model_call_wall_s", "outside_model_call_wall_s", "backend_non_model_wall_s",
              "outside_backend_eval_ms_per_action", "outside_backend_eval_ms_per_committed_token",
              "model_calls", "evaluated_input_positions")
    return {field: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for field in fields if (values := [metric[field] for metric in metrics if metric[field] is not None])}


def _readable(report: dict, path: Path) -> Path:
    counts = report["counts"]
    lines = [f"SPE real-model report: {report['profile']['name']}",
             f"Model: {report['model_path']}",
             f"Selected {counts['selected']}; executed {counts['executed']}; passed {counts['passed']}; failed {counts['failed']}; skipped {counts['skipped']}",
             "Times are elapsed wall time. Outside-backend includes sampling, ledger, and scenario work."]
    for name in report["profile"]["scenarios"]:
        cases = [sample for sample in report["samples"] if sample["scenario"] == name]
        status = ", ".join(sample["status"] for sample in cases) or "unexecuted"
        lines.append(f"\n{name}: {status}")
        metrics = report["summary"].get(name, {})
        for field in ("active_wall_s", "outside_backend_eval_wall_s", "model_call_wall_s", "evaluated_input_positions"):
            if field in metrics:
                lines.append(f"  {field}: median {metrics[field]['median']:.6g}; range {metrics[field]['min']:.6g}..{metrics[field]['max']:.6g}")
        for sample in cases:
            if sample["status"] == "failed":
                lines.append(f"  failure: {sample['error']}")
                partial = sample.get("partial_result", {})
                if "logit_trajectory" in partial:
                    lines.append(
                        f"  completed {partial['completed_actions']}/{partial['requested_actions']} continuation actions"
                    )
                    for point in partial["logit_trajectory"]:
                        lines.append(
                            f"  checkpoint {point['checkpoint']}: max error "
                            f"{point['max_abs_logit_error']:.4f}; top token "
                            f"{'agrees' if point['top_token_agrees'] else 'DIFFERS'}"
                        )
    if report.get("fatal"):
        lines.append(f"Fatal: {report['fatal']}")
    summary_path = path.with_suffix(".txt")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_profile(profile: dict, output_dir: Path) -> tuple[Path, bool]:
    report = {"format": FORMAT, "harness_version": HARNESS_VERSION,
              "scenario_fixture_version": 2,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "git": _git_identity(), "hardware": _hardware(),
              "profile": {key: value for key, value in profile.items() if key not in {"options", "launch_tokens", "model_path"}},
              "model_path": str(profile["model_path"]), "model": None, "runtime_identity": None,
              "preflight": None, "load_wall_s": None, "preflight_wall_s": None,
              "provenance_hash_wall_s": None, "samples": [], "summary": {},
              "counts": {"selected": len(profile["scenarios"]) * profile["repeats"],
                         "executed": 0, "passed": 0, "failed": 0, "skipped": 0},
              "timing_boundary": ("active interval starts before scenario prompt/context construction and ends after "
                                  "the specified final action; it includes runner observation, sampling, ledger, "
                                  "and in-process finalization. JSONL scenarios include parsing/export. "
                                  "Reference oracle, preflight, warmup, hashing, and report serialization are excluded. "
                                  "CLI outer wall includes interpreter/model load and shutdown. No human wait is timed."),
              "fine_boundary": ("llama.cpp: synchronous Python Llama.eval call, including native-library bookkeeping. "
                                "Transformers CPU: synchronous model forward call; output transfer and cache handling "
                                "remain inside backend service. Accelerator forward dispatch has no validated "
                                "completion interval and is null. No extra GPU sync is inserted. Input positions "
                                "are counts, not attention FLOPs.")}
    backend = None
    try:
        with deadline(profile["timeout_s"]):
            start = perf_counter()
            backend = load_backend(profile["options"])
            report["load_wall_s"] = perf_counter() - start
            start = perf_counter()
            report["preflight"] = _preflight(backend)
            report["preflight_wall_s"] = perf_counter() - start
            start = perf_counter()
            report["model"] = backend.provenance(include_model_sha256=True)
            report["runtime_identity"] = _runtime_identity(backend, profile)
            report["provenance_hash_wall_s"] = perf_counter() - start
        for _ in range(profile["warmup"]):
            with deadline(profile["timeout_s"]):
                continuation(backend, sampler_from_args(profile["options"]),
                             rtol=profile["rtol"], atol=profile["atol"])
        sampling = sampler_from_args(profile["options"])
        for name in profile["scenarios"]:
            for repeat in range(profile["repeats"]):
                sample = {"scenario": name, "repeat": repeat, "status": "failed"}
                report["counts"]["executed"] += 1
                try:
                    with deadline(profile["timeout_s"]):
                        sample["result"] = SCENARIOS[name](
                            backend, sampling, rtol=profile["rtol"], atol=profile["atol"],
                            profile=profile, model_path=profile["model_path"],
                            timeout_s=profile["timeout_s"], provenance=report["model"],
                        )
                    sample["status"] = "passed"
                    report["counts"]["passed"] += 1
                except Exception as exc:
                    sample["error_kind"] = "assertion" if isinstance(exc, AssertionError) else "inference"
                    sample["error"] = f"{type(exc).__name__}: {exc}"
                    if hasattr(exc, "partial_result"):
                        sample["partial_result"] = exc.partial_result
                    report["counts"]["failed"] += 1
                report["samples"].append(sample)
                report["summary"][name] = _summary([item for item in report["samples"] if item["scenario"] == name])
                interim = _save(report, output_dir)
                print(f"{profile['name']} {name} #{repeat + 1}: {sample['status']} ({interim})", flush=True)
    except Exception as exc:
        report["fatal"] = {"kind": "configuration" if backend is None else "preflight",
                           "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if backend is not None:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
            del backend
    path = _save(report, output_dir)
    summary_path = _readable(report, path)
    counts = report["counts"]
    success = not report.get("fatal") and counts["executed"] > 0 and counts["failed"] == 0 and counts["passed"] == counts["selected"]
    print(f"Report: {path}\nSummary: {summary_path}\nSelected {counts['selected']}, executed {counts['executed']}, passed {counts['passed']}, failed {counts['failed']}, skipped {counts['skipped']}")
    if report.get("fatal"):
        print(report["fatal"], file=sys.stderr)
    return path, success


def compare(first: Path, second: Path) -> int:
    left = json.loads(first.read_text(encoding="utf-8"))
    right = json.loads(second.read_text(encoding="utf-8"))
    mismatches = []
    for label, a, b in (
        ("format", left.get("format"), right.get("format")),
        ("harness_version", left.get("harness_version"), right.get("harness_version")),
        ("fixture version", left.get("scenario_fixture_version"), right.get("scenario_fixture_version")),
        ("model identity", (left.get("model") or {}).get("model_sha256"), (right.get("model") or {}).get("model_sha256")),
        ("runtime identity", left.get("runtime_identity"), right.get("runtime_identity")),
        ("backend", left["profile"].get("backend"), right["profile"].get("backend")),
        ("launch", left["profile"].get("launch"), right["profile"].get("launch")),
        ("tolerances", (left["profile"].get("rtol"), left["profile"].get("atol")),
         (right["profile"].get("rtol"), right["profile"].get("atol"))),
        ("hardware", left.get("hardware"), right.get("hardware")),
        ("scenarios", left["profile"].get("scenarios"), right["profile"].get("scenarios")),
        ("result status", left.get("counts", {}).get("failed"), right.get("counts", {}).get("failed")),
    ):
        if a != b or a is None:
            mismatches.append(label)
    if left.get("counts", {}).get("failed") or right.get("counts", {}).get("failed"):
        mismatches.append("failed checks")
    keys = ("visible_token_ids", "retained_token_ids", "replay_visible_token_ids",
            "resumed_visible_token_ids", "child_visible_token_ids", "surviving_token_ids",
            "final_text_sha256", "prompt_tokens", "replayed_actions")
    for name in left["profile"]["scenarios"]:
        a = [{key: result[key] for key in keys if key in result}
             for item in left["samples"] if item["scenario"] == name and item["status"] == "passed"
             for result in [item.get("result", {})]]
        b = [{key: result[key] for key in keys if key in result}
             for item in right["samples"] if item["scenario"] == name and item["status"] == "passed"
             for result in [item.get("result", {})]]
        if a != b:
            mismatches.append(f"{name} realized workload")
    if mismatches:
        print("Incompatible performance baselines: " + ", ".join(mismatches))
        print(f"Side by side reports: {first} | {second}")
        return 2
    for name in left["profile"]["scenarios"]:
        a = left["summary"].get(name, {}).get("outside_backend_eval_wall_s", {}).get("median")
        b = right["summary"].get(name, {}).get("outside_backend_eval_wall_s", {}).get("median")
        if a is not None and b is not None:
            change = ((b / a) - 1) * 100 if a else float("nan")
            print(f"{name}: outside backend evaluation {a * 1000:.3f} -> {b * 1000:.3f} ms ({change:+.1f}%)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--profile", type=Path, action="append", required=True)
    run.add_argument("--model-root", type=Path)
    run.add_argument("--model")
    run.add_argument("--backend", choices=("llama.cpp", "transformers"))
    run.add_argument("--scenario", action="append", choices=tuple(SCENARIOS))
    run.add_argument("--repeats", type=int)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    cmp = sub.add_parser("compare")
    cmp.add_argument("first", type=Path)
    cmp.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare(args.first, args.second)
    okay = True
    for path in args.profile:
        try:
            profile = load_profile(path, args.model_root, args.model, args.backend, args.scenario, args.repeats)
        except (ProfileError, EditorError) as exc:
            print(f"Configuration failure for {path}: {exc}", file=sys.stderr)
            failed = {"format": FORMAT, "harness_version": HARNESS_VERSION,
                      "scenario_fixture_version": 2,
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "profile": {"name": path.stem, "scenarios": args.scenario or []},
                      "model_path": args.model, "model": None, "samples": [], "summary": {},
                      "counts": {"selected": len(args.scenario or []), "executed": 0,
                                 "passed": 0, "failed": 0, "skipped": 0},
                      "fatal": {"kind": "configuration", "error": str(exc)}}
            artifact = _save(failed, args.output_dir)
            _readable(failed, artifact)
            print(f"Partial report: {artifact}", file=sys.stderr)
            return 2
        _, passed = run_profile(profile, args.output_dir)
        okay &= passed
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
