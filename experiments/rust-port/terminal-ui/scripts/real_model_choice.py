#!/usr/bin/env python3
"""Run two Rust Choice decisions through one real SPE inference worker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unicodedata
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "core/src"), str(ROOT)]

# This is the pre-0 context: its token-prefix SHA-256 is the stream's 256-bit
# identity, while generated-token sampling boundaries start at 0.
PROMPT = "A short list of everyday objects:"
SAMPLER = {
    "temperature": 0.8,
    "top_k": 5,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "tail_free_z": 1.0,
    "draw_kernel": "categorical",
    "gaussian_noise_std": 1.0,
    "perturb_noise_std": 1.0,
    "student_t_df": 3.0,
    "seed": 17,
    "gumbel_noise_address": "token-id",
    "gumbel_noise_scale": 1.0,
}
LAYOUT_PATH = ROOT / "experiments/rust-port/terminal-ui/fixtures/real-model-screen-layout.json"
WORKER_PATH = ROOT / "experiments/rust-port/terminal-ui/scripts/real_model_worker.py"
RUST_BINARY = ROOT / "experiments/rust-port/target/debug/rust-terminal-ui-smoke"


def _safe_text(value: str) -> str:
    return "".join(
        f"<U+{ord(character):04X}>"
        if unicodedata.category(character) == "Cc"
        else character
        for character in str(value)
    )


def _fit_line(value: str, width: int) -> str:
    from rich.cells import cell_len

    output = []
    used = 0
    for character in _safe_text(value):
        cells = cell_len(character)
        if used + cells > width:
            break
        output.append(character)
        used += cells
    return "".join(output) + " " * (width - used)


def expected_real_frame(checkpoint: str, decision_index: int, size: list[int],
                        profile_name: str, decision: dict, layout: dict) -> dict:
    """Build expected cells from the Python backend transcript and oracle."""
    from rich.cells import cell_len

    width, height = size
    lines = [" " * width for _ in range(height)]
    title = f"Rust Choice / real model / turn {decision_index + 1}/2"
    if checkpoint == "choice.real.ready":
        rows = layout["ready_rows"]
        lines[rows["title"]] = title
        lines[rows["profile"]] = f"Profile: {profile_name}"
        lines[rows["boundary"]] = f"Context @ boundary {decision['boundary']}"
        lines[rows["context"]] = f"Context: {decision['context_text']}"
        lines[rows["candidate_header"]] = "Rank  Token  Text"
        for offset, candidate in enumerate(decision["candidates"]):
            marker = ">" if candidate["token_id"] == decision["proposal_token_id"] else " "
            suffix = " [EOG]" if candidate["is_eog"] else ""
            candidate_text = _safe_text(candidate["text"].lstrip())
            lines[rows["candidate_start"] + offset] = (
                f"{marker} {candidate['raw_rank']:>5} {candidate['token_id']:>6}  "
                f"\"{candidate_text}\"{suffix}"
            )
        proposal = next(
            item for item in decision["candidates"]
            if item["token_id"] == decision["proposal_token_id"]
        )
        lines[rows["proposal"]] = (
            f"Proposal: rank {proposal['raw_rank']} / token {proposal['token_id']} / "
            f"\"{_safe_text(proposal['text'])}\""
        )
        lines[rows["input"]] = "Choice >"
        lines[rows["help"]] = "Enter accepts · q exits"
        cursor = list(layout["cursor"]["ready"])
    elif checkpoint == "choice.real.accepted":
        rows = layout["accepted_rows"]
        accepted = decision["accepted_token"]
        lines[rows["title"]] = title
        lines[rows["profile"]] = f"Profile: {profile_name}"
        lines[rows["boundary"]] = f"Context @ boundary {decision['boundary_after']}"
        lines[rows["context"]] = f"Context: {decision['accepted_context_text']}"
        lines[rows["action"]] = "Action: accept"
        lines[rows["accepted_token"]] = (
            f"Accepted token: {accepted['token_id']} / \"{_safe_text(accepted['text'])}\""
        )
        agreement = "yes" if accepted["token_id"] == decision["proposal_token_id"] else "no"
        lines[rows["evidence"]] = (
            f"Evidence: boundary {decision['boundary']} / sampling {decision['boundary']} / "
            f"agreement {agreement}"
        )
        lines[rows["visible_ids"]] = (
            "Visible token IDs: "
            + ", ".join(str(value) for value in decision["visible_token_ids"])
        )
        lines[rows["committed"]] = "Choice committed"
        lines[rows["help"]] = "Press q to exit"
        cursor = layout["cursor"]["accepted"]
    else:
        raise AssertionError(f"unexpected real-model frame checkpoint {checkpoint!r}")

    lines = [_fit_line(line, width) for line in lines]
    assert len(lines) == height and all(cell_len(line) == width for line in lines)
    return {"size": size, "lines": lines, "cursor": cursor}


def _assert_expected(actual: dict, expected: dict, label: str) -> None:
    from rich.cells import cell_len

    assert actual["size"] == expected["size"], f"{label}: size differs"
    width, height = expected["size"]
    assert len(actual["lines"]) == height, f"{label}: actual row count differs"
    assert all(cell_len(row) == width for row in actual["lines"]), (
        f"{label}: Rust frame does not contain complete cell rows"
    )
    assert [line.rstrip() for line in actual["lines"]] == [
        line.rstrip() for line in expected["lines"]
    ], f"{label}: cell grid differs"
    assert actual["cursor"] == expected["cursor"], f"{label}: cursor differs"


def verify_pty_frames(raw: bytes, frames: list[dict], decisions: list[dict],
                      profile_name: str, layout: dict) -> bool:
    import pyte

    assert frames, "PTY capture has no frames"
    screen = pyte.Screen(*frames[0]["size"])
    stream = pyte.ByteStream(screen)
    position = 0
    for frame in frames:
        width, height = frame["size"]
        if (screen.columns, screen.lines) != (width, height):
            screen.resize(height, width)
        checkpoint = frame.get("checkpoint")
        index = frame.get("decision_index")
        if type(index) is not int or not 0 <= index < len(decisions):
            raise AssertionError(f"frame has invalid decision index {index!r}")
        expected = expected_real_frame(
            checkpoint, index, [width, height], profile_name, decisions[index], layout
        )
        _assert_expected(frame, expected, f"frame {frame['sequence']} report")
        stream.feed(raw[position:frame["end_offset"]])
        position = frame["end_offset"]
        got_lines = [line.rstrip() for line in screen.display]
        want_lines = [line.rstrip() for line in expected["lines"]]
        assert got_lines == want_lines, (
            f"pyte screen differs from independent oracle at frame {frame['sequence']}"
        )
        if expected["cursor"] is None:
            assert screen.cursor.hidden, f"frame {frame['sequence']}: cursor should be hidden"
        else:
            assert (screen.cursor.x, screen.cursor.y) == tuple(expected["cursor"])
            assert not screen.cursor.hidden

    # Negative control: deleting the oracle-derived proposal row must fail.
    ready_frame = next(frame for frame in frames
                       if frame["checkpoint"] == "choice.real.ready")
    ready_index = ready_frame["decision_index"]
    broken = expected_real_frame(
        "choice.real.ready", ready_index, ready_frame["size"], profile_name,
        decisions[ready_index], layout,
    )
    proposal_row = layout["ready_rows"]["proposal"]
    broken["lines"][proposal_row] = " " * ready_frame["size"][0]
    try:
        _assert_expected(ready_frame, broken, "negative control")
    except AssertionError:
        return True
    raise AssertionError("negative control did not reject a missing proposal row")


def _load_json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _wait_for_frame(ui, predicate, timeout: float = 30.0, what: str = "frame") -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ui.pump(0.02)
        done, status = os.waitpid(ui.pid, os.WNOHANG)
        if done:
            ui.status = status
            ui.pump(0.2)
            raise RuntimeError(
                f"Rust terminal process exited before {what} (wait status {status})"
            )
        try:
            frames = ui.frames()
        except json.JSONDecodeError:
            # Frame JSON is written while the PTY child is live; retry if a
            # reader catches the file between its payload and trailing newline.
            continue
        if frames and predicate(frames[-1]):
            return frames[-1]
    try:
        frames = ui.frames()
        last = "\n".join(frames[-1]["lines"]) if frames else "(no frame)"
    except json.JSONDecodeError:
        last = "(frame log ended during a JSON write)"
    raise AssertionError(f"timed out waiting for {what}; last frame:\n{last}")


def _python_sampler():
    from trajectory_editor.core.sampler_config import SamplerConfig

    return SamplerConfig(**SAMPLER)


def _replay_full_prefix(backend, root_prefix: list[int], prefix: list[int]) -> None:
    """Build a fresh backend state using the live prefill/incremental split.

    Some quantized llama.cpp kernels produce slightly different logits when a
    generated token is evaluated in a one-token incremental batch versus being
    folded into the original prompt batch. Replaying the exact root prefix and
    each generated token separately checks the same backend path on a fresh
    context while retaining the exact complete prefix.
    """
    if not root_prefix or prefix[:len(root_prefix)] != root_prefix:
        raise AssertionError("fresh-prefix replay does not begin with the exact root prefix")
    backend.reset(root_prefix)
    for token_id in prefix[len(root_prefix):]:
        backend.eval([token_id])


def _oracle_decision(backend, prefix: list[int], boundary: int, fingerprint: str,
                     logits=None, replay_root: list[int] | None = None) -> dict:
    import numpy as np

    from trajectory_editor.core.sampling import (
        SparseDistribution,
        _softmax,
        apply_candidate_filter,
        draw_token,
        raw_rank,
    )

    sampler = _python_sampler()
    if logits is None:
        if replay_root is None:
            backend.reset(prefix)
        else:
            _replay_full_prefix(backend, replay_root, prefix)
        logits = backend.last_logits()
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or len(logits) != backend.vocabulary_size() or not np.all(np.isfinite(logits)):
        raise AssertionError(
            f"fresh-prefix logits must be finite with shape ({backend.vocabulary_size()},), got {logits.shape}"
        )
    filtered = apply_candidate_filter(logits, sampler)
    candidate_ids = filtered.stages["after_min_p"]
    scores = filtered.scaled_logits[candidate_ids]
    probabilities = _softmax(scores)
    distribution = SparseDistribution(candidate_ids, probabilities, scores)
    proposal = draw_token(
        distribution,
        seed=sampler.seed,
        stream_fingerprint=fingerprint,
        aligned_step=boundary,
        kernel=sampler.draw_kernel,
        gumbel_noise_address=sampler.gumbel_noise_address,
        gumbel_noise_scale=sampler.gumbel_noise_scale,
    )
    candidates = [
        {
            "token_id": int(token_id),
            "raw_rank": raw_rank(logits, int(token_id)),
            "text": backend.token_text(int(token_id)),
            "probability": float(probabilities[index]),
            "is_eog": bool(backend.is_eog(int(token_id))),
        }
        for index, token_id in enumerate(candidate_ids)
    ]
    return {
        "prefix_token_ids": list(prefix),
        "boundary": boundary,
        "stream_fingerprint": fingerprint,
        "context_text": backend.render(prefix),
        "proposal_token_id": int(proposal),
        "candidate_token_ids": [item["token_id"] for item in candidates],
        "candidate_raw_ranks": [item["raw_rank"] for item in candidates],
        "candidate_texts": [item["text"] for item in candidates],
        "candidate_probabilities": [item["probability"] for item in candidates],
        "candidate_is_eog": [item["is_eog"] for item in candidates],
        "candidates": candidates,
        "accepted_context_text": backend.render([*prefix, int(proposal)]),
        "fresh_logits": logits,
    }


def compare_decisions(backend, semantic: list[dict], metadata: dict,
                      capture_dir: Path) -> list[dict]:
    import numpy as np

    from trajectory_editor.episode_hash import token_prefix_sha256

    if len(semantic) != 2:
        raise AssertionError(f"expected two accepted Rust decisions, got {len(semantic)}")
    root = backend.tokenize(PROMPT, add_bos=True, special=True)
    if not root:
        raise AssertionError("Python oracle tokenizer returned an empty root prefix")
    root = [int(value) for value in root]
    fingerprint = token_prefix_sha256(root)
    if root != semantic[0]["root_token_ids"]:
        raise AssertionError("Rust and Python worker tokenized different root token IDs")
    if metadata.get("tokenizer_id") != backend.tokenizer_id():
        raise AssertionError("fresh Python oracle tokenizer identity differs from the worker")

    comparisons = []
    selected_prefix = root.copy()
    selected_visible_token_ids = []
    for index, record in enumerate(semantic):
        prefix = selected_prefix.copy()
        boundary = index
        if record.get("decision_index") != index:
            raise AssertionError(f"Rust decision index mismatch at {index}")
        if record.get("prefix_token_ids") != prefix:
            raise AssertionError(f"Rust decision {index} used a different full prefix")
        if record.get("root_token_ids") != root:
            raise AssertionError(f"Rust decision {index} changed the root token IDs")
        if record.get("stream_fingerprint") != fingerprint:
            raise AssertionError(f"Rust decision {index} used a different root fingerprint")
        if record.get("sampling_boundary") != boundary:
            raise AssertionError(f"Rust decision {index} used the wrong sampling boundary")
        if record.get("boundary_before") != boundary:
            raise AssertionError(f"Rust decision {index} used the wrong visible boundary")
        if record.get("boundary_after") != boundary + 1:
            raise AssertionError(f"Rust decision {index} advanced to the wrong visible boundary")
        evidence = record.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("boundary") != boundary:
            raise AssertionError(f"Rust decision {index} recorded evidence at the wrong boundary")
        if record.get("sampler") != SAMPLER:
            raise AssertionError(f"Rust decision {index} used unexpected sampler settings")
        if record.get("action") != {"kind": "accept"}:
            raise AssertionError(f"Rust decision {index} did not accept its proposal")

        live_path = capture_dir / f"live-boundary-{boundary}.f64le"
        if not live_path.is_file():
            raise AssertionError(f"Rust did not preserve live logits for boundary {boundary}")
        live_logits = np.fromfile(live_path, dtype="<f8")
        live_reference = _oracle_decision(
            backend, prefix, boundary, fingerprint, logits=live_logits
        )
        if record.get("decision_context_text") != live_reference["context_text"]:
            raise AssertionError(f"Rust decision {index} displayed a different backend context")
        for key in (
            "candidate_token_ids", "candidate_raw_ranks", "candidate_texts", "candidate_is_eog"
        ):
            if record.get(key) != live_reference[key]:
                raise AssertionError(f"Rust decision {index} differs from Python in {key}")
        if not np.allclose(
            record.get("candidate_probabilities", []),
            live_reference["candidate_probabilities"],
            rtol=1e-12,
            atol=1e-14,
        ):
            raise AssertionError(f"Rust decision {index} candidate probabilities differ")
        if record.get("proposal_token_id") != live_reference["proposal_token_id"]:
            raise AssertionError(f"Rust decision {index} proposal differs from Python sampler")
        if record.get("selected_token_id") != live_reference["proposal_token_id"]:
            raise AssertionError(f"Rust decision {index} accepted a token other than the proposal")

        fresh_reference = _oracle_decision(
            backend, prefix, boundary, fingerprint, replay_root=root
        )
        fresh_logits = fresh_reference.pop("fresh_logits")
        if fresh_reference["proposal_token_id"] != record.get("selected_token_id"):
            raise AssertionError(
                f"fresh-prefix decision at boundary {boundary} sampled "
                f"{fresh_reference['proposal_token_id']} instead of Rust token "
                f"{record.get('selected_token_id')}"
            )
        if fresh_reference["context_text"] != live_reference["context_text"]:
            raise AssertionError(f"fresh-prefix context changed at boundary {boundary}")
        if live_logits.shape != fresh_logits.shape:
            raise AssertionError(f"live and fresh logits have different shapes at boundary {boundary}")
        difference = np.abs(live_logits - fresh_logits)
        live_top = np.argsort(-live_logits, kind="stable")[:10].astype(int).tolist()
        fresh_top = np.argsort(-fresh_logits, kind="stable")[:10].astype(int).tolist()
        selected_token_id = int(record["selected_token_id"])
        accepted_token = next(
            candidate for candidate in live_reference["candidates"]
            if candidate["token_id"] == selected_token_id
        )
        reference = live_reference
        reference.pop("fresh_logits", None)
        reference["accepted_context_text"] = backend.render([*prefix, selected_token_id])
        reference.update({
            "decision_index": index,
            "selected_token_id": selected_token_id,
            "selected_token_agrees": selected_token_id == reference["proposal_token_id"],
            "accepted_token": accepted_token,
            "boundary_after": boundary + 1,
            "visible_token_ids": [*selected_visible_token_ids, selected_token_id],
            "logit_diagnostics": {
                "max_abs_logit_delta": float(np.max(difference)),
                "mean_abs_logit_delta": float(np.mean(difference)),
                "live_top_token_id": int(np.argmax(live_logits)),
                "fresh_top_token_id": int(np.argmax(fresh_logits)),
                "top_token_agrees": int(np.argmax(live_logits)) == int(np.argmax(fresh_logits)),
                "live_top_10_token_ids": live_top,
                "fresh_top_10_token_ids": fresh_top,
                "top_10_order_agrees": live_top == fresh_top,
                "fresh_prefix_candidate_order_agrees": (
                    fresh_reference["candidate_token_ids"]
                    == live_reference["candidate_token_ids"]
                ),
                "fresh_prefix_candidate_probabilities_max_abs_delta": float(np.max(np.abs(
                    np.asarray(fresh_reference["candidate_probabilities"], dtype=np.float64)
                    - np.asarray(live_reference["candidate_probabilities"], dtype=np.float64)
                ))),
                "fresh_prefix_proposal_token_id": fresh_reference["proposal_token_id"],
                "live_logits_bytes": live_path.stat().st_size,
            },
        })
        comparisons.append(reference)
        selected_visible_token_ids.append(selected_token_id)
        selected_prefix.append(selected_token_id)
    return comparisons


def _artifact(path: Path, root: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run(args) -> tuple[Path, bool]:
    from benchmarks.real_model import load_profile
    from tests.core.test_live_terminal_pty import Session

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    run_dir.mkdir()
    paths = {
        "frames": run_dir / "frames.jsonl",
        "semantic": run_dir / "semantic.jsonl",
        "raw_pty": run_dir / "pty.raw",
        "worker_stderr": run_dir / "worker.stderr.log",
        "worker_metadata": run_dir / "worker-metadata.json",
        "protocol_metrics": run_dir / "protocol-metrics.json",
        "capture_dir": run_dir / "live-logits",
        "report": run_dir / "report.json",
    }
    report = {
        "format": "spe-rust-real-model-choice-v1",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "profile": str(args.profile.resolve()),
        "model_root": str(args.model_root.resolve()),
        "sampler": SAMPLER,
        "prompt": PROMPT,
        "protocol_version": 1,
        "decisions": [],
        "fresh_prefix_comparisons": [],
        "operation_metrics": None,
        "failures": [],
        "artifacts": {},
    }
    ui = None
    try:
        if not RUST_BINARY.is_file():
            raise FileNotFoundError(
                f"Rust terminal binary not found at {RUST_BINARY}; build it with the documented cargo command"
            )
        profile = load_profile(args.profile, args.model_root, None, None, None, None)
        report.update({
            "model_path": str(profile["model_path"]),
            "backend": profile["backend"],
            "profile_name": profile["name"],
        })
        paths["capture_dir"].mkdir()
        env = {
            "SPE_REAL_MODEL_PYTHON": sys.executable,
            "SPE_REAL_MODEL_WORKER": str(WORKER_PATH),
            "SPE_REAL_MODEL_PROFILE": str(profile["path"]),
            "SPE_REAL_MODEL_ROOT": str(args.model_root.resolve()),
            "SPE_REAL_MODEL_CAPTURE_DIR": str(paths["capture_dir"]),
            "SPE_REAL_MODEL_METRICS_PATH": str(paths["protocol_metrics"]),
            "SPE_REAL_MODEL_METADATA_PATH": str(paths["worker_metadata"]),
            "SPE_REAL_MODEL_WORKER_STDERR": str(paths["worker_stderr"]),
        }
        ui = Session(
            run_dir,
            "rust-real-model-choice",
            size=(100, 30),
            env=env,
            argv=[str(RUST_BINARY), "--real-model"],
        )
        timeout = max(profile["timeout_s"] + 120, 180)
        _wait_for_frame(
            ui,
            lambda frame: frame.get("checkpoint") == "choice.real.ready"
            and frame.get("decision_index") == 0,
            timeout=timeout,
            what="first real-model Choice",
        )
        ui.send("\r")
        _wait_for_frame(
            ui,
            lambda frame: frame.get("checkpoint") == "choice.real.ready"
            and frame.get("decision_index") == 1,
            timeout=profile["timeout_s"],
            what="second real-model Choice",
        )
        ui.resize(80, 24)
        _wait_for_frame(
            ui,
            lambda frame: frame.get("checkpoint") == "choice.real.ready"
            and frame.get("decision_index") == 1 and frame["size"] == [80, 24],
            what="real-model Choice compact resize",
        )
        ui.resize(100, 30)
        _wait_for_frame(
            ui,
            lambda frame: frame.get("checkpoint") == "choice.real.ready"
            and frame.get("decision_index") == 1 and frame["size"] == [100, 30],
            what="real-model Choice restored resize",
        )
        ui.send("\r")
        _wait_for_frame(
            ui,
            lambda frame: frame.get("checkpoint") == "choice.real.accepted"
            and frame.get("decision_index") == 1,
            timeout=profile["timeout_s"],
            what="second accepted real-model Choice",
        )
        ui.send("q")
        process_output = ui.finish(timeout=profile["timeout_s"] + 30)
        if not (os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0):
            raise RuntimeError("Rust real-model process failed: " + process_output[-2000:])

        raw = bytes(ui.output)
        paths["raw_pty"].write_bytes(raw)
        frames = ui.frames()
        semantic = _load_json_lines(ui.semantic_path)
        metadata = json.loads(paths["worker_metadata"].read_text(encoding="utf-8"))
        report["worker_metadata"] = metadata
        report["operation_metrics"] = json.loads(
            paths["protocol_metrics"].read_text(encoding="utf-8")
        )
        from trajectory_editor.episode_backend_loader import load_backend

        # The live worker has exited. A fresh backend instance now replays both
        # exact prefixes, so this oracle cannot alter the incremental live path.
        oracle_backend = load_backend(profile["options"])
        try:
            comparisons = compare_decisions(
                oracle_backend, semantic, metadata, paths["capture_dir"]
            )
        finally:
            close = getattr(oracle_backend, "close", None)
            if callable(close):
                close()
        report["decisions"] = semantic
        report["fresh_prefix_comparisons"] = comparisons
        layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
        negative_control_passed = verify_pty_frames(
            raw, frames, comparisons, profile["name"], layout
        )
        if not any(frame["size"] == [80, 24] for frame in frames):
            raise AssertionError("PTY path did not capture the compact resize")
        if not any(frame["size"] == [100, 30] for frame in frames):
            raise AssertionError("PTY path did not capture the restored resize")
        provenance = metadata.get("provenance", {})
        report["model_identity"] = {
            "model_path": str(profile["model_path"]),
            "model_id": provenance.get("model_id"),
            "model_sha256": provenance.get("model_sha256"),
            "tokenizer_id": metadata.get("tokenizer_id"),
            "vocabulary_size": metadata.get("vocabulary_size"),
        }
        report["screen_oracle"] = {
            "frames_checked": len(frames),
            "negative_control_rejected_missing_proposal": negative_control_passed,
            "sizes": [frame["size"] for frame in frames],
        }
        report["status"] = "passed"
    except BaseException as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        if ui is not None:
            if ui.status is None:
                try:
                    os.kill(ui.pid, signal.SIGKILL)
                    os.waitpid(ui.pid, 0)
                except OSError:
                    pass
            ui.pump(0.2)
            paths["raw_pty"].write_bytes(bytes(ui.output))
            report["decisions"] = _load_json_lines(ui.semantic_path)
            report["frames"] = ui.frames()
        if paths["worker_metadata"].is_file():
            report["worker_metadata"] = json.loads(
                paths["worker_metadata"].read_text(encoding="utf-8")
            )
        if paths["protocol_metrics"].is_file():
            report["operation_metrics"] = json.loads(
                paths["protocol_metrics"].read_text(encoding="utf-8")
            )
    finally:
        for name, path in paths.items():
            if name != "capture_dir":
                report["artifacts"][name] = (
                    path.name if name == "report" else _artifact(path, run_dir)
                )
        if paths["capture_dir"].is_dir():
            report["artifacts"]["live_logits"] = [
                _artifact(path, run_dir)
                for path in sorted(paths["capture_dir"].glob("*.f64le"))
            ]
        paths["report"].write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    return paths["report"], report["status"] == "passed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "spe-rust-real-model-choice",
    )
    args = parser.parse_args(argv)
    report_path, passed = run(args)
    print(f"Real-model Choice {'passed' if passed else 'failed'}: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
