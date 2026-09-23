"""Bounded real inference scenarios. Imported only by the opt-in harness."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np

from trajectory_editor.chord import ActionSequencePolicy
from trajectory_editor.core.actions import Accept, Hold, Write
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_runner import LiveSessionRunner
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.teacher_plan import export_live_teacher_tape, load_teacher_tape_jsonl

from benchmarks.real_model_metrics import Measurement


PROMPT = "A short list of everyday objects:"
ACTION_RECORDS = (
    {"step": 0, "action": {"kind": "write", "text": " The next item is", "mode": "exact"}},
    {"step": 1, "action": {"kind": "accept"}},
    {"step": 2, "action": {"kind": "hold", "limit": 2}},
)


class ScenarioFailure(AssertionError):
    def __init__(self, message: str, partial_result: dict) -> None:
        super().__init__(message)
        self.partial_result = partial_result


class LogitMismatch(AssertionError):
    def __init__(self, diagnostics: dict) -> None:
        super().__init__(f"same-prefix logits differ; max absolute error {diagnostics['max_abs_logit_error']:g}")
        self.diagnostics = diagnostics


def _session(backend, sampling, *, prompt: str = PROMPT, max_tokens: int = 64) -> LiveSession:
    return LiveSession(
        EpisodeEngine(backend, sampling=sampling, max_tokens=max_tokens, initial_text=prompt),
        prompt=prompt,
    )


def _oracle(backend, prefix: list[int], observed: np.ndarray, *, rtol: float, atol: float) -> dict:
    # Direct fresh full-prefix inference is independent of runner/replay logic.
    backend.reset(prefix)
    fresh = np.asarray(backend.last_logits(), dtype=np.float64)
    actual = np.asarray(observed, dtype=np.float64)
    if not np.allclose(actual, fresh, rtol=rtol, atol=atol):
        difference = np.abs(actual - fresh)
        raise LogitMismatch({"max_abs_logit_error": float(np.max(difference)),
                             "mean_abs_logit_error": float(np.mean(difference)),
                             "argmax_equal": bool(np.argmax(actual) == np.argmax(fresh)),
                             "observed_top_token_id": int(np.argmax(actual)),
                             "fresh_top_token_id": int(np.argmax(fresh))})
    return {"max_abs_logit_error": float(np.max(np.abs(actual - fresh)))}


def compare_prefix_trajectory(backend, captures, *, rtol: float, atol: float) -> list[dict]:
    """Compare every captured incremental prefix with direct fresh evaluation.

    Captures come from an uninterrupted production run. Reference resets happen
    afterward, so the oracle cannot perturb the measured continuation.
    """
    diagnostics = []
    for index, (prefix, observed, committed_token_id) in enumerate(captures):
        backend.reset(list(prefix))
        fresh = np.asarray(backend.last_logits(), dtype=np.float64)
        actual = np.asarray(observed, dtype=np.float64)
        if fresh.shape != actual.shape or not np.all(np.isfinite(fresh)):
            raise AssertionError(f"invalid fresh logits at checkpoint {index}: {fresh.shape}")
        difference = np.abs(actual - fresh)
        observed_top = int(np.argmax(actual))
        fresh_top = int(np.argmax(fresh))
        diagnostics.append({
            "checkpoint": index,
            "prefix_tokens": len(prefix),
            "committed_token_id": committed_token_id,
            "next_committed_token_id": captures[index + 1][2] if index + 1 < len(captures) else None,
            "max_abs_logit_error": float(np.max(difference)),
            "mean_abs_logit_error": float(np.mean(difference)),
            "within_tolerance": bool(np.allclose(actual, fresh, rtol=rtol, atol=atol)),
            "observed_top_token_id": observed_top,
            "fresh_top_token_id": fresh_top,
            "top_token_agrees": observed_top == fresh_top,
        })
    return diagnostics


@contextmanager
def _no_database():
    def blocked(*_args, **_kwargs):
        raise AssertionError("database access attempted in a persistence-free scenario")

    from trajectory_editor.episode_store import EpisodeStore
    with patch.object(EpisodeStore, "__init__", blocked), patch.object(sqlite3, "connect", blocked):
        yield


def continuation(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    meter = Measurement()
    with meter.attach(backend), meter.active():
        with meter.phase("initial_prompt"):
            session = _session(backend, sampling)
        with meter.phase("steady_continuation"):
            result = LiveSessionRunner(session).run(
                live_policy=ActionSequencePolicy([Accept(), Accept()]), max_live_actions=2
            )
        prefix = [*session.engine.initial_token_ids, *session.engine.visible_token_ids]
        logits = backend.last_logits().copy()
        generated = len(session.engine.visible_token_ids)
        with meter.phase("finalization"):
            session.quit()
    if len(result.outcomes) != 2 or generated != 2:
        raise AssertionError(f"continuation stopped early: {len(result.outcomes)} actions, {generated} tokens")
    metrics = meter.result(actions=2, committed_tokens=generated)
    return {"metrics": metrics, "prompt_tokens": len(session.engine.initial_token_ids),
            "visible_token_ids": prefix[len(session.engine.initial_token_ids):],
            "oracle": _oracle(backend, prefix, logits, rtol=rtol, atol=atol)}


def controlled_write(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    del rtol, atol
    # The actions are fixed before measuring either code version. No generated
    # text is substituted into this controlled-work comparison.
    from trajectory_editor.run_loop import ReplayPlan, TapeStep
    actions = (Write(" An apple", mode="exact"), Write(" and a pear", mode="exact"))
    meter = Measurement()
    with meter.attach(backend), meter.active():
        with meter.phase("initial_prompt"):
            session = _session(backend, sampling)
        with meter.phase("controlled_actions"):
            result = LiveSessionRunner(session, divergence_policy="ballistic").run(
                tape=ReplayPlan(tuple(TapeStep(action, None) for action in actions))
            )
        ledger = tuple(session.engine.visible_token_ids)
        with meter.phase("finalization"):
            session.quit()
    if result.replayed_actions != 2 or not result.replay_exhausted or not ledger:
        raise AssertionError("controlled write workload was not fully executed")
    return {"metrics": meter.result(actions=2, committed_tokens=len(ledger)),
            "visible_token_ids": ledger, "prompt_tokens": len(session.engine.initial_token_ids)}


def candidate_refresh(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    del rtol, atol
    meter = Measurement()
    with meter.attach(backend), meter.active():
        with meter.phase("initial_prompt"):
            session = _session(backend, sampling)
        with meter.phase("candidate_refresh"):
            initial = session.engine.observe()
            before = session.engine.candidates(initial, count=12)
            result = LiveSessionRunner(session).run(
                live_policy=ActionSequencePolicy([Accept()]), max_live_actions=1
            )
            after = session.engine.candidates(session.engine.observe(), count=12)
        ledger = tuple(session.engine.visible_token_ids)
        with meter.phase("finalization"):
            session.quit()
    if len(result.outcomes) != 1 or not before or not after:
        raise AssertionError("candidate refresh did not traverse the real observation path")
    return {"metrics": meter.result(actions=1, committed_tokens=len(ledger)),
            "visible_token_ids": ledger, "candidate_counts": [len(before), len(after)],
            "ui_mode": "headless-candidate-preparation"}


def instrumentation_parity(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    meter = Measurement()
    with meter.attach(backend), meter.active():
        measured = _session(backend, sampling)
        measured_run = LiveSessionRunner(measured).run(
            live_policy=ActionSequencePolicy([Accept(), Accept()]), max_live_actions=2
        )
        measured_tokens = tuple(measured.engine.visible_token_ids)
        measured_logits = backend.last_logits().copy()
        measured.quit()
    metrics = meter.result(actions=2, committed_tokens=len(measured_tokens))
    start = perf_counter()
    ordinary = _session(backend, sampling)
    ordinary_run = LiveSessionRunner(ordinary).run(
        live_policy=ActionSequencePolicy([Accept(), Accept()]), max_live_actions=2
    )
    ordinary_tokens = tuple(ordinary.engine.visible_token_ids)
    ordinary_logits = backend.last_logits().copy()
    ordinary.quit()
    ordinary_wall = perf_counter() - start
    if len(measured_run.outcomes) != 2 or len(ordinary_run.outcomes) != 2:
        raise AssertionError("measurement parity continuation stopped early")
    if measured_tokens != ordinary_tokens or not np.allclose(measured_logits, ordinary_logits, rtol=rtol, atol=atol):
        raise AssertionError("measurement changed semantic token or logit results")
    return {"metrics": metrics, "ordinary_active_wall_s": ordinary_wall,
            "visible_token_ids": measured_tokens,
            "timing_note": "Single alternating pair is diagnostic only; no ratio threshold"}


def action_jsonl(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    del rtol, atol
    with TemporaryDirectory(prefix="spe-real-plan-") as directory:
        path = Path(directory) / "handwritten.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in ACTION_RECORDS), encoding="utf-8")
        meter = Measurement()
        with _no_database(), meter.attach(backend), meter.active():
            with meter.phase("jsonl_parse"):
                tape = load_teacher_tape_jsonl(path)
            with meter.phase("initial_prompt"):
                session = _session(backend, sampling)
            with meter.phase("teacher_plan"):
                result = LiveSessionRunner(session, divergence_policy="ballistic").run(tape=tape.plan)
            ledger = tuple(session.engine.visible_token_ids)
            live_edge = not session.engine.ended and not session.engine.checkpointed
            with meter.phase("finalization"):
                session.quit()
        if result.replayed_actions != len(ACTION_RECORDS) or len(result.outcomes) != len(ACTION_RECORDS):
            raise AssertionError(f"action plan incomplete: replayed {result.replayed_actions}/{len(ACTION_RECORDS)}")
        if not result.replay_exhausted or result.handed_off or not live_edge:
            raise AssertionError("plan did not return to the expected usable live edge")
        if tuple(token for outcome in result.outcomes for token in outcome.visible_token_ids) != ledger:
            raise AssertionError("outcome token ledger does not match live edge")
        if not ledger or any(outcome.status != "completed" for outcome in result.outcomes):
            raise AssertionError("action plan did not complete with a visible ledger")
        return {"metrics": meter.result(actions=len(result.outcomes), committed_tokens=len(ledger)),
                "replayed_actions": result.replayed_actions,
                "stop_reasons": [outcome.stop_reason for outcome in result.outcomes],
                "visible_token_ids": ledger, "live_edge": live_edge}


def observed_jsonl(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    del rtol, atol
    with TemporaryDirectory(prefix="spe-real-observed-") as directory:
        path = Path(directory) / "export.jsonl"
        meter = Measurement()
        with _no_database(), meter.attach(backend), meter.active():
            with meter.phase("initial_prompt"):
                source = _session(backend, sampling)
            with meter.phase("source_generation"):
                made = LiveSessionRunner(source).run(
                    live_policy=ActionSequencePolicy([Write(" An apple", mode="exact"), Accept()]),
                    max_live_actions=2,
                )
            if len(made.outcomes) != 2:
                raise AssertionError("source session ended before observed export")
            source_ledger = tuple(source.engine.visible_token_ids)
            with meter.phase("jsonl_export_parse"):
                export_live_teacher_tape(source, path)
                source.discard()
                tape = load_teacher_tape_jsonl(path, require_observations=True)
            with meter.phase("restore_prompt"):
                dest = _session(backend, sampling)
            with meter.phase("teacher_plan"):
                replay = LiveSessionRunner(dest, divergence_policy="handoff").run(tape=tape.plan)
            dest_ledger = tuple(dest.engine.visible_token_ids)
            live_edge = not dest.engine.ended and not dest.engine.checkpointed
            with meter.phase("finalization"):
                dest.quit()
        if replay.handed_off or replay.replayed_actions != len(tape.plan):
            raise AssertionError("same-model observed plan failed to replay")
        if source_ledger != dest_ledger or len(tape.plan) != 2 or not live_edge:
            raise AssertionError("observed replay changed the surviving prefix or edge")
        # Deliberate divergence at a valid token ID, never model randomness.
        first = tape.plan.steps[0]
        from trajectory_editor.core.results import ReplayExpectation
        from trajectory_editor.run_loop import ReplayPlan, TapeStep
        expected = first.expectation
        assert expected is not None and expected.token_ids
        changed = (expected.token_ids[0] + 1) % backend.vocabulary_size()
        altered = ReplayPlan(steps=(TapeStep(first.action, ReplayExpectation(
            (changed, *expected.token_ids[1:]), expected.terminal_token_id, expected.stop_reason
        )), *tape.plan.steps[1:]), follow_source_sampling=False)
        with _no_database():
            divergent = _session(backend, sampling)
            handoff = LiveSessionRunner(divergent, divergence_policy="handoff").run(tape=altered)
            usable = not divergent.engine.ended and not divergent.engine.checkpointed
            divergent.discard()
        if (not handoff.handed_off or handoff.replayed_actions != 0 or not usable
                or len(handoff.outcomes) != 1
                or handoff.outcomes[0].status != "handed-off"
                or handoff.outcomes[0].boundary_before != 0
                or handoff.outcomes[0].divergence is None):
            raise AssertionError("deliberately altered expectation did not hand off at first step")
        return {"metrics": meter.result(actions=len(made.outcomes) + len(replay.outcomes),
                                        committed_tokens=len(dest_ledger)),
                "source_visible_token_ids": source_ledger,
                "replay_visible_token_ids": dest_ledger,
                "replayed_actions": replay.replayed_actions,
                "deliberate_handoff_boundary": handoff.outcomes[0].boundary_before,
                "deliberate_handoff_reason": handoff.outcomes[0].divergence.reason}


def rewind_replace(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    meter = Measurement()
    with meter.attach(backend), meter.active():
        with meter.phase("initial_prompt"):
            session = _session(backend, sampling)
        with meter.phase("steady_continuation"):
            original = LiveSessionRunner(session).run(
                live_policy=ActionSequencePolicy([Accept(), Accept()]), max_live_actions=2
            )
        if len(original.outcomes) != 2:
            raise AssertionError("initial continuation ended early")
        survivor = session.engine.visible_token_ids[0]
        abandoned = session.engine.visible_token_ids[1]
        with meter.phase("rewind_replacement"):
            session.rewind(1)
            replaced = session.generate(Write(" an alternative", mode="exact"))
        ledger = tuple(session.engine.visible_token_ids)
        prefix = [*session.engine.initial_token_ids, *ledger]
        logits = backend.last_logits().copy()
        with meter.phase("finalization"):
            session.quit()
    if ledger != (survivor, *replaced.visible_token_ids):
        raise AssertionError("rewind/replacement ledger did not retain the exact surviving prefix")
    return {"metrics": meter.result(actions=3, committed_tokens=len(ledger)),
            "retained_token_ids": ledger, "abandoned_token_id": abandoned,
            "oracle": _oracle(backend, prefix, logits, rtol=rtol, atol=atol)}


def fork_switch(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    meter = Measurement()
    with meter.attach(backend), meter.active():
        with meter.phase("initial_prompt"):
            session = _session(backend, sampling)
        root = session.branch.branch_id
        session.generate(Write(" An apple", mode="exact"))
        parent = tuple(session.engine.visible_token_ids)
        with meter.phase("fork_switch"):
            child = session.fork(boundary=len(parent))
            session.activate(child.branch.branch_id)
            session.generate(Write(" and a pear", mode="exact"))
            child_ledger = tuple(session.engine.visible_token_ids)
            session.activate(root)
        parent_again = tuple(session.engine.visible_token_ids)
        parent_prefix = [*session.engine.initial_token_ids, *parent_again]
        parent_logits = backend.last_logits().copy()
        with meter.phase("fork_switch"):
            session.activate(child.branch.branch_id)
        child_again = tuple(session.engine.visible_token_ids)
        child_prefix = [*session.engine.initial_token_ids, *child_again]
        child_logits = backend.last_logits().copy()
        with meter.phase("finalization"):
            session.discard()
    if parent != parent_again or child_ledger != child_again or child_ledger[:len(parent)] != parent:
        raise AssertionError("fork/switch changed independent branch histories")
    metrics = meter.result(actions=2, committed_tokens=len(child_ledger))
    child_check = _oracle(backend, child_prefix, child_logits, rtol=rtol, atol=atol)
    parent_check = _oracle(backend, parent_prefix, parent_logits, rtol=rtol, atol=atol)
    return {"metrics": metrics, "parent_visible_token_ids": parent,
            "child_visible_token_ids": child_ledger,
            "oracle": {"parent": parent_check, "child": child_check}}


def save_resume(backend, sampling, *, rtol: float, atol: float, provenance: dict, **_kwargs) -> dict:
    from trajectory_editor.episode_lifecycle import _restore_engine
    from trajectory_editor.episode_materializer import materialize_live_branch
    from trajectory_editor.episode_runner import EpisodeRunner
    from trajectory_editor.episode_store import EpisodeStore

    with TemporaryDirectory(prefix="spe-real-save-") as directory:
        meter = Measurement()
        with meter.attach(backend), meter.active():
            with meter.phase("initial_prompt"):
                source = _session(backend, sampling)
            source.generate(Write(" An apple", mode="exact"))
            saved_ledger = tuple(source.engine.visible_token_ids)
            with EpisodeStore(Path(directory) / "episode.sqlite3") as store:
                with meter.phase("persistence"):
                    identifier = materialize_live_branch(store, source, source.branch_state(), provenance)
                source.discard()
                with meter.phase("resume"):
                    restored = _restore_engine(store, identifier, backend, max_tokens=None,
                                               sampling_override=None, notice=lambda _message: None)
                result = EpisodeRunner(restored, store, identifier).run(
                    live_policy=ActionSequencePolicy([Accept()]), max_live_actions=1
                )
                ledger = tuple(restored.visible_token_ids)
                prefix = [*restored.initial_token_ids, *ledger]
                logits = backend.last_logits().copy()
                recorded = tuple(row["token_id"] for row in store.tokens(identifier)
                                if row["realized_visible"])
        if len(result.outcomes) != 1 or ledger[:len(saved_ledger)] != saved_ledger or recorded != ledger:
            raise AssertionError("durable resume did not preserve and extend the recorded ledger")
        return {"metrics": meter.result(actions=2, committed_tokens=len(ledger)),
                "saved_visible_token_ids": saved_ledger, "resumed_visible_token_ids": ledger,
                "oracle": _oracle(backend, prefix, logits, rtol=rtol, atol=atol)}


def long_context(backend, sampling, *, rtol: float, atol: float, **_kwargs) -> dict:
    prompt = "The scene contains ordinary objects. " * 10
    requested_actions = 8
    ids = backend.tokenize(prompt, add_bos=True, special=True)
    limit = getattr(getattr(backend, "settings", None), "n_ctx", None) or getattr(backend, "_context_limit", None)
    if limit is not None and len(ids) + requested_actions > limit:
        raise AssertionError(f"long context needs {len(ids) + requested_actions} positions, limit is {limit}")
    meter = Measurement()
    captures = []
    completed_actions = 0
    stop_reason = None
    with meter.attach(backend), meter.active():
        with meter.phase("initial_prompt"):
            session = _session(backend, sampling, prompt=prompt)
        initial_prefix = tuple(session.engine.initial_token_ids)
        captures.append((initial_prefix, backend.last_logits().copy(), None))
        with meter.phase("steady_continuation"):
            for _ in range(requested_actions):
                result = LiveSessionRunner(session).run(
                    live_policy=ActionSequencePolicy([Accept()]), max_live_actions=1
                )
                if len(result.outcomes) != 1:
                    stop_reason = result.handoff_reason or "no completed action"
                    break
                outcome = result.outcomes[0]
                if not outcome.visible_token_ids:
                    stop_reason = outcome.stop_reason
                    break
                completed_actions += 1
                prefix = tuple((*session.engine.initial_token_ids, *session.engine.visible_token_ids))
                captures.append((prefix, backend.last_logits().copy(), outcome.visible_token_ids[-1]))
                if session.engine.ended or session.engine.checkpointed:
                    stop_reason = outcome.stop_reason
                    break
        ledger = tuple(session.engine.visible_token_ids)
        with meter.phase("finalization"):
            if not session.engine.ended:
                session.quit()
    partial = {
        "metrics": meter.result(actions=completed_actions, committed_tokens=len(ledger)),
        "prompt_tokens": len(ids),
        "visible_token_ids": ledger,
        "requested_actions": requested_actions,
        "completed_actions": completed_actions,
        "stop_reason": stop_reason,
    }
    # Inspect all comparable prefixes before deciding pass/fail. A numerical
    # mismatch at one checkpoint cannot hide a later top-token divergence.
    partial["logit_trajectory"] = compare_prefix_trajectory(
        backend, captures, rtol=rtol, atol=atol
    )
    partial["numerical_mismatch_checkpoints"] = [
        item["checkpoint"] for item in partial["logit_trajectory"]
        if not item["within_tolerance"]
    ]
    partial["top_token_mismatch_checkpoints"] = [
        item["checkpoint"] for item in partial["logit_trajectory"]
        if not item["top_token_agrees"]
    ]
    problems = []
    if completed_actions != requested_actions:
        problems.append(f"continued {completed_actions}/{requested_actions} actions ({stop_reason})")
    if partial["top_token_mismatch_checkpoints"]:
        problems.append(f"top-token divergence at checkpoints {partial['top_token_mismatch_checkpoints']}")
    if partial["numerical_mismatch_checkpoints"]:
        problems.append(f"logit tolerance exceeded at checkpoints {partial['numerical_mismatch_checkpoints']}")
    if problems:
        raise ScenarioFailure("; ".join(problems), partial)
    return partial


def cache_compare(backend, sampling, *, profile, rtol: float, atol: float, **_kwargs) -> dict:
    from trajectory_editor.episode_backend_loader import load_backend

    if profile["options"].cache != "auto":
        raise AssertionError("cache comparison requires a cache-auto profile")
    options = copy.copy(profile["options"])
    options.cache = "off"
    alternate = load_backend(options)
    try:
        results = {}
        for label, selected in (("auto", backend), ("off", alternate)):
            meter = Measurement()
            with meter.attach(selected), meter.active():
                session = _session(selected, sampling)
                run = LiveSessionRunner(session).run(
                    live_policy=ActionSequencePolicy([Accept(), Accept()]), max_live_actions=2
                )
                ledger = tuple(session.engine.visible_token_ids)
                logits = selected.last_logits().copy()
                session.quit()
            if len(run.outcomes) != 2:
                raise AssertionError(f"cache-{label} continuation stopped early")
            results[label] = {"metrics": meter.result(actions=2, committed_tokens=len(ledger)),
                              "visible_token_ids": ledger, "logits": logits}
        if results["auto"]["visible_token_ids"] != results["off"]["visible_token_ids"]:
            raise AssertionError("cache modes produced different controlled token ledgers")
        if not np.allclose(results["auto"]["logits"], results["off"]["logits"], rtol=rtol, atol=atol):
            raise AssertionError("cache modes disagree on same-prefix logits")
        return {"metrics": results["auto"]["metrics"],
                "cache_off_metrics": results["off"]["metrics"],
                "visible_token_ids": results["auto"]["visible_token_ids"],
                "max_abs_logit_error": float(np.max(np.abs(results["auto"]["logits"] - results["off"]["logits"]))) }
    finally:
        close = getattr(alternate, "close", None)
        if callable(close):
            close()


def cfg_lifecycle(backend, sampling, *, profile, provenance: dict, rtol: float,
                  atol: float, **_kwargs) -> dict:
    from trajectory_editor.episode_backend_loader import load_cfg_guidance_backend

    if sampling.cfg_unconditional_prompt is None:
        raise AssertionError("CFG scenario requires cfg-unconditional-prompt in launch settings")
    guidance = load_cfg_guidance_backend(profile["options"], provenance)
    try:
        meter = Measurement()
        with meter.attach(backend, role="conditional"), meter.attach(guidance, role="unconditional"), meter.active():
            engine = EpisodeEngine(backend, sampling=sampling, max_tokens=64,
                                   initial_text=PROMPT, guidance_backend=guidance)
            session = LiveSession(engine, prompt=PROMPT)
            first = session.generate(Accept())
            second = session.generate(Accept())
            if len(first.visible_token_ids) != 1 or len(second.visible_token_ids) != 1:
                raise AssertionError("CFG continuation ended before rewind")
            session.rewind(1)
            observation = session.engine.observe()
            conditional_prefix = [*session.engine.initial_token_ids, *session.engine.visible_token_ids]
            conditional_logits = backend.last_logits().copy()
            guidance_prefix = guidance.tokenize(sampling.cfg_unconditional_prompt, add_bos=True, special=True)
            guidance_prefix += session.engine.visible_token_ids
            guidance_logits = guidance.last_logits().copy()
            session.quit()
        work = meter.result(actions=3, committed_tokens=1)
        if not any(item["role"] == "conditional" for item in work["work"]) or not any(item["role"] == "unconditional" for item in work["work"]):
            raise AssertionError("CFG work accounting omitted a model branch")
        return {"metrics": work, "surviving_token_ids": conditional_prefix[len(engine.initial_token_ids):],
                "observation_boundary": observation.boundary,
                "oracle": {"conditional": _oracle(backend, conditional_prefix, conditional_logits, rtol=rtol, atol=atol),
                           "unconditional": _oracle(guidance, guidance_prefix, guidance_logits, rtol=rtol, atol=atol)}}
    finally:
        close = getattr(guidance, "close", None)
        if callable(close):
            close()


def cli_jsonl(backend, sampling, *, profile, model_path: Path, timeout_s: int, **_kwargs) -> dict:
    del backend, sampling
    with TemporaryDirectory(prefix="spe-real-cli-") as directory:
        folder = Path(directory)
        # sitecustomize loads before the CLI entry point. Its audit hook makes
        # even a transient SQLite connection or workspace open a hard failure.
        (folder / "sitecustomize.py").write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['SPE_BENCH_AUDIT_MARKER']).write_text('loaded')\n"
            "def audit(event, args):\n"
            "    if event == 'sqlite3.connect':\n"
            "        raise RuntimeError('SQLite connection attempted in ephemeral CLI')\n"
            "    if event == 'open':\n"
            "        try: path = os.path.abspath(os.fspath(args[0]))\n"
            "        except TypeError: return\n"
            "        for key in ('SPE_BENCH_SELECTED_WORKSPACE', 'SPE_BENCH_DEFAULT_WORKSPACE'):\n"
            "            target = os.environ[key]\n"
            "            if path == target or path.startswith(target + '-'):\n"
            "                raise RuntimeError('workspace opened in ephemeral CLI: ' + path)\n"
            "sys.addaudithook(audit)\n", encoding="utf-8"
        )
        elapsed = {}
        lengths = {}
        hashes = {}
        for variant in ("action-only", "embedded", "sidecar"):
            plan = folder / f"{variant}.jsonl"
            output = folder / f"{variant}.txt"
            workspace = folder / f"never-open-{variant}.sqlite3"
            envelope = {"type": "serial-policy-tape", "version": 1, "prompt": PROMPT,
                        "environment": {"backend": {"model_path": "/missing/source/model.gguf"},
                                        "source_episode_id": "absent-source"}}
            lines = ([json.dumps(envelope)] if variant == "embedded" else []) + [json.dumps(row) for row in ACTION_RECORDS]
            plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
            command = [sys.executable, "-m", "trajectory_editor", "--ephemeral", "--plain-ui",
                       "--teacher-plan", str(plan), "--divergence-policy", "ballistic",
                       "--model", str(model_path), "--backend", profile["backend"],
                       "--workspace", str(workspace), "--output", str(output),
                       *profile["launch_tokens"]]
            if variant == "action-only":
                command += ["--new-prompt", PROMPT]
            if variant == "sidecar":
                sidecar = folder / "sidecar.json"
                sidecar.write_text(json.dumps({"format": "serial-policy-tape", **{k: v for k, v in envelope.items() if k != "type"}}), encoding="utf-8")
                command += ["--teacher-plan-envelope", str(sidecar)]
            marker = folder / f"audit-{variant}.txt"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(folder) + os.pathsep + environment.get("PYTHONPATH", "")
            environment["SPE_BENCH_AUDIT_MARKER"] = str(marker)
            environment["SPE_BENCH_SELECTED_WORKSPACE"] = str(workspace)
            environment["SPE_BENCH_DEFAULT_WORKSPACE"] = str(folder / "episodes.sqlite3")
            start = perf_counter()
            completed = subprocess.run(command, input="end\n", text=True, cwd=folder,
                                       capture_output=True, timeout=timeout_s, env=environment)
            elapsed[variant] = perf_counter() - start
            if not marker.is_file():
                raise AssertionError(f"{variant} subprocess did not load the database audit hook")
            if completed.returncode != 0:
                raise AssertionError(f"{variant} CLI exit {completed.returncode}: {completed.stdout[-1000:]} {completed.stderr[-1000:]}")
            if "Teacher plan exhausted" not in completed.stdout or not output.exists():
                raise AssertionError(f"{variant} CLI did not reach live edge and end normally: {completed.stdout[-1000:]}")
            final_text = output.read_text(encoding="utf-8")
            if " The next item is" not in final_text:
                raise AssertionError(f"{variant} CLI final text omitted handwritten write action")
            lengths[variant] = len(final_text)
            hashes[variant] = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        if any(folder.glob("*.sqlite3*")) or workspace.exists():
            raise AssertionError("CLI created a database, journal, or WAL")
        return {"outer_subprocess_wall_s": elapsed, "replayed_actions": len(ACTION_RECORDS),
                "final_text_characters": lengths, "final_text_sha256": hashes,
                "database_files": 0, "metrics": None}


SCENARIOS = {
    "continuation": continuation,
    "controlled-write": controlled_write,
    "candidate-refresh": candidate_refresh,
    "instrumentation-parity": instrumentation_parity,
    "action-jsonl": action_jsonl,
    "observed-jsonl": observed_jsonl,
    "rewind-replace": rewind_replace,
    "fork-switch": fork_switch,
    "save-resume": save_resume,
    "long-context": long_context,
    "cache-compare": cache_compare,
    "cfg-lifecycle": cfg_lifecycle,
    "cli-jsonl": cli_jsonl,
}
