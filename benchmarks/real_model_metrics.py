"""Optional, process-local elapsed-time accounting for real-model runs.

The coarse service interval covers reset/eval/branch_to_prefix, including
adapter work and logit transfer. Nested service calls count once. The narrower
llama.cpp interval covers its synchronous Python ``Llama.eval`` call; it still
includes native-library bookkeeping and is not a kernel/CPU-use timer.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from time import perf_counter_ns
from typing import Callable, Iterator


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative measurement interval")
        if start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


class Measurement:
    def __init__(self, clock: Callable[[], int] = perf_counter_ns) -> None:
        self.clock = clock
        self.service_intervals: list[tuple[int, int]] = []
        self.model_intervals: list[tuple[int, int]] = []
        self.work: list[dict] = []
        self.cache_fallbacks: list[str] = []
        self.phases: dict[str, list[tuple[int, int]]] = {}
        self._depth = 0
        self._service_start = 0
        self._active = False
        self._start = 0
        self.active_ns = 0
        self.fine_supported = False
        self.work_supported = False

    @contextmanager
    def active(self) -> Iterator[None]:
        if self._active:
            raise RuntimeError("measurement interval already active")
        self._active = True
        self._start = self.clock()
        try:
            yield
        finally:
            self.active_ns += self.clock() - self._start
            self._active = False
            if self._depth:
                raise RuntimeError("backend service escaped measurement interval")

    @contextmanager
    def service(self) -> Iterator[None]:
        if not self._active:
            yield
            return
        if self._depth == 0:
            self._service_start = self.clock()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self.service_intervals.append((self._service_start, self.clock()))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self._active:
            yield
            return
        start = self.clock()
        try:
            yield
        finally:
            self.phases.setdefault(name, []).append((start, self.clock()))

    @contextmanager
    def model_call(self, kind: str, positions: int, context_length: int,
                   role: str = "conditional") -> Iterator[None]:
        if not self._active:
            yield
            return
        start = self.clock()
        completed = False
        try:
            yield
            completed = True
        finally:
            stop = self.clock()
            self.model_intervals.append((start, stop))
            self.work.append({
                "kind": kind, "role": role, "input_positions": positions,
                "context_length": context_length,
                "completed": completed,
            })

    def cache_fallback(self, reason: str) -> None:
        if self._active:
            self.cache_fallbacks.append(reason)

    @contextmanager
    def attach(self, backend, *, role: str = "conditional") -> Iterator[None]:
        originals = {}
        for name in ("reset", "eval", "branch_to_prefix"):
            method = getattr(backend, name, None)
            if not callable(method):
                continue
            originals[name] = method

            @wraps(method)
            def wrapped(*args, _method=method, **kwargs):
                with self.service():
                    return _method(*args, **kwargs)

            setattr(backend, name, wrapped)
        prior = getattr(backend, "_real_model_probe", None)
        if hasattr(backend, "_real_model_probe"):
            backend._real_model_probe = _RoleProbe(self, role)
            # A CPU forward finishes before returning. CUDA/accelerator work
            # can be asynchronous, so its dispatch interval is not additive
            # with host wall time and is deliberately withheld.
            input_device = getattr(backend, "_input_device", None)
            eligible = (
                input_device is None or
                (str(input_device) == "cpu" and all(str(p.device) == "cpu" for p in backend._model.parameters()))
            )
            self.fine_supported = (self.fine_supported and eligible) if self.work_supported else eligible
            self.work_supported = True
        try:
            yield
        finally:
            for name, method in originals.items():
                setattr(backend, name, method)
            if hasattr(backend, "_real_model_probe"):
                backend._real_model_probe = prior

    def result(self, *, actions: int, committed_tokens: int) -> dict:
        service = union_ns(self.service_intervals)
        model = union_ns(self.model_intervals) if self.fine_supported else None
        if service > self.active_ns or (model is not None and model > service):
            raise RuntimeError("invalid measurement residual: nested intervals exceed parent")
        outside = self.active_ns - service
        phase_intervals = [interval for values in self.phases.values() for interval in values]
        phase_total = union_ns(phase_intervals)
        if phase_total > self.active_ns:
            raise RuntimeError("invalid phase accounting residual")
        result = {
            "active_wall_s": self.active_ns / 1e9,
            "backend_eval_wall_s": service / 1e9,
            "outside_backend_eval_wall_s": outside / 1e9,
            "outside_backend_eval_percent": 100 * outside / self.active_ns if self.active_ns else None,
            "outside_backend_eval_ms_per_action": outside / 1e6 / actions if actions else None,
            "outside_backend_eval_ms_per_committed_token": outside / 1e6 / committed_tokens if committed_tokens else None,
            "model_call_wall_s": model / 1e9 if model is not None else None,
            "outside_model_call_wall_s": (self.active_ns - model) / 1e9 if model is not None else None,
            "outside_model_call_percent": 100 * (self.active_ns - model) / self.active_ns if model is not None and self.active_ns else None,
            "backend_non_model_wall_s": (service - model) / 1e9 if model is not None else None,
            "fine_unavailable_reason": None if model is not None else "model-call wall time has no validated completion boundary on this device/backend",
            "phase_wall_s": {name: union_ns(values) / 1e9 for name, values in self.phases.items()},
            "unclassified_phase_wall_s": (self.active_ns - phase_total) / 1e9,
            "model_calls": len(self.work) if self.work_supported else None,
            "evaluated_input_positions": sum(item["input_positions"] for item in self.work) if self.work_supported else None,
            "cache_fallbacks": list(self.cache_fallbacks),
            "generated_or_inserted_tokens_per_s": committed_tokens / (self.active_ns / 1e9) if self.active_ns and committed_tokens else None,
            "work": list(self.work),
            "work_by_kind": {
                kind: {"calls": sum(item["kind"] == kind for item in self.work),
                       "input_positions": sum(item["input_positions"] for item in self.work if item["kind"] == kind)}
                for kind in sorted({item["kind"] for item in self.work})
            },
            "actions": actions,
            "committed_tokens": committed_tokens,
        }
        return result


class _RoleProbe:
    def __init__(self, measurement: Measurement, role: str) -> None:
        self.measurement = measurement
        self.role = role

    def model_call(self, kind: str, positions: int, context_length: int):
        return self.measurement.model_call(kind, positions, context_length, self.role)

    def cache_fallback(self, reason: str) -> None:
        self.measurement.cache_fallback(f"{self.role}:{reason}")
