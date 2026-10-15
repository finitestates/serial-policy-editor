"""Small, read-only behavioral probes and immutable learning UI snapshots.

No model copies, optimizer controls, or terminal dependencies. Root probes use
episode-start predictions; correction baselines mean first seen, not original
checkpoint behavior. Only the episode owner evaluates probes.
"""

from dataclasses import dataclass, replace
import numpy as np


HISTORY_LIMIT = 64
CORRECTION_PROBE_LIMIT = 12


@dataclass(frozen=True)
class TokenComparison:
    target_id: int
    target_text: str
    target_logit: float
    rival_id: int
    rival_text: str
    rival_logit: float
    margin: float
    probability: float
    rank: int
    argmax_id: int
    argmax_text: str


def probabilities(logits):
    row = np.asarray(logits, dtype=np.float64)
    if row.ndim != 1 or row.size < 2 or not np.isfinite(row).all():
        raise ValueError("learning diagnostics require finite vocabulary logits")
    values = np.exp(row - row.max())
    return values / values.sum()


def compare(logits, target, token_text):
    row = np.asarray(logits)
    probs = probabilities(row)
    rivals = row.copy()
    rivals[target] = -np.inf
    rival = int(np.argmax(rivals))
    top = int(np.argmax(row))
    return TokenComparison(
        int(target), token_text(int(target)), float(row[target]), rival,
        token_text(rival), float(row[rival]), float(row[target] - row[rival]),
        float(probs[target]), 1 + int(np.count_nonzero(row > row[target])),
        top, token_text(top),
    )


def distribution_drift(before, after):
    """Jensen-Shannon divergence in bits: zero unchanged, at most one."""
    midpoint = (before + after) / 2
    def part(values):
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))
    return max(0.0, (part(before) + part(after)) / 2)


@dataclass(frozen=True)
class LearningStep:
    comparison: TokenComparison
    step: int
    limit: int | None
    goal: float | None
    targeted: bool
    loss: float
    gradient_norm: float | None
    revision: int
    status: str = "training"
    target_delta: float | None = None
    rival_delta: float | None = None
    margin_delta: float | None = None
    rival_changed: bool = False


@dataclass(frozen=True)
class ProbeReadout:
    label: str
    context: str
    baseline_kind: str
    baseline: TokenComparison
    current: TokenComparison
    drift: float
    revision: int
    trained: bool
    taught_margin: float | None
    history: tuple[float, ...]


@dataclass(frozen=True)
class LearningSnapshot:
    revision: int
    corrections: int
    optimizer_steps: int
    lr: float
    default_steps: int
    max_step: int
    current: LearningStep | None = None
    history: tuple[float, ...] = ()
    probes: tuple[ProbeReadout, ...] = ()
    style: str = "sgd-only"
    freeze: tuple[str, ...] = ()
    thaw: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()


class _Probe:
    def __init__(self, prefix, target, logits, backend, label, baseline_kind):
        self.prefix = tuple(prefix)
        self.target = int(target)
        self.distribution = probabilities(logits)
        sample = compare(logits, target, backend.token_text)
        self.readout = ProbeReadout(
            label, backend.render(list(prefix[-32:]), special=True), baseline_kind,
            sample, sample, 0.0, backend.weight_revision, False, None,
            (sample.margin,),
        )

    def update(self, logits, backend, *, taught=False):
        sample = compare(logits, self.target, backend.token_text)
        self.readout = replace(
            self.readout, current=sample,
            drift=distribution_drift(self.distribution, probabilities(logits)),
            revision=backend.weight_revision,
            trained=self.readout.trained or taught,
            taught_margin=sample.margin if taught else self.readout.taught_margin,
            history=(*self.readout.history, sample.margin)[-HISTORY_LIMIT:],
        )


class LearningDiagnostics:
    def __init__(self, backend, root):
        self.backend = backend
        self.root = tuple(root)
        self.roots = []
        self.corrections = []
        self.current = None
        self.history = ()
        self.steps = 0
        self._active = None
        self.reset()

    def reset(self):
        self.roots.clear()
        self.corrections.clear()
        self.current = None
        self.history = ()
        self.steps = 0
        self._active = None
        # Fixed contexts from this episode's prompt, captured before SGD.
        for length in sorted({1, max(1, len(self.root) // 2), len(self.root)}):
            if not length:
                continue
            prefix = self.root[:length]
            logits = self.backend.learning_probe_logits(prefix)
            self.roots.append(_Probe(prefix, int(np.argmax(logits)), logits,
                                     self.backend, f"root · {length} tokens", "episode start"))

    def start(self, prefix, target, boundary):
        self.current = None
        self.history = ()
        self._active = (tuple(prefix), target, boundary)

    def _watch(self, prefix, target, boundary, logits):
        prefix = tuple(prefix)
        probe = next((p for p in self.corrections
                      if (p.prefix, p.target) == (prefix, target)), None)
        if probe is None:
            probe = _Probe(prefix, target, logits, self.backend,
                           f"correction · boundary {boundary}", "first seen")
        else:
            self.corrections.remove(probe)
            probe.update(logits, self.backend)
        self.corrections.append(probe)
        self.corrections = self.corrections[-CORRECTION_PROBE_LIMIT:]

    def watch(self, example):
        """A paused correction is still worth watching, without an SGD trace."""
        prefix, corrected, _, boundary = example
        self._watch(prefix, corrected[0], boundary, self.backend.learning_probe_logits(prefix))

    def record(self, step, logits):
        prefix, target, boundary = self._active
        if step.step == 0:
            self._watch(prefix, target, boundary, logits)
        else:
            self.steps += 1
        self.current = step
        self.history = (*self.history, step.comparison.margin)[-HISTORY_LIMIT:]
        for probe in self.roots + self.corrections:
            if probe.prefix == prefix:
                probe.update(logits, self.backend, taught=step.step > 0)

    def finish(self, status):
        if self.current is not None:
            self.current = replace(self.current, status=status)

    def refresh(self):
        rows = {}
        for probe in self.roots + self.corrections:
            if probe.prefix not in rows:
                rows[probe.prefix] = self.backend.learning_probe_logits(probe.prefix)
            probe.update(rows[probe.prefix], self.backend)

    def snapshot(self, config, corrections):
        return LearningSnapshot(
            self.backend.weight_revision, corrections, self.steps,
            config.lr, config.steps, config.max_step, self.current, self.history,
            tuple(p.readout for p in self.roots + self.corrections),
            config.style, tuple(config.freeze), tuple(config.thaw), tuple(config.layers),
        )
