"""Learning dock and an inspectable, live document over immutable snapshots."""

from rich.text import Text

from ..learning_diagnostics import LearningSnapshot
from ..tui_render import _safe_context_text
from .overlays import DocumentOverlay


def sparkline(values, width=24):
    """Trailing samples, individually scaled; numbers carry the absolute scale."""
    values = tuple(values)[-max(1, width):]
    if not values:
        return "—"
    low, high = min(values), max(values)
    if high == low:
        return "▄" * len(values)
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(7, int((value - low) / (high - low) * 7))] for value in values)


def spelling(value, limit=22):
    safe = _safe_context_text(value).replace("\n", "\\n").replace("\t", "\\t")
    return repr(safe[:limit] + ("…" if len(safe) > limit else ""))


def delta(value):
    return "—" if value is None else f"{value:+.3f}"


def dock_height(width, height, enabled):
    if not enabled or width < 20 or height < 6:
        return 0
    return 5 if height >= 20 else 2 if height >= 12 else 1


def dock_lines(snapshot, status, width, height):
    if snapshot is None:
        return ["─ Online SGD · F3 overview", status or "Waiting for learning diagnostics"][:height]
    step = snapshot.current
    if step is None:
        return [
            f"─ Online SGD · {snapshot.optimizer_steps} steps · F3 overview",
            "No SGD trace yet · a human correction starts one",
            f"lr {snapshot.lr:g} · default {snapshot.default_steps} steps · ceiling {snapshot.max_step}",
            f"{len(snapshot.probes)} fixed root contexts · comparisons in F3",
            "Margin history: —",
        ][:height]
    point = step.comparison
    limit = str(step.limit) if step.limit is not None else "∞"
    title = f"─ Online SGD · {step.step}/{limit} · {step.status} · F3 overview"
    goal = f"goal ≥{step.goal:g}" if step.goal is not None else "goal argmax" if step.targeted else "fixed steps"
    if height < 5:
        summary = f"SGD {step.step}/{limit} {step.status} · margin {point.margin:+.3f} · F3"
        return [summary, f"{spelling(point.target_text, 12)} vs {spelling(point.rival_text, 12)}  {sparkline(snapshot.history, 12)}"][:height]
    return [
        title,
        f"Target {spelling(point.target_text):<26} {point.target_logit:8.3f}  Δ {delta(step.target_delta):>7}  p {point.probability:.1%}  rank {point.rank}",
        f"Rival  {spelling(point.rival_text):<26} {point.rival_logit:8.3f}  Δ {delta(step.rival_delta):>7}" + ("  rival changed" if step.rival_changed else ""),
        f"Margin {point.margin:+.3f}  Δ {delta(step.margin_delta)} · {goal} · {sparkline(snapshot.history, min(24, max(4, width - 54)))}",
        f"Loss {step.loss:.5f} · gradient {step.gradient_norm:.4g}" if step.gradient_norm is not None else f"Loss {step.loss:.5f} · before updates",
    ]


def draw_dock(ctx, snapshot, status, y, height):
    for offset, line in enumerate(dock_lines(snapshot, status, ctx.canvas.width, height)):
        ctx.canvas.put(0, y + offset, line,
                       ctx.styles("section" if offset == 0 else "table-row" if offset < 4 else "muted"))


def overview_document(snapshot: LearningSnapshot | None, refreshing=False, notice=""):
    doc = Text()
    if snapshot is None:
        doc.append("Enable --online-sgd to inspect episode learning.\n")
        return doc
    doc.append("EPISODE LEARNING\n", style="bold")
    doc.append(f"{snapshot.optimizer_steps} SGD steps · {snapshot.corrections} corrections · revision {snapshot.revision}\n")
    doc.append(f"lr {snapshot.lr:g} · default steps {snapshot.default_steps} · ceiling {snapshot.max_step}\n")
    if refreshing:
        doc.append("Refreshing comparisons on the episode thread…\n")
    elif notice:
        doc.append(notice + "\n")
    step = snapshot.current
    if step is not None:
        doc.append(f"\nLATEST TRAINING TRACE · {step.status}\n", style="bold")
        point = step.comparison
        goal = f"≥{step.goal:g}" if step.goal is not None else "argmax" if step.targeted else "fixed steps"
        doc.append(f"{spelling(point.target_text)} vs {spelling(point.rival_text)} · margin {point.margin:+.3f} · goal {goal}\n")
        doc.append(f"{sparkline(snapshot.history)}  [{min(snapshot.history):+.3f}, {max(snapshot.history):+.3f}] · loss {step.loss:.5f}\n")
    roots = [p for p in snapshot.probes if p.baseline_kind == "episode start"]
    corrections = [p for p in snapshot.probes if p.baseline_kind == "first seen"]
    for title, probes in [("FIXED ROOT CONTEXTS", roots), ("RECENT CORRECTIONS (up to 12)", corrections)]:
        doc.append("\n" + title + "\n", style="bold")
        if not probes:
            doc.append("No corrections yet.\n")
        else:
            stale = sum(p.revision != snapshot.revision for p in probes)
            flips = sum(p.baseline.argmax_id != p.current.argmax_id for p in probes)
            drift = sum(p.drift for p in probes) / len(probes)
            doc.append(f"{flips}/{len(probes)} argmax flips · mean drift {drift:.3g} bits")
            if stale:
                doc.append(f" · {stale} awaiting R")
            doc.append("\n")
        for probe in probes:
            before, now = probe.baseline, probe.current
            stale = " · stale; R to refresh" if probe.revision != snapshot.revision else ""
            taught = " · trained context" if probe.trained else " · not trained here"
            doc.append(f"\n{probe.label}{taught}{stale}\n", style="bold")
            doc.append(f"  …{spelling(probe.context, 70)}\n")
            label = "Original choice" if probe.baseline_kind == "episode start" else "Target"
            doc.append(f"  {label} {spelling(now.target_text)} · margin {before.margin:+.3f} → {now.margin:+.3f}")
            if probe.taught_margin is not None and probe.baseline_kind == "first seen":
                doc.append(f" · Δ since taught {now.margin - probe.taught_margin:+.3f}")
            doc.append("\n")
            doc.append(f"  p {before.probability:.1%} → {now.probability:.1%} · rank {before.rank} → {now.rank} · drift {probe.drift:.3g} bits\n")
            doc.append(f"  Argmax {spelling(before.argmax_text)} → {spelling(now.argmax_text)}\n")
            doc.append(f"  {sparkline(probe.history)}  [{min(probe.history):+.2f}, {max(probe.history):+.2f}] · revision {probe.revision}\n")
    doc.append("\nREADING THESE NUMBERS\n", style="bold")
    doc.append("Comparisons refresh on open / R. Training trace updates live.\n")
    doc.append("Root baselines: episode start. Corrections: when first seen.\n")
    doc.append("Δ since taught shows later reinforcement / interference.\n")
    doc.append("Drift = JS divergence, 0–1 bits; measures these contexts only.\n")
    doc.append("Sparklines scale each row; samples are SGD steps / panel refreshes.\n")
    doc.append("For multi-token corrections, the trace follows the first token.\n")
    return doc


class LearningOverlay(DocumentOverlay):
    def __init__(self, app):
        super().__init__("Learning overview", Text(), close_keys={"escape", "q", "f3"})
        self.app = app

    def on_key(self, key):
        if key.char in {"r", "R"}:
            self.app.refresh_learning()
            return False
        return super().on_key(key)

    def hint(self):
        return "R refresh · PgUp/PgDn ↑↓ scroll · F3/Esc closes"

    def lines(self, ctx, width):
        self.document = overview_document(self.app.learning_snapshot,
                                          self.app.learning_refreshing,
                                          self.app.learning_notice)
        return super().lines(ctx, width)
