"""Episode-local direct supervision; no sampling controls or torch dependency here."""
from dataclasses import asdict, dataclass, replace
import json
import math
import re

from .core.errors import EditorError


@dataclass
class LearningConfig:
    lr: float = 1e-5
    steps: int = 1
    max_step: int = 5

    style: str = "sgd-only"
    freeze: tuple = ()
    thaw: tuple = ()
    layers: tuple = ()

    def validate(self):
        if self.style not in ("sgd-only", "sgd-and-sga"):
            raise EditorError("learning style must be sgd-only or sgd-and-sga")
        if self.freeze and self.thaw:
            raise EditorError("freeze and thaw are mutually exclusive")
        if type(self.max_step) is not int or self.max_step < 1:
            raise EditorError("max_step must be a positive integer")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise EditorError("online learning rate must be finite and positive")
        if type(self.steps) is not int or self.steps < 0:
            raise EditorError("online steps must be a nonnegative integer")


class EpisodeLearner:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = replace(config)
        self.events = []
        self.corrections = 0
        self.last = None
        self.progress = None
        self.display = None
        engine.backend.start_learning_episode(self)
        from .learning_diagnostics import LearningDiagnostics
        self.diagnostics = LearningDiagnostics(engine.backend, engine.initial_token_ids)

    def snapshot(self):
        return self.diagnostics.snapshot(self.config, self.corrections)

    def publish(self):
        if self.display is not None:
            self.display(self.snapshot())

    def overview(self):
        self.require_owner()
        self.diagnostics.refresh()
        self.publish()
        return self.snapshot()

    def _diagnostic(self, step, logits):
        self.diagnostics.record(step, logits)
        self.publish()

    def require_owner(self):
        if self.engine.backend._learning_owner is not self:
            raise EditorError("online episode is no longer active; switching learned episodes is unsupported")

    def train(self, example, steps, mode=None, margin=None):
        self.require_owner()
        prefix, corrected, proposal, boundary = example
        self.engine.discard_speculative_accept()
        self.diagnostics.start(prefix, corrected[0], boundary)
        try:
            metrics = self.engine.backend.supervised_update(prefix, corrected, self.config.lr, steps,
                **(dict(until_argmax=True, max_steps=self.config.max_step if mode == "?" else None,
                        progress=self.progress, margin=margin, diagnostic=self._diagnostic)
                   if mode else dict(progress=self.progress, diagnostic=self._diagnostic)))
            steps = metrics.pop("steps_run", steps)
            self.diagnostics.finish(metrics['stop_reason'])
        except EditorError:
            self.diagnostics.finish("failed · learn reset required")
            self.publish()
            raise
        finally:
            self.engine._invalidate_observation()
        event = dict(prefix_token_ids=list(prefix), corrected_token_ids=list(corrected),
                     proposal_token_id=proposal, boundary=boundary, lr=self.config.lr,
                     learning_config=asdict(self.config), steps=steps, **metrics)
        self.events.append(event)
        self.engine.backend.refresh_learning_prefix(self.engine.token_ids)
        self.publish()
        return event

    def correction(self, example, mode=None, margin=None):
        self.corrections += 1
        self.last = example
        if mode or self.config.steps:
            self.train(example, self.config.steps, mode, margin)
        else:
            self.diagnostics.watch(example)
            self.publish()

    def command(self, raw):
        self.require_owner()
        if raw == "learn status":
            return json.dumps(dict(**asdict(self.config), corrections=self.corrections,
                optimizer_steps=sum(e['steps'] for e in self.events),
                targets=getattr(self.engine.backend, "learning_selection", {}),
                weight_revision=self.engine.backend.weight_revision,
                recent_events=self.events[-3:]), sort_keys=True)
        if raw == "learn reset":
            self.engine.discard_speculative_accept()
            self.engine.backend.restore_learning_base()
            self.events.clear()
            self.corrections = 0
            self.last = None
            self.diagnostics.reset()
            self.engine._invalidate_observation()
            self.engine.backend.reset(self.engine.token_ids)
            self.publish()
            return "Original episode weights restored; token trajectory retained."
        match = re.fullmatch(r"learn (freeze|thaw|layers|style) (.+)", raw)
        if match:
            key, value = match.groups()
            updated = replace(self.config)
            if key == "style":
                updated.style = value
            else:
                setattr(updated, key, () if value == "all" else tuple(value.split()))
                if key in ("freeze", "thaw"):
                    setattr(updated, "thaw" if key == "freeze" else "freeze", ())
            updated.validate()
            self.engine.backend.configure_learning(updated)
            self.config = updated
            self.publish()
            return self.command("learn status")
        if raw == "learn targets":
            return json.dumps(self.engine.backend.learning_targets.inventory(), sort_keys=True)
        if raw.startswith("max_step="):
            raw = "learn " + raw
        match = re.fullmatch(r"learn (steps|lr|max_step)=(\S+)", raw)
        if match:
            key, value = match.groups()
            updated = replace(self.config)
            try:
                setattr(updated, key, float(value) if key == 'lr' else int(value))
                updated.validate()
            except (ValueError, OverflowError) as exc:
                raise EditorError("invalid learning configuration") from exc
            self.engine.backend.configure_learning(updated)
            self.config = updated
            self.publish()
            return self.command("learn status")
        match = re.fullmatch(r"teach last steps=(\d+)", raw)
        if match:
            if self.last is None:
                raise EditorError("no human correction is available in this episode")
            steps = int(match[1])
            if steps < 1:
                raise EditorError("teach last requires positive steps")
            return json.dumps(self.train(self.last, steps), sort_keys=True)
        raise EditorError("use learn status/reset/targets, learn steps=N/lr=X, learn style/freeze/thaw/layers VALUE..., or teach last steps=N")
