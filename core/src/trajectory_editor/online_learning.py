"""Episode-local direct supervision; no sampling controls or torch dependency here."""
from dataclasses import asdict, dataclass
import json
import math
import re

from .core.errors import EditorError


@dataclass
class LearningConfig:
    lr: float = 1e-5
    steps: int = 1

    def validate(self):
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise EditorError("online learning rate must be finite and positive")
        if type(self.steps) is not int or self.steps < 0:
            raise EditorError("online steps must be a nonnegative integer")


class EpisodeLearner:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = LearningConfig(config.lr, config.steps)
        self.events = []
        self.corrections = 0
        self.last = None
        engine.backend.start_learning_episode(self)

    def require_owner(self):
        if self.engine.backend._learning_owner is not self:
            raise EditorError("online episode is no longer active; switching learned episodes is unsupported")

    def train(self, example, steps):
        self.require_owner()
        prefix, corrected, proposal, boundary = example
        self.engine.discard_speculative_accept()
        try:
            metrics = self.engine.backend.supervised_update(prefix, corrected, self.config.lr, steps)
        finally:
            self.engine._invalidate_observation()
        event = dict(prefix_token_ids=list(prefix), corrected_token_ids=list(corrected),
                     proposal_token_id=proposal, boundary=boundary, lr=self.config.lr,
                     steps=steps, **metrics)
        self.events.append(event)
        self.engine.backend.reset(self.engine.token_ids)
        return event

    def correction(self, example):
        self.corrections += 1
        self.last = example
        if self.config.steps:
            self.train(example, self.config.steps)

    def command(self, raw):
        self.require_owner()
        if raw == "learn status":
            return json.dumps(dict(**asdict(self.config), corrections=self.corrections,
                optimizer_steps=sum(e['steps'] for e in self.events),
                weight_revision=self.engine.backend.weight_revision,
                recent_events=self.events[-3:]), sort_keys=True)
        if raw == "learn reset":
            self.engine.discard_speculative_accept()
            self.engine.backend.restore_learning_base()
            self.events.clear()
            self.corrections = 0
            self.last = None
            self.engine._invalidate_observation()
            self.engine.backend.reset(self.engine.token_ids)
            return "Original episode weights restored; token trajectory retained."
        match = re.fullmatch(r"learn (steps|lr)=(\S+)", raw)
        if match:
            key, value = match.groups()
            updated = LearningConfig(self.config.lr, self.config.steps)
            try:
                setattr(updated, key, int(value) if key == 'steps' else float(value))
                updated.validate()
            except (ValueError, OverflowError) as exc:
                raise EditorError("invalid learning configuration") from exc
            self.config = updated
            return self.command("learn status")
        match = re.fullmatch(r"teach last steps=(\d+)", raw)
        if match:
            if self.last is None:
                raise EditorError("no human correction is available in this episode")
            steps = int(match[1])
            if steps < 1:
                raise EditorError("teach last requires positive steps")
            return json.dumps(self.train(self.last, steps), sort_keys=True)
        raise EditorError("use learn status/reset, learn steps=N/lr=X, or teach last steps=N")
