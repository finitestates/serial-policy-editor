from __future__ import annotations

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine


class LiveScriptedIO(ScriptedIO):
    def __init__(self, responses: list[str | None]) -> None:
        super().__init__(responses)
        self.initial_commands: list[str | None] = []
        self.remaining_tokens: list[int] = []

    @property
    def supports_live_choices(self) -> bool:
        return True

    def read_choice(self, state):
        self.initial_commands.append(state.initial_command)
        self.remaining_tokens.append(state.remaining_tokens)
        if not self.responses:
            raise AssertionError("unexpected live input request")
        return self.responses.pop(0)


class NoEogBackend(ConformingFakeBackend):
    def last_logits(self):
        logits = super().last_logits()
        logits[0] = -100.0
        return logits


def engine(backend=None, *, max_tokens: int = 2) -> EpisodeEngine:
    return EpisodeEngine(
        backend or ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        max_tokens=max_tokens,
        initial_text="P",
        initial_token_ids=[7],
    )
