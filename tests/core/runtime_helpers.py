from __future__ import annotations

from contextlib import contextmanager

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.terminal_contracts import TerminalCapabilities


class LiveScriptedIO(ScriptedIO):
    """Request-level live terminal fake with no renderer selection in callers."""

    def __init__(self, responses: list[str | None]) -> None:
        super().__init__(responses)
        self.initial_commands: list[str | None] = []
        self.remaining_tokens: list[int] = []
        self.entered = 0

    @property
    def capabilities(self) -> TerminalCapabilities:
        return TerminalCapabilities(live_views=True, seamless_review=True)

    def read_choice(self, state):
        self.choice_requests.append(state)
        self.initial_commands.append(state.initial_command)
        self.remaining_tokens.append(state.remaining_tokens)
        if not self.responses:
            raise AssertionError("unexpected live input request")
        return self.responses.pop(0)

    def read_edge(self, state):
        self.edge_requests.append(state)
        return self.read("live edge> ")

    @contextmanager
    def session(self):
        self.entered += 1
        yield self


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
