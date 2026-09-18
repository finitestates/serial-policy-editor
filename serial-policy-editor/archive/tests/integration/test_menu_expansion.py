import numpy as np
from unittest.mock import patch

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.live_tui import _visible_candidates, _render_choice


class LargeBackend(ConformingFakeBackend):
    def vocabulary_size(self):
        return 1024

    def last_logits(self):
        return np.linspace(10, -10, self.vocabulary_size())

    def token_text(self, token_id):
        return f" token{token_id}"

    def render(self, token_ids, *, special=False):
        return "".join(self.token_text(token) for token in token_ids)


def engine():
    return EpisodeEngine(LargeBackend(), initial_token_ids=[7],
                         sampling=SamplingConfig(temperature=0))


def test_repeated_expansion_and_neighborhood_do_not_inflate_main_count():
    runtime = engine()

    class InspectIO:
        supports_live_choices = True

        def __init__(self):
            self.commands = iter(["m10"] * 20 + ["ms 900", "m10", "212"])
            self.main_sizes = []
            self.views = []

        def read_choice(self, choice, **kwargs):
            self.main_sizes.append(len(choice.candidates))
            self.views.append(tuple(row.rank for row in kwargs["display_candidates"]))
            return next(self.commands)

    io = InspectIO()
    action = InteractivePolicy(io=io).choose(runtime, runtime.observe())
    assert io.main_sizes[20:24] == [212, 212, 222]
    assert 900 in io.views[21]
    assert io.views[22] == tuple(range(1, 223))
    assert action.rank == 212
    assert runtime.boundary == 0


def test_oversized_request_stops_at_vocabulary_size():
    runtime = engine()

    class InspectIO:
        supports_live_choices = True

        def __init__(self):
            self.commands = iter(["m 999999999999", "m10", "1024"])
            self.sizes = []

        def read_choice(self, choice, **kwargs):
            self.sizes.append(len(choice.candidates))
            return next(self.commands)

    io = InspectIO()
    action = InteractivePolicy(io=io).choose(runtime, runtime.observe())
    assert io.sizes == [12, 1024, 1024]
    assert action.rank == 1024
    assert runtime.boundary == 0


def test_large_menu_windows_around_focus_and_renders():
    runtime = engine()
    observation = runtime.observe()
    rows = runtime.candidates(observation, count=300)
    shown, before, after = _visible_candidates(rows, 212, 12)
    assert len(shown) == 12
    assert any(row.rank == 212 for row in shown)
    assert before + len(shown) + after == 300
    choice = _choice_from_observation(runtime, observation, rows, context_characters=100, serial=1)
    with patch("trajectory_editor.live_tui._terminal_size", return_value=(100, 40)):
        fragments = _render_choice(choice, rows, "212", None, lambda text, mode: text, None)
    assert "token211" in "".join(text for _, text in fragments)


def test_plain_menu_can_expand_past_old_cap():
    runtime = engine()
    action = InteractivePolicy(io=ScriptedIO(["m 200", "212"])).choose(runtime, runtime.observe())
    assert action.rank == 212
