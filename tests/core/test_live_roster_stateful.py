"""Generated session journeys with an independent model of retained histories.

No model downloads, terminal driver, sleeps, or sampling randomness are needed.
Hypothesis shrinks a failing write/fork/root/switch sequence into a reproducer.
"""

from unittest.mock import patch

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_session import LiveSession, LiveSessionRoster


class LiveRosterMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.backend = ConformingFakeBackend()
        self.backend.reset([7])
        session = LiveSession(
            EpisodeEngine(
                self.backend, initial_text="P", initial_token_ids=[7],
                sampling=SamplerConfig(),
            ),
            prompt="P",
        )
        self.roster = LiveSessionRoster(session)
        # Numbers are public roster addresses; expected histories never come
        # from production branch state or the engine being checked.
        self.histories = {1: ()}
        self.parents = {1: None}
        self.active = 1

    @rule(token=st.sampled_from((1, 2, 5, 6)))
    def write(self, token):
        self.roster.active_session.generate(
            Write(self.backend.pieces[token], mode="exact")
        )
        self.histories[self.active] += (token,)

    @rule(position=st.integers(min_value=0, max_value=1000))
    def fork(self, position):
        history = self.histories[self.active]
        boundary = position % (len(history) + 1)
        child = self.roster.fork(boundary=boundary)
        number = self.roster.number_for(child.session, child.branch.branch_id)
        self.histories[number] = history[:boundary]
        self.parents[number] = self.roster.resolve(self.active).branch_id
        # Fork retains the parent as active until an explicit switch.

    @rule(position=st.integers(min_value=0, max_value=1000))
    def switch(self, position):
        number = sorted(self.histories)[position % len(self.histories)]
        self.roster.switch(f"#{number}")
        self.active = number

    @rule(prompt=st.sampled_from(("P", "Q", "new root")))
    def new_root(self, prompt):
        root = self.roster.new_root(prompt)
        number = self.roster.number_for(root, root.branch.branch_id)
        self.histories[number] = ()
        self.parents[number] = None
        self.active = number

    @invariant()
    def histories_and_backend_match_model(self):
        entries = self.roster.entries()
        assert tuple(entry.number for entry in entries) == tuple(self.histories)
        for entry in entries:
            state = entry.session.branch_state(entry.branch_id)
            expected = self.histories[entry.number]
            assert state.visible_token_ids == expected, "retained branch history changed"
            assert state.boundary == len(expected)
            assert len(state.history_tape) == len(expected)
            assert state.identity.parent_id == self.parents[entry.number]
        active = self.roster.resolve(self.active)
        assert self.roster.active_session is active.session
        assert self.roster.active_branch.branch.branch_id == active.branch_id
        assert self.roster.active_session.engine.visible_token_ids == list(
            self.histories[self.active]
        )
        # Cache positioning is intentionally lazy. Observation is the public
        # boundary at which the active ledger must reach the shared backend.
        self.roster.active_session.engine.observe()
        assert self.backend.tokens == [7, *self.histories[self.active]], (
            "active backend prefix was not restored"
        )

    def teardown(self):
        self.roster.discard()


TestLiveRosterJourneys = LiveRosterMachine.TestCase
TestLiveRosterJourneys.pytestmark = pytest.mark.invariant


@pytest.mark.invariant
def test_roster_oracle_rejects_missing_backend_restore():
    """Deliberately broken activation must fail the same generated oracle."""
    machine = LiveRosterMachine()
    try:
        machine.write(1)
        machine.new_root("Q")
        machine.write(2)
        machine.switch(0)
        with patch.object(EpisodeEngine, "_ensure_backend_positioned", return_value=None):
            with pytest.raises(AssertionError, match="active backend prefix"):
                machine.histories_and_backend_match_model()
    finally:
        machine.teardown()
