"""Contracts for persistence-free live sessions and branch handles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor import EpisodeEngine, LiveSession, SamplerConfig
from trajectory_editor.core.actions import Accept, Hold, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import ReplayExpectation
from tests.core.test_lifecycle_contracts import NoEogBackend


def engine() -> EpisodeEngine:
    return EpisodeEngine(
        ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0)
    )


@pytest.mark.current_workflow
def test_live_session_owns_records_metadata_and_adapter_hooks():
    events = []
    exports = []
    session = LiveSession(
        engine(),
        prompt="Prompt",
        environment_stamp={"model": "fake", "revision": "test"},
        recorder=events.append,
        export_targets={"memory": lambda session: exports.append(session.history_tape)},
    )
    episode = session.branch_handle(session.branch.branch_id)

    outcome = episode.generate(Accept())
    result = episode.export("memory")

    assert outcome.visible_token_ids == (1,)
    assert episode.prompt == "Prompt"
    assert episode.sampler.temperature == 0.0
    assert episode.environment_stamp["model"] == "fake"
    assert episode.tape[0].expectation == outcome.expectation()
    assert result is None
    assert exports == [episode.history_tape]
    assert [event.kind for event in events] == ["created", "generated", "exported"]


@pytest.mark.invariant
def test_rewind_trims_records_restores_live_engine_and_retains_tail_for_review():
    session = LiveSession(engine())
    episode = session.branch_handle(session.branch.branch_id)
    episode.generate(Write(" A B", mode="exact"))

    rewind = episode.rewind(1)

    assert episode.engine.visible_token_ids == [1]
    assert episode.tape[0].action == Write(" A", mode="exact")
    assert episode.outcomes[0].visible_token_ids == (1,)
    assert rewind.boundary == 1
    assert len(rewind.discarded_tape) == 1


@pytest.mark.invariant
def test_rewind_inside_a_conditional_hold_clears_the_discarded_stop_condition():
    session = LiveSession(
        EpisodeEngine(
            NoEogBackend(),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        )
    )
    episode = session.branch_handle(session.branch.branch_id)
    episode.generate(Hold(3, "sentence"))

    episode.rewind(1)

    assert episode.tape[0].action == Hold(1)
    assert episode.tape[0].expectation.stop_reason == "requested-length"


@pytest.mark.invariant
def test_rewind_removes_a_zero_width_replay_handoff_at_the_target_boundary():
    session = LiveSession(engine())
    episode = session.branch_handle(session.branch.branch_id)
    expected = ReplayExpectation((2,))
    outcome = episode.generate(Accept(), expectation=expected, replay=True)

    assert outcome.status == "handed-off"
    episode.rewind(0)

    assert episode.history_tape == ()
    assert episode.rewind_state is not None
    assert len(episode.rewind_state.discarded_tape) == 1
    assert episode.rewind_state.discarded_tape[0].action == Accept()


@pytest.mark.invariant
def test_forked_branch_can_rewind_before_its_fork_point_and_continue_locally():
    session = LiveSession(engine(), branch_id="root")
    session.generate(Accept())
    child = session.fork(boundary=1, branch_id="child")

    child.rewind(0)
    child.generate(Write("C", mode="exact"))

    assert child.history_visible_token_ids == (3,)
    assert child.tape == (child.history_tape[0],)
    assert child.tape[0].action == Write("C", mode="exact")


@pytest.mark.invariant
def test_fork_creates_an_independent_child_with_lineage_and_inherited_history():
    session = LiveSession(engine(), branch_id="root")
    parent = session.branch_handle("root")
    parent.generate(Accept())

    child = parent.fork(boundary=1, branch_id="child")

    assert child.branch.parent_id == "root"
    assert child.branch.fork_boundary == 1
    # A session keeps root-relative boundaries for every branch.  The child
    # resumes on the one shared backend instead of receiving a new model.
    assert child.engine.initial_token_ids == (7,)
    assert child.engine.visible_token_ids == [1]
    assert child.history_tape == parent.tape
    assert parent.fork_state is not None
    assert parent.fork_state.child == child.branch


@pytest.mark.current_workflow
def test_live_session_reactivates_branch_records_on_one_backend():
    backend = ConformingFakeBackend()
    session = LiveSession(
        EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0)),
        branch_id="root",
    )
    session.generate(Accept())
    child = session.fork(boundary=1, branch_id="child")
    child.generate(Write(" B", mode="exact"))

    session.activate("root")
    session.generate(Write("C", mode="exact"))
    assert session.engine.backend is backend
    assert session.history_visible_token_ids == (1, 3)

    session.activate("child")
    assert session.engine.backend is backend
    assert session.history_visible_token_ids == (1, 2)
    assert [outcome.boundary_after for outcome in session.history_outcomes] == [1, 2]


@pytest.mark.invariant
def test_nested_fork_truncates_a_partial_action_in_root_boundaries():
    session = LiveSession(engine(), branch_id="root")
    session.generate(Accept())
    child = session.fork(boundary=1, branch_id="child")
    child.generate(Write(" A B", mode="exact"))
    grandchild = child.fork(boundary=2, branch_id="grandchild")

    assert grandchild.history_visible_token_ids == (1, 1)
    assert [outcome.boundary_after for outcome in grandchild.history_outcomes] == [1, 2]
    assert grandchild.history_tape[-1].action == Write(" A", mode="exact")


@pytest.mark.invariant
def test_branch_reactivation_restores_sampler_and_budget_at_the_fork_point():
    session = LiveSession(
        EpisodeEngine(
            ConformingFakeBackend(),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
            max_tokens=1,
        ),
        branch_id="root",
    )
    session.generate(Accept())
    session.resume(max_tokens=3)
    expected_sampler = replace(session.sampler, temperature=0.7)
    session.set_sampler(expected_sampler)
    child = session.fork(boundary=1, branch_id="child")

    assert child.engine.sampling == expected_sampler
    assert child.engine.max_tokens == 3
    assert child.engine.checkpoint_boundary == 4

    session.activate("root")
    session.set_sampler(replace(session.sampler, temperature=0.2))
    session.activate("child")
    assert session.sampler == expected_sampler
    assert session.engine.remaining == 3


@pytest.mark.invariant
def test_optional_cache_snapshots_do_not_change_branch_semantics():
    class SnapshotBackend(ConformingFakeBackend):
        def snapshot_cache(self):
            return tuple(self.tokens)

        def restore_cache(self, snapshot):
            self.tokens = list(snapshot)

    backend = SnapshotBackend()
    session = LiveSession(
        EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0)),
        branch_id="root",
    )
    session.generate(Accept())
    child = session.fork(boundary=1, branch_id="child")
    child.generate(Write(" B", mode="exact"))

    session.activate("root")
    assert session.branch_states["root"].backend_cache_snapshot is not None
    session.activate("child")
    assert session.history_visible_token_ids == (1, 2)


@pytest.mark.invariant
def test_quit_and_discard_need_no_persistence_and_prevent_further_generation():
    session = LiveSession(engine())
    episode = session.branch_handle(session.branch.branch_id)
    episode.quit("user-quit")

    assert episode.status == "quit"
    assert episode.engine.terminal_reason == "user-quit"
    with pytest.raises(EditorError, match="quit"):
        episode.generate()

    discarded_session = LiveSession(engine())
    discarded = discarded_session.branch_handle(discarded_session.branch.branch_id)
    discarded.generate(Accept())
    discarded.discard()

    assert discarded.status == "discarded"
    assert discarded.tape == ()
    with pytest.raises(EditorError, match="discarded"):
        discarded.generate()


@pytest.mark.current_workflow
def test_live_session_import_does_not_load_episode_store():
    # Check the actual package import graph in a clean interpreter; other test
    # modules are allowed to use persistence in the main test process.
    source_root = Path(__file__).parents[2] / "core" / "src"
    environment = dict(os.environ, PYTHONPATH=str(source_root))
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from trajectory_editor import LiveSession; "
            "assert 'trajectory_editor.episode_store' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert check.returncode == 0, check.stderr
