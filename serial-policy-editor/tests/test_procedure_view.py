from dataclasses import replace
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from tests.test_replay_eog import engine, create
from tests.test_episode_runtime import NoEogBackend
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_actions import SelectRawRank, Write, Hold
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_projector import project_procedure
from trajectory_editor.episode_cli import main
from trajectory_editor.domain import SamplingConfig


def test_procedure_metadata_whitespace_writes_and_sampler_transitions(tmp_path):
    e = engine()
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = store.create_episode(
            initial_text="Initial\nprefix", initial_token_ids=[7],
            sampling=e.sampling, stream_fingerprint=e.stream_fingerprint,
            coordinate_offset=0, max_tokens=None,
            backend={"backend": "llama.cpp", "model_path": "/private/models/Llama.gguf"},
        )
        first = e.apply(SelectRawRank(1))
        store.record_action(identifier, 0, first)
        second = e.apply(Write(r"literal \n # text", "exact"))
        store.record_action(identifier, 1, second)
        next_sampling = replace(e.sampling, temperature=0.6, seed=987)
        store.record_sampling_segment(
            identifier, start_boundary=2, sampling=next_sampling,
            stream_fingerprint=e.stream_fingerprint, coordinate_offset=0)
        store.record_action(identifier, 2, e.apply(Write("hello")))
        final = replace(next_sampling, seed=432)
        store.record_sampling_segment(
            identifier, start_boundary=3, sampling=final,
            stream_fingerprint=e.stream_fingerprint, coordinate_offset=0)
        text = project_procedure(store, identifier)
    assert "MODEL   : Llama.gguf" in text
    assert "/private" not in text
    assert "BACKEND : llama.cpp" in text
    assert "SAMPLER : temperature=0" in text
    assert "P       : Initial\\nprefix" in text
    assert "# A" in text
    assert "1 : x literal \\n # text" in text
    assert "2 : q\n2 : s temperature=0.6 seed=987\n2 : c" in text
    assert "2 : t hello" in text
    assert text.endswith("3 : q\n3 : s seed=432")
    assert "history_scope" not in text


def test_derived_hold_and_rejected_attempts(tmp_path):
    e = EpisodeEngine(NoEogBackend(), initial_token_ids=[7],
                      sampling=SamplingConfig(temperature=0))
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = create(store, e)
        outcome = e.apply(Hold(5), expectation=ReplayExpectation((1, 2, 6)),
                          replay=True, divergence_policy="handoff")
        store.record_action(identifier, 0, outcome)
        store.record_interaction(identifier, 2, "seamless-rewind", {})
        text = project_procedure(store, identifier)
        step = store.replay_procedure(identifier)[0]
        assert step["action"] == store.replay_tape(identifier)[0][0] == Hold(2)
    assert "0 : h 2 # A B" in text
    assert text.endswith("2 : q")
    assert "rewind" not in text


def test_empty_procedure_cli_does_not_load_model(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    with EpisodeStore(path) as store:
        create(store, engine())
    output = StringIO()
    with patch("trajectory_editor.episode_cli._backend", side_effect=AssertionError("no model")), redirect_stdout(output):
        assert main(["--workspace", str(path), "--project", "test", "--procedure"]) == 0
    assert output.getvalue().rstrip().endswith("0 : q")


def test_comments_distinguish_leading_space_and_newline(tmp_path):
    e = engine()
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = create(store, e)
        for ordinal, text in enumerate(["The", " The", "\n"]):
            outcome = e.apply(Write("hello", "exact"))
            # Preserve real evidence structure while supplying representative
            # tokenizer text forms absent from this tiny fake vocabulary.
            token = replace(outcome.evidence[0], text=text)
            outcome = replace(outcome, action=SelectRawRank(5), evidence=(token,))
            store.record_action(identifier, ordinal, outcome)
        text = project_procedure(store, identifier)
    assert "#The\n" in text
    assert "# The\n" in text
    assert "#\\n\n" in text
