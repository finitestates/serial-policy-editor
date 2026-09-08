from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Accept, Hold, SelectRawRank
from trajectory_editor.episode_cli import (
    _live_edge_menu,
    build_parser,
    main as episode_main,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_projector import (
    project_episode,
    project_fork_map,
    project_lineage,
)
from trajectory_editor.episode_store import EpisodeStore


class StructuredLiveEdgeIO(ScriptedIO):
    @property
    def supports_live_choices(self) -> bool:
        return True

    def __init__(self, responses: list[str | None]) -> None:
        super().__init__(responses)
        self.edge_calls: list[dict[str, object]] = []

    def read_live_edge_command(self, **kwargs: object) -> str | None:
        self.edge_calls.append(kwargs)
        return self.responses.pop(0)


def engine(*, max_tokens: int = 4) -> EpisodeEngine:
    return EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplingConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        max_tokens=max_tokens,
        initial_text="P",
        initial_token_ids=[7],
    )


def create_episode(
    store: EpisodeStore,
    runtime: EpisodeEngine,
    episode_id: str,
    *,
    parent_episode_id: str | None = None,
    fork_boundary: int | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    store.create_episode(
        episode_id=episode_id,
        initial_text="P",
        initial_token_ids=runtime.initial_token_ids,
        sampling=runtime.sampling,
        stream_fingerprint=runtime.stream_fingerprint,
        coordinate_offset=0,
        max_tokens=runtime.max_tokens,
        backend={"backend": "fake"},
        parent_episode_id=parent_episode_id,
        fork_boundary=fork_boundary,
        metadata=metadata,
    )


def persist_open_episode(store: EpisodeStore, runtime: EpisodeEngine, episode_id: str) -> None:
    store.update_episode(
        episode_id,
        visible_text=runtime.backend.render(runtime.visible_token_ids),
        max_tokens=runtime.max_tokens,
    )


def test_lineage_separates_forks_from_replays_and_tracks_status(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "root")
        store.update_episode(store.get_episode("root")["episode_id"], visible_text="", max_tokens=4)

        create_episode(
            store,
            runtime,
            "branch-a",
            parent_episode_id="root",
            fork_boundary=2,
            metadata={"mode": "fork"},
        )
        store.finish_episode(
            "branch-a",
            visible_text="",
            terminal_token_id=None,
            terminal_reason="menu-end",
        )
        create_episode(
            store,
            runtime,
            "branch-b",
            parent_episode_id="root",
            fork_boundary=1,
            metadata={"mode": "fork"},
        )
        store.update_episode("branch-b", visible_text="", max_tokens=4)
        create_episode(
            store,
            runtime,
            "replay",
            parent_episode_id="branch-b",
            fork_boundary=0,
            metadata={
                "mode": "serial-policy-replay",
                "spr_source": "branch-a",
            },
        )
        store.finish_episode(
            "replay",
            visible_text="",
            terminal_token_id=None,
            terminal_reason="menu-end",
        )
        create_episode(
            store,
            runtime,
            "replay-fork",
            parent_episode_id="replay",
            fork_boundary=1,
            metadata={"mode": "fork"},
        )
        store.update_episode("replay-fork", visible_text="", max_tokens=4)

        details = store.lineage("branch-b")
        rendered = project_lineage(store, "branch-b")
        replay_details = store.lineage("replay")
        replay_fork_details = store.lineage("replay-fork")

    assert details['family_root_id'] == 'root'
    assert details['tree']['episode_id'] == 'root'
    assert [child['episode_id'] for child in details['tree']['children']] == ['branch-a', 'branch-b']
    assert details['tree']['children'][0]['status'] == 'completed'
    assert [item['episode_id'] for item in details['replays']] == ['replay']
    assert details['replays'][0]['replay_source_episode_id'] == 'branch-a'
    assert [item['episode_id'] for item in details['replay_derived_forks']] == ['replay-fork']
    assert replay_details['family_root_id'] == 'root'
    assert replay_details['selected_episode_id'] == 'replay'
    assert replay_fork_details['family_root_id'] == 'replay-fork'
    assert [item['episode_id'] for item in replay_fork_details['replays']] == ['replay']
    assert 'fork family:' in rendered
    assert '* branch-b [open]' in rendered
    assert 'branch-a [completed] · fork@2' in rendered
    assert 'replay [completed] · source=branch-a · context=branch-b' in rendered
    assert 'replay-fork [open] · parent=replay · fork@1' in rendered


def test_projector_can_append_lineage_metadata_without_changing_default(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "project-lineage")
        ordinary = project_episode(store, "project-lineage").text
        with_metadata = project_episode(
            store,
            "project-lineage",
            with_lineage=True,
        ).text

    assert ordinary == 'P'
    assert 'P\n\n--- lineage ---' in with_metadata
    assert 'family root: project-lineage' in with_metadata


def test_cli_lists_lineage_without_loading_a_backend(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        runtime = engine()
        create_episode(store, runtime, "listed-lineage")
    output = StringIO()
    with redirect_stdout(output):
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--list",
                "--lineage",
                "listed-lineage",
            ]
        )

    assert status == 0
    assert 'family root: listed-lineage' in output.getvalue()


def test_live_edge_uses_the_structured_surface_when_available(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine(max_tokens=4)
        create_episode(store, runtime, "structured-edge")
        io = StructuredLiveEdgeIO(["n 6"])

        action = _live_edge_menu(io, store, "structured-edge", runtime)

    assert action == ('continue', 6)
    assert len(io.edge_calls) == 1
    assert io.edge_calls[0]['episode_id'] == '#1  P'
    assert io.edge_calls[0]['current_budget'] == 4
    assert io.output == []


def test_fork_map_marks_exact_visible_boundaries_including_zero(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "fork-map")

        selected = runtime.apply(Accept())
        store.record_action("fork-map", 0, selected)
        delegated = runtime.apply(Hold(1))
        store.record_action("fork-map", 1, delegated)
        persist_open_episode(store, runtime, "fork-map")

        text = project_fork_map(store, "fork-map")

    assert text == 'P|0| A|1| B|2|'


def test_live_edge_fork_map_returns_the_selected_absolute_boundary(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "fork-map-live")
        selected = runtime.apply(Accept())
        store.record_action("fork-map-live", 0, selected)
        persist_open_episode(store, runtime, "fork-map-live")
        io = ScriptedIO(["fm", "0"])

        action, value = _live_edge_menu(
            io, store, "fork-map-live", runtime
        )

    assert (action, value) == ('fork', 0)
    rendered = "".join(io.output)
    assert '[fm] fork map' in rendered
    assert 'P|0| A|1|' in rendered
    assert 'Fork boundary (0..1; blank cancels)' in rendered


def test_live_edge_fork_map_reprompts_bad_boundaries_and_blank_cancels(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "fork-map-cancel")
        selected = runtime.apply(Accept())
        store.record_action("fork-map-cancel", 0, selected)
        persist_open_episode(store, runtime, "fork-map-cancel")
        io = ScriptedIO(["fm", "nope", "9", "", "e"])

        action, value = _live_edge_menu(
            io, store, "fork-map-cancel", runtime
        )

    assert (action, value) == ('end', None)
    rendered = "".join(io.output)
    assert 'Fork boundary must be an integer.' in rendered
    assert 'Fork boundary must be 0..1.' in rendered


def test_full_evidence_footnotes_teacher_tokens_but_not_hold_tokens(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "project")

        selected = runtime.apply(Accept())
        store.record_action("project", 0, selected)
        delegated = runtime.apply(Hold(1))
        store.record_action("project", 1, delegated)
        persist_open_episode(store, runtime, "project")

        text = project_episode(store, "project", full_evidence=True).text

    assert 'P A[^1] B' in text
    assert "token=' A'" in text
    assert 'teacher=accept' in text
    assert 'proposal=agree' in text
    assert 'nll=' in text
    assert 'raw-rank=1' in text
    assert 'policy-rank=' in text
    assert text.count('[^1]') == 2
    assert 'teacher=hold' not in text
    assert 'proposal-id' not in text
    assert 'MODEL[' not in text


def test_model_probabilities_are_opt_in_and_structurally_distinct(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "model-probs")
        selected = runtime.apply(SelectRawRank(2))
        store.record_action("model-probs", 0, selected)
        persist_open_episode(store, runtime, "model-probs")

        ordinary = project_episode(
            store, "model-probs", full_evidence=True
        ).text
        with_model = project_episode(
            store,
            "model-probs",
            full_evidence=True,
            with_model_probs=True,
        ).text
        model_only = project_episode(
            store, "model-probs", with_model_probs=True
        ).text

    assert 'MODEL[' not in ordinary
    assert 'MODEL[raw-p=' in with_model
    assert 'decoder-p=' in with_model
    assert 'nll=' in with_model
    assert 'proposal=different' in with_model
    assert 'MODEL[raw-p=' in model_only
    assert 'nll=' not in model_only


def test_annotation_and_evidence_footnotes_share_numbering_cleanly(tmp_path) -> None:
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        runtime = engine()
        create_episode(store, runtime, "annotated")
        selected = runtime.apply(Accept())
        store.record_action("annotated", 0, selected)
        store.record_interaction(
            "annotated", 0, "note-before", {"text": "teacher note"}
        )
        persist_open_episode(store, runtime, "annotated")

        projection = project_episode(
            store,
            "annotated",
            annotations="footnotes",
            full_evidence=True,
        )

    assert 'P[^1] A[^2]' in projection.text
    assert '[^1]: teacher note' in projection.text
    assert "[^2]: token=' A'" in projection.text
    assert projection.annotations == ('teacher note',)


def test_with_model_alias_sets_same_projection_option() -> None:
    parser = build_parser()
    long_form = parser.parse_args(["--project", "ep", "--with-model-probs"])
    alias = parser.parse_args(["--project", "ep", "--with-model"])
    assert long_form.with_model_probs
    assert alias.with_model_probs
    assert parser.parse_args(['--project', 'ep', '--with-lineage']).with_lineage


def test_live_edge_project_uses_full_evidence_without_model_probs(tmp_path) -> None:
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["accept", "p", "e"])
    with (
        patch(
            "trajectory_editor.episode_cli._backend",
            return_value=ConformingFakeBackend(),
        ),
        patch("trajectory_editor.episode_cli.TerminalIO", return_value=io),
    ):
        status = episode_main(
            [
                "--workspace",
                str(workspace),
                "--model",
                "fake.gguf",
                "--new-prompt",
                "P",
                "--episode-id",
                "live-project",
                "--max-tokens",
                "1",
                "--plain-ui",
            ]
        )

    rendered = "".join(io.output)
    assert status == 0
    assert 'teacher=select-raw-rank' in rendered
    assert 'proposal=agree' in rendered
    assert 'MODEL[' not in rendered
