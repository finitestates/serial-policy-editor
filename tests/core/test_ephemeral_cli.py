"""CLI contracts for persistence-free live sessions."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.actions import Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import _ephemeral_edge_menu, main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_replay_source import replay_procedure
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.teacher_plan import load_teacher_tape_jsonl


def _run(
    tmp_path,
    commands: list[str | None],
    *flags: str,
    plain_ui: bool = True,
) -> ScriptedIO:
    io = ScriptedIO(commands)
    ui_flag = ["--plain-ui"] if plain_ui else []
    with patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        side_effect=lambda _args: ConformingFakeBackend(),
    ), patch("trajectory_editor.episode_cli.TerminalIO", return_value=io):
        assert main(["--ephemeral", "--model", "fake", *ui_flag, *flags]) == 0
    return io


def test_ephemeral_quit_never_opens_the_default_or_selected_workspace(tmp_path):
    workspace = tmp_path / "should-not-exist.sqlite3"

    _run(tmp_path, ["q", "q"], "--workspace", str(workspace), "--new-prompt", "P")

    assert not workspace.exists()


def test_ephemeral_export_and_save_materialize_only_the_selected_branch(tmp_path):
    exported = tmp_path / "branch.jsonl"
    workspace = tmp_path / "saved.sqlite3"

    _run(
        tmp_path,
        ["1", "q", f"export {exported}", f"save {workspace} selected", "q"],
        "--new-prompt", "P",
    )

    tape = load_teacher_tape_jsonl(exported)
    assert len(tape.plan) == 1
    with EpisodeStore(workspace) as store:
        saved = store.get_episode("selected")
        assert saved["metadata"]["mode"] == "ephemeral-save"
        assert len(replay_procedure(store, "selected")) == 1


def test_ephemeral_fork_selects_a_new_live_branch_without_a_workspace(tmp_path):
    workspace = tmp_path / "should-not-exist.sqlite3"

    io = _run(
        tmp_path,
        ["1", "q", "fork 1", "q", "branches", "q"],
        "--workspace", str(workspace), "--new-prompt", "P",
    )

    assert not workspace.exists()
    assert any("Forked live branch" in item for item in io.output)
    assert any("Live branches:" in item for item in io.output)


def test_ephemeral_rewind_can_move_a_forked_branch_before_its_fork_point(tmp_path):
    workspace = tmp_path / "rewound-child.sqlite3"

    _run(
        tmp_path,
        [
            "1", "q", "fork 1", "q", "rewind 0", "1", "q",
            f"save {workspace} selected", "q",
        ],
        "--new-prompt", "P",
    )

    with EpisodeStore(workspace) as store:
        saved = store.get_episode("selected")
        assert saved["visible_text"] == " A"
        assert len(store.actions("selected")) == 1


def test_ephemeral_nested_fork_save_keeps_only_the_selected_prefix(tmp_path):
    workspace = tmp_path / "selected.sqlite3"

    _run(
        tmp_path,
        [
            "1", "q", "fork 1", "x  A B", "q", "fork 2", "q",
            f"save {workspace} selected", "q",
        ],
        "--new-prompt", "P",
    )

    with EpisodeStore(workspace) as store:
        saved = store.get_episode("selected")
        assert saved["visible_text"] == " A A"
        assert [
            (row["boundary_before"], row["boundary_after"])
            for row in store.actions("selected")
        ] == [(0, 1), (1, 2)]


def test_ephemeral_fork_map_uses_root_relative_boundaries():
    session = LiveSession(
        EpisodeEngine(
            ConformingFakeBackend(),
            initial_text="P",
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        ),
        prompt="P",
    )
    session.generate(Write(" A", mode="exact"))
    io = ScriptedIO(["fm", "", "q"])

    action, value = _ephemeral_edge_menu(io, session)

    assert (action, value) == ("quit", None)
    assert any(item == "P|0| A|1|" for item in io.output)


def test_ephemeral_branches_use_numeric_aliases_for_switching():
    session = LiveSession(
        EpisodeEngine(
            ConformingFakeBackend(),
            initial_text="P",
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        ),
        prompt="P",
    )
    session.generate(Write(" A", mode="exact"))
    child = session.fork(boundary=1)
    session.activate(child.branch.branch_id)
    io = ScriptedIO(["branches", "switch 1"])

    action, value = _ephemeral_edge_menu(io, session)

    assert action == "switch"
    assert value == next(iter(session.branch_tree.nodes))
    listing = next(item for item in io.output if item.startswith("Live branches:"))
    assert "  1  from=root" in listing
    assert "live-" not in listing


def test_ephemeral_save_family_materializes_live_lineage(tmp_path):
    workspace = tmp_path / "family.sqlite3"

    _run(
        tmp_path,
        ["1", "q", "fork 1", "1", "q", f"save-family {workspace} root-save", "q"],
        "--new-prompt", "P",
    )

    with EpisodeStore(workspace) as store:
        root = store.get_episode("root-save")
        children = [row for row in store.list_episodes() if row["parent_episode_id"] == "root-save"]
        assert root["parent_episode_id"] is None
        assert len(children) == 1
        assert children[0]["fork_boundary"] == 1
        assert store.get_episode(children[0]["episode_id"])["metadata"]["mode"] == "ephemeral-family-save"


class _LiveContextIO(ScriptedIO):
    def __init__(self, responses, *, supports_live_choices=True):
        super().__init__(responses)
        self.entered = 0
        self._supports_live_choices = supports_live_choices
        self.edge_modes: list[str] = []

    @property
    def supports_live_choices(self):
        return self._supports_live_choices

    def read_choice(self, choice, **kwargs):
        del choice, kwargs
        return self.read("live choice> ")

    def read_live_edge_command(self, **kwargs):
        self.edge_modes.append(kwargs["mode"])
        return self.read("live edge> ")

    @contextmanager
    def live_session(self):
        self.entered += 1
        yield self


def test_ephemeral_uses_live_ui_by_default_and_plain_ui_is_an_opt_out(tmp_path):
    captured: list[bool] = []
    io = _LiveContextIO(["q", "q"])

    def terminal_io(*, live_choices, live_theme):
        del live_theme
        captured.append(live_choices)
        return io

    with patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        side_effect=lambda _args: ConformingFakeBackend(),
    ), patch("trajectory_editor.episode_cli.TerminalIO", side_effect=terminal_io):
        assert main(["--ephemeral", "--model", "fake", "--new-prompt", "P"]) == 0

    assert captured == [True]
    assert io.entered == 1
    assert io.edge_modes == ["session"]

    captured.clear()
    plain = _LiveContextIO(["q", "q"], supports_live_choices=False)
    with patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        side_effect=lambda _args: ConformingFakeBackend(),
    ), patch(
        "trajectory_editor.episode_cli.TerminalIO",
        side_effect=lambda *, live_choices, live_theme: (
            captured.append(live_choices) or plain
        ),
    ):
        assert main([
            "--ephemeral", "--model", "fake", "--plain-ui", "--new-prompt", "P",
        ]) == 0

    assert captured == [False]


def test_ephemeral_live_policy_keeps_seamless_review_enabled():
    from trajectory_editor.episode_cli import build_parser
    from trajectory_editor.episode_policy_setup import ephemeral_policy

    live = _LiveContextIO([])
    args = build_parser(include_vector=False).parse_args([])

    live_policy = ephemeral_policy(args, live)
    plain_policy = ephemeral_policy(args, ScriptedIO([]))

    assert live_policy.seamless is True
    assert plain_policy.seamless is False
    assert live_policy.view_preferences is plain_policy.view_preferences
