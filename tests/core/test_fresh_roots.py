from __future__ import annotations

from unittest.mock import patch

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.actions import Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_materializer import save_live_family
from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.ephemeral_runtime import ephemeral_edge_menu
from trajectory_editor.fresh_episode import fresh_root_from


class DurableFakeBackend(ConformingFakeBackend):
    def provenance(self, *, include_model_sha256: bool = True):
        del include_model_sha256
        return {
            "backend": "llama.cpp",
            "vocabulary_size": self.vocabulary_size(),
        }


def _session(*, guidance: bool = False, max_tokens: int = 3) -> LiveSession:
    backend = ConformingFakeBackend()
    sampling = SamplerConfig(
        temperature=0.25,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
        cfg_unconditional_prompt="guide" if guidance else None,
        cfg_prefix_tokens=4 if guidance else 0,
    )
    engine = EpisodeEngine(
        backend,
        initial_text="P",
        initial_token_ids=[7],
        sampling=sampling,
        max_tokens=max_tokens,
        guidance_backend=ConformingFakeBackend() if guidance else None,
    )
    return LiveSession(engine, prompt="P")


def test_fresh_root_factory_copies_runtime_settings_but_starts_a_new_ledger():
    source = _session(max_tokens=4)
    source.generate(Write(" A", mode="exact"))

    fresh = fresh_root_from(source.engine, "Q")

    assert fresh.backend is source.engine.backend
    assert fresh.guidance_backend is source.engine.guidance_backend
    assert fresh.sampling == source.sampler
    assert fresh.max_tokens == 4
    assert fresh.remaining == 4
    assert fresh.boundary == 0
    assert fresh.visible_token_ids == []
    assert fresh.coordinate_offset == 0
    assert fresh.stream_fingerprint is not None
    assert fresh.initial_text == "Q"


def test_detached_root_rebuilds_primary_and_guidance_prefix():
    session = _session(guidance=True)
    session.generate(Write(" A", mode="exact"))
    roster = LiveSessionRoster(session)
    roster.new_root("Q")

    roster.switch("#1")

    assert session.engine.backend.tokens == [7, 1]
    assert session.engine.guidance_backend is not None
    session.engine.observe()  # Guidance positioning is lazy.
    assert session.engine.guidance_backend.tokens == [7, 1]
    # Roster inspection is semantic-only while the root is detached.
    session.suspend()
    assert session.branch_states[session.branch.branch_id].visible_token_ids == (1,)


def test_ephemeral_help_and_bare_new_expose_the_polished_commands():
    from trajectory_editor.edge_tui import _edge_header

    help_text = "".join(
        fragment
        for _, fragment in _edge_header(
            episode_id="#1",
            boundary=0,
            current_budget=3,
            remaining_tokens=3,
            sampler_summary="temp=1",
            mode="session",
        )
    )
    assert "#N" in help_text
    assert "new TEXT" in help_text

    class PromptIO(ScriptedIO):
        def __init__(self):
            super().__init__(["new", "Q"])
            self.requests = []

        def prompt(self, request):
            self.requests.append(request)
            return super().prompt(request)

    io = PromptIO()
    action, value = ephemeral_edge_menu(io, _session())
    assert (action, value) == ("new", "Q")
    assert len(io.requests) == 1 and io.requests[0].multiline


def test_plain_bare_new_uses_line_prompt_without_cli_or_live_application(monkeypatch):
    import builtins

    from trajectory_editor.tui import TerminalIO

    replies = iter(("new", "", "Q"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))
    original_import = builtins.__import__

    def no_cli_import(name, *args, **kwargs):
        if name == "episode_cli" or name.endswith(".episode_cli"):
            raise AssertionError("plain composer imported the CLI")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_cli_import)
    terminal = TerminalIO(live_choices=False)
    with terminal.session():
        assert ephemeral_edge_menu(terminal, _session()) == ("new", "Q")


def test_bare_new_cancellation_returns_to_ephemeral_edge():
    io = ScriptedIO(["new", None, "q"])
    assert ephemeral_edge_menu(io, _session()) == ("quit", None)


def test_ephemeral_new_has_one_model_load_and_global_stable_addresses(tmp_path):
    workspace = tmp_path / "must-not-exist.sqlite3"
    backend = ConformingFakeBackend()
    loads: list[object] = []
    new_root_samplers: list[SamplerConfig] = []
    io = ScriptedIO(
        [
            "1",       # original root action
            "fork 1",  # original root #1 -> branch #2
            "s temperature=0.7",  # change the active sampler at EDGE
            "new Q",   # unrelated root #3
            "1",       # action in root #3
            "fork 1",  # root #3 -> branch #4
            "branches",
            "#1",      # bare global address returns to the original root
            "q",
        ]
    )

    def load(_args):
        loads.append(object())
        return backend

    from trajectory_editor import ephemeral_runtime

    class RecordingRoster(LiveSessionRoster):
        def new_root(self, prompt):
            root = super().new_root(prompt)
            new_root_samplers.append(root.sampler)
            return root

    with patch("trajectory_editor.episode_backend_loader.load_backend", side_effect=load), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ), patch.object(ephemeral_runtime, "LiveSessionRoster", RecordingRoster):
        assert main(
            [
                "--ephemeral",
                "--plain-ui",
                "--model",
                "fake",
                "--max-tokens",
                "1",
                "--workspace",
                str(workspace),
                "--new-prompt",
                "P",
            ]
        ) == 0

    assert len(loads) == 1
    assert not workspace.exists()
    assert new_root_samplers[0].temperature == 0.7
    assert any("[#N] switch" in item for item in io.output)
    listing = next(item for item in io.output if item.startswith("Live branches:"))
    assert "#1" in listing and "#2" in listing
    assert "#3" in listing and "#4" in listing


def test_durable_new_is_parentless_and_bare_number_returns_to_prior_episode(tmp_path):
    workspace = tmp_path / "episodes.sqlite3"
    backend = DurableFakeBackend()
    io = ScriptedIO(["1", "s temperature=0.7", "new Q", "#1", "q"])

    with patch("trajectory_editor.episode_backend_loader.load_backend", return_value=backend), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(
            [
                "--plain-ui",
                "--model",
                "fake",
                "--max-tokens",
                "1",
                "--workspace",
                str(workspace),
                "--new-prompt",
                "P",
            ]
        ) == 0

    with EpisodeStore(workspace) as store:
        episodes = store.episode_relation_rows()
        assert len(episodes) == 2
        original = store.resolve_id("#1")
        fresh = store.resolve_id("#2")
        assert store.get_episode(fresh)["parent_episode_id"] is None
        assert store.get_episode(fresh)["initial_text"] == "Q"
        assert store.get_episode(fresh)["max_tokens"] == 1
        assert store.get_episode(fresh)["checkpoint_boundary"] == 1
        assert store.sampling_segment(fresh, 0)["sampling"]["temperature"] == 0.7
        assert store.actions(fresh) == []
        assert store.tokens(fresh) == []
        assert store.get_episode(original)["parent_episode_id"] is None
        assert store.label(original).startswith("#1")


def test_save_family_only_materializes_the_selected_root_family(tmp_path):
    session = _session()
    roster = LiveSessionRoster(session)
    session.generate(Write(" A", mode="exact"))
    roster.fork(boundary=1)
    roster.new_root("Q")
    new_session = roster.active_session
    new_session.generate(Write(" B", mode="exact"))
    roster.fork(boundary=1)

    workspace = tmp_path / "family.sqlite3"
    identifiers = save_live_family(
        new_session,
        workspace,
        {"backend": "fake"},
        root_episode_id="new-root",
    )

    with EpisodeStore(workspace) as store:
        assert set(identifiers) == {
            new_session.branch.branch_id,
            next(
                branch_id
                for branch_id, state in new_session.branch_states.items()
                if state.identity.parent_id == new_session.branch.branch_id
            ),
        }
        assert len(store.episode_relation_rows()) == 2
        assert store.get_episode("new-root")["initial_text"] == "Q"
