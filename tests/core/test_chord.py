"""Chord previews never enter episode history; selection uses ordinary actions."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO, SnapshotFakeBackend
from trajectory_editor.chord import Chord, _recent_context, chord_menu, parse_chord
from trajectory_editor.core.actions import Accept, SelectRawRank
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_replay_source import replay_procedure
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import _choice_from_observation
from trajectory_editor.episode_hash import token_prefix_sha256
from trajectory_editor.live_tui import action_preview
from trajectory_editor.persistent_tui import PersistentTerminalSession, _Request
from trajectory_editor.terminal_contracts import PromptRequest
from trajectory_editor.teacher_plan import load_teacher_tape_jsonl


def engine(*, budget=None, backend=None, guidance=None, sampling=None):
    return EpisodeEngine(
        backend or ConformingFakeBackend(), initial_text="P",
        sampling=sampling or SamplerConfig(), max_tokens=budget,
        guidance_backend=guidance,
    )


class OrderedSnapshotBackend(SnapshotFakeBackend):
    """Context-sensitive token paths for preview-to-commit invariants."""

    pieces = {
        0: "<EOG>",
        1: " you",
        2: " if",
        3: " can",
        4: " walk",
        5: " follow",
        6: " onward",
        7: "P",
    }

    def last_logits(self):
        order = {
            (7,): (1, 2, 0),
            (7, 1): (3, 4, 0),
            (7, 1, 3): (4, 6, 0),
            (7, 2): (5, 4, 0),
            (7, 2, 5): (6, 4, 0),
        }.get(tuple(self.tokens), (0, 6, 4))
        logits = np.full(self.vocabulary_size(), -100.0, dtype=np.float32)
        for score, token_id in enumerate(reversed(order), start=1):
            logits[token_id] = score
        return logits

    def branch_to_prefix(self, prefix_token_ids):
        self.tokens = list(prefix_token_ids)


@pytest.mark.invariant
def test_chord_discard_and_survivor_match_ordinary_actions():
    original = engine()
    before = (original.boundary, tuple(original.token_ids), original.sampling)
    chord = Chord(original, (1, 2, 4))
    assert [path.state for path in chord.paths] == ["live", "live", "EOG"]
    assert chord.advance()
    assert chord.paths[2].state == "EOG"
    assert chord.paths[0].starting_rank == 1
    assert chord.paths[0].engine.boundary == 2
    assert chord.rewind()
    assert chord.paths[0].engine.boundary == 1
    assert original.boundary == 0
    assert tuple(original.token_ids) == before[1]
    chord.discard()
    assert (original.boundary, tuple(original.token_ids), original.sampling) == before
    assert original.backend.tokens == list(before[1])

    chord = Chord(original, (1, 2, 4))
    chord.advance()
    actions = chord.select("a")
    assert actions == (SelectRawRank(1), Accept())
    for action in actions:
        original.apply(action)
    manual = engine()
    for action in (SelectRawRank(1), Accept()):
        manual.apply(action)
    assert original.token_ids == manual.token_ids
    assert original.observe().proposal_token_id == manual.observe().proposal_token_id

    zero = engine()
    initial = Chord(zero, (1, 2))
    assert initial.select("2") == (SelectRawRank(2),)
    assert zero.boundary == 0


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warmed"])
@pytest.mark.parametrize("branching", [False, True], ids=["reset", "branch"])
@pytest.mark.parametrize(
    ("selection", "expected_ids"),
    [("1", (1, 3)), ("b", (2, 5))],
    ids=["rank-alias", "letter-label"],
)
@pytest.mark.invariant
def test_chord_menu_preview_matches_committed_tokens_after_switch_and_rewind(
    warm, branching, selection, expected_ids,
):
    backend = OrderedSnapshotBackend()
    if not branching:
        backend.branch_to_prefix = None
    original = engine(backend=backend, sampling=SamplerConfig(temperature=0.0))
    base = tuple(original.token_ids)
    observation = original.observe()
    if warm:
        assert original.speculate_accept(
            observation, raw_rank=1, token_id=1, generation=1,
        )
        assert backend.tokens == list(base)

    chord = Chord(original, (1, 2))
    assert chord.advance()
    assert chord.advance()
    assert chord.rewind()

    selected_path = chord.paths[0] if selection == "1" else chord.paths[1]
    preview_ids = tuple(selected_path.token_ids)
    preview_text = backend.render(list(preview_ids))
    assert preview_ids == expected_ids
    assert tuple(
        selected_path.engine.visible_token_ids[len(chord.base_visible):]
    ) == preview_ids

    io = ScriptedIO([selection])
    result, actions = chord_menu(io, chord)
    assert result == "select"
    assert actions == (
        SelectRawRank(selected_path.starting_rank), Accept(),
    )
    assert any(
        f"{selected_path.label}  rank {selected_path.starting_rank}  LIVE\n"
        f"    {preview_text.lstrip()}" in request.body
        for request in io.prompt_requests
    )

    outcomes = [original.apply(action) for action in actions]
    committed_ids = tuple(
        token_id
        for outcome in outcomes
        for token_id in outcome.resolved_token_ids
    )
    assert committed_ids == preview_ids
    assert tuple(original.token_ids) == (*base, *preview_ids)
    assert original.backend.render(original.token_ids) == f"P{preview_text}"
    assert original.backend.tokens == original.token_ids


@pytest.mark.current_workflow
def test_chord_eog_and_budget_paths_and_stable_rank_selection():
    original = engine(budget=2)
    chord = Chord(original, (1, 2, 4))
    assert chord.paths[2].actions == [SelectRawRank(4)]
    assert chord.paths[2].state == "EOG"
    assert chord.advance()
    assert chord.paths[0].state == "budget reached"
    assert chord.paths[1].state == "EOG"
    assert chord.rewind()
    assert [path.state for path in chord.paths] == ["live", "live", "EOG"]
    assert chord.select("4") == (SelectRawRank(4),)
    assert original.boundary == 0
    outcome = original.apply(SelectRawRank(4))
    assert outcome.stop_reason == "eog"


@pytest.mark.current_workflow
def test_chord_selects_eog_after_other_paths_keep_advancing():
    original = engine()
    chord = Chord(original, (1, 2, 4))
    assert chord.advance()
    assert chord.paths[1].state == "EOG"
    assert chord.paths[0].state == "live"
    assert chord.advance()
    assert chord.paths[0].state == "EOG"
    assert chord.paths[1].actions == [SelectRawRank(2), Accept()]
    actions = chord.select("1")
    assert actions == (SelectRawRank(1), Accept(), Accept())
    for action in actions:
        original.apply(action)
    manual = engine()
    for action in actions:
        manual.apply(action)
    assert original.terminal_reason == manual.terminal_reason == "teacher-eog"
    assert original.token_ids == manual.token_ids


@pytest.mark.current_workflow
def test_chord_parser_rejects_invalid_ranks():
    assert parse_chord("chord 3 5 15", 20) == (3, 5, 15)
    for command in ("chord 1", "chord 1 1", "chord 0 2", "chord 2 20", "chord a 2"):
        with pytest.raises(EditorError):
            parse_chord(command, 8)


@pytest.mark.current_workflow
def test_chord_shows_bounded_shared_context_above_stable_paths():
    assert _recent_context(
        "first\nsecond\nthird\nfourth\nfifth", width=20, lines=3,
    ) == "…third\nfourth\nfifth"
    assert _recent_context("1234567890", width=4, lines=2) == "…567\n90"

    original = engine()
    original.apply(SelectRawRank(1))
    chord = Chord(original, (1, 2))
    heading = chord.display(width=20).split("\n\n", 1)[0]
    assert heading == "Shared context (last 4 lines):\nP A"
    assert "Paths:\na  rank 1  LIVE\n    B\nb  rank 2  LIVE" in chord.display(width=20)
    chord.advance()
    assert chord.display(width=20).split("\n\n", 1)[0] == heading
    chord.discard()


@pytest.mark.current_workflow
def test_chord_display_tracks_live_locked_and_rewound_paths():
    chord = Chord(engine(budget=2), (1, 2, 4))
    initial = chord.display(width=40)
    assert "a  rank 1  LIVE\n    A" in initial
    assert "b  rank 2  LIVE\n    B" in initial
    assert "c  rank 4  EOG\n   (no visible continuation)" in initial

    assert chord.advance()
    advanced = chord.display(width=40)
    assert "a  rank 1  BUDGET REACHED\n    A B" in advanced
    assert "b  rank 2  EOG\n    B" in advanced
    assert "c  rank 4  EOG\n   (no visible continuation)" in advanced
    assert not chord.advance()
    assert chord.display(width=40) == advanced

    assert chord.rewind()
    assert chord.display(width=40) == initial
    chord.discard()

    chord = Chord(engine(), (1, 2))
    assert chord.advance()
    ended = "b  rank 2  EOG\n    B"
    assert ended in chord.display(width=40)
    assert chord.advance()
    assert ended in chord.display(width=40)
    assert "a  rank 1  EOG\n    A B" in chord.display(width=40)
    assert chord.rewind()
    assert ended in chord.display(width=40)
    assert "a  rank 1  LIVE\n    A B" in chord.display(width=40)
    assert chord.rewind()
    assert "a  rank 1  LIVE\n    A" in chord.display(width=40)
    assert "b  rank 2  LIVE\n    B" in chord.display(width=40)
    chord.discard()


@pytest.mark.current_workflow
def test_chord_display_wraps_full_continuation_at_narrow_width():
    class LongBackend(ConformingFakeBackend):
        pieces = {**ConformingFakeBackend.pieces,
                  1: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}

    chord = Chord(engine(backend=LongBackend()), (1, 2, 4))
    body = chord.display(width=24).split("Paths:\n", 1)[1]
    first = body.split("\nb  rank", 1)[0].splitlines()
    assert first[0] == "a  rank 1  LIVE"
    assert "".join(line[3:] for line in first[1:]) == LongBackend.pieces[1]
    assert all(len(line) <= 24 for line in body.splitlines())

    narrow = chord.display(width=8).split("Paths:\n", 1)[1]
    assert all(len(line) <= 8 for line in narrow.splitlines())
    assert "LIVE" in narrow and "EOG" in narrow
    assert narrow.startswith("a ") and "\nb " in narrow and "\nc " in narrow
    chord.discard()


@pytest.mark.current_workflow
def test_chord_prompts_distinguish_options_and_help_explains_commit():
    io = ScriptedIO(["?", "q", "?", "c", "a"])
    result, actions = chord_menu(io, Chord(engine(), (1, 2)))
    assert result == "select" and actions == (SelectRawRank(1),)
    output = "".join(io.output)
    assert "Enter: advance live paths | rewind: undo one round" in output
    assert "a–z or starting rank: choose and commit | q: options" in output
    assert "c: resume chord | discard: restore episode | q: quit editor" in output
    assert "commit that path's actions and drop the other previews" in output

    locked = ScriptedIO(["", "a"])
    result, actions = chord_menu(locked, Chord(engine(budget=1), (1, 2, 4)))
    assert result == "select" and actions == (SelectRawRank(1),)
    assert "All chord paths are locked." in "".join(locked.output)


@pytest.mark.current_workflow
def test_live_choice_preview_recognizes_chord_and_validates_ranks():
    runtime = engine()
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=4)
    choice = _choice_from_observation(
        runtime, observation, candidates,
        context_text_tail=observation.context_text,
        context_token_sha256=token_prefix_sha256(list(observation.prefix_token_ids)),
        serial=1,
    )
    before = tuple(runtime.token_ids)
    preview = action_preview(choice, "chord 1 2 4", candidates, lambda text, mode: text)
    assert preview.valid and preview.label == "chord preview"
    assert "No episode action is recorded" in preview.detail
    invalid = action_preview(choice, "chord 1 9", candidates, lambda text, mode: text)
    assert not invalid.valid and invalid.label == "invalid chord"
    assert tuple(runtime.token_ids) == before


@pytest.mark.current_workflow
def test_live_chord_prompt_uses_only_current_preview_body():
    terminal = PersistentTerminalSession()
    terminal._notice = "Model loaded."
    terminal.application = SimpleNamespace(
        layout=SimpleNamespace(focus=lambda control: None), invalidate=lambda: None,
    )
    terminal._show(_Request(PromptRequest(
        "Chord > ", body="a (1) | b (2)", isolated=True,
    )))
    assert terminal._prompt_view.body.text == "a (1) | b (2)"
    assert terminal._notice == ""
    terminal._show(_Request(PromptRequest("Next > ")))
    assert terminal._prompt_view.body.text == ""

    class ChordIO(ScriptedIO):
        def __init__(self):
            super().__init__(["", "q", "c", "a"])
            self.bodies = []

        def prompt(self, request):
            assert request.isolated
            self.bodies.append(request.body)
            return self.read(request.prompt)

    io = ChordIO()
    result, actions = chord_menu(io, Chord(engine(), (1, 2)))
    assert result == "select" and actions == (SelectRawRank(1), Accept())
    assert len(io.bodies) == 4
    assert not any("Model loaded" in body for body in io.bodies)


class BranchBackend(ConformingFakeBackend):
    def __init__(self):
        super().__init__()
        self.work = []

    def reset(self, prefix_token_ids):
        self.work.append(("reset", tuple(prefix_token_ids)))
        super().reset(prefix_token_ids)

    def eval(self, token_ids):
        self.work.append(("eval", tuple(token_ids)))
        super().eval(token_ids)

    def branch_to_prefix(self, prefix_token_ids):
        prefix = list(prefix_token_ids)
        if self.tokens[:len(prefix)] == prefix:
            self.tokens = prefix
            self.work.append(("branch", tuple(prefix)))
        else:
            self.reset(prefix)


@pytest.mark.current_workflow
def test_chord_switches_primary_and_cfg_by_shared_prefix():
    primary, guidance = BranchBackend(), BranchBackend()
    sampling = SamplerConfig(cfg_unconditional_prompt="P", cfg_scale=1.4, cfg_prefix_tokens=5)
    original = engine(backend=primary, guidance=guidance, sampling=sampling)
    original.observe()
    chord = Chord(original, (1, 2, 4))
    for _ in range(2):
        chord.advance()
    chord.rewind()
    chord.discard()
    assert primary.tokens == original.token_ids
    original.observe()
    assert guidance.tokens == [*guidance.tokenize("P", add_bos=True, special=True), *original.visible_token_ids]
    # Switching works from the common prefix, with only path suffixes evaluated.
    assert not any(kind == "reset" and len(tokens) > len(original.token_ids)
                   for kind, tokens in primary.work)
    assert not any(kind == "reset" and len(tokens) > 1
                   for kind, tokens in guidance.work)


@pytest.mark.parametrize("branching", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.invariant
def test_chord_keeps_active_survivor_cache_and_repositions_after_rewind(branching, cfg):
    primary, guidance = BranchBackend(), BranchBackend()
    if not branching:
        primary.branch_to_prefix = guidance.branch_to_prefix = None
    sampling = SamplerConfig(cfg_unconditional_prompt="P", cfg_scale=1.4) if cfg else SamplerConfig()
    original = engine(backend=primary, guidance=guidance if cfg else None, sampling=sampling)
    chord = Chord(original, (4, 1))  # EOG first; the last path remains live.
    primary.work.clear()
    guidance.work.clear()
    assert chord.advance()
    assert primary.work == [("eval", (2,))]
    assert guidance.work == ([("eval", (1,))] if cfg else [])
    assert chord.paths[1].token_ids == [1, 2]

    # Observe EOG so CFG reaches the two-token prefix, then rewind both
    # rounds. Guidance is now ahead of the one-token surviving continuation.
    assert chord.advance()
    assert chord.paths[1].state == "EOG"
    assert chord.rewind()
    assert chord.rewind()
    assert chord.rewind() is False
    assert chord.advance()
    actions = chord.select("b")
    for action in actions:
        original.apply(action)
    manual = engine(backend=BranchBackend(), guidance=BranchBackend() if cfg else None,
                    sampling=sampling)
    for action in actions:
        manual.apply(action)
    assert original.token_ids == manual.token_ids
    assert original.observe().proposal_token_id == manual.observe().proposal_token_id
    if cfg:
        assert guidance.tokens == manual.guidance_backend.tokens


@pytest.mark.current_workflow
def test_chord_shared_context_wrap_is_reused_until_width_changes():
    chord = Chord(engine(), (1, 2))
    with patch("trajectory_editor.chord._recent_context", wraps=_recent_context) as wrap:
        first = chord.display(width=40)
        assert chord.display(width=40) == first
        chord.advance()
        assert chord.display(width=40) != first
        assert wrap.call_count == 1
        chord.display(width=20)
        assert wrap.call_count == 2
    chord.discard()


class DurableFakeBackend(ConformingFakeBackend):
    def provenance(self, *, include_model_sha256=True):
        return {"backend": "llama.cpp", "vocabulary_size": self.vocabulary_size()}


class LiveIO(ScriptedIO):
    @property
    def capabilities(self):
        from trajectory_editor.terminal_contracts import TerminalCapabilities
        return TerminalCapabilities(live_views=True, seamless_review=True)

    def read_choice(self, state):
        return self.read("live choice> ")

    def read_edge(self, state):
        return self.read("live edge> ")

    @contextmanager
    def session(self):
        yield self


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("ephemeral", [False, True])
@pytest.mark.invariant
def test_chord_cli_records_only_survivor_and_export(tmp_path, live, ephemeral):
    workspace = tmp_path / "episode.sqlite3"
    export = tmp_path / "selected.jsonl"
    commands = ["chord 1 2 4", "", "q", "s temperature=0.7", "c", "a", "q"]
    if ephemeral:
        commands += [f"export {export}", "q"]
    else:
        commands += ["q"]
    io = LiveIO(commands) if live else ScriptedIO(commands)
    flags = ["--ephemeral"] if ephemeral else ["--workspace", str(workspace)]
    if not live:
        flags.append("--plain-ui")
    with patch("trajectory_editor.episode_backend_loader.load_backend",
               side_effect=lambda _args: ConformingFakeBackend() if ephemeral else DurableFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--model", "fake", "--new-prompt", "P", *flags]) == 0
    assert any("a  rank 1" in item and "b  rank 2" in item and "c  rank 4" in item
               for item in io.output)
    assert any(
        "a  rank 1  LIVE\n    A B" in request.body
        for request in io.prompt_requests
    )
    assert any("Resolve the chord before changing the episode" in item for item in io.output)
    if ephemeral:
        tape = load_teacher_tape_jsonl(export)
        assert [step.action for step in tape.plan] == [SelectRawRank(1), Accept()]
    else:
        with EpisodeStore(workspace) as store:
            episode_id = store.resolve_id("#1")
            assert [step["action"] for step in replay_procedure(store, episode_id)] == [SelectRawRank(1), Accept()]
            tokens = store.tokens(episode_id)
            assert [token["token_id"] for token in tokens] == [1, 2]
            assert [token["text"] for token in tokens] == [" A", " B"]


@pytest.mark.invariant
def test_chord_discard_cli_keeps_durable_actions_empty(tmp_path):
    workspace = tmp_path / "episode.sqlite3"
    io = ScriptedIO(["chord 1 2", "", "q", "discard", "q", "q"])
    with patch("trajectory_editor.episode_backend_loader.load_backend", return_value=DurableFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--model", "fake", "--plain-ui", "--new-prompt", "P",
                     "--workspace", str(workspace)]) == 0
    with EpisodeStore(workspace) as store:
        episode_id = store.resolve_id("#1")
        assert store.actions(episode_id) == []
        assert store.get_episode(episode_id)["visible_text"] == ""


@pytest.mark.parametrize("choice, expected", [("4", ["select-raw-rank"]),
                                            ("b", ["select-raw-rank", "accept"])])
@pytest.mark.invariant
def test_durable_chord_eog_selection_uses_ordinary_terminal_action(tmp_path, choice, expected):
    workspace = tmp_path / "episode.sqlite3"
    responses = ["chord 1 2 4"]
    if choice == "b":
        responses.append("")
    responses.append(choice)
    io = ScriptedIO(responses)
    with patch("trajectory_editor.episode_backend_loader.load_backend", return_value=DurableFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--model", "fake", "--plain-ui", "--new-prompt", "P",
                     "--workspace", str(workspace)]) == 0
    with EpisodeStore(workspace) as store:
        episode_id = store.resolve_id("#1")
        assert [row["kind"] for row in store.actions(episode_id)] == expected
        assert store.get_episode(episode_id)["terminal_reason"] == "teacher-eog"


@pytest.mark.parametrize("ephemeral", [False, True])
@pytest.mark.invariant
def test_chord_budget_selection_stops_at_checkpoint(tmp_path, ephemeral):
    workspace = tmp_path / "episode.sqlite3"
    io = ScriptedIO(["chord 1 2", "", "a", "q"])
    flags = ["--ephemeral"] if ephemeral else ["--workspace", str(workspace)]
    with patch("trajectory_editor.episode_backend_loader.load_backend",
               return_value=ConformingFakeBackend() if ephemeral else DurableFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--model", "fake", "--plain-ui", "--new-prompt", "P",
                     "--max-tokens", "1", *flags]) == 0
    assert any("BUDGET REACHED" in item for item in io.output)
    if not ephemeral:
        with EpisodeStore(workspace) as store:
            episode_id = store.resolve_id("#1")
            assert [row["kind"] for row in store.actions(episode_id)] == ["select-raw-rank"]
