from pathlib import Path

from trajectory_editor.episode_cli import build_parser
from trajectory_editor.runtime_setup import (
    RuntimePlan,
    apply_setup_command,
    run_runtime_setup_menu,
    sampler_summary,
    setup_summary,
)


class Store:
    def __init__(self):
        self.path = Path("episodes.sqlite3")

    def resolve_id(self, value):
        assert value in {"1", "2", "#1", "#2"}
        return {"1": "episode-one", "2": "episode-two", "#1": "episode-one", "#2": "episode-two"}[value]

    def get_episode(self, value):
        assert value in {"episode-one", "episode-two"}
        return {
            "status": "paused",
            "parent_episode_id": None,
            "fork_boundary": None,
            "initial_text": "A prompt",
        }

    def tokens(self, value):
        return [{"realized_visible": True}, {"realized_visible": True}]

    def actions(self, value):
        return [{"kind": "accept"}]

    def label(self, value):
        return {"episode-one": "#1  Example episode", "episode-two": "#2  Other episode"}[value]

    def workspace_list(self, *, include_finished):
        assert include_finished
        return "#1  Example episode  (paused)"

    def switch_workspace(self, path):
        self.path = Path(path)


class ScriptedIO:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.writes = []

    def read(self, prompt):
        self.writes.append(prompt)
        return next(self.commands, None)

    def write(self, text="", *, end="\n"):
        self.writes.append(text)

    def page(self, text):
        self.writes.append(text)


def _args():
    args = build_parser().parse_args([])
    args._explicit_options = set()
    return args


def test_setup_commands_build_a_core_runtime_selection(tmp_path):
    args = _args()
    plan = RuntimePlan()

    assert apply_setup_command('prompt "A careful beginning"', plan) == "continue"
    assert apply_setup_command(f"model {tmp_path / 'model.gguf'}", plan) == "continue"
    assert apply_setup_command("backend llama.cpp", plan) == "continue"
    assert apply_setup_command(f"vector {tmp_path / 'calm.json'}", plan) == "continue"
    assert apply_setup_command("sampler temp=.8 top_k=24", plan) == "continue"
    assert apply_setup_command("seed 91", plan) == "continue"
    plan.apply_to_args(args)

    assert args.new_prompt == "A careful beginning"
    assert args.model == tmp_path / "model.gguf"
    assert args.backend == "llama.cpp"
    assert args.activation_vector == tmp_path / "calm.json"
    assert args.temperature == 0.8
    assert args.top_k == 24
    assert args.seed == 91
    assert {"model", "backend", "activation_vector", "temperature", "top_k", "seed"} <= args._explicit_options


def test_setup_menu_requires_a_source_and_supports_cancellation():
    args = _args()
    io = ScriptedIO(["go", 'prompt "Prompt"', "go"])
    assert run_runtime_setup_menu(io, args) is True
    assert args.new_prompt == "Prompt"
    assert any("choose a prompt or source" in text for text in io.writes if isinstance(text, str))

    cancelled = _args()
    cancel_io = ScriptedIO(["quit"])
    assert run_runtime_setup_menu(cancel_io, cancelled) is False


def test_setup_summary_and_sampler_panel_expose_core_surface(tmp_path):
    plan = RuntimePlan()
    apply_setup_command('prompt "Prompt"', plan)
    apply_setup_command(f"steering {tmp_path / 'vector.json'}", plan)
    apply_setup_command("sampler temperature=.7", plan)

    rendered = setup_summary(plan)
    assert "PRE-RUNTIME SETUP" in rendered
    assert "new prompt: Prompt" in rendered
    assert str(tmp_path / "vector.json") in rendered
    assert "temperature=0.7" in rendered
    assert "learning|group" not in rendered
    assert "preference [key=value]" not in rendered
    assert "budget N|off" not in rendered

    replay_plan = RuntimePlan(replay="#1")
    assert "inherited from source" in sampler_summary(replay_plan)


def test_setup_menu_switches_workspace_and_accepts_prompt_file(tmp_path):
    args = _args()
    store = Store()
    selected = tmp_path / "other.sqlite3"
    prompt_file = tmp_path / "prompt.txt"
    io = ScriptedIO([f"workspace {selected}", f"prompt-file {prompt_file}", "go"])

    assert run_runtime_setup_menu(io, args, store=store) is True
    assert store.path == selected
    assert args.workspace == selected
    assert args.new_prompt_file == prompt_file


def test_setup_menu_episode_inspection_and_fork_map_are_read_only(monkeypatch):
    from trajectory_editor import episode_projector

    monkeypatch.setattr(
        episode_projector,
        "project_fork_map",
        lambda store, episode_id: f"FORK MAP FOR {episode_id}",
    )
    args = _args()
    io = ScriptedIO(["#1", "fm #1", "prompt New", "go"])

    assert run_runtime_setup_menu(io, args, store=Store()) is True
    assert any("EPISODE #1" in text for text in io.writes if isinstance(text, str))
    assert any("FORK MAP FOR episode-one" in text for text in io.writes if isinstance(text, str))
    assert args.new_prompt == "New"


def test_runtime_plan_round_trips_core_paths_and_explicit_choices(tmp_path):
    plan = RuntimePlan(
        workspace=tmp_path / "work.sqlite3",
        model=tmp_path / "model.gguf",
        replay="#2",
        activation_vector=tmp_path / "vector.json",
        temperature=.8,
        top_k=17,
        explicit_options={"workspace", "model", "replay", "activation_vector", "temperature", "top_k"},
    )

    restored = RuntimePlan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.workspace == tmp_path / "work.sqlite3"
    assert restored.model == tmp_path / "model.gguf"
    assert restored.activation_vector == tmp_path / "vector.json"
