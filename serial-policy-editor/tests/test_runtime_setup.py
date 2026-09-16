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


def _plan():
    return RuntimePlan()


def test_setup_commands_build_a_new_runtime_selection(tmp_path):
    args = _args()
    plan = _plan()

    assert apply_setup_command('prompt "A careful beginning"', plan) == "continue"
    assert apply_setup_command(f"model {tmp_path / 'model.gguf'}", plan) == "continue"
    assert apply_setup_command("backend llama.cpp", plan) == "continue"
    assert apply_setup_command(f"activation {tmp_path / 'calm.json'}", plan) == "continue"
    assert apply_setup_command("sampler temp=.8 top_k=24", plan) == "continue"
    assert apply_setup_command("budget 32", plan) == "continue"
    assert apply_setup_command("seed 91", plan) == "continue"
    assert apply_setup_command("learning on", plan) == "continue"
    assert apply_setup_command("preference on", plan) == "continue"
    plan.apply_to_args(args)

    assert args.new_prompt == "A careful beginning"
    assert args.model == Path(tmp_path / "model.gguf")
    assert args.backend == "llama.cpp"
    assert args.activation_vector == Path(tmp_path / "calm.json")
    assert args.temperature == 0.8
    assert args.top_k == 24
    assert args.max_tokens == 32
    assert args.seed == 91
    assert not args.random_seed
    assert args.online_learning and args.token_preference
    assert {"model", "backend", "activation_vector", "temperature", "top_k", "max_tokens", "seed"} <= args._explicit_options


def test_setup_source_accepts_workspace_episode_references():
    args = build_parser().parse_args(["--new-prompt", "old", "--temperature", ".6"])
    args._explicit_options = {"new_prompt", "temperature"}
    plan = RuntimePlan.from_args(args)
    store = Store()

    apply_setup_command("source replay #2", plan, store=store)
    assert plan.replay == "#2"
    assert plan.new_prompt is None
    assert "new_prompt" not in plan.explicit_options
    assert "temperature" in plan.explicit_options

    apply_setup_command("source fork #1 at 7", plan, store=store)
    assert plan.fork_from == "#1"
    assert plan.at == 7
    assert plan.replay is None

    plan.apply_to_args(args)
    assert args.fork_from == "#1"
    assert args.temperature == 0.6
    assert "new_prompt" not in args._explicit_options


def test_setup_menu_requires_a_source_before_go(tmp_path):
    args = _args()
    io = ScriptedIO(["go", f'prompt "Prompt"', "go"])

    assert run_runtime_setup_menu(io, args) is True
    assert args.new_prompt == "Prompt"
    assert any("choose a prompt or source" in text for text in io.writes if isinstance(text, str))


def test_setup_menu_can_be_cancelled():
    args = _args()
    io = ScriptedIO(["quit"])

    assert run_runtime_setup_menu(io, args) is False
    assert any("cancelled" in text.lower() for text in io.writes if isinstance(text, str))


def test_setup_summary_shows_effective_selections(tmp_path):
    plan = _plan()
    apply_setup_command('prompt "Prompt"', plan)
    apply_setup_command(f"activation {tmp_path / 'vector.json'}", plan)
    apply_setup_command("sampler temperature=.7", plan)

    rendered = setup_summary(plan)
    assert "PRE-RUNTIME SETUP" in rendered
    assert "new prompt: Prompt" in rendered
    assert str(tmp_path / "vector.json") in rendered
    assert "temperature=0.7" in rendered


def test_setup_menu_can_collect_a_prompt_and_list_episodes(tmp_path):
    args = _args()
    io = ScriptedIO(["prompt", "A prompt from the setup menu", "ls", "go"])

    assert run_runtime_setup_menu(io, args, store=Store()) is True
    assert args.new_prompt == "A prompt from the setup menu"
    assert "episode" in " ".join(str(value) for value in io.writes).lower()


def test_setup_menu_can_select_a_prompt_file(tmp_path):
    args = _args()
    plan = _plan()
    path = tmp_path / "prompt.txt"

    apply_setup_command(f"prompt-file {path}", plan)
    plan.apply_to_args(args)
    assert args.new_prompt is None
    assert args.new_prompt_file == path


def test_sampler_without_arguments_describes_defaults_and_inheritance():
    new_plan = _plan()
    result = apply_setup_command("sampler", new_plan)
    assert result == "show-sampler"
    assert "temperature" in sampler_summary(new_plan)
    assert "1.0" in sampler_summary(new_plan)

    replay_plan = RuntimePlan(replay="#1")
    assert "inherited from source" in sampler_summary(replay_plan)


def test_bare_episode_number_inspects_and_fork_map_is_read_only(monkeypatch):
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


def test_setup_menu_switches_workspace_before_go(tmp_path):
    args = _args()
    store = Store()
    selected = tmp_path / "other.sqlite3"
    io = ScriptedIO([f"workspace {selected}", "prompt New", "go"])

    assert run_runtime_setup_menu(io, args, store=store) is True
    assert store.path == selected
    assert args.workspace == selected
    assert args.new_prompt == "New"
