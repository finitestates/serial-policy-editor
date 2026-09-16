from pathlib import Path

from trajectory_editor.episode_cli import build_parser
from trajectory_editor.runtime_setup import (
    apply_setup_command,
    run_runtime_setup_menu,
    setup_summary,
)


class Store:
    def resolve_id(self, value):
        assert value in {"#1", "#2"}
        return {"#1": "episode-one", "#2": "episode-two"}[value]

    def workspace_list(self, *, include_finished):
        assert include_finished
        return "#1  Example episode  (paused)"


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


def test_setup_commands_build_a_new_runtime_selection(tmp_path):
    args = _args()

    assert apply_setup_command('prompt "A careful beginning"', args) == "continue"
    assert apply_setup_command(f"model {tmp_path / 'model.gguf'}", args) == "continue"
    assert apply_setup_command("backend llama.cpp", args) == "continue"
    assert apply_setup_command(f"activation {tmp_path / 'calm.json'}", args) == "continue"
    assert apply_setup_command("sampler temp=.8 top_k=24", args) == "continue"
    assert apply_setup_command("budget 32", args) == "continue"
    assert apply_setup_command("seed 91", args) == "continue"
    assert apply_setup_command("learning on", args) == "continue"
    assert apply_setup_command("preference on", args) == "continue"

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
    args = _args()
    store = Store()

    apply_setup_command("source replay #2", args, store=store)
    assert args.replay == "#2"
    assert args.new_prompt is None

    apply_setup_command("source fork #1 at 7", args, store=store)
    assert args.fork_from == "#1"
    assert args.at == 7
    assert args.replay is None


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
    args = _args()
    apply_setup_command('prompt "Prompt"', args)
    apply_setup_command(f"activation {tmp_path / 'vector.json'}", args)
    apply_setup_command("sampler temperature=.7", args)

    rendered = setup_summary(args)
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
    path = tmp_path / "prompt.txt"

    apply_setup_command(f"prompt-file {path}", args)
    assert args.new_prompt is None
    assert args.new_prompt_file == path
